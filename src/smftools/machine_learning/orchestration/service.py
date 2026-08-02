"""Atomic lifecycle execution for framework-independent ML job operations."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, TypeVar

from smftools.logging_utils import get_logger
from smftools.readwrite import atomic_write_json

from ..artifacts import (
    ArtifactReference,
    FailureRecord,
    RunManifest,
    file_sha256,
    new_run_id,
    publish_bundle,
)
from .contracts import (
    JobArtifact,
    JobCancellationToken,
    JobDryRun,
    JobExecutionContext,
    JobExecutionOutcome,
    JobOperationResult,
    MLJobCancellationRequested,
    MLJobCancelledError,
    MLJobExecutionError,
    MLJobServiceError,
    ResolvedJob,
    _thaw_json,
)

T = TypeVar("T")
_LOGGER = get_logger(__name__)
_ACTION_SERVICES = frozenset({"train", "apply", "evaluate", "explain", "plot"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reference(role: str, relative_path: str, source: Path, media_type: str) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=relative_path,
        sha256=file_sha256(source),
        size_bytes=source.stat().st_size,
        media_type=media_type,
    )


def _log_state(job: ResolvedJob, run_id: str, state: str, *, phase: str) -> None:
    _LOGGER.info(
        "ML job state transition",
        extra={
            "ml_action": job.job.action,
            "ml_job_name": job.job_name,
            "ml_run_id": run_id,
            "ml_state": state,
            "ml_phase": phase,
        },
    )


def _work_root(job: ResolvedJob, run_id: str) -> Path:
    if job.workspace.runs_root.is_symlink():
        raise MLJobServiceError("runs root cannot be a symbolic link")
    job.workspace.runs_root.mkdir(parents=True, exist_ok=True)
    work_parent = job.workspace.runs_root / ".work"
    if work_parent.is_symlink():
        raise MLJobServiceError("run staging parent cannot be a symbolic link")
    work_parent.mkdir(parents=True, exist_ok=True)
    try:
        work_parent.resolve().relative_to(job.workspace.root)
    except ValueError as exc:
        raise MLJobServiceError("run staging parent escapes the active ML workspace") from exc
    path = Path(mkdtemp(prefix=f"{run_id}.", dir=work_parent)).resolve()
    try:
        path.relative_to(job.workspace.root)
    except ValueError as exc:
        raise MLJobServiceError("run staging root escapes the active ML workspace") from exc
    return path


def _base_sources(
    job: ResolvedJob, root: Path
) -> tuple[ArtifactReference, ArtifactReference, dict]:
    plan_path = atomic_write_json(root / "resolved_plan.json", job.plan.to_dict())
    config_path = atomic_write_json(root / "resolved_config.json", _thaw_json(job.resolved_config))
    plan_reference = _reference(
        "resolved_plan", "resolved_plan.json", plan_path, "application/json"
    )
    config_reference = _reference(
        "resolved_config", "resolved_config.json", config_path, "application/json"
    )
    return (
        plan_reference,
        config_reference,
        {
            plan_reference.relative_path: plan_path,
            config_reference.relative_path: config_path,
        },
    )


def _planned_manifest(
    job: ResolvedJob,
    run_id: str,
    plan_reference: ArtifactReference,
    config_reference: ArtifactReference,
    created_at: str,
) -> RunManifest:
    return RunManifest.create(
        run_id=run_id,
        workspace_id=job.workspace.workspace_id,
        action=job.job.action,
        job_name=job.job_name,
        plan_hash=job.plan.plan_hash,
        resolved_plan=plan_reference,
        resolved_config=config_reference,
        dataset_snapshot_id=job.dataset_snapshot_id,
        split_id=job.split_id,
        model_keys=job.job.models if job.job.action == "train" else (),
        source_model_ids=job.source_model_ids,
        source_run_ids=job.source_run_ids,
        environment=job.environment,
        seeds=job.seeds,
        device=job.device,
        created_at=created_at,
    )


def _artifact_sources(
    root: Path,
    artifacts: tuple[JobArtifact, ...],
) -> tuple[tuple[ArtifactReference, ...], dict[str, Path]]:
    references: list[ArtifactReference] = []
    sources: dict[str, Path] = {}
    roles: set[str] = set()
    for artifact in artifacts:
        if artifact.role in roles:
            raise MLJobServiceError(f"duplicate run artifact role {artifact.role!r}")
        roles.add(artifact.role)
        source = (root / artifact.relative_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise MLJobServiceError("run artifact escapes the active staging root") from exc
        if source.is_symlink() or not source.is_file():
            raise MLJobServiceError(
                f"run artifact {artifact.relative_path!r} is missing or not a regular file"
            )
        reference = _reference(
            artifact.role,
            artifact.relative_path,
            source,
            artifact.media_type,
        )
        references.append(reference)
        sources[reference.relative_path] = source
    return tuple(references), sources


def _outcome(
    job: ResolvedJob,
    manifest: RunManifest,
    sources: Mapping[str, Path],
    history: tuple[str, ...],
    value: T | None,
) -> JobExecutionOutcome[T]:
    bundle = publish_bundle(job.workspace, manifest, sources=sources)
    return JobExecutionOutcome(
        manifest=manifest,
        bundle=bundle,
        state_history=history,
        value=value,
    )


def _failure_message(error: BaseException) -> str:
    message = " ".join(str(error).split())[:2000]
    return message or "operation failed without an error message"


def dry_run_job(job: ResolvedJob, *, run_id: str | None = None) -> JobDryRun:
    """Resolve intended paths and immutable inputs without creating files."""
    resolved_run_id = run_id or new_run_id()
    return JobDryRun(
        run_id=resolved_run_id,
        action=job.job.action,
        job_name=job.job_name,
        workspace=job.workspace.to_job_dry_run_dict(
            action=job.job.action,
            run_id=resolved_run_id,
        ),
        model_selections=tuple(selection.to_dict() for selection in job.model_selections),
        source_run_ids=job.source_run_ids,
    )


def _execute_action(
    expected_action: str,
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    if expected_action not in _ACTION_SERVICES:
        raise MLJobServiceError(f"unsupported job service action {expected_action!r}")
    if job.job.action != expected_action:
        raise MLJobServiceError(
            f"{expected_action} service cannot execute a {job.job.action!r} plan job"
        )
    token = cancellation or JobCancellationToken()
    run_id = new_run_id()
    root = _work_root(job, run_id)
    context = JobExecutionContext(
        run_id=run_id,
        action=expected_action,
        staging_root=root,
        cancellation=token,
    )
    try:
        plan_reference, config_reference, sources = _base_sources(job, root)
        manifest = _planned_manifest(
            job,
            run_id,
            plan_reference,
            config_reference,
            clock(),
        )
        history = ("planned",)
        _log_state(job, run_id, "planned", phase="preflight")
        try:
            context.raise_if_cancelled()
            manifest = manifest.transition("running", at=clock())
            history += ("running",)
            _log_state(job, run_id, "running", phase=context.phase)
            result = operation(context)
            if not isinstance(result, JobOperationResult):
                raise MLJobServiceError("job operation must return JobOperationResult")
            context.raise_if_cancelled()
            references, result_sources = _artifact_sources(root, result.artifacts)
            sources.update(result_sources)
            manifest = manifest.transition("completed", at=clock(), artifacts=references)
            history += ("completed",)
            _log_state(job, run_id, "completed", phase=context.phase)
            return _outcome(job, manifest, sources, history, result.value)
        except MLJobCancellationRequested as error:
            failure = FailureRecord(
                error_type=type(error).__name__,
                message=_failure_message(error),
                phase=context.phase,
            )
            manifest = manifest.transition("cancelled", at=clock(), failure=failure)
            history += ("cancelled",)
            _log_state(job, run_id, "cancelled", phase=context.phase)
            outcome = _outcome(job, manifest, sources, history, None)
            raise MLJobCancelledError(outcome) from error
        except KeyboardInterrupt as error:
            failure = FailureRecord(
                error_type="KeyboardInterrupt",
                message="execution interrupted by user",
                phase=context.phase,
            )
            manifest = manifest.transition("cancelled", at=clock(), failure=failure)
            history += ("cancelled",)
            _log_state(job, run_id, "cancelled", phase=context.phase)
            _outcome(job, manifest, sources, history, None)
            raise
        except Exception as error:
            failure = FailureRecord(
                error_type=type(error).__name__,
                message=_failure_message(error),
                phase=context.phase,
            )
            manifest = manifest.transition("failed", at=clock(), failure=failure)
            history += ("failed",)
            _LOGGER.error(
                "ML job failed",
                extra={
                    "ml_action": job.job.action,
                    "ml_job_name": job.job_name,
                    "ml_run_id": run_id,
                    "ml_state": "failed",
                    "ml_phase": context.phase,
                    "ml_error_type": type(error).__name__,
                },
            )
            outcome = _outcome(job, manifest, sources, history, None)
            raise MLJobExecutionError(outcome) from error
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.parent.rmdir()
        except OSError:
            pass


def run_train_job(
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    """Execute one train operation with immutable lifecycle publication."""
    return _execute_action("train", job, operation, cancellation=cancellation, clock=clock)


def run_apply_job(
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    """Execute one model-application operation without a training code path."""
    return _execute_action("apply", job, operation, cancellation=cancellation, clock=clock)


def run_evaluate_job(
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    """Execute evaluation over stored predictions without model fitting."""
    return _execute_action("evaluate", job, operation, cancellation=cancellation, clock=clock)


def run_explain_job(
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    """Execute explanation over one immutable fitted model without training."""
    return _execute_action("explain", job, operation, cancellation=cancellation, clock=clock)


def run_plot_job(
    job: ResolvedJob,
    operation: Callable[[JobExecutionContext], JobOperationResult[T]],
    *,
    cancellation: JobCancellationToken | None = None,
    clock: Callable[[], str] = _now,
) -> JobExecutionOutcome[T]:
    """Execute plot/compare work from immutable source runs and contained paths."""
    return _execute_action("plot", job, operation, cancellation=cancellation, clock=clock)
