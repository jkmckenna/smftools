"""Tests for backend-neutral ML job resolution and lifecycle services."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pytest

from smftools.machine_learning.artifacts import (
    ArtifactReference,
    EnvironmentRecord,
    ModelLineage,
    ModelManifest,
    ResolvedDefinition,
    SerializationPolicy,
    file_sha256,
    publish_bundle,
)
from smftools.machine_learning.evaluation import PredictionResult
from smftools.machine_learning.orchestration import (
    JobArtifact,
    JobCancellationToken,
    JobOperationResult,
    MLJobCancelledError,
    MLJobExecutionError,
    MLJobServiceError,
    ModelMetricCandidate,
    ModelSelectionRequest,
    ResolvedJob,
    ResolvedModelSelection,
    dry_run_job,
    evaluate_prediction_result,
    resolve_model_selection,
    run_apply_job,
    run_evaluate_job,
    run_explain_job,
    run_plot_job,
    run_train_job,
)
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.workspace import MLWorkspace
from smftools.readwrite import atomic_write_json

pytestmark = pytest.mark.unit

DATASET_ID = "d" * 64
SPLIT_ID = "e" * 64
NOW = "2026-08-01T12:00:00Z"


def _plan():
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                },
                "new_reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                },
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample"],
                    "train_groups": ["sample-a"],
                    "validation_groups": ["sample-b"],
                    "test_groups": ["sample-c"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                },
                "apply": {"action": "apply", "dataset": "new_reads", "model": "nb"},
                "evaluate": {
                    "action": "evaluate",
                    "dataset": "reads",
                    "source_job": "train",
                    "evaluate": ["validation"],
                },
                "explain": {
                    "action": "explain",
                    "dataset": "reads",
                    "model": "nb",
                    "source_job": "train",
                    "explain": ["native"],
                },
                "plot": {"action": "plot", "runs": ["train"], "plots": ["roc_pr"]},
            },
        }
    )


def _workspace(tmp_path: Path) -> MLWorkspace:
    return MLWorkspace(
        scope_kind="project",
        scope_id="project-a",
        owner_root=tmp_path,
        root=tmp_path / "project_outputs" / "ml",
    )


def _environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        smftools_version="2.20.0.dev0",
        python_version="3.12.4",
        platform="test",
        code_revision="abc123",
        dirty_tree=False,
        dependencies={"numpy": "2.0"},
    )


def _model_manifest(
    tmp_path: Path,
    workspace: MLWorkspace,
    *,
    source_run_id: str,
    content: bytes = b"model-a",
) -> ModelManifest:
    source = tmp_path / f"{uuid.uuid4().hex}.bin"
    source.write_bytes(content)
    reference = ArtifactReference(
        role="model",
        relative_path="payload/model.bin",
        sha256=file_sha256(source),
        size_bytes=source.stat().st_size,
        media_type="application/octet-stream",
    )
    manifest = ModelManifest.create(
        model_key="nb",
        backend="sklearn",
        family="bernoulli_nb",
        task_type="binary_classification",
        originating_run_id=source_run_id,
        workspace_id=workspace.workspace_id,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        input_schema_hash="a" * 64,
        label_schema_hash="b" * 64,
        architecture=ResolvedDefinition.create(
            name="bernoulli_nb",
            version="1",
            parameters={"alpha": 1.0},
        ),
        lineage=ModelLineage(
            kind="from_scratch",
            parent_model_ids=(),
            parent_roles=(),
        ),
        artifact=reference,
        serialization=SerializationPolicy(
            format="opaque-test",
            loader="test.loader",
            requires_unsafe_load=False,
            allowed_types=(),
            package_versions={"test": "1"},
        ),
        environment=_environment(),
        created_at=NOW,
    )
    publish_bundle(workspace, manifest, sources={reference.relative_path: source})
    return manifest


def _resolved_job(
    tmp_path: Path,
    job_name: str,
    *,
    selection: ResolvedModelSelection | None = None,
    source_run_id: str | None = None,
) -> ResolvedJob:
    plan = _plan()
    workspace = _workspace(tmp_path)
    action = plan.jobs[job_name].action
    return ResolvedJob(
        plan=plan,
        workspace=workspace,
        job_name=job_name,
        environment=_environment(),
        resolved_config={"job": job_name, "action": action},
        dataset_snapshot_id=(
            DATASET_ID if action in {"train", "apply", "evaluate", "explain"} else None
        ),
        split_id=SPLIT_ID if action == "train" else None,
        model_selections=() if selection is None else (selection,),
        source_run_ids=(source_run_id,) if source_run_id is not None else (),
        seeds={"model": 7},
        device="cpu",
    )


def _selection(tmp_path: Path) -> ResolvedModelSelection:
    workspace = _workspace(tmp_path)
    manifest = _model_manifest(
        tmp_path,
        workspace,
        source_run_id=str(uuid.uuid4()),
    )
    return ResolvedModelSelection(manifest=manifest, selection_kind="exact")


def test_dry_run_is_read_only_and_records_immutable_model_resolution(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    job = _resolved_job(tmp_path, "apply", selection=selection)
    runs_root = job.workspace.runs_root

    report = dry_run_job(job)

    assert report.action == "apply"
    assert report.model_selections[0]["model_id"] == selection.model_id
    assert report.workspace["run"]["root"].endswith(report.run_id)
    assert not runs_root.exists()


def test_repeated_apply_attempts_publish_distinct_completed_runs(tmp_path: Path) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))

    def operation(context):
        path = context.output_path("predictions/inference.json")
        atomic_write_json(path, {"run_id": context.run_id})
        return JobOperationResult(
            value=context.run_id,
            artifacts=(
                JobArtifact(
                    role="predictions:inference",
                    relative_path="predictions/inference.json",
                    media_type="application/json",
                ),
            ),
        )

    first = run_apply_job(job, operation)
    second = run_apply_job(job, operation)

    assert first.manifest.run_id != second.manifest.run_id
    assert first.state_history == ("planned", "running", "completed")
    assert first.manifest.state == second.manifest.state == "completed"
    assert first.value == first.manifest.run_id
    assert (first.bundle.path / "predictions" / "inference.json").is_file()
    assert first.manifest.source_model_ids == job.source_model_ids


def test_lifecycle_emits_structured_package_log_records(tmp_path: Path, caplog) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))
    caplog.set_level(
        logging.INFO,
        logger="smftools.machine_learning.orchestration.service",
    )

    outcome = run_apply_job(job, lambda _context: JobOperationResult(value=None))

    states = [
        record.ml_state
        for record in caplog.records
        if getattr(record, "ml_run_id", None) == outcome.manifest.run_id
    ]
    assert states == ["planned", "running", "completed"]


def test_failure_publishes_diagnostic_run_without_completed_marker(tmp_path: Path) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))

    def fail(context):
        context.advance_phase("prediction")
        raise ValueError("synthetic failure")

    with pytest.raises(MLJobExecutionError) as caught:
        run_apply_job(job, fail)

    outcome = caught.value.outcome
    assert outcome.manifest.state == "failed"
    assert outcome.state_history == ("planned", "running", "failed")
    assert outcome.manifest.failure.phase == "prediction"
    assert outcome.manifest.failure.error_type == "ValueError"
    with (outcome.bundle.path / "run_manifest.json").open(encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["state"] == "failed"
    assert stored["finished_at"] is not None


def test_prestart_cancellation_publishes_cancelled_run_and_skips_operation(
    tmp_path: Path,
) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))
    token = JobCancellationToken()
    token.cancel()
    called = False

    def operation(_context):
        nonlocal called
        called = True
        return JobOperationResult(value=None)

    with pytest.raises(MLJobCancelledError) as caught:
        run_apply_job(job, operation, cancellation=token)

    assert called is False
    assert caught.value.outcome.manifest.state == "cancelled"
    assert caught.value.outcome.state_history == ("planned", "cancelled")
    assert caught.value.outcome.manifest.started_at is None


def test_keyboard_interrupt_publishes_cancelled_run_then_propagates(tmp_path: Path) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))

    def interrupt(context):
        context.advance_phase("prediction")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_apply_job(job, interrupt)

    runs = [path for path in job.workspace.runs_root.iterdir() if not path.name.startswith(".")]
    assert len(runs) == 1
    with (runs[0] / "run_manifest.json").open(encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["state"] == "cancelled"
    assert stored["failure"]["phase"] == "prediction"


def test_operation_output_cannot_escape_active_run_workspace(tmp_path: Path) -> None:
    job = _resolved_job(tmp_path, "apply", selection=_selection(tmp_path))

    def escape(context):
        context.output_path("../outside.json")
        return JobOperationResult(value=None)

    with pytest.raises(MLJobExecutionError) as caught:
        run_apply_job(job, escape)

    assert isinstance(caught.value.__cause__, MLJobServiceError)
    assert not (job.workspace.root / "outside.json").exists()


def test_action_specific_service_rejects_wrong_plan_action(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    job = _resolved_job(tmp_path, "evaluate", selection=selection)

    with pytest.raises(MLJobServiceError, match="cannot execute"):
        run_apply_job(job, lambda _context: JobOperationResult(value=None))
    assert not job.workspace.runs_root.exists()


@pytest.mark.parametrize(
    ("job_name", "runner"),
    [
        ("train", run_train_job),
        ("apply", run_apply_job),
        ("evaluate", run_evaluate_job),
        ("explain", run_explain_job),
        ("plot", run_plot_job),
    ],
)
def test_every_action_service_is_callable_without_click(
    tmp_path: Path,
    job_name: str,
    runner,
) -> None:
    if job_name == "train":
        job = _resolved_job(tmp_path, job_name)
    elif job_name == "plot":
        job = _resolved_job(tmp_path, job_name, source_run_id=str(uuid.uuid4()))
    else:
        job = _resolved_job(tmp_path, job_name, selection=_selection(tmp_path))

    outcome = runner(job, lambda context: JobOperationResult(value=context.action))

    assert outcome.value == job_name
    assert outcome.manifest.action == job_name
    assert outcome.manifest.state == "completed"


def test_best_from_run_resolves_validation_metric_once_and_deterministically(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    source_run_id = str(uuid.uuid4())
    lower = _model_manifest(
        tmp_path,
        workspace,
        source_run_id=source_run_id,
        content=b"lower",
    )
    higher = _model_manifest(
        tmp_path,
        workspace,
        source_run_id=source_run_id,
        content=b"higher",
    )
    request = ModelSelectionRequest(
        kind="best_from_run",
        source_run_id=source_run_id,
        metric_name="roc_auc",
        direction="maximize",
        cohort="validation-natural",
    )

    resolved = resolve_model_selection(
        workspace,
        request,
        candidates=(
            ModelMetricCandidate(
                model_id=lower.model_id,
                source_run_id=source_run_id,
                metric_name="roc_auc",
                metric_value=0.7,
                cohort="validation-natural",
            ),
            ModelMetricCandidate(
                model_id=higher.model_id,
                source_run_id=source_run_id,
                metric_name="roc_auc",
                metric_value=0.9,
                cohort="validation-natural",
            ),
        ),
    )

    assert resolved.model_id == higher.model_id
    assert resolved.metric_value == 0.9
    assert resolved.metric_direction == "maximize"


def test_evaluation_service_consumes_predictions_without_training(monkeypatch) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("evaluation must not train")

    monkeypatch.setattr(
        "smftools.machine_learning.orchestration.actions.fit_sklearn_partition_model",
        fail_training,
    )
    monkeypatch.setattr(
        "smftools.machine_learning.orchestration.actions.fit_torch_partition_model",
        fail_training,
    )
    predictions = PredictionResult(
        molecule_uids=("m0", "m1", "m2", "m3"),
        class_ids=np.asarray([0, 1, 0, 1]),
        scores=np.log(np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])),
        probabilities=np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
        class_order=("inactive", "active"),
        split="validation",
        experiment_uids=("exp",) * 4,
        modalities=("deaminase",) * 4,
        truth_class_ids=np.asarray([0, 1, 0, 1]),
        positive_class="active",
        cohort="validation-natural",
        model_id="f" * 64,
    )

    result = evaluate_prediction_result(predictions)

    assert result.predictions is predictions
    assert any(metric.name == "accuracy" for metric in result.metrics)
