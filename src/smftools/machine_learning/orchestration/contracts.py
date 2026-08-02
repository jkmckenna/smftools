"""Typed, framework-independent contracts for machine-learning job services."""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from ..artifacts import EnvironmentRecord, ModelManifest, PublishedBundle, RunManifest
from ..plan import JobSpec, MLPlan
from ..workspace import MLWorkspace

T = TypeVar("T")
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_RESERVED_RUN_PAYLOADS = frozenset(
    {"run_manifest.json", "resolved_plan.json", "resolved_config.json"}
)


class MLJobServiceError(RuntimeError):
    """Raised when a resolved job cannot be planned or executed safely."""


class MLJobCancellationRequested(MLJobServiceError):
    """Signal cooperative cancellation from within a running operation."""


class MLJobExecutionError(MLJobServiceError):
    """Expose the published failed-run record while preserving the original cause."""

    def __init__(self, outcome: JobExecutionOutcome[Any]):
        self.outcome = outcome
        super().__init__(f"ML job failed; diagnostic run {outcome.manifest.run_id} was published")


class MLJobCancelledError(MLJobServiceError):
    """Expose the published cancelled-run record to the caller."""

    def __init__(self, outcome: JobExecutionOutcome[Any]):
        self.outcome = outcome
        super().__init__(f"ML job {outcome.manifest.run_id} was cancelled")


def _nonempty(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MLJobServiceError(f"{path} must be a non-empty string")
    return value.strip()


def _digest(value: str, path: str) -> str:
    value = _nonempty(value, path)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MLJobServiceError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _portable_payload_path(value: str) -> str:
    value = _nonempty(value, "artifact.relative_path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) == 0
        or "\\" in value
        or "://" in value
        or value in _RESERVED_RUN_PAYLOADS
    ):
        raise MLJobServiceError("artifact.relative_path must be a non-reserved portable child path")
    return path.as_posix()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers from orchestration's immutable values."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MLJobServiceError(f"{path} must be a mapping with string keys")
    try:
        payload = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise MLJobServiceError(f"{path} must contain only finite JSON values") from exc
    return _freeze_json(payload)


@dataclass(frozen=True)
class ModelMetricCandidate:
    """One immutable model and held-out metric eligible for explicit selection."""

    model_id: str
    source_run_id: str
    metric_name: str
    metric_value: float
    split: str = "validation"
    cohort: str | None = None

    def __post_init__(self) -> None:
        _digest(self.model_id, "candidate.model_id")
        try:
            uuid.UUID(self.source_run_id)
        except (TypeError, ValueError) as exc:
            raise MLJobServiceError("candidate.source_run_id must be a UUID") from exc
        _nonempty(self.metric_name, "candidate.metric_name")
        if isinstance(self.metric_value, bool) or not isinstance(self.metric_value, (int, float)):
            raise MLJobServiceError("candidate.metric_value must be numeric")
        if not math.isfinite(float(self.metric_value)):
            raise MLJobServiceError("candidate.metric_value must be finite")
        if self.split != "validation":
            raise MLJobServiceError("model selection metrics must come from validation")
        if self.cohort is not None:
            _nonempty(self.cohort, "candidate.cohort")


@dataclass(frozen=True)
class ModelSelectionRequest:
    """Request exact, alias, or validation-metric model resolution."""

    kind: str
    model_id: str | None = None
    alias: str | None = None
    source_run_id: str | None = None
    metric_name: str | None = None
    direction: str | None = None
    cohort: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"exact", "alias", "best_from_run"}:
            raise MLJobServiceError("selection.kind must be 'exact', 'alias', or 'best_from_run'")
        required = {
            "exact": {"model_id"},
            "alias": {"alias"},
            "best_from_run": {"source_run_id", "metric_name", "direction"},
        }[self.kind]
        present = {
            name
            for name in ("model_id", "alias", "source_run_id", "metric_name", "direction")
            if getattr(self, name) is not None
        }
        if present != required:
            raise MLJobServiceError(
                f"{self.kind} selection fields must be exactly {sorted(required)}"
            )
        for name in present:
            _nonempty(getattr(self, name), f"selection.{name}")
        if self.model_id is not None:
            _digest(self.model_id, "selection.model_id")
        if self.direction is not None and self.direction not in {"maximize", "minimize"}:
            raise MLJobServiceError("selection.direction must be 'maximize' or 'minimize'")
        if self.source_run_id is not None:
            try:
                uuid.UUID(self.source_run_id)
            except (TypeError, ValueError) as exc:
                raise MLJobServiceError("selection.source_run_id must be a UUID") from exc
        if self.kind != "best_from_run" and self.cohort is not None:
            raise MLJobServiceError("selection.cohort is only valid for best_from_run")
        if self.cohort is not None:
            _nonempty(self.cohort, "selection.cohort")


@dataclass(frozen=True)
class ResolvedModelSelection:
    """One selector resolved once to a validated immutable model manifest."""

    manifest: ModelManifest
    selection_kind: str
    metric_name: str | None = None
    metric_value: float | None = None
    metric_direction: str | None = None
    metric_cohort: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ModelManifest):
            raise MLJobServiceError("resolved selection requires a ModelManifest")
        if self.selection_kind not in {"exact", "alias", "best_from_run"}:
            raise MLJobServiceError("resolved selection kind is invalid")
        metric_values = (self.metric_name, self.metric_value, self.metric_direction)
        if self.selection_kind == "best_from_run":
            if any(value is None for value in metric_values):
                raise MLJobServiceError("best-from-run resolution requires metric provenance")
            if self.metric_direction not in {"maximize", "minimize"}:
                raise MLJobServiceError("resolved metric direction is invalid")
            if isinstance(self.metric_value, bool) or not isinstance(
                self.metric_value, (int, float)
            ):
                raise MLJobServiceError("resolved metric value must be numeric")
            if not math.isfinite(float(self.metric_value)):
                raise MLJobServiceError("resolved metric value must be finite")
            if self.metric_cohort is not None:
                _nonempty(self.metric_cohort, "resolved.metric_cohort")
        elif any(value is not None for value in (*metric_values, self.metric_cohort)):
            raise MLJobServiceError("exact and alias resolutions cannot contain metric provenance")

    @property
    def model_id(self) -> str:
        """Return the immutable model identity selected for execution."""
        return self.manifest.model_id

    def to_dict(self) -> dict[str, Any]:
        """Return display-safe immutable resolution provenance."""
        return {
            "model_id": self.model_id,
            "model_key": self.manifest.model_key,
            "backend": self.manifest.backend,
            "family": self.manifest.family,
            "selection_kind": self.selection_kind,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_direction": self.metric_direction,
            "metric_cohort": self.metric_cohort,
        }


@dataclass(frozen=True)
class ResolvedJob:
    """Complete runtime metadata for one plan job in one resolved workspace."""

    plan: MLPlan
    workspace: MLWorkspace
    job_name: str
    environment: EnvironmentRecord
    resolved_config: Mapping[str, Any]
    dataset_snapshot_id: str | None = None
    split_id: str | None = None
    model_selections: tuple[ResolvedModelSelection, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    seeds: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MLPlan):
            raise MLJobServiceError("plan must be a resolved MLPlan")
        if not isinstance(self.workspace, MLWorkspace):
            raise MLJobServiceError("workspace must be a resolved MLWorkspace")
        if not isinstance(self.environment, EnvironmentRecord):
            raise MLJobServiceError("environment must be an EnvironmentRecord")
        job_name = _nonempty(self.job_name, "job_name")
        if job_name not in self.plan.jobs:
            raise MLJobServiceError(f"job_name {job_name!r} is not declared by the ML plan")
        if self.plan.scope.kind != self.workspace.scope_kind:
            raise MLJobServiceError("ML plan scope differs from the resolved workspace")
        object.__setattr__(self, "job_name", job_name)
        object.__setattr__(self, "resolved_config", _json_mapping(self.resolved_config, "config"))
        if not isinstance(self.seeds, Mapping) or not all(
            isinstance(name, str)
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed >= 0
            for name, seed in self.seeds.items()
        ):
            raise MLJobServiceError("seeds must map names to non-negative integers")
        object.__setattr__(self, "seeds", MappingProxyType(dict(sorted(self.seeds.items()))))
        object.__setattr__(self, "device", _nonempty(self.device, "device"))
        if not isinstance(self.model_selections, tuple) or not all(
            isinstance(selection, ResolvedModelSelection) for selection in self.model_selections
        ):
            raise MLJobServiceError("model_selections must contain resolved model selections")
        if not isinstance(self.source_run_ids, tuple):
            raise MLJobServiceError("source_run_ids must be a tuple")
        job = self.plan.jobs[job_name]
        if job.action in {"train", "apply", "evaluate", "explain"}:
            if self.dataset_snapshot_id is None:
                raise MLJobServiceError(f"dataset_snapshot_id is required for {job.action}")
            _digest(self.dataset_snapshot_id, "dataset_snapshot_id")
        elif self.dataset_snapshot_id is not None:
            _digest(self.dataset_snapshot_id, "dataset_snapshot_id")
        if self.split_id is not None:
            _digest(self.split_id, "split_id")
        if job.action == "train":
            if self.split_id is None:
                raise MLJobServiceError("split_id is required for train")
            if self.model_selections or self.source_run_ids:
                raise MLJobServiceError("train jobs cannot consume fitted model or run selections")
        elif job.action in {"apply", "evaluate", "explain"}:
            if len(self.model_selections) != 1:
                raise MLJobServiceError(f"{job.action} requires exactly one resolved model")
            selection = self.model_selections[0]
            if selection.manifest.workspace_id != self.workspace.workspace_id:
                raise MLJobServiceError("resolved model belongs to a different ML workspace")
            if job.model is not None:
                if job.model.startswith("model:"):
                    if selection.model_id != job.model.removeprefix("model:"):
                        raise MLJobServiceError("resolved model ID differs from the plan job")
                elif selection.manifest.model_key != job.model:
                    raise MLJobServiceError("resolved model key differs from the plan job")
            if self.source_run_ids:
                raise MLJobServiceError(f"{job.action} source runs come from model provenance")
        elif job.action == "plot":
            if self.dataset_snapshot_id is not None or self.split_id is not None:
                raise MLJobServiceError("plot jobs cannot bind a new dataset or split")
            if self.model_selections:
                raise MLJobServiceError("plot jobs consume immutable source runs, not models")
            if not self.source_run_ids:
                raise MLJobServiceError("plot jobs require immutable source run IDs")
        normalized_run_ids: list[str] = []
        for run_id in self.source_run_ids:
            try:
                normalized_run_ids.append(str(uuid.UUID(run_id)))
            except (TypeError, ValueError) as exc:
                raise MLJobServiceError("source_run_ids must contain UUIDs") from exc
        if len(set(normalized_run_ids)) != len(normalized_run_ids):
            raise MLJobServiceError("source_run_ids cannot contain duplicates")
        object.__setattr__(self, "source_run_ids", tuple(normalized_run_ids))

    @property
    def job(self) -> JobSpec:
        """Return the immutable plan job declaration."""
        return self.plan.jobs[self.job_name]

    @property
    def source_model_ids(self) -> tuple[str, ...]:
        """Return immutable model IDs selected before execution."""
        return tuple(selection.model_id for selection in self.model_selections)


class JobCancellationToken:
    """Thread-safe cooperative cancellation signal for bounded job phases."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation at the next explicit boundary."""
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise the cooperative cancellation signal when requested."""
        if self.cancelled:
            raise MLJobCancellationRequested("cancellation requested")


@dataclass(frozen=True)
class JobArtifact:
    """One operation payload to include in the terminal immutable run bundle."""

    role: str
    relative_path: str
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty(self.role, "artifact.role"))
        object.__setattr__(self, "relative_path", _portable_payload_path(self.relative_path))
        object.__setattr__(
            self,
            "media_type",
            _nonempty(self.media_type, "artifact.media_type"),
        )


@dataclass(frozen=True)
class JobOperationResult(Generic[T]):
    """In-memory result plus staged files selected for immutable publication."""

    value: T
    artifacts: tuple[JobArtifact, ...] = ()


class JobExecutionContext:
    """Contained run staging paths, phase tracking, and cancellation boundary."""

    def __init__(
        self,
        *,
        run_id: str,
        action: str,
        staging_root: Path,
        cancellation: JobCancellationToken,
    ) -> None:
        self.run_id = run_id
        self.action = action
        self._staging_root = staging_root.resolve()
        self._cancellation = cancellation
        self._phase = action

    @property
    def phase(self) -> str:
        """Return the current diagnostic execution phase."""
        return self._phase

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancellation.cancelled

    def raise_if_cancelled(self) -> None:
        """Stop at a cooperative cancellation boundary."""
        self._cancellation.raise_if_cancelled()

    def advance_phase(self, phase: str) -> None:
        """Record a concise diagnostic phase and check cancellation."""
        self._phase = _nonempty(phase, "phase")
        self.raise_if_cancelled()

    def output_path(self, relative_path: str) -> Path:
        """Resolve a contained non-reserved staging path for an operation output."""
        relative = _portable_payload_path(relative_path)
        path = (self._staging_root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            path.relative_to(self._staging_root)
        except ValueError as exc:
            raise MLJobServiceError("operation output escapes the active run staging root") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class JobExecutionOutcome(Generic[T]):
    """Terminal run manifest, publication result, state history, and optional value."""

    manifest: RunManifest
    bundle: PublishedBundle
    state_history: tuple[str, ...]
    value: T | None

    def __post_init__(self) -> None:
        if self.manifest.state not in _TERMINAL_STATES:
            raise MLJobServiceError("job outcomes require a terminal run manifest")
        if not self.state_history or self.state_history[-1] != self.manifest.state:
            raise MLJobServiceError("state history must end at the terminal manifest state")


@dataclass(frozen=True)
class JobDryRun:
    """Read-only job resolution report with no filesystem side effects."""

    run_id: str
    action: str
    job_name: str
    workspace: Mapping[str, Any]
    model_selections: tuple[Mapping[str, Any], ...]
    source_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.run_id)
        except (TypeError, ValueError) as exc:
            raise MLJobServiceError("dry_run.run_id must be a UUID") from exc
        object.__setattr__(self, "action", _nonempty(self.action, "dry_run.action"))
        object.__setattr__(self, "job_name", _nonempty(self.job_name, "dry_run.job_name"))
        object.__setattr__(self, "workspace", _json_mapping(self.workspace, "dry_run.workspace"))
        object.__setattr__(
            self,
            "model_selections",
            tuple(
                _json_mapping(value, "dry_run.model_selection") for value in self.model_selections
            ),
        )
