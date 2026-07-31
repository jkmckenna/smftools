"""Scope-safe path resolution for experiment and project ML outputs.

Workspace resolution is read-only. It identifies where later artifact services
may write, but does not create directories, publish files, or inspect matrices.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from smftools.constants import (
    ML_DATASETS_DIR,
    ML_EXPERIMENT_OUTPUTS_DIR,
    ML_INDEX_DIR,
    ML_MODELS_DIR,
    ML_PROJECT_OUTPUTS_DIR,
    ML_RUNS_DIR,
)

from .plan import SUPPORTED_JOB_ACTIONS

if TYPE_CHECKING:
    from smftools.config.experiment_config import ExperimentConfig

PROJECT_OUTPUTS_DIR = "project_outputs"
PROJECT_REGISTRY_FILENAME = "registry.json"
WORKSPACE_SCOPE_KINDS = frozenset({"experiment", "project"})
_SAFE_COMPONENT = re.compile(r"^[0-9A-Za-z._-]+$")


class MLWorkspaceError(ValueError):
    """Raised when an ML output workspace or child path is invalid."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MLWorkspaceError(f"{path}: must be a non-empty string")
    return value.strip()


def _component(value: Any, path: str) -> str:
    result = _string(value, path)
    candidate = PurePosixPath(result)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name in {".", ".."}
        or "\\" in result
        or "://" in result
        or _SAFE_COMPONENT.fullmatch(result) is None
    ):
        raise MLWorkspaceError(f"{path}: must be one filesystem-safe path component, not a path")
    return result


def _canonical_hash(value: dict[str, str]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _assert_contained(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MLWorkspaceError(f"{label}: path escapes the active ML workspace") from exc


@dataclass(frozen=True)
class MLRunPaths:
    """Deterministic paths for one execution below an active ML workspace."""

    run_id: str
    root: Path
    manifest: Path
    resolved_plan: Path
    resolved_config: Path
    environment: Path
    history: Path
    metrics: Path
    predictions: Path
    plots: Path
    explanations: Path
    checkpoints: Path
    logs: Path

    def __post_init__(self) -> None:
        _component(self.run_id, "run_id")
        root = self.root.resolve()
        object.__setattr__(self, "root", root)
        expected = {
            "manifest": root / "run_manifest.json",
            "resolved_plan": root / "resolved_plan.json",
            "resolved_config": root / "resolved_config.json",
            "environment": root / "environment.json",
            "history": root / "history.parquet",
            "metrics": root / "metrics.parquet",
            "predictions": root / "predictions",
            "plots": root / "plots",
            "explanations": root / "explanations",
            "checkpoints": root / "checkpoints",
            "logs": root / "logs",
        }
        for field_name, expected_path in expected.items():
            path = getattr(self, field_name).resolve()
            if path != expected_path:
                raise MLWorkspaceError(f"run_paths.{field_name}: must resolve to {expected_path}")
            _assert_contained(path, root, f"run_paths.{field_name}")
            object.__setattr__(self, field_name, path)

    def prediction_path(self, cohort: str) -> Path:
        """Return the intended Parquet path for one named prediction cohort."""
        return self.predictions / f"{_component(cohort, 'cohort')}.parquet"

    def explanation_dir(self, explanation_id: str) -> Path:
        """Return the intended directory for one immutable explanation result."""
        return self.explanations / _component(explanation_id, "explanation_id")

    def to_dict(self) -> dict[str, str]:
        """Return absolute intended paths for display by a dry run."""
        return {
            "run_id": self.run_id,
            "root": self.root.as_posix(),
            "manifest": self.manifest.as_posix(),
            "resolved_plan": self.resolved_plan.as_posix(),
            "resolved_config": self.resolved_config.as_posix(),
            "environment": self.environment.as_posix(),
            "history": self.history.as_posix(),
            "metrics": self.metrics.as_posix(),
            "predictions": self.predictions.as_posix(),
            "plots": self.plots.as_posix(),
            "explanations": self.explanations.as_posix(),
            "checkpoints": self.checkpoints.as_posix(),
            "logs": self.logs.as_posix(),
        }


@dataclass(frozen=True)
class MLWorkspace:
    """Resolved output ownership for exactly one experiment or project scope."""

    scope_kind: str
    scope_id: str
    owner_root: Path
    root: Path

    def __post_init__(self) -> None:
        if self.scope_kind not in WORKSPACE_SCOPE_KINDS:
            raise MLWorkspaceError(f"scope_kind: must be one of {sorted(WORKSPACE_SCOPE_KINDS)}")
        scope_id = _string(self.scope_id, "scope_id")
        owner_root = self.owner_root.resolve()
        root = self.root.resolve()
        expected = (
            owner_root / ML_EXPERIMENT_OUTPUTS_DIR
            if self.scope_kind == "experiment"
            else owner_root / PROJECT_OUTPUTS_DIR / ML_PROJECT_OUTPUTS_DIR
        )
        if root != expected:
            raise MLWorkspaceError(f"root: {self.scope_kind} workspace must resolve to {expected}")
        _assert_contained(root, owner_root, "root")
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "owner_root", owner_root)
        object.__setattr__(self, "root", root)

    @property
    def workspace_id(self) -> str:
        """Return a path-neutral identity that survives workspace relocation."""
        return _canonical_hash(
            {
                "scope_kind": self.scope_kind,
                "scope_id": self.scope_id,
            }
        )

    @property
    def datasets_root(self) -> Path:
        """Return the intended root for immutable dataset artifacts."""
        return self.root / ML_DATASETS_DIR

    @property
    def runs_root(self) -> Path:
        """Return the intended root for execution artifacts."""
        return self.root / ML_RUNS_DIR

    @property
    def models_root(self) -> Path:
        """Return the intended root for reusable model artifacts."""
        return self.root / ML_MODELS_DIR

    @property
    def index_root(self) -> Path:
        """Return the intended root for rebuildable workspace indexes."""
        return self.root / ML_INDEX_DIR

    def dataset_dir(self, snapshot_id: str) -> Path:
        """Return the intended directory for one immutable dataset snapshot."""
        return self.datasets_root / _component(snapshot_id, "snapshot_id")

    def model_dir(self, model_id: str) -> Path:
        """Return the intended directory for one reusable model artifact."""
        return self.models_root / _component(model_id, "model_id")

    def run_paths(self, run_id: str) -> MLRunPaths:
        """Return every standard path for one run without creating directories."""
        run_id = _component(run_id, "run_id")
        root = (self.runs_root / run_id).resolve()
        _assert_contained(root, self.root, "run_id")
        return MLRunPaths(
            run_id=run_id,
            root=root,
            manifest=root / "run_manifest.json",
            resolved_plan=root / "resolved_plan.json",
            resolved_config=root / "resolved_config.json",
            environment=root / "environment.json",
            history=root / "history.parquet",
            metrics=root / "metrics.parquet",
            predictions=root / "predictions",
            plots=root / "plots",
            explanations=root / "explanations",
            checkpoints=root / "checkpoints",
            logs=root / "logs",
        )

    def portable_reference(self, path: str | Path) -> str:
        """Serialize one contained path relative to this relocatable workspace."""
        candidate = _resolved(path)
        _assert_contained(candidate, self.root, "path")
        relative = candidate.relative_to(self.root)
        if not relative.parts:
            raise MLWorkspaceError("path: must identify a child of the ML workspace")
        return relative.as_posix()

    def resolve_reference(self, reference: str) -> Path:
        """Resolve a portable child reference and reject absolute or escaping paths."""
        value = _string(reference, "reference")
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in value or "://" in value:
            raise MLWorkspaceError(
                "reference: must be a portable path contained by the ML workspace"
            )
        resolved = (self.root / Path(*candidate.parts)).resolve()
        _assert_contained(resolved, self.root, "reference")
        if resolved == self.root:
            raise MLWorkspaceError("reference: must identify a child of the ML workspace")
        return resolved

    def to_dry_run_dict(self, *, run_id: str | None = None) -> dict[str, Any]:
        """Return the resolved ownership and optional intended run paths."""
        result: dict[str, Any] = {
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "workspace_id": self.workspace_id,
            "owner_root": self.owner_root.as_posix(),
            "root": self.root.as_posix(),
            "datasets_root": self.datasets_root.as_posix(),
            "runs_root": self.runs_root.as_posix(),
            "models_root": self.models_root.as_posix(),
            "index_root": self.index_root.as_posix(),
        }
        if run_id is not None:
            result["run"] = self.run_paths(run_id).to_dict()
        return result

    def to_job_dry_run_dict(self, *, action: str, run_id: str) -> dict[str, Any]:
        """Report intended paths for one validated ML-plan job action."""
        action = _string(action, "action").lower()
        if action not in SUPPORTED_JOB_ACTIONS:
            raise MLWorkspaceError(f"action: must be one of {sorted(SUPPORTED_JOB_ACTIONS)}")
        result = self.to_dry_run_dict(run_id=run_id)
        result["action"] = action
        return result


def resolve_ml_workspace(
    *,
    experiment_config: ExperimentConfig | Any | None = None,
    project_dir: str | Path | None = None,
    scope_id: str | None = None,
    selected_experiment_ids: Sequence[str] = (),
) -> MLWorkspace:
    """Resolve exactly one experiment- or project-owned ML workspace.

    Args:
        experiment_config: Config-like object exposing ``output_directory`` and
            normally ``experiment_name``.
        project_dir: Initialized smftools project directory.
        scope_id: Optional stable owner identity. Experiment name and project
            directory name are used as compatibility defaults.
        selected_experiment_ids: Experiments selected by the resolved dataset.
            Experiment scope rejects selections spanning multiple experiments.

    Returns:
        A read-only workspace value object. No directories are created.
    """
    supplied = int(experiment_config is not None) + int(project_dir is not None)
    if supplied != 1:
        raise MLWorkspaceError("exactly one of experiment_config or project_dir must be supplied")
    selected = tuple(
        dict.fromkeys(
            _string(value, f"selected_experiment_ids[{index}]")
            for index, value in enumerate(selected_experiment_ids)
        )
    )

    if experiment_config is not None:
        if len(selected) > 1:
            raise MLWorkspaceError(
                "experiment-scoped ML work cannot select more than one experiment"
            )
        output_directory = getattr(experiment_config, "output_directory", None)
        if output_directory is None or not str(output_directory).strip():
            raise MLWorkspaceError(
                "experiment_config.output_directory: is required for ML workspace resolution"
            )
        owner_root = _resolved(output_directory)
        default_scope_id = (
            selected[0] if selected else getattr(experiment_config, "experiment_name", None)
        )
        resolved_scope_id = _string(
            scope_id if scope_id is not None else default_scope_id,
            "scope_id",
        )
        return MLWorkspace(
            scope_kind="experiment",
            scope_id=resolved_scope_id,
            owner_root=owner_root,
            root=owner_root / ML_EXPERIMENT_OUTPUTS_DIR,
        )

    owner_root = _resolved(project_dir)
    registry = owner_root / PROJECT_REGISTRY_FILENAME
    if not registry.is_file():
        raise MLWorkspaceError(
            f"project_dir: no project registry at {registry}; initialize the project first"
        )
    resolved_scope_id = _string(
        scope_id if scope_id is not None else owner_root.name,
        "scope_id",
    )
    return MLWorkspace(
        scope_kind="project",
        scope_id=resolved_scope_id,
        owner_root=owner_root,
        root=owner_root / PROJECT_OUTPUTS_DIR / ML_PROJECT_OUTPUTS_DIR,
    )
