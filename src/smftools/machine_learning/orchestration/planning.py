"""Read-only planning for complete user-declared machine-learning workflows."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..contracts import InputSchema, LabelSchema
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry
from ..plan import DatasetSpec, MLPlan, ModelSpec, SplitSpec
from ..selection import MLDataSelectionPlan, plan_ml_dataset
from ..splitting import MLSplitResolution, plan_ml_splits
from ..workspace import MLWorkspace, resolve_ml_workspace


class MLWorkflowPlanningError(ValueError):
    """Raised with a plan field path when a workflow cannot be previewed."""


@dataclass(frozen=True)
class MLWorkflowDryRun:
    """Immutable, JSON-ready preview of a fully validated ML workflow."""

    plan_hash: str
    workspace: Mapping[str, Any]
    datasets: tuple[Mapping[str, Any], ...]
    splits: tuple[Mapping[str, Any], ...]
    models: tuple[Mapping[str, Any], ...]
    jobs: tuple[Mapping[str, Any], ...]
    optional_dependencies: tuple[Mapping[str, Any], ...]
    execution: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", _freeze_mapping(self.workspace))
        object.__setattr__(self, "datasets", _freeze_records(self.datasets))
        object.__setattr__(self, "splits", _freeze_records(self.splits))
        object.__setattr__(self, "models", _freeze_records(self.models))
        object.__setattr__(self, "jobs", _freeze_records(self.jobs))
        object.__setattr__(
            self,
            "optional_dependencies",
            _freeze_records(self.optional_dependencies),
        )
        object.__setattr__(self, "execution", _freeze_mapping(self.execution))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-serializable report."""
        return {
            "plan_hash": self.plan_hash,
            "workspace": _thaw(self.workspace),
            "datasets": _thaw(self.datasets),
            "splits": _thaw(self.splits),
            "models": _thaw(self.models),
            "jobs": _thaw(self.jobs),
            "optional_dependencies": _thaw(self.optional_dependencies),
            "execution": _thaw(self.execution),
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze(dict(value))


def _freeze_records(values: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_freeze_mapping(value) for value in values)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _raise(path: str, exc: Exception) -> None:
    raise MLWorkflowPlanningError(f"{path}: {exc}") from exc


def _input_schema(
    plan: MLPlan,
    dataset_name: str,
    selection: MLDataSelectionPlan,
) -> InputSchema:
    references = sorted(
        {reference for source in selection.sources for reference in source.canonical_references}
    )
    if len(references) != 1:
        raise MLWorkflowPlanningError(
            f"datasets.{dataset_name}.references: dry-run model tensors currently require "
            f"exactly one resolved canonical reference; found {references}"
        )
    try:
        return InputSchema.from_dataset(
            plan.datasets[dataset_name],
            reference=references[0],
            n_positions=selection.n_features,
        )
    except ValueError as exc:
        _raise(f"datasets.{dataset_name}.channels", exc)


def _selection_report(
    selection: MLDataSelectionPlan,
    dataset: DatasetSpec,
    input_schema: InputSchema,
    label_schema: LabelSchema | None,
) -> dict[str, Any]:
    report = selection.to_dry_run_dict()
    report["selectors"] = {
        "experiments": {
            "include": list(dataset.experiments.include),
            "exclude": list(dataset.experiments.exclude),
        },
        "samples": {
            "include": list(dataset.samples.include),
            "exclude": list(dataset.samples.exclude),
        },
        "references": list(dataset.references),
    }
    report["input_schema"] = input_schema.to_dict()
    report["input_schema_hash"] = input_schema.schema_hash
    report["label_schema"] = None if label_schema is None else label_schema.to_dict()
    return report


def _overlap_checks(
    selection: MLDataSelectionPlan,
    resolution: MLSplitResolution,
) -> dict[str, Any]:
    selected = set(selection.identity_table["molecule_uid"].astype(str))
    assigned = set(resolution.assignments)
    role_members = {
        role: {
            uid for uid, assigned_role in resolution.assignments.items() if assigned_role == role
        }
        for role in ("train", "validation", "test")
    }
    overlaps = {
        f"{left}_{right}": len(role_members[left].intersection(role_members[right]))
        for index, left in enumerate(role_members)
        for right in tuple(role_members)[index + 1 :]
    }
    if any(overlaps.values()) or selected != assigned:
        raise MLWorkflowPlanningError(
            f"splits.{resolution.split_name}: assignments do not form a disjoint, "
            "complete partition of selected observations"
        )
    return {
        "observation_roles_disjoint": True,
        "group_roles_disjoint": True,
        "role_overlap_counts": overlaps,
        "unassigned_observations": len(selected.difference(assigned)),
        "unknown_observations": len(assigned.difference(selected)),
    }


def _split_report(
    dataset_name: str,
    selection: MLDataSelectionPlan,
    resolution: MLSplitResolution,
    spec: SplitSpec,
) -> dict[str, Any]:
    report = resolution.to_dry_run_dict()
    report["dataset_name"] = dataset_name
    report["requested_groups"] = {
        "train": list(spec.train_groups),
        "validation": list(spec.validation_groups),
        "test": list(spec.test_groups),
    }
    report["overlap_checks"] = _overlap_checks(selection, resolution)
    report["estimated_materialization_bytes_by_role"] = {
        summary.split: math.ceil(
            selection.estimated_materialization_bytes
            * summary.n_observations
            / selection.n_observations
        )
        for summary in resolution.summaries
    }
    return report


def _resolved_model(
    name: str,
    spec: ModelSpec,
    *,
    dataset_name: str,
    input_schema: InputSchema,
    registry: ModelRegistry,
) -> dict[str, Any]:
    try:
        if spec.backend == "sklearn":
            assert spec.family is not None
            resolved = registry.resolve(
                spec.family,
                input_schema=input_schema,
                parameters=spec.parameters,
            )
        else:
            assert spec.recipe is not None
            recipe = registry.recipe(spec.recipe)
            if recipe.backend != spec.backend:
                raise ValueError(
                    f"recipe {spec.recipe!r} uses backend {recipe.backend!r}, not {spec.backend!r}"
                )
            resolved = registry.resolve(
                recipe.family,
                input_schema=input_schema,
                parameters=spec.overrides,
                recipe=spec.recipe,
            )
    except ValueError as exc:
        _raise(f"models.{name}", exc)
    if resolved.backend != spec.backend:
        raise MLWorkflowPlanningError(
            f"models.{name}.backend: declared {spec.backend!r} but registry resolved "
            f"{resolved.backend!r}"
        )
    return {
        "model_name": name,
        "dataset_name": dataset_name,
        "backend": resolved.backend,
        "family": resolved.family,
        "recipe_id": resolved.recipe_id,
        "architecture_schema_version": resolved.architecture_schema_version,
        "architecture": resolved.architecture.to_dict(),
        "capabilities": resolved.capabilities.to_dict(),
        "initialization": _thaw(spec.initialization),
    }


def _job_model_names(plan: MLPlan, job_name: str) -> tuple[str, ...]:
    job = plan.jobs[job_name]
    if job.models:
        return job.models
    if job.model is not None and not job.model.startswith("model:"):
        return (job.model,)
    return ()


def _job_report(plan: MLPlan, job_name: str, workspace: MLWorkspace) -> dict[str, Any]:
    job = plan.jobs[job_name]
    if job.action == "train":
        outputs = ("history", "metrics", "checkpoints", "models")
    elif job.action == "apply":
        outputs = ("predictions",)
    elif job.action == "evaluate":
        outputs = ("metrics",)
    elif job.action == "explain":
        outputs = ("explanations",)
    else:
        outputs = ("plots",)
    balancing_policy = None
    if job.balancing is not None:
        declared = plan.balancing[job.balancing]
        balancing_policy = {
            "train": declared.train.method,
            "validation": declared.validation.method,
            "test": declared.test.method,
        }
    return {
        "job_name": job_name,
        "action": job.action,
        "dataset": job.dataset,
        "split": job.split,
        "balancing": job.balancing,
        "balancing_policy": balancing_policy,
        "models": list(_job_model_names(plan, job_name)),
        "model_selector": job.model,
        "source_job": job.source_job,
        "source_runs": list(job.runs),
        "evaluate": list(job.evaluate),
        "explain": list(job.explain),
        "plots": list(job.plots),
        "output_root": workspace.runs_root.as_posix(),
        "output_layout": "runs/<fresh-run-uuid>",
        "expected_artifacts": list(outputs),
    }


def _dependency_available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _optional_dependencies(
    plan: MLPlan,
    checker: Callable[[str], bool],
) -> tuple[dict[str, Any], ...]:
    requested: dict[str, set[str]] = {}

    def add(package: str, feature: str) -> None:
        requested.setdefault(package, set()).add(feature)

    if plan.tracking is not None and plan.tracking.provider != "none":
        if plan.tracking.provider != "wandb":
            raise MLWorkflowPlanningError(
                f"tracking.provider: unsupported provider {plan.tracking.provider!r}; "
                "supported providers are 'none' and 'wandb'"
            )
        add("wandb", "experiment tracking")

    captum_methods = {
        "saliency",
        "inputxgradient",
        "integratedgradients",
        "deeplift",
        "gradientshap",
        "layergradcam",
        "guidedgradcam",
    }
    shap_methods = {"treeshap", "kernelshap"}
    for job_name, job in plan.jobs.items():
        for method in job.explain:
            normalized = method.replace("_", "").replace("-", "").lower()
            if normalized in captum_methods:
                add("captum", f"jobs.{job_name}.explain:{method}")
            if normalized in shap_methods:
                add("shap", f"jobs.{job_name}.explain:{method}")

    return tuple(
        {
            "package": package,
            "extra": "ml-extended",
            "available": bool(checker(package)),
            "required_for": sorted(features),
        }
        for package, features in sorted(requested.items())
    )


def plan_ml_workflow(
    plan: MLPlan,
    *,
    experiment_config: Any | None = None,
    project_dir: str | Path | None = None,
    experiment_id: str | None = None,
    model_registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
    dependency_checker: Callable[[str], bool] | None = None,
) -> MLWorkflowDryRun:
    """Resolve a complete ML plan without training or writing artifacts.

    Exactly one scope input is mandatory and must match ``plan.scope``. No path
    is inferred from the process working directory.
    """
    if not isinstance(plan, MLPlan):
        raise MLWorkflowPlanningError("plan: must be a parsed MLPlan")
    if plan.scope.kind == "experiment":
        if experiment_config is None or project_dir is not None:
            raise MLWorkflowPlanningError(
                "scope: experiment plans require experiment_config and cannot use project_dir"
            )
        experiment_dir = getattr(experiment_config, "output_directory", None)
        if experiment_dir is None:
            raise MLWorkflowPlanningError("scope.experiment_config.output_directory: is required")
    else:
        if project_dir is None or experiment_config is not None:
            raise MLWorkflowPlanningError(
                "scope: project plans require project_dir and cannot use experiment_config"
            )
        experiment_dir = None

    selections: dict[str, MLDataSelectionPlan] = {}
    schemas: dict[str, InputSchema] = {}
    labels: dict[str, LabelSchema | None] = {}
    for dataset_name, dataset in plan.datasets.items():
        try:
            selection = plan_ml_dataset(
                plan,
                dataset_name,
                experiment_dir=experiment_dir,
                project_dir=project_dir,
                experiment_id=experiment_id,
            )
        except ValueError as exc:
            _raise(f"datasets.{dataset_name}", exc)
        selections[dataset_name] = selection
        schemas[dataset_name] = _input_schema(plan, dataset_name, selection)
        try:
            labels[dataset_name] = (
                None if dataset.labels is None else LabelSchema.from_plan_label(dataset.labels)
            )
        except ValueError as exc:
            _raise(f"datasets.{dataset_name}.labels", exc)

    selected_experiments = tuple(
        sorted(
            {
                source.experiment_id
                for selection in selections.values()
                for source in selection.sources
            }
        )
    )
    try:
        workspace = resolve_ml_workspace(
            experiment_config=experiment_config,
            project_dir=project_dir,
            selected_experiment_ids=selected_experiments,
        )
    except ValueError as exc:
        _raise("scope", exc)
    if workspace.scope_kind != plan.scope.kind:
        raise MLWorkflowPlanningError("scope.kind: differs from the resolved output workspace")
    for dataset_name, selection in selections.items():
        if selection.scope_kind != workspace.scope_kind or selection.scope_id != workspace.scope_id:
            raise MLWorkflowPlanningError(
                f"datasets.{dataset_name}.scope: selection owner "
                f"{selection.scope_kind}:{selection.scope_id} differs from workspace owner "
                f"{workspace.scope_kind}:{workspace.scope_id}"
            )

    split_reports = []
    split_pairs = sorted(
        {
            (job.dataset, job.split)
            for job in plan.jobs.values()
            if job.dataset is not None and job.split is not None
        }
    )
    for dataset_name, split_name in split_pairs:
        assert dataset_name is not None and split_name is not None
        try:
            resolutions = plan_ml_splits(plan, split_name, selections[dataset_name])
        except ValueError as exc:
            _raise(f"splits.{split_name}", exc)
        split_reports.extend(
            _split_report(
                dataset_name,
                selections[dataset_name],
                resolution,
                plan.splits[split_name],
            )
            for resolution in resolutions
        )

    model_reports: dict[tuple[str, str], dict[str, Any]] = {}
    for job_name, job in plan.jobs.items():
        if job.dataset is None:
            continue
        for model_name in _job_model_names(plan, job_name):
            key = (model_name, job.dataset)
            if key not in model_reports:
                model_reports[key] = _resolved_model(
                    model_name,
                    plan.models[model_name],
                    dataset_name=job.dataset,
                    input_schema=schemas[job.dataset],
                    registry=model_registry,
                )

    checker = dependency_checker or _dependency_available
    return MLWorkflowDryRun(
        plan_hash=plan.plan_hash,
        workspace=workspace.to_dry_run_dict(),
        datasets=tuple(
            _selection_report(
                selections[name],
                plan.datasets[name],
                schemas[name],
                labels[name],
            )
            for name in sorted(selections)
        ),
        splits=tuple(split_reports),
        models=tuple(model_reports[key] for key in sorted(model_reports)),
        jobs=tuple(_job_report(plan, name, workspace) for name in plan.jobs),
        optional_dependencies=_optional_dependencies(plan, checker),
        execution={
            "writes_files": False,
            "trains_models": False,
            "backend_services": ["sklearn", "pytorch"],
            "lightning_required": False,
            "hydra_required": False,
            "tracking_provider": ("none" if plan.tracking is None else plan.tracking.provider),
        },
    )
