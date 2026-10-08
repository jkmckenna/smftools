"""Train a bound plan job and publish it as one self-describing run (`MLR-01`).

`run_bound_train_job` returns fitted fold models and their evaluations in
memory. `train_and_publish` runs the same job through the train job service so
the result is an immutable run in the ML workspace:

- the run manifest (plan and plan hash, resolved options, dataset snapshot
  id, split id, environment, seeds) plus the resolved plan and config;
- ``tags.json``: the caller's labels (e.g. a project's task id);
- ``data/membership.parquet``: per fold, every molecule's role and class --
  the exact training and evaluation sets; ``data/splits.json``: per fold, the
  split id, held-out groups and role / class counts;
- ``models.json``: per model and fold, the published model id (each fold
  model is its own immutable model bundle under ``models/``, reloadable with
  ``load_published_sklearn_model`` / ``load_published_torch_model``);
- ``predictions/test.parquet``: per model, fold and held-out molecule, the
  truth, predicted class and class probabilities;
- ``metrics.parquet``: per model and fold, every evaluation metric, plus
  average precision normalised by the cohort's class fraction and at a fixed
  positive prevalence (`average_precision_at_prevalence`);
- ``curves.parquet``: ROC / PR / calibration curve points;
- ``history.parquet``: training events (torch: per-epoch losses);
- ``summary.json``: per model, the mean and SD across folds of the pooled
  metrics, which the workspace run index carries with the tags.

Fold models are published as they are trained and released, so memory holds
one fitted model at a time. With ``final_model=True`` each model is also fit
on every selected row (fold ``"final"``; `MLR-02`), the model to reuse.

`apply_and_publish` (`MLR-02`) applies one published model to a plan's apply
job dataset -- after checking it has the model's channels, positions and
classes -- and publishes an apply run naming the source model: the applied
molecules, predictions and, for labeled data, metrics, curves and summary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.readwrite import atomic_write_json

from ..artifacts import (
    EnvironmentRecord,
    capture_environment_record,
    rebuild_workspace_indexes,
)
from ..evaluation import (
    average_precision_at_prevalence,
    sklearn_training_history,
    torch_training_history,
)
from ..manifests import SplitManifest
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry
from ..workspace import MLWorkspace, resolve_ml_workspace
from .actions import (
    SklearnTrainOptions,
    TorchTrainOptions,
    apply_partition_model,
    evaluate_prediction_result,
)
from .binding import (
    FINAL_FOLD,
    BoundDataset,
    BoundJob,
    FoldRun,
    _concat_predictions,
    bind_ml_dataset,
    final_training_split,
    iter_bound_final_models,
    iter_bound_train_job,
)
from .contracts import (
    JobArtifact,
    JobExecutionContext,
    JobExecutionOutcome,
    JobOperationResult,
    MLJobServiceError,
    ModelSelectionRequest,
    ResolvedJob,
)
from .resolution import resolve_model_selection
from .service import _now, run_apply_job, run_train_job

# Run payloads (paths inside the run bundle) and their roles.
TAGS = "tags.json"
MEMBERSHIP = "data/membership.parquet"
SPLITS = "data/splits.json"
MODELS = "models.json"
PREDICTIONS = "predictions/test.parquet"
METRICS = "metrics.parquet"
CURVES = "curves.parquet"
HISTORY = "history.parquet"
SUMMARY = "summary.json"
_PAYLOADS = (
    ("tags", TAGS, "application/json"),
    ("membership", MEMBERSHIP, "application/vnd.apache.parquet"),
    ("splits", SPLITS, "application/json"),
    ("models", MODELS, "application/json"),
    ("predictions", PREDICTIONS, "application/vnd.apache.parquet"),
    ("metrics", METRICS, "application/vnd.apache.parquet"),
    ("curves", CURVES, "application/vnd.apache.parquet"),
    ("history", HISTORY, "application/vnd.apache.parquet"),
    ("summary", SUMMARY, "application/json"),
)


@dataclass(frozen=True)
class PublishedTrainRun:
    """One published train run: its id, bundle path, fold model ids and summary."""

    run_id: str
    path: Path
    workspace: MLWorkspace
    model_ids: Mapping[str, Mapping[str, str]]  # model name -> fold name -> model id
    summary: Mapping[str, Any]
    outcome: JobExecutionOutcome

    def read(self, payload: str) -> Any:
        """One run payload (e.g. ``METRICS``): a DataFrame or parsed JSON."""
        path = self.path / payload
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())


def _held_out(fold_name: str | None) -> str:
    return "single" if fold_name is None else fold_name.split("=", 1)[-1]


def _fold(fold_name: str | None) -> str:
    return fold_name or "single"


def _jsonable(value: Any) -> Any:
    """Options as JSON: dataclasses field by field, anything else by ``str``."""
    if value is None:
        return None
    payload = asdict(value) if is_dataclass(value) else value
    return json.loads(json.dumps(payload, default=str, sort_keys=True))


def _split_family_id(bound: BoundJob, final_split: SplitManifest | None = None) -> str:
    """One id for the job's folds (and final split): a digest of their split
    ids in fold order."""
    ids = [fold.split.split_id for fold in bound.folds]
    if final_split is not None:
        ids.append(final_split.split_id)
    joined = "\n".join(ids)
    return hashlib.sha256(joined.encode()).hexdigest()


def _tags(tags: Mapping[str, Any] | None) -> dict[str, str]:
    result = {}
    for key, value in (tags or {}).items():
        if not isinstance(key, str) or not key.strip():
            raise MLJobServiceError("tag names must be non-empty strings")
        result[key] = str(value)
    return dict(sorted(result.items()))


def _membership(
    bound: BoundJob, final_split: SplitManifest | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    observations = pd.DataFrame(
        {
            "molecule_uid": [item.molecule_uid for item in bound.snapshot.observations],
            "experiment_uid": [item.experiment_uid for item in bound.snapshot.observations],
            "sample_id": [item.sample_id for item in bound.snapshot.observations],
            "reference": [item.reference for item in bound.snapshot.observations],
            "class_id": pd.array(
                [item.class_id for item in bound.snapshot.observations], dtype="Int64"
            ),
        }
    ).set_index("molecule_uid")
    folds = [
        (
            _fold(fold.fold_name),
            _held_out(fold.fold_name),
            fold.split.split_id,
            dict(fold.resolution.assignments),
        )
        for fold in bound.folds
    ]
    if final_split is not None:
        folds.append(
            (
                FINAL_FOLD,
                None,
                final_split.split_id,
                {member.molecule_uid: member.split for member in final_split.members},
            )
        )
    frames, splits = [], []
    for fold_name, held_out, split_id, assignments in folds:
        roles = pd.Series(assignments, name="role")
        frame = observations.loc[roles.index].rename_axis("molecule_uid")
        frame = frame.assign(role=roles.to_numpy())
        frame.insert(0, "fold", fold_name)
        frames.append(frame.reset_index())
        counts = frame.groupby(["role", "class_id"], dropna=False).size()
        splits.append(
            {
                "fold": fold_name,
                "held_out": held_out,
                "split_id": split_id,
                "n_by_role": {
                    role: int(n) for role, n in frame["role"].value_counts().sort_index().items()
                },
                "n_by_role_and_class": [
                    {"role": role, "class_id": None if pd.isna(cls) else int(cls), "n": int(n)}
                    for (role, cls), n in counts.items()
                ],
            }
        )
    return pd.concat(frames, ignore_index=True), splits


def _prediction_rows(run: FoldRun, model_id: str) -> pd.DataFrame:
    predictions = run.predictions
    classes = list(predictions.class_order)
    frame = pd.DataFrame(
        {
            "model": run.model_name,
            "fold": _fold(run.fold_name),
            "held_out": _held_out(run.fold_name),
            "model_id": model_id,
            "molecule_uid": list(predictions.molecule_uids),
            "experiment_uid": list(predictions.experiment_uids) or None,
            "modality": list(predictions.modalities) or None,
            "truth": (
                None
                if predictions.truth_class_ids is None
                else [classes[i] for i in np.asarray(predictions.truth_class_ids)]
            ),
            "predicted": [classes[i] for i in np.asarray(predictions.class_ids)],
        }
    )
    probabilities = np.asarray(predictions.probabilities)
    for index, name in enumerate(classes):
        frame[f"p_{name}"] = probabilities[:, index]
    return frame


def _metric_rows(
    run: FoldRun,
    *,
    positive_class: str | None,
    prevalence: float | None,
    draws: int,
    seed: int,
) -> list[dict]:
    base = {"model": run.model_name, "fold": _fold(run.fold_name)}
    rows = [
        {
            **base,
            "name": metric.name,
            "value": metric.value,
            "n_observations": metric.n_observations,
            "scope": metric.scope,
            "modality": metric.modality,
            "class_name": metric.class_name,
            "prevalence": None,
        }
        for metric in run.evaluation.metrics
    ]
    # Average precision over the class's own fraction: chance is 1.
    fractions = {
        (record.scope, record.modality, record.class_name): record.fraction
        for record in run.evaluation.class_balance
    }
    for row in list(rows):
        fraction = fractions.get((row["scope"], row["modality"], row["class_name"]))
        if row["name"] == "average_precision" and row["value"] is not None and fraction:
            rows.append(
                {**row, "name": "normalized_average_precision", "value": row["value"] / fraction}
            )
    predictions = run.predictions
    if prevalence is not None and positive_class is not None:
        if predictions.truth_class_ids is not None:
            index = list(predictions.class_order).index(positive_class)
            fixed = average_precision_at_prevalence(
                np.asarray(predictions.truth_class_ids) == index,
                np.asarray(predictions.probabilities)[:, index],
                prevalence=prevalence,
                draws=draws,
                seed=seed,
            )
            if fixed is not None:
                for name, value in (
                    ("average_precision_at_prevalence_reweighted", fixed.reweighted),
                    (
                        "normalized_average_precision_at_prevalence_reweighted",
                        fixed.normalized_reweighted,
                    ),
                    ("average_precision_at_prevalence_subsampled", fixed.subsampled),
                    ("average_precision_at_prevalence_subsampled_sd", fixed.subsampled_sd),
                    (
                        "normalized_average_precision_at_prevalence_subsampled",
                        fixed.normalized_subsampled,
                    ),
                ):
                    rows.append(
                        {
                            **base,
                            "name": name,
                            "value": value,
                            "n_observations": predictions.n_observations,
                            "scope": "pooled",
                            "modality": None,
                            "class_name": positive_class,
                            "prevalence": prevalence,
                        }
                    )
    return rows


def _curve_rows(run: FoldRun) -> list[pd.DataFrame]:
    frames = []
    for curve in run.evaluation.curves:
        thresholds = (
            np.full(curve.x.shape, np.nan)
            if curve.thresholds is None or curve.thresholds.shape != curve.x.shape
            else curve.thresholds
        )
        frames.append(
            pd.DataFrame(
                {
                    "model": run.model_name,
                    "fold": _fold(run.fold_name),
                    "kind": curve.kind,
                    "scope": curve.scope,
                    "modality": curve.modality,
                    "class_name": curve.class_name,
                    "point": np.arange(curve.x.size),
                    "x": curve.x,
                    "y": curve.y,
                    "threshold": thresholds,
                }
            )
        )
    return frames


def _history_rows(run: FoldRun, backend: str) -> list[dict]:
    model = run.training.model
    history = (
        torch_training_history(model) if backend == "torch" else sklearn_training_history(model)
    )
    rows = []
    for event in history.events:
        base = {
            "model": run.model_name,
            "fold": _fold(run.fold_name),
            "event_index": event.event_index,
            "event_type": event.event_type,
            "epoch": event.epoch,
            "step": event.step,
        }
        if not event.metrics:
            rows.append({**base, "metric": None, "value": None})
        for name, value in event.metrics.items():
            rows.append({**base, "metric": name, "value": value})
    return rows


def _summary(metrics: pd.DataFrame, positive_class: str | None) -> dict[str, dict]:
    """Per model: mean / SD / n across folds of each pooled metric, for the
    positive class (binary) or class-free metrics."""
    pooled = metrics[metrics["scope"] == "pooled"]
    keep = pooled["class_name"].isna()
    if positive_class is not None:
        keep |= pooled["class_name"] == positive_class
    pooled = pooled[keep & pooled["value"].notna()]
    summary: dict[str, dict] = {}
    for (model, name), values in pooled.groupby(["model", "name"])["value"]:
        summary.setdefault(str(model), {})[str(name)] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_folds": int(len(values)),
        }
    return summary


def _publish_model(run: FoldRun, backend: str, workspace, *, model_key, run_id, environment):
    if backend == "sklearn":
        from ..models.sklearn_artifacts import publish_sklearn_model as publish
    elif backend == "torch":
        from ..models.torch_artifacts import publish_torch_model as publish
    else:
        raise MLJobServiceError(f"no model publisher for backend {backend!r}")
    return publish(
        run.training.model,
        workspace,
        model_key=model_key,
        originating_run_id=run_id,
        environment=environment,
        created_at=_now(),
    )


def train_and_publish(
    bound: BoundJob,
    *,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    tags: Mapping[str, Any] | None = None,
    sklearn_options: SklearnTrainOptions | None = None,
    torch_options: TorchTrainOptions | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
    prevalence: float | None = 0.10,
    prevalence_draws: int = 50,
    seed: int = 0,
    environment: EnvironmentRecord | None = None,
    rebuild_index: bool = True,
    final_model: bool = False,
) -> PublishedTrainRun:
    """Train every model of a bound train job per fold and publish the run.

    Args:
        bound: The job from `bind_ml_job`.
        workspace / project_dir: Where to publish -- a resolved workspace, or
            a project directory (its ``project_outputs/ml`` workspace).
        tags: Caller labels stored with the run and in the run index.
        sklearn_options / torch_options / registry: As `run_bound_train_job`.
            Fold models are published through the built-in registry's
            publishers, so a family outside it cannot be published.
        prevalence: Positive prevalence for the fixed-prevalence average
            precision (binary tasks with a positive class); ``None`` skips it.
        prevalence_draws / seed: Subsampling draws and their seed.
        environment: Recorded execution environment (captured by default).
        rebuild_index: Rebuild the workspace run / model index afterwards.
        final_model: Also fit each model on every selected row (fold
            ``"final"`` in ``model_ids`` and the records; no held-out
            evaluation -- the folds estimate its performance), the model to
            apply to new data with `apply_and_publish`. Not yet for torch
            models (they need a validation role).

    Returns:
        The published run. Failures still publish a failed run manifest and
        raise `MLJobExecutionError`.
    """
    if (workspace is None) == (project_dir is None):
        raise MLJobServiceError("pass exactly one of workspace or project_dir")
    if workspace is None:
        workspace = resolve_ml_workspace(project_dir=project_dir)
    job_spec = bound.plan.jobs[bound.job_name]
    if job_spec.action != "train":
        raise MLJobServiceError(f"job {bound.job_name!r} is a {job_spec.action!r} job, not train")
    environment = environment or capture_environment_record()
    tags = _tags(tags)
    label_schema = bound.snapshot.label_schema
    positive_class = None if label_schema is None else label_schema.positive_class
    backends = {name: bound.plan.models[name].backend for name in job_spec.models}
    final = None
    if final_model:
        final = final_training_split(bound)
        torch_models = sorted(name for name, backend in backends.items() if backend == "torch")
        if torch_models and not any(m.split == "validation" for m in final[0].members):
            raise MLJobServiceError(
                f"final models for torch models {torch_models} need a validation role; "
                "declare the split's validation_fraction"
            )
    job = ResolvedJob(
        plan=bound.plan,
        workspace=workspace,
        job_name=bound.job_name,
        environment=environment,
        resolved_config={
            "tags": tags,
            "sklearn_options": _jsonable(sklearn_options),
            "torch_options": _jsonable(torch_options),
            "prevalence": prevalence,
            "prevalence_draws": prevalence_draws,
            "folds": [
                {"fold": _fold(fold.fold_name), "split_id": fold.split.split_id}
                for fold in bound.folds
            ],
            "final_split_id": None if final is None else final[0].split_id,
        },
        dataset_snapshot_id=bound.snapshot.snapshot_id,
        split_id=_split_family_id(bound, None if final is None else final[0]),
        seeds={"prevalence_subsampling": seed},
    )
    state: dict[str, Any] = {}

    def operation(context: JobExecutionContext) -> JobOperationResult[None]:
        context.advance_phase("record_data")
        atomic_write_json(context.output_path(TAGS), tags)
        membership, splits = _membership(bound, None if final is None else final[0])
        membership.to_parquet(context.output_path(MEMBERSHIP), index=False)
        atomic_write_json(context.output_path(SPLITS), splits)
        model_ids: dict[str, dict[str, str]] = {}
        models, predictions, metrics, curves, history = [], [], [], [], []

        def publish(run: FoldRun) -> str:
            context.advance_phase(f"train:{run.model_name}:{_fold(run.fold_name)}")
            backend = backends[run.model_name]
            published = _publish_model(
                run,
                backend,
                workspace,
                model_key=run.model_name,
                run_id=context.run_id,
                environment=environment,
            )
            model_id = published.manifest.model_id
            model_ids.setdefault(run.model_name, {})[_fold(run.fold_name)] = model_id
            models.append(
                {
                    "model": run.model_name,
                    "fold": _fold(run.fold_name),
                    "held_out": None if run.fold_name == FINAL_FOLD else _held_out(run.fold_name),
                    "model_id": model_id,
                    "backend": backend,
                    "family": published.manifest.family,
                    "split_id": published.manifest.split_id,
                    "n_train": run.training.n_training_observations,
                    "train_class_counts": list(run.training.class_counts),
                }
            )
            history.extend(_history_rows(run, backend))
            return model_id

        for run in iter_bound_train_job(
            bound,
            sklearn_options=sklearn_options,
            torch_options=torch_options,
            registry=registry,
        ):
            model_id = publish(run)
            predictions.append(_prediction_rows(run, model_id))
            metrics.extend(
                _metric_rows(
                    run,
                    positive_class=positive_class,
                    prevalence=prevalence,
                    draws=prevalence_draws,
                    seed=seed,
                )
            )
            curves.extend(_curve_rows(run))
        if final is not None:
            for run in iter_bound_final_models(
                bound,
                final=final,
                sklearn_options=sklearn_options,
                torch_options=torch_options,
                registry=registry,
            ):
                publish(run)
        context.advance_phase("record_evaluation")
        atomic_write_json(context.output_path(MODELS), models)
        pd.concat(predictions, ignore_index=True).to_parquet(
            context.output_path(PREDICTIONS), index=False
        )
        metric_frame = pd.DataFrame(metrics)
        metric_frame.to_parquet(context.output_path(METRICS), index=False)
        (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()).to_parquet(
            context.output_path(CURVES), index=False
        )
        pd.DataFrame(history).to_parquet(context.output_path(HISTORY), index=False)
        summary = _summary(metric_frame, positive_class)
        atomic_write_json(context.output_path(SUMMARY), summary)
        state.update(model_ids=model_ids, summary=summary)
        return JobOperationResult(
            value=None,
            artifacts=tuple(
                JobArtifact(role=role, relative_path=path, media_type=media_type)
                for role, path, media_type in _PAYLOADS
            ),
        )

    outcome = run_train_job(job, operation)
    if rebuild_index:
        rebuild_workspace_indexes(workspace)
    return PublishedTrainRun(
        run_id=outcome.manifest.run_id,
        path=outcome.bundle.path,
        workspace=workspace,
        model_ids=state["model_ids"],
        summary=state["summary"],
        outcome=outcome,
    )


# Apply-run payloads.
APPLIED_MOLECULES = "data/molecules.parquet"
APPLIED_PREDICTIONS = "predictions/applied.parquet"
APPLIED_MODEL = "model.json"
APPLIED_COHORT = "applied"


@dataclass(frozen=True)
class PublishedApplyRun:
    """One published apply run: a saved model's predictions on a dataset."""

    run_id: str
    path: Path
    workspace: MLWorkspace
    model_id: str
    labeled: bool
    summary: Mapping[str, Any]
    outcome: JobExecutionOutcome

    def read(self, payload: str) -> Any:
        """One run payload (e.g. ``APPLIED_PREDICTIONS``): a DataFrame or parsed JSON."""
        path = self.path / payload
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())


def _load_model(workspace: MLWorkspace, model_id: str, backend: str):
    if backend == "sklearn":
        from ..models.sklearn_artifacts import load_published_sklearn_model

        return load_published_sklearn_model(workspace, model_id)
    if backend == "torch":
        from ..models.torch_artifacts import load_published_torch_model

        return load_published_torch_model(workspace, model_id)
    raise MLJobServiceError(f"no model loader for backend {backend!r}")


def _check_applicable(model: Any, bound: BoundDataset) -> None:
    """Refuse data the model was not fit on: channels, positions, classes."""
    channels = tuple(channel.name for channel in bound.snapshot.input_schema.channels)
    fitted_channels = tuple(model.transform.channel_names)
    if channels != fitted_channels:
        raise MLJobServiceError(
            f"dataset channels {list(channels)} differ from the model's {list(fitted_channels)}"
        )
    coordinates = tuple(int(value) for value in bound.dataset.plan.coordinates)
    fitted = tuple(int(value) for value in model.transform.coordinates)
    if coordinates != fitted:
        missing = sorted(set(fitted) - set(coordinates))
        extra = sorted(set(coordinates) - set(fitted))
        raise MLJobServiceError(
            "dataset positions differ from the model's: "
            f"{len(missing)} missing (first {missing[:5]}), {len(extra)} extra "
            f"(first {extra[:5]})"
        )
    labels = bound.snapshot.label_schema
    if labels is not None and tuple(labels.class_order) != tuple(model.label_schema.class_order):
        raise MLJobServiceError(
            f"dataset classes {list(labels.class_order)} differ from the model's "
            f"{list(model.label_schema.class_order)}"
        )


def apply_and_publish(
    plan,
    job_name: str,
    *,
    model_id: str | None = None,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    tags: Mapping[str, Any] | None = None,
    policy=None,
    prevalence: float | None = 0.10,
    prevalence_draws: int = 50,
    seed: int = 0,
    environment: EnvironmentRecord | None = None,
    rebuild_index: bool = True,
) -> PublishedApplyRun:
    """Apply one published model to a plan's apply-job dataset and publish the run.

    The job (``action: apply``) names a dataset and a model: ``model:<id>``
    for an exact published model, or a plan model key with ``model_id``
    given here (typically a train run's ``model_ids[key]["final"]``). The
    dataset must have the model's channels, positions and (when labeled)
    classes. The run records the applied molecules, predictions and -- when
    the dataset has labels -- metrics, curves and a summary as in a train run;
    its manifest names the source model.

    Args:
        plan / job_name: The plan and its apply job.
        model_id: The published model, when the job names a model key.
        workspace / project_dir: Where the model lives and the run is published.
        tags, prevalence, prevalence_draws, seed, environment, rebuild_index:
            As `train_and_publish`.
        policy: `PartitionReadPolicy` for reading the dataset.
    """
    if (workspace is None) == (project_dir is None):
        raise MLJobServiceError("pass exactly one of workspace or project_dir")
    if workspace is None:
        workspace = resolve_ml_workspace(project_dir=project_dir)
    if job_name not in plan.jobs:
        raise MLJobServiceError(f"unknown job {job_name!r}")
    job_spec = plan.jobs[job_name]
    if job_spec.action != "apply":
        raise MLJobServiceError(f"job {job_name!r} is a {job_spec.action!r} job, not apply")
    if job_spec.model.startswith("model:"):
        named = job_spec.model.removeprefix("model:")
        if model_id is not None and model_id != named:
            raise MLJobServiceError(f"job {job_name!r} names model {named}, not {model_id}")
        model_id = named
    elif model_id is None:
        raise MLJobServiceError(
            f"job {job_name!r} names model key {job_spec.model!r}: pass the published model_id"
        )
    selection = resolve_model_selection(
        workspace, ModelSelectionRequest(kind="exact", model_id=model_id)
    )
    manifest = selection.manifest
    model = _load_model(workspace, model_id, manifest.backend)
    owner = {"experiment_dir": workspace.owner_root}
    if workspace.scope_kind == "project":
        owner = {"project_dir": workspace.owner_root}
    bound = bind_ml_dataset(plan, job_spec.dataset, policy=policy, **owner)
    _check_applicable(model, bound)
    labeled = bound.snapshot.label_schema is not None
    positive_class = model.label_schema.positive_class
    environment = environment or capture_environment_record()
    tags = _tags(tags)
    job = ResolvedJob(
        plan=plan,
        workspace=workspace,
        job_name=job_name,
        environment=environment,
        resolved_config={
            "tags": tags,
            "model_id": model_id,
            "prevalence": prevalence,
            "prevalence_draws": prevalence_draws,
        },
        dataset_snapshot_id=bound.snapshot.snapshot_id,
        model_selections=(selection,),
        seeds={"prevalence_subsampling": seed},
    )
    state: dict[str, Any] = {}

    def operation(context: JobExecutionContext) -> JobOperationResult[None]:
        context.advance_phase("record_data")
        atomic_write_json(context.output_path(TAGS), tags)
        atomic_write_json(
            context.output_path(APPLIED_MODEL),
            {
                "model_id": model_id,
                "model_key": manifest.model_key,
                "backend": manifest.backend,
                "family": manifest.family,
                "originating_run_id": manifest.originating_run_id,
                "split_id": manifest.split_id,
            },
        )
        identity = bound.identity.copy()
        identity.to_parquet(context.output_path(APPLIED_MOLECULES), index=False)
        context.advance_phase("apply")
        parts = [
            apply_partition_model(
                model,
                batch,
                phase="test" if labeled else "inference",
                cohort=APPLIED_COHORT,
                model_id=model_id,
            )
            for batch in bound.iter_batches()
        ]
        predictions = _concat_predictions(parts)
        # Whole-dataset batches may carry an internal "train" role: label the
        # cohort as what it is.
        predictions = replace(predictions, split="test" if labeled else "inference")
        run = FoldRun(
            fold_name=APPLIED_COHORT,
            model_name=manifest.model_key,
            training=None,
            predictions=predictions,
            evaluation=evaluate_prediction_result(predictions) if labeled else None,
        )
        _prediction_rows(run, model_id).to_parquet(
            context.output_path(APPLIED_PREDICTIONS), index=False
        )
        artifacts = [
            JobArtifact(role="tags", relative_path=TAGS, media_type="application/json"),
            JobArtifact(role="model", relative_path=APPLIED_MODEL, media_type="application/json"),
            JobArtifact(
                role="molecules",
                relative_path=APPLIED_MOLECULES,
                media_type="application/vnd.apache.parquet",
            ),
            JobArtifact(
                role="predictions",
                relative_path=APPLIED_PREDICTIONS,
                media_type="application/vnd.apache.parquet",
            ),
        ]
        summary: dict = {}
        if labeled:
            context.advance_phase("evaluate")
            metrics = pd.DataFrame(
                _metric_rows(
                    run,
                    positive_class=positive_class,
                    prevalence=prevalence,
                    draws=prevalence_draws,
                    seed=seed,
                )
            )
            metrics.to_parquet(context.output_path(METRICS), index=False)
            curves = _curve_rows(run)
            (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()).to_parquet(
                context.output_path(CURVES), index=False
            )
            summary = _summary(metrics, positive_class)
            atomic_write_json(context.output_path(SUMMARY), summary)
            artifacts += [
                JobArtifact(
                    role=role,
                    relative_path=path,
                    media_type=media_type,
                )
                for role, path, media_type in _PAYLOADS
                if role in {"metrics", "curves", "summary"}
            ]
        state["summary"] = summary
        return JobOperationResult(value=None, artifacts=tuple(artifacts))

    outcome = run_apply_job(job, operation)
    if rebuild_index:
        rebuild_workspace_indexes(workspace)
    return PublishedApplyRun(
        run_id=outcome.manifest.run_id,
        path=outcome.bundle.path,
        workspace=workspace,
        model_id=model_id,
        labeled=labeled,
        summary=state["summary"],
        outcome=outcome,
    )
