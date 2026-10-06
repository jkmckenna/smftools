"""Bind a resolved ML plan job to partition datasets and run its folds (`MLX-06`).

`plan_ml_workflow` resolves a plan without reading data. This module takes the
same resolution one step further: it turns a job's dataset selection into a
`DatasetSnapshotManifest`, each resolved split fold into a `SplitManifest`
(through `MLSplitResolution.to_manifest`, which re-checks that both describe
the same rows), and binds both to the experiments' stage spines as a
`PartitionDataset`. `run_bound_train_job` then fits every declared model on
each fold's train role and evaluates it on the fold's test role, predicted
batch by batch so no split has to fit in memory at once.

Results are returned in memory; publishing them through the job service as
immutable run artifacts is a separate step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import LabelSchema
from ..data.partition_dataset import (
    ExperimentPartitionSource,
    PartitionDataset,
    PartitionReadPolicy,
    build_partition_data_plan,
)
from ..manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    GenomicInterval,
    SourceArtifactReference,
    SplitManifest,
)
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry
from ..plan import MLPlan
from ..selection import MLDataSelectionPlan, SelectedExperimentSource, plan_ml_dataset
from ..splitting import MLSplitResolution, plan_ml_splits
from .actions import (
    SklearnTrainOptions,
    TorchTrainOptions,
    apply_partition_model,
    evaluate_prediction_result,
    train_partition_model,
)
from .contracts import MLJobServiceError
from .planning import _input_schema, resolve_plan_model

# DatasetObservation fields; any other group field travels in group_values.
_OBSERVATION_FIELDS = frozenset(
    {"molecule_uid", "experiment_uid", "read_id", "sample_id", "reference", "modality"}
)


@dataclass(frozen=True)
class BoundFold:
    """One resolved split fold bound to readable partition data."""

    fold_name: str | None
    resolution: MLSplitResolution
    split: SplitManifest
    dataset: PartitionDataset


@dataclass(frozen=True)
class BoundJob:
    """A plan job's dataset snapshot and every fold, ready to read."""

    plan: MLPlan
    job_name: str
    dataset_name: str
    selection: MLDataSelectionPlan
    snapshot: DatasetSnapshotManifest
    folds: tuple[BoundFold, ...]


@dataclass(frozen=True)
class FoldRun:
    """One model trained on one fold and evaluated on its test role."""

    fold_name: str | None
    model_name: str
    training: Any
    predictions: Any
    evaluation: Any


def _artifact_path(path: Path, run_root: Path | None) -> str:
    if run_root is not None:
        try:
            return path.resolve().relative_to(run_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _experiment_source(source: SelectedExperimentSource) -> ExperimentSource:
    stages = sorted(source.stage_spines) or sorted({channel.stage for channel in source.channels})
    return ExperimentSource(
        experiment_id=source.experiment_id,
        experiment_uid=source.experiment_uid,
        modality=source.modality,
        stage="+".join(stages),
        stage_generation_id="+".join(
            source.stage_generations.get(stage, "current") for stage in stages
        ),
        membership_fingerprint=source.membership_fingerprint,
        feature_fingerprint=source.feature_fingerprint,
        artifacts=(
            SourceArtifactReference(
                artifact_id=f"{source.experiment_id}:molecule_index",
                kind="molecule_index",
                relative_path=_artifact_path(source.membership_artifact, source.run_root),
                sha256=source.membership_artifact_sha256,
            ),
        ),
    )


def _observations(
    identity: pd.DataFrame, group_by: Sequence[str]
) -> tuple[DatasetObservation, ...]:
    extra = [field for field in group_by if field not in _OBSERVATION_FIELDS]
    absent = [field for field in extra if field not in identity]
    if absent:
        raise MLJobServiceError(f"selection identity lacks group fields {absent}")
    records = identity.to_dict("records")
    return tuple(
        DatasetObservation(
            molecule_uid=str(row["molecule_uid"]),
            experiment_uid=str(row["experiment_uid"]),
            read_id=str(row["read_id"]),
            sample_id=str(row["sample_id"]),
            reference=str(row["reference"]),
            modality=str(row["modality"]),
            class_id=None if pd.isna(row["class_id"]) else int(row["class_id"]),
            group_values={field: str(row[field]) for field in extra},
        )
        for row in records
    )


def snapshot_from_selection(
    plan: MLPlan, selection: MLDataSelectionPlan
) -> DatasetSnapshotManifest:
    """The immutable dataset snapshot a resolved selection describes."""
    dataset = plan.datasets[selection.dataset_name]
    input_schema = _input_schema(plan, selection.dataset_name, selection)
    start = int(dataset.filters.get("start", 0))
    windows = (
        dataset.positions.windows()
        if dataset.positions is not None
        else ((start, start + selection.n_features),)
    )
    return DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind=selection.scope_kind,
            scope_id=selection.scope_id,
            set_name=selection.set_name,
            dataset_name=selection.dataset_name,
            plan_hash=selection.plan_hash,
            samples=tuple(sorted(set(selection.identity_table["sample_id"].astype(str)))),
            # Every reference molecules come from; a coordinate frame maps the
            # others onto the schema reference (`MLX-03`).
            references=tuple(
                sorted({input_schema.reference, *selection.identity_table["reference"].astype(str)})
            ),
            intervals=tuple(
                GenomicInterval(input_schema.reference, window_start, window_end)
                for window_start, window_end in windows
            ),
            filters=dict(dataset.filters),
        ),
        input_schema=input_schema,
        label_schema=(
            None if dataset.labels is None else LabelSchema.from_plan_label(dataset.labels)
        ),
        sources=tuple(_experiment_source(source) for source in selection.sources),
        observations=_observations(selection.identity_table, selection.group_by),
    )


def _partition_sources(selection: MLDataSelectionPlan) -> tuple[ExperimentPartitionSource, ...]:
    sources = []
    for source in selection.sources:
        if not source.stage_spines:
            raise MLJobServiceError(
                f"experiment {source.experiment_id!r} has no stage spine for its channels"
            )
        sources.append(
            ExperimentPartitionSource(
                experiment_uid=source.experiment_uid,
                modality=source.modality,
                stage_spines=dict(source.stage_spines),
                # Lets the reader go partition by partition (`MLX-09`).
                stage_read_indexes=dict(source.stage_read_indexes),
            )
        )
    return tuple(sources)


def _concat_predictions(parts: Sequence[Any]) -> Any:
    """One prediction table from per-batch tables, rows in batch order."""
    first = parts[0]
    values = {}
    for item in fields(first):
        column = [getattr(part, item.name) for part in parts]
        if isinstance(column[0], np.ndarray):
            values[item.name] = np.concatenate(column)
        elif isinstance(column[0], tuple) and item.name != "class_order":
            values[item.name] = tuple(value for part in column for value in part)
        elif column[0] is None:
            values[item.name] = None
        else:
            if any(value != column[0] for value in column[1:]):
                raise MLJobServiceError(f"prediction batches disagree on {item.name!r}")
            values[item.name] = column[0]
    return type(first)(**values)


def _predict_split(training: Any, dataset: PartitionDataset, split: str, model_id: str) -> Any:
    """Predict a split batch by batch: a held-out experiment can exceed the
    materialization budget (one real test fold estimated 2.5 GB)."""
    return _concat_predictions(
        [
            apply_partition_model(training.model, batch, phase=split, model_id=model_id)
            for batch in dataset.iter_batches(split)
        ]
    )


def bind_ml_job(
    plan: MLPlan,
    job_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    experiment_id: str | None = None,
    policy: PartitionReadPolicy | None = None,
) -> BoundJob:
    """Resolve a job's selection, snapshot and folds, bound to readable data."""
    if job_name not in plan.jobs:
        raise MLJobServiceError(f"unknown job {job_name!r}")
    job = plan.jobs[job_name]
    if job.dataset is None or job.split is None:
        raise MLJobServiceError(f"job {job_name!r} declares no dataset and split to bind")
    selection = plan_ml_dataset(
        plan,
        job.dataset,
        project_dir=project_dir,
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
    )
    snapshot = snapshot_from_selection(plan, selection)
    partition_sources = _partition_sources(selection)
    folds = []
    for resolution in plan_ml_splits(plan, job.split, selection):
        split = resolution.to_manifest(snapshot)
        read_plan = build_partition_data_plan(
            snapshot,
            split,
            partition_sources,
            policy=policy,
            coordinate_maps=selection.coordinate_maps,
        )
        folds.append(
            BoundFold(
                fold_name=resolution.fold_name,
                resolution=resolution,
                split=split,
                dataset=PartitionDataset(read_plan),
            )
        )
    return BoundJob(
        plan=plan,
        job_name=job_name,
        dataset_name=job.dataset,
        selection=selection,
        snapshot=snapshot,
        folds=tuple(folds),
    )


@dataclass(frozen=True)
class BoundDataset:
    """Every selected row of one dataset, ready to read without a split (`MLX-10`).

    ``identity`` is the selection's row table (molecule, experiment, sample,
    reference, label and any requested group columns, e.g. label-table
    columns), in snapshot order.
    """

    plan: MLPlan
    dataset_name: str
    selection: MLDataSelectionPlan
    snapshot: DatasetSnapshotManifest
    dataset: PartitionDataset

    @property
    def identity(self) -> pd.DataFrame:
        return self.selection.identity_table

    def iter_batches(self, *, worker_id: int = 0, num_workers: int = 1):
        """Batches of every row; with ``num_workers`` > 1, this worker's blocks only.

        Workers split whole blocks (`F69`), so N processes each reading their
        share decode every row exactly once between them.
        """
        return self.dataset.iter_batches(_ALL_ROWS, worker_id=worker_id, num_workers=num_workers)

    def materialize(self):
        return self.dataset.materialize(_ALL_ROWS)


# A split role covering every row; the reader needs a role, an embedding none.
_ALL_ROWS = "train"


def bind_ml_dataset(
    plan: MLPlan,
    dataset_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    experiment_id: str | None = None,
    group_by: Sequence[str] = (),
    policy: PartitionReadPolicy | None = None,
) -> BoundDataset:
    """Bind one dataset's selected rows for reading, with no train/test split.

    For embeddings and other whole-cohort analyses that need the ML data path
    (label tables, QC filters, position masks, coordinate maps, partition-major
    reads) but no folds. ``group_by`` names extra per-row columns to carry --
    a label table's group column, ``Barcode`` -- into ``identity``.
    """
    if dataset_name not in plan.datasets:
        raise MLJobServiceError(f"unknown dataset {dataset_name!r}")
    selection = plan_ml_dataset(
        plan,
        dataset_name,
        project_dir=project_dir,
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        group_by=tuple(group_by),
    )
    snapshot = snapshot_from_selection(plan, selection)
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("experiment_uid",),
        assignments={item.molecule_uid: _ALL_ROWS for item in snapshot.observations},
    )
    read_plan = build_partition_data_plan(
        snapshot,
        split,
        _partition_sources(selection),
        policy=policy,
        coordinate_maps=selection.coordinate_maps,
    )
    return BoundDataset(
        plan=plan,
        dataset_name=dataset_name,
        selection=selection,
        snapshot=snapshot,
        dataset=PartitionDataset(read_plan),
    )


def run_bound_train_job(
    bound: BoundJob,
    *,
    sklearn_options: SklearnTrainOptions | None = None,
    torch_options: TorchTrainOptions | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> tuple[FoldRun, ...]:
    """Fit each of the job's models per fold and evaluate on that fold's test role.

    The plan's balancing profile, when the job names one, applies to training;
    explicit options take precedence over it.
    """
    job = bound.plan.jobs[bound.job_name]
    if job.action != "train":
        raise MLJobServiceError(f"job {bound.job_name!r} is a {job.action!r} job, not train")
    balancing = bound.plan.balancing[job.balancing] if job.balancing is not None else None
    runs = []
    for model_name in job.models:
        spec = bound.plan.models[model_name]
        resolved = resolve_plan_model(
            model_name,
            spec,
            input_schema=bound.snapshot.input_schema,
            registry=registry,
        )
        sk_options = sklearn_options
        th_options = torch_options
        if spec.backend == "sklearn" and sk_options is None:
            sk_options = SklearnTrainOptions(balancing=balancing)
        if spec.backend == "torch" and th_options is None:
            th_options = TorchTrainOptions(balancing=balancing)
        for fold in bound.folds:
            training = train_partition_model(
                fold.dataset,
                resolved,
                sklearn_options=sk_options if spec.backend == "sklearn" else None,
                torch_options=th_options if spec.backend == "torch" else None,
                registry=registry,
            )
            predictions = _predict_split(
                training,
                fold.dataset,
                "test",
                model_id=f"{bound.job_name}:{model_name}:{fold.fold_name or 'single'}",
            )
            runs.append(
                FoldRun(
                    fold_name=fold.fold_name,
                    model_name=model_name,
                    training=training,
                    predictions=predictions,
                    evaluation=evaluate_prediction_result(predictions),
                )
            )
    return tuple(runs)
