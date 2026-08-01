from __future__ import annotations

from pathlib import Path

import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import (
    ExperimentPartitionSource,
    MLMemoryBudgetError,
    MLPartitionDataError,
    PartitionDataset,
    PartitionReadPolicy,
    build_partition_data_plan,
)
from smftools.machine_learning.manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    GenomicInterval,
    SourceArtifactReference,
    SplitManifest,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit

EXPERIMENT_UID = "12345678-1234-5678-1234-567812345678"


def _manifests() -> tuple[DatasetSnapshotManifest, SplitManifest]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["sample-a", "sample-b"],
                    "validation_groups": ["sample-c"],
                    "test_groups": ["sample-d"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                }
            },
        }
    )
    dataset_spec = plan.datasets["reads"]
    observations = tuple(
        DatasetObservation(
            molecule_uid=molecule_uid(EXPERIMENT_UID, f"read-{index}"),
            experiment_uid=EXPERIMENT_UID,
            read_id=f"read-{index}",
            sample_id=f"sample-{'a' if index == 0 else 'b'}",
            reference="locus",
            modality="deaminase",
            class_id=index,
            group_values={},
        )
        for index in range(2)
    )
    snapshot = DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="experiment",
            scope_id="exp-a",
            set_name=None,
            dataset_name="reads",
            plan_hash=plan.plan_hash,
            samples=("sample-a", "sample-b"),
            references=("locus",),
            intervals=(GenomicInterval("locus", 2, 6),),
            filters={},
        ),
        input_schema=InputSchema.from_dataset(
            dataset_spec,
            reference="locus",
            n_positions=4,
        ),
        label_schema=LabelSchema.from_plan_label(dataset_spec.labels),
        sources=(
            ExperimentSource(
                experiment_id="exp-a",
                experiment_uid=EXPERIMENT_UID,
                modality="deaminase",
                stage="preprocess",
                stage_generation_id="generation-1",
                membership_fingerprint="members",
                feature_fingerprint="features",
                artifacts=(
                    SourceArtifactReference(
                        artifact_id="spine",
                        kind="spine",
                        relative_path="preprocess_outputs/spine.h5ad",
                        sha256="a" * 64,
                    ),
                ),
            ),
        ),
        observations=observations,
    )
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("sample_id",),
        assignments={item.molecule_uid: "train" for item in observations},
    )
    return snapshot, split


def _source(path: Path, *, modality: str = "deaminase") -> ExperimentPartitionSource:
    path.touch()
    return ExperimentPartitionSource(
        experiment_uid=EXPERIMENT_UID,
        modality=modality,
        stage_spines={"preprocess": path},
    )


def test_plan_resolves_interval_and_clamps_batch_size_to_memory(tmp_path: Path) -> None:
    snapshot, split = _manifests()
    policy = PartitionReadPolicy(
        batch_size=10,
        max_batch_bytes=180,
        max_materialization_bytes=10_000,
    )

    first = build_partition_data_plan(
        snapshot,
        split,
        [_source(tmp_path / "spine.h5ad")],
        policy=policy,
    )
    second = build_partition_data_plan(
        snapshot,
        split,
        [_source(tmp_path / "spine.h5ad")],
        policy=policy,
    )

    assert first.plan_id == second.plan_id
    assert first.coordinates.tolist() == [2, 3, 4, 5]
    assert first.bytes_per_row == 90
    assert first.effective_batch_size == 2
    assert [entry.molecule_uid for entry in first.entries] == sorted(
        entry.molecule_uid for entry in first.entries
    )


def test_plan_rejects_one_row_larger_than_batch_budget(tmp_path: Path) -> None:
    snapshot, split = _manifests()

    with pytest.raises(MLMemoryBudgetError, match="one decoded row"):
        build_partition_data_plan(
            snapshot,
            split,
            [_source(tmp_path / "spine.h5ad")],
            policy=PartitionReadPolicy(max_batch_bytes=89),
        )


def test_plan_rejects_binding_with_wrong_modality(tmp_path: Path) -> None:
    snapshot, split = _manifests()

    with pytest.raises(MLPartitionDataError, match="expected 'deaminase'"):
        build_partition_data_plan(
            snapshot,
            split,
            [_source(tmp_path / "spine.h5ad", modality="conversion")],
        )


def test_full_materialization_refuses_before_opening_partitions(tmp_path: Path) -> None:
    snapshot, split = _manifests()
    plan = build_partition_data_plan(
        snapshot,
        split,
        [_source(tmp_path / "not-a-real-spine.h5ad")],
        policy=PartitionReadPolicy(
            max_batch_bytes=1_000,
            max_materialization_bytes=1,
        ),
    )

    with pytest.raises(MLMemoryBudgetError, match="use iter_batches"):
        PartitionDataset(plan).materialize("train")
