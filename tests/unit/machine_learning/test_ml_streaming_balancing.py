from __future__ import annotations

import numpy as np
import pytest

from smftools.machine_learning.contracts import LabelSchema
from smftools.machine_learning.data.balancing import (
    MLBalanceError,
    resolve_role_balance,
    resolve_role_balance_from_plan,
)
from smftools.machine_learning.data.partition_dataset import (
    MLMaterializedPartitionData,
    PartitionReadEntry,
)
from smftools.machine_learning.plan import BalanceRoleSpec, BalancingSpec, LabelSpec

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64
EXPERIMENT_UID = "12345678-1234-5678-1234-567812345678"

BALANCE_METHODS = ("natural", "class_weight", "weighted_sampler", "downsample", "upsample")


def _label_schema() -> LabelSchema:
    return LabelSchema.from_plan_label(
        LabelSpec(column="activity", classes={"inactive": 0, "active": 1})
    )


def _class_ids(n_rows: int) -> list[int]:
    # Deliberately imbalanced so downsample and upsample actually move rows.
    return [0 if index % 3 else 1 for index in range(n_rows)]


class _PlanStub:
    """Minimal stand-in exposing only what the metadata path consumes."""

    def __init__(self, entries: tuple[PartitionReadEntry, ...]) -> None:
        self._entries = entries

    def entries_for(self, split: str) -> tuple[PartitionReadEntry, ...]:
        rows = tuple(entry for entry in self._entries if entry.split == split)
        if not rows:
            raise ValueError(f"split role {split!r} is absent")
        return rows


def _entries(class_ids: list[int], split: str = "train") -> tuple[PartitionReadEntry, ...]:
    return tuple(
        PartitionReadEntry(
            order_index=index,
            molecule_uid=f"molecule-{index}",
            experiment_uid=EXPERIMENT_UID,
            read_id=f"read-{index}",
            reference="locus",
            modality="deaminase",
            class_id=class_id,
            split=split,
        )
        for index, class_id in enumerate(class_ids)
    )


def _materialized(class_ids: list[int], split: str = "train") -> MLMaterializedPartitionData:
    n_rows = len(class_ids)
    n_positions = 3
    tensor = np.zeros((n_rows, n_positions, 1), dtype=np.float32)
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=(EXPERIMENT_UID,) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.arange(n_positions, dtype=np.int64),
        channel_names=("accessibility",),
        values=tensor,
        labels=np.asarray(class_ids, dtype=np.int64),
        observed_mask=np.ones_like(tensor, dtype=bool),
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=np.ones((n_positions, 1), dtype=bool),
        padding_mask=np.zeros((n_rows, n_positions), dtype=bool),
    )


@pytest.mark.parametrize("method", BALANCE_METHODS)
def test_metadata_resolution_matches_materialized_resolution(method: str) -> None:
    # Balancing reads labels and identities, never feature values, so resolving
    # from the read plan must be indistinguishable from resolving after a full
    # materialization -- including the seeded resampling order.
    class_ids = _class_ids(60)
    balancing = BalancingSpec(train=BalanceRoleSpec(method=method))
    schema = _label_schema()

    reference = resolve_role_balance(
        _materialized(class_ids),
        schema,
        balancing,
        seed=7,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )
    streamed = resolve_role_balance_from_plan(
        _PlanStub(_entries(class_ids)),
        schema,
        balancing,
        seed=7,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )

    assert streamed.resolution_id == reference.resolution_id
    np.testing.assert_array_equal(streamed.selected_indices, reference.selected_indices)
    assert streamed.source_counts == reference.source_counts
    assert streamed.result_counts == reference.result_counts
    assert streamed.selected_molecule_digest == reference.selected_molecule_digest


def test_metadata_resolution_requires_supervised_labels() -> None:
    entries = _entries(_class_ids(12))
    unlabeled = (*entries[:-1], PartitionReadEntry(**{**entries[-1].__dict__, "class_id": None}))

    with pytest.raises(MLBalanceError, match="supervised labels"):
        resolve_role_balance_from_plan(
            _PlanStub(unlabeled),
            _label_schema(),
            BalancingSpec(),
            seed=0,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
        )


def test_metadata_resolution_rejects_unknown_roles() -> None:
    with pytest.raises(MLBalanceError, match="unsupported split role"):
        resolve_role_balance_from_plan(
            _PlanStub(_entries(_class_ids(12))),
            _label_schema(),
            BalancingSpec(),
            role="holdout",
            seed=0,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
        )


def test_evaluation_roles_keep_natural_prevalence_from_metadata() -> None:
    class_ids = _class_ids(30)
    balancing = BalancingSpec(validation=BalanceRoleSpec(method="downsample"))

    with pytest.raises(MLBalanceError, match="natural prevalence"):
        resolve_role_balance_from_plan(
            _PlanStub(_entries(class_ids, split="validation")),
            _label_schema(),
            balancing,
            role="validation",
            seed=0,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
        )
