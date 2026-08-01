from __future__ import annotations

import numpy as np
import pytest

from smftools.machine_learning.contracts import LabelSchema
from smftools.machine_learning.data.balancing import (
    MLBalanceError,
    balance_counts,
    resolve_evaluation_sensitivity,
    resolve_role_balance,
)
from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.plan import BalanceRoleSpec, BalancingSpec, LabelSpec

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64


def _schema() -> LabelSchema:
    return LabelSchema.from_plan_label(
        LabelSpec(column="activity", classes={"inactive": 0, "active": 1})
    )


def _data(split: str, labels: list[int]) -> MLMaterializedPartitionData:
    n_rows = len(labels)
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"{split}-molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.asarray([10], dtype=np.int64),
        channel_names=("accessibility",),
        values=np.ones((n_rows, 1, 1), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        observed_mask=np.ones((n_rows, 1, 1), dtype=bool),
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=np.ones((1, 1), dtype=bool),
        padding_mask=np.zeros((n_rows, 1), dtype=bool),
    )


def _resolve(data: MLMaterializedPartitionData, method: str):
    return resolve_role_balance(
        data,
        _schema(),
        BalancingSpec(train=BalanceRoleSpec(method)),
        seed=41,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )


def test_class_weights_follow_persisted_class_order() -> None:
    resolution = _resolve(_data("train", [0, 0, 0, 0, 1, 1]), "class_weight")

    assert resolution.class_order == ("inactive", "active")
    assert resolution.source_counts == (4, 2)
    assert resolution.class_weights.tolist() == pytest.approx([0.75, 1.5])
    assert resolution.sample_weights is None
    assert balance_counts(resolution) == {"inactive": 4, "active": 2}
    assert not resolution.class_weights.flags.writeable


def test_weighted_sampler_maps_weights_to_source_labels_deterministically() -> None:
    resolution = _resolve(_data("train", [0, 0, 0, 0, 1, 1]), "weighted_sampler")

    assert resolution.sample_weights.tolist() == pytest.approx([0.75, 0.75, 0.75, 0.75, 1.5, 1.5])
    assert list(resolution.torch_weighted_sampler()) == list(resolution.torch_weighted_sampler())


@pytest.mark.parametrize("method", ["downsample", "upsample"])
def test_training_resampling_is_balanced_and_reproducible(method: str) -> None:
    data = _data("train", [0, 0, 0, 0, 1, 1])
    first = _resolve(data, method)
    second = _resolve(data, method)

    assert first.result_counts[0] == first.result_counts[1]
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert first.resolution_id == second.resolution_id
    if method == "downsample":
        assert len(first.selected_indices) == 4
    else:
        assert len(first.selected_indices) == 8


def test_validation_and_test_primary_cohorts_remain_natural() -> None:
    balancing = BalancingSpec(train=BalanceRoleSpec("downsample"))
    for role in ("validation", "test"):
        data = _data(role, [0, 0, 0, 1])
        resolution = resolve_role_balance(
            data,
            _schema(),
            balancing,
            seed=41,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
        )

        assert resolution.method == "natural"
        assert resolution.purpose == "primary"
        assert resolution.result_counts == (3, 1)
        np.testing.assert_array_equal(resolution.selected_indices, np.arange(4))


def test_evaluation_resampling_is_separately_named_and_does_not_mutate_source() -> None:
    validation = _data("validation", [0, 0, 0, 1])
    original_labels = validation.labels.copy()

    resolution = resolve_evaluation_sensitivity(
        validation,
        _schema(),
        name="balanced_prevalence",
        seed=41,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )

    assert resolution.purpose == "evaluation_sensitivity:balanced_prevalence"
    assert resolution.method == "downsample"
    assert resolution.source_counts == (3, 1)
    assert resolution.result_counts == (1, 1)
    np.testing.assert_array_equal(validation.labels, original_labels)


def test_missing_persisted_class_is_rejected() -> None:
    with pytest.raises(MLBalanceError, match="missing persisted classes"):
        _resolve(_data("train", [0, 0, 0]), "class_weight")
