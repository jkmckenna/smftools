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


def _capped(labels, method, cap, seed=None, job_seed=41):
    return resolve_role_balance(
        _data("train", labels),
        _schema(),
        BalancingSpec(train=BalanceRoleSpec(method, max_per_class=cap, seed=seed)),
        seed=job_seed,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )


def test_max_per_class_caps_training_counts() -> None:
    labels = [0] * 10 + [1] * 6
    # downsample: every class at min(cap, smallest class)
    assert _capped(labels, "downsample", 4).result_counts == (4, 4)
    assert _capped(labels, "downsample", 8).result_counts == (6, 6)
    # natural / class_weight: each class capped on its own
    assert _capped(labels, "natural", 8).result_counts == (8, 6)
    weighted = _capped(labels, "class_weight", 8)
    assert weighted.result_counts == (8, 6)
    assert weighted.class_weights.tolist() == pytest.approx([14 / 16, 14 / 12])
    capped = _capped(labels, "downsample", 4)
    assert len(set(capped.selected_indices.tolist())) == 8  # without replacement
    assert capped.max_per_class == 4 and capped.to_dict()["max_per_class"] == 4


def test_cap_seed_redraws_the_cohort_independently_of_the_job_seed() -> None:
    labels = [0] * 50 + [1] * 50

    def drawn(**kwargs):
        return sorted(_capped(labels, "downsample", 10, **kwargs).selected_indices.tolist())

    assert drawn(seed=1, job_seed=5) == drawn(seed=1, job_seed=6)
    assert drawn(seed=1) != drawn(seed=2)
    assert drawn(job_seed=5) != drawn(job_seed=6)  # no cap seed: the job seed draws


def test_uncapped_resolutions_keep_their_identity() -> None:
    plain = _resolve(_data("train", [0, 0, 0, 0, 1, 1]), "class_weight")
    assert "max_per_class" not in plain.to_dict()


@pytest.mark.parametrize(
    ("train", "message"),
    [
        ({"method": "upsample", "max_per_class": 10}, "applies to methods"),
        ({"method": "weighted_sampler", "max_per_class": 10}, "applies to methods"),
        ({"method": "downsample", "max_per_class": 0}, "integer >= 1"),
        ({"method": "downsample", "seed": -1}, "integer >= 0"),
    ],
)
def test_plan_validates_the_cap(train, message) -> None:
    from tests.fixtures.ml_project import ml_plan

    from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

    document = ml_plan().to_dict()
    document["balancing"] = {"capped": {"train": train}}
    document["jobs"]["train"]["balancing"] = "capped"
    with pytest.raises(MLPlanValidationError, match=message):
        parse_ml_plan(document)


def test_cap_applies_only_to_training() -> None:
    from tests.fixtures.ml_project import ml_plan

    from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

    document = ml_plan().to_dict()
    document["balancing"] = {"capped": {"test": {"method": "natural", "max_per_class": 5}}}
    with pytest.raises(MLPlanValidationError):
        parse_ml_plan(document)
