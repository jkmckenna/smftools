from __future__ import annotations

import numpy as np
import pytest

from smftools.machine_learning.data.partition_dataset import (
    MLMaterializedPartitionData,
    MLPartitionBatch,
)
from smftools.machine_learning.data.streaming_transforms import (
    MAX_MODE_CARDINALITY,
    fit_feature_transform_streaming,
    plan_transform_fit,
)
from smftools.machine_learning.data.transforms import (
    FeatureTransformSpec,
    MLTransformError,
    fit_feature_transform,
)

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64


def _matrix(n_rows: int, n_positions: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2, size=(n_rows, n_positions)).astype(np.float32)
    values[rng.random(size=values.shape) < 0.25] = np.nan
    return values


def _materialized(matrix: np.ndarray, split: str = "train") -> MLMaterializedPartitionData:
    tensor = matrix[:, :, np.newaxis]
    n_rows, n_positions, _ = tensor.shape
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.arange(10, 10 + n_positions, dtype=np.int64),
        channel_names=("accessibility",),
        values=tensor,
        labels=None,
        observed_mask=np.isfinite(tensor),
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=np.ones((n_positions, 1), dtype=bool),
        padding_mask=np.zeros((n_rows, n_positions), dtype=bool),
    )


class _BatchSource:
    """Yields the same rows as ``_materialized`` in fixed-size batches."""

    def __init__(self, matrix: np.ndarray, batch_size: int) -> None:
        self._matrix = matrix
        self._batch_size = batch_size

    def iter_batches(self, split: str):
        n_rows, n_positions = self._matrix.shape
        for start in range(0, n_rows, self._batch_size):
            chunk = self._matrix[start : start + self._batch_size]
            tensor = chunk[:, :, np.newaxis]
            rows = tensor.shape[0]
            yield MLPartitionBatch(
                order_indices=np.arange(start, start + rows, dtype=np.int64),
                molecule_uids=tuple(f"molecule-{start + i}" for i in range(rows)),
                read_ids=tuple(f"read-{start + i}" for i in range(rows)),
                experiment_uids=("experiment",) * rows,
                modalities=("deaminase",) * rows,
                coordinates=np.arange(10, 10 + n_positions, dtype=np.int64),
                channel_names=("accessibility",),
                values=tensor,
                labels=None,
                observed_mask=np.isfinite(tensor),
                availability_mask=np.ones((rows, 1), dtype=bool),
                design_mask=np.ones((n_positions, 1), dtype=bool),
                padding_mask=np.zeros((rows, n_positions), dtype=bool),
            )


def _stream(source: _BatchSource, spec: FeatureTransformSpec, matrix: np.ndarray):
    n_rows, n_positions = matrix.shape
    return fit_feature_transform_streaming(
        source,
        spec,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        coordinates=tuple(range(10, 10 + n_positions)),
        channel_names=("accessibility",),
        molecule_uids=tuple(f"molecule-{index}" for index in range(n_rows)),
    )


STREAMABLE_SPECS = [
    FeatureTransformSpec(),
    FeatureTransformSpec(imputation="constant", scaling="standard"),
    FeatureTransformSpec(imputation="mean", scaling="none"),
    FeatureTransformSpec(imputation="mean", scaling="standard"),
    FeatureTransformSpec(imputation="most_frequent", scaling="none"),
    FeatureTransformSpec(imputation="most_frequent", scaling="standard"),
]


@pytest.mark.parametrize(
    ("imputation", "scaling", "expected_passes"),
    [
        ("constant", "none", 0),
        ("constant", "standard", 1),
        ("mean", "none", 1),
        ("most_frequent", "none", 1),
        ("mean", "standard", 2),
        ("most_frequent", "standard", 2),
    ],
)
def test_pass_count_is_resolved_from_the_spec_before_any_read(
    imputation: str, scaling: str, expected_passes: int
) -> None:
    plan = plan_transform_fit(FeatureTransformSpec(imputation=imputation, scaling=scaling))

    assert plan.passes == expected_passes
    assert plan.rationale


def test_default_spec_needs_no_data_passes() -> None:
    # The common case declares both statistics rather than learning them, so a
    # streaming fit reads nothing at all.
    plan = plan_transform_fit(FeatureTransformSpec())

    assert plan.passes == 0
    assert not plan.needs_fill_pass
    assert not plan.needs_scaling_pass


def test_median_is_refused_with_actionable_guidance_rather_than_approximated() -> None:
    with pytest.raises(MLTransformError) as error:
        plan_transform_fit(FeatureTransformSpec(imputation="median"))

    message = str(error.value)
    assert "median" in message
    assert "mean" in message and "most_frequent" in message


@pytest.mark.parametrize("spec", STREAMABLE_SPECS, ids=lambda s: f"{s.imputation}-{s.scaling}")
def test_streamed_fit_matches_materialized_fit_exactly(spec: FeatureTransformSpec) -> None:
    matrix = _matrix(48, 6)
    reference = fit_feature_transform(
        _materialized(matrix),
        spec,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )

    streamed = _stream(_BatchSource(matrix, batch_size=7), spec, matrix)

    assert streamed.transform_id == reference.transform_id
    np.testing.assert_array_equal(streamed.fill_values, reference.fill_values)
    np.testing.assert_allclose(streamed.centers, reference.centers)
    np.testing.assert_allclose(streamed.scales, reference.scales)


@pytest.mark.parametrize("spec", STREAMABLE_SPECS, ids=lambda s: f"{s.imputation}-{s.scaling}")
def test_transform_id_does_not_depend_on_batch_size(spec: FeatureTransformSpec) -> None:
    # Regression guard. transform_id used to hash unrounded float64 moments, so
    # summation order leaked into it: batch_size -- a pure performance knob --
    # produced four different identities for one dataset, and therefore four
    # different lineages for models fitted from it.
    matrix = _matrix(48, 6)

    identities = {
        _stream(_BatchSource(matrix, batch_size=size), spec, matrix).transform_id
        for size in (1, 5, 16, 48)
    }

    assert len(identities) == 1


def test_streaming_fit_refuses_roles_other_than_train() -> None:
    matrix = _matrix(12, 4)

    with pytest.raises(MLTransformError, match="train"):
        fit_feature_transform_streaming(
            _BatchSource(matrix, batch_size=4),
            FeatureTransformSpec(),
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
            coordinates=tuple(range(10, 14)),
            channel_names=("accessibility",),
            molecule_uids=tuple(f"molecule-{index}" for index in range(12)),
            split="validation",
        )


def test_most_frequent_refuses_unbounded_cardinality() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.random(size=(MAX_MODE_CARDINALITY * 3, 4)).astype(np.float32)

    with pytest.raises(MLTransformError, match="most_frequent"):
        _stream(
            _BatchSource(matrix, batch_size=8),
            FeatureTransformSpec(imputation="most_frequent"),
            matrix,
        )


def test_columns_without_valid_observations_keep_the_declared_fill_value() -> None:
    matrix = _matrix(24, 4)
    matrix[:, 2] = np.nan
    spec = FeatureTransformSpec(imputation="mean", fill_value=0.5)

    streamed = _stream(_BatchSource(matrix, batch_size=5), spec, matrix)
    reference = fit_feature_transform(
        _materialized(matrix), spec, dataset_snapshot_id=DATASET_ID, split_id=SPLIT_ID
    )

    assert streamed.fill_values[2] == pytest.approx(0.5)
    assert streamed.transform_id == reference.transform_id
