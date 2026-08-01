from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.data.transforms import (
    FeatureTransformSpec,
    FittedFeatureTransform,
    MLTransformError,
    TorchFeatureTransform,
    build_sklearn_preprocessing_pipeline,
    fit_feature_transform,
)

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64


def _data(
    split: str,
    values: list[list[float]],
    labels: list[int] | None = None,
    *,
    observed_mask: np.ndarray | None = None,
    design_mask: np.ndarray | None = None,
) -> MLMaterializedPartitionData:
    matrix = np.asarray(values, dtype=np.float32)
    tensor = matrix[:, :, np.newaxis]
    n_rows, n_positions, _ = tensor.shape
    observed = (
        np.isfinite(tensor) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    )
    design = (
        np.ones((n_positions, 1), dtype=bool)
        if design_mask is None
        else np.asarray(design_mask, dtype=bool)
    )
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"{split}-molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.arange(10, 10 + n_positions, dtype=np.int64),
        channel_names=("accessibility",),
        values=tensor,
        labels=None if labels is None else np.asarray(labels, dtype=np.int64),
        observed_mask=observed,
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=design,
        padding_mask=np.zeros((n_rows, n_positions), dtype=bool),
    )


def test_fit_uses_training_values_only_and_reuses_immutable_state() -> None:
    train = _data("train", [[1.0, np.nan], [3.0, 5.0]], [0, 1])
    validation = _data("validation", [[101.0, np.nan], [103.0, 205.0]], [0, 1])
    fitted = fit_feature_transform(
        train,
        FeatureTransformSpec(imputation="mean", scaling="standard", indicators=()),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )

    assert fitted.fill_values.tolist() == pytest.approx([2.0, 5.0])
    assert fitted.centers.tolist() == pytest.approx([2.0, 5.0])
    before = fitted.to_dict()
    transformed = fitted.transform(validation)

    assert transformed[0, 0] == pytest.approx(99.0)
    assert transformed[0, 1] == pytest.approx(0.0)
    assert fitted.to_dict() == before
    assert not fitted.fill_values.flags.writeable


def test_fit_rejects_validation_to_make_leakage_visible() -> None:
    validation = _data("validation", [[1.0, 2.0], [3.0, 4.0]], [0, 1])

    with pytest.raises(MLTransformError, match="only be fit.*train"):
        fit_feature_transform(
            validation,
            FeatureTransformSpec(),
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
        )


def test_observed_and_design_indicators_survive_feature_transformation() -> None:
    observed = np.asarray([[[True], [False]], [[True], [True]]])
    design = np.asarray([[True], [False]])
    train = _data(
        "train",
        [[1.0, 2.0], [3.0, 4.0]],
        [0, 1],
        observed_mask=observed,
        design_mask=design,
    )
    fitted = fit_feature_transform(
        train,
        FeatureTransformSpec(indicators=("observed", "design")),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )

    transformed = fitted.transform(train)

    assert transformed.shape == (2, 6)
    np.testing.assert_array_equal(transformed[:, 2:4], observed.reshape(2, 2))
    np.testing.assert_array_equal(
        transformed[:, 4:6],
        np.broadcast_to(design, train.values.shape).reshape(2, 2),
    )
    assert fitted.feature_names == (
        "signal:accessibility@10",
        "signal:accessibility@11",
        "observed:accessibility@10",
        "observed:accessibility@11",
        "design:accessibility@10",
        "design:accessibility@11",
    )


def test_fitted_state_round_trips_and_rejects_tampering() -> None:
    train = _data("train", [[1.0, np.nan], [3.0, 5.0]], [0, 1])
    fitted = fit_feature_transform(
        train,
        FeatureTransformSpec(imputation="median", scaling="standard"),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )
    restored = FittedFeatureTransform.from_dict(fitted.to_dict())

    np.testing.assert_array_equal(restored.transform(train), fitted.transform(train))
    assert restored.to_dict() == fitted.to_dict()

    tampered = deepcopy(fitted.to_dict())
    tampered["fill_values"][0] += 1.0
    with pytest.raises(MLTransformError, match="transform_id"):
        FittedFeatureTransform.from_dict(tampered)


def test_sklearn_pipeline_and_torch_adapter_share_fitted_state() -> None:
    train = _data("train", [[1.0, np.nan], [3.0, 5.0]], [0, 1])
    validation = _data("validation", [[7.0, 9.0]], [1])
    pipeline = build_sklearn_preprocessing_pipeline(
        FeatureTransformSpec(imputation="mean", indicators=("observed",)),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )
    pipeline.fit(train, train.labels)
    sklearn_features = pipeline.transform(validation)
    fitted = pipeline.named_steps["features"].fitted_transform_

    torch_batch = TorchFeatureTransform(fitted)(validation)

    assert tuple(torch_batch.values.shape) == (1, 1, 2)
    np.testing.assert_array_equal(
        torch_batch.values.numpy().transpose(0, 2, 1).reshape(1, 2),
        sklearn_features[:, :2],
    )
    np.testing.assert_array_equal(torch_batch.labels.numpy(), validation.labels)
    np.testing.assert_array_equal(
        torch_batch.observed_mask.numpy(),
        validation.observed_mask.transpose(0, 2, 1),
    )
    np.testing.assert_array_equal(
        torch_batch.availability_mask.numpy(), validation.availability_mask
    )
    np.testing.assert_array_equal(torch_batch.design_mask.numpy(), validation.design_mask.T)
    np.testing.assert_array_equal(torch_batch.padding_mask.numpy(), validation.padding_mask)
