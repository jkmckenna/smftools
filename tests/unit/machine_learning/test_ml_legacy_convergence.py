"""Compatibility and ownership checks for the staged legacy ML transition."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute import ml_cnn, ml_explanations, ml_metrics
from smftools.analysis.compute.ml_splits import build_leave_one_group_out_splits
from smftools.machine_learning.compatibility._warnings import LEGACY_ML_REMOVAL_VERSION
from smftools.machine_learning.inference import run_sklearn_inference, sliding_window_inference
from smftools.machine_learning.models import MLPClassifier
from smftools.machine_learning.training import train_sklearn_model

pytestmark = pytest.mark.unit


def test_analysis_matrix_cnn_delegates_to_machine_learning_compatibility() -> None:
    values = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)

    with pytest.warns(FutureWarning, match=r"build_cnn_input.*3\.0\.0.*PartitionDataset"):
        encoded = ml_cnn.build_cnn_input(values)

    assert encoded.shape == (2, 2, 2)
    assert ml_cnn.build_cnn_input.__wrapped__.__module__ == (
        "smftools.machine_learning.compatibility.matrix_cnn"
    )


def test_analysis_sklearn_compatibility_preserves_fit_and_prediction() -> None:
    features = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    with pytest.warns(FutureWarning, match=r"build_binary_classifier.*BUILTIN_MODEL_REGISTRY"):
        estimator = ml_metrics.build_binary_classifier("bernoulli_nb")
    with pytest.warns(FutureWarning, match=r"fit_classifier.*train_partition_model"):
        fitted = ml_metrics.fit_classifier(estimator, features, labels)
    with pytest.warns(FutureWarning, match=r"predict_binary_scores.*apply_partition_model"):
        scores = ml_metrics.predict_binary_scores(fitted, features)

    assert scores.shape == (4,)
    assert scores[0] < scores[-1]
    assert ml_metrics.fit_classifier.__wrapped__.__module__ == (
        "smftools.machine_learning.compatibility.classical_models"
    )


def test_analysis_explanation_delegates_and_preserves_nb_identity() -> None:
    features = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fitted = ml_metrics.fit_classifier(
            ml_metrics.build_binary_classifier("bernoulli_nb", alpha=1.0),
            features,
            labels,
        )

    with pytest.warns(FutureWarning, match=r"bernoulli_nb.*explain_sklearn_model"):
        contributions, prior = ml_explanations.bernoulli_nb_logodds_contributions(
            fitted,
            features,
        )

    reconstructed = prior + contributions.sum(axis=1)
    expected = fitted.predict_log_proba(features)[:, 1] - fitted.predict_log_proba(features)[:, 0]
    np.testing.assert_allclose(reconstructed, expected)
    assert ml_explanations.bernoulli_nb_logodds_contributions.__wrapped__.__module__ == (
        "smftools.machine_learning.compatibility.classical_explanations"
    )


def test_pure_analysis_metric_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = ml_metrics.logit_from_probability(np.array([0.25, 0.75]))

    assert not caught
    np.testing.assert_allclose(values, [-1.09861229, 1.09861229])


def test_legacy_split_warns_but_preserves_group_disjoint_folds() -> None:
    metadata = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
        }
    )

    with pytest.warns(FutureWarning, match=r"build_leave_one_group_out_splits.*plan_ml_splits"):
        folds = build_leave_one_group_out_splits(metadata, "group", "label")

    assert len(folds) == 2
    for fold in folds:
        assert set(fold["train_idx"]).isdisjoint(fold["test_idx"])


@pytest.mark.parametrize(
    ("function", "symbol"),
    [
        (train_sklearn_model, "train_sklearn_model"),
        (run_sklearn_inference, "run_sklearn_inference"),
        (sliding_window_inference, "sliding_window_inference"),
    ],
)
def test_legacy_job_entry_points_warn_before_delegating(function, symbol: str) -> None:
    with pytest.warns(FutureWarning, match=rf"{symbol}.*{LEGACY_ML_REMOVAL_VERSION}"):
        with pytest.raises(TypeError):
            function()


def test_prototype_model_warns_with_removal_version() -> None:
    with pytest.warns(FutureWarning, match=rf"MLPClassifier.*{LEGACY_ML_REMOVAL_VERSION}"):
        model = MLPClassifier(input_dim=2, hidden_dims=[2], use_batchnorm=False)

    assert model.model is not None
