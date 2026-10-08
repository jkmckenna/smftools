"""MLR-05: the bootstrap's fast metrics equal scikit-learn's."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from smftools.machine_learning.evaluation import average_precision_at_prevalence
from smftools.machine_learning.orchestration.comparison import (
    _metric_functions,
    average_precision,
    roc_auc,
)

pytestmark = pytest.mark.unit


def _cohort(seed: int, *, ties: bool = False):
    rng = np.random.default_rng(seed)
    truth = rng.random(400) < 0.3
    score = rng.normal(size=400) + truth
    if ties:
        score = np.round(score, 1)  # many tied scores
    return truth, score


@pytest.mark.parametrize("ties", [False, True])
def test_auroc_and_average_precision_match_sklearn(ties: bool) -> None:
    for seed in range(5):
        truth, score = _cohort(seed, ties=ties)
        assert roc_auc(truth, score) == pytest.approx(roc_auc_score(truth, score))
        assert average_precision(truth, score) == pytest.approx(
            average_precision_score(truth, score)
        )
        weights = np.where(truth, 0.5, 2.0)
        assert average_precision(truth, score, weights) == pytest.approx(
            average_precision_score(truth, score, sample_weight=weights)
        )


def test_prevalence_metric_matches_the_evaluation_metric() -> None:
    truth, score = _cohort(3)
    expected = average_precision_at_prevalence(truth, score, prevalence=0.10)
    value = _metric_functions(0.10)["normalized_average_precision_at_prevalence"](truth, score)
    assert value == pytest.approx(expected.normalized_reweighted)


def test_single_class_gives_nan() -> None:
    truth = np.zeros(10, bool)
    assert np.isnan(roc_auc(truth, np.arange(10.0)))
    assert np.isnan(average_precision(truth, np.arange(10.0)))


def test_sign_flip_p_is_exact_for_few_folds() -> None:
    from smftools.machine_learning.orchestration.comparison import sign_flip_p

    # All five differences positive: only the all-plus and all-minus patterns
    # reach the observed |mean| -> 2 / 32, the smallest attainable with 5 folds.
    assert sign_flip_p(np.array([0.5, 0.4, 0.6, 0.3, 0.7])) == pytest.approx(2 / 32)
    # Symmetric differences: every pattern is at least as extreme.
    assert sign_flip_p(np.array([1.0, -1.0])) == pytest.approx(1.0)
    assert sign_flip_p(np.array([])) is None
    # Beyond the exact limit a seeded sample is used, and is reproducible.
    many = np.linspace(0.1, 1.0, 20)
    assert sign_flip_p(many) == sign_flip_p(many) < 0.001


def test_fold_interval_resamples_folds() -> None:
    import pandas as pd

    from smftools.machine_learning.orchestration.comparison import _fold_interval

    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    interval = _fold_interval(values, 0.025, np.random.default_rng(0))
    assert 1.0 <= interval["fold_ci_low"] < 3.0 < interval["fold_ci_high"] <= 5.0
    assert _fold_interval(pd.Series([1.0]), 0.025, np.random.default_rng(0))["fold_ci_low"] is None
