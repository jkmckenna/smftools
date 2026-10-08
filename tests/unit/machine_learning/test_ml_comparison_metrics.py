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
