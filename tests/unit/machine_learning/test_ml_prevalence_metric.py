"""MLR-01: average precision at a fixed positive prevalence."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from smftools.machine_learning.evaluation import (
    EvaluationContractError,
    average_precision_at_prevalence,
)


def _cohort(n_positive: int, n_negative: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = np.r_[np.ones(n_positive, bool), np.zeros(n_negative, bool)]
    score = rng.normal(size=truth.size) + 1.0 * truth
    return truth, score


def test_at_the_cohorts_own_prevalence_it_is_plain_average_precision() -> None:
    truth, score = _cohort(100, 400)  # 20 % positive
    result = average_precision_at_prevalence(truth, score, prevalence=0.20)
    assert result.reweighted == pytest.approx(average_precision_score(truth, score))
    assert result.n_positive_subsampled == 100 and result.n_negative_subsampled == 400


def test_reweighting_and_subsampling_agree() -> None:
    truth, score = _cohort(600, 1400)  # 30 % positive, thinned to 10 %
    result = average_precision_at_prevalence(truth, score, prevalence=0.10, draws=200)
    assert result.subsampled == pytest.approx(result.reweighted, abs=0.02)
    assert result.n_negative_subsampled == 1400
    assert result.n_positive_subsampled == round(1400 / 9)
    assert result.normalized_reweighted == pytest.approx(result.reweighted / 0.10)


def test_too_few_positives_thins_the_negatives() -> None:
    truth, score = _cohort(20, 1000)  # 2 % positive
    result = average_precision_at_prevalence(truth, score, prevalence=0.10)
    assert result.n_positive_subsampled == 20
    assert result.n_negative_subsampled == 180


def test_subsampling_is_seeded() -> None:
    truth, score = _cohort(300, 700)
    first = average_precision_at_prevalence(truth, score, seed=3)
    second = average_precision_at_prevalence(truth, score, seed=3)
    assert first == second


def test_a_missing_class_gives_none() -> None:
    truth = np.zeros(10, bool)
    assert average_precision_at_prevalence(truth, np.arange(10.0)) is None


@pytest.mark.parametrize("prevalence", [0.0, 1.0, -0.1])
def test_prevalence_must_be_a_proportion(prevalence: float) -> None:
    truth, score = _cohort(10, 10)
    with pytest.raises(EvaluationContractError):
        average_precision_at_prevalence(truth, score, prevalence=prevalence)
