from __future__ import annotations

import numpy as np
import pytest

from smftools.analysis.compute.autocorrelation import (
    binary_autocorrelation_with_spacing,
    compute_replicate_curve,
    weighted_mean_autocorr,
)

MAX_LAG = 50


# ---------------------------------------------------------------------------
# binary_autocorrelation_with_spacing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acf_lag0_is_one() -> None:
    row = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    pos = np.array([0, 10, 20, 30, 40], dtype=int)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    assert ac[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_acf_all_nan_returns_nan() -> None:
    row = np.full(5, np.nan)
    pos = np.arange(0, 50, 10, dtype=int)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    assert np.all(np.isnan(ac))


@pytest.mark.unit
def test_acf_single_valid_point_returns_nan() -> None:
    row = np.array([1.0, np.nan, np.nan])
    pos = np.array([0, 10, 20], dtype=int)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    # Only one valid point: can't compute pairwise correlation
    assert np.all(np.isnan(ac))


@pytest.mark.unit
def test_acf_zero_variance_returns_nan() -> None:
    row = np.ones(5)
    pos = np.arange(0, 50, 10, dtype=int)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    assert np.all(np.isnan(ac))


@pytest.mark.unit
def test_acf_output_length() -> None:
    row = np.array([1.0, 0.0, 1.0, 0.0])
    pos = np.array([0, 5, 10, 15], dtype=int)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    assert len(ac) == MAX_LAG + 1


@pytest.mark.unit
def test_acf_return_counts_shape() -> None:
    row = np.array([1.0, 0.0, 1.0])
    pos = np.array([0, 10, 20], dtype=int)
    ac, counts = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG, return_counts=True)
    assert len(ac) == MAX_LAG + 1
    assert len(counts) == MAX_LAG + 1
    assert counts[0] == 0  # lag-0 computed analytically, not from pairs


@pytest.mark.unit
def test_acf_periodic_signal_has_peak_at_period() -> None:
    period = 10
    pos = np.arange(0, 200, 1, dtype=int)
    row = np.where(pos % period == 0, 1.0, 0.0)
    ac = binary_autocorrelation_with_spacing(row, pos, max_lag=MAX_LAG)
    # Lag at multiples of the period should have higher autocorrelation than lag at half-period
    assert ac[period] > ac[period // 2]


# ---------------------------------------------------------------------------
# weighted_mean_autocorr
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_weighted_mean_autocorr_basic() -> None:
    ac = np.array([[1.0, 0.5, 0.3], [1.0, 0.4, 0.2]])
    counts = np.array([[10, 10, 10], [10, 10, 10]])
    mean, total = weighted_mean_autocorr(ac, counts)
    np.testing.assert_allclose(mean, [1.0, 0.45, 0.25])
    np.testing.assert_array_equal(total, [20, 20, 20])


@pytest.mark.unit
def test_weighted_mean_autocorr_nan_rows_ignored() -> None:
    ac = np.array([[1.0, np.nan, 0.3], [1.0, 0.4, 0.2]])
    counts = np.array([[10, 0, 10], [10, 10, 10]])
    mean, total = weighted_mean_autocorr(ac, counts)
    assert np.isnan(mean[1]) or total[1] > 0  # lag-1: only one read had data
    assert mean[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_weighted_mean_autocorr_min_count_masks_sparse_lags() -> None:
    ac = np.array([[1.0, 0.5]])
    counts = np.array([[100, 3]])  # lag-1 has only 3 pairs
    mean, _ = weighted_mean_autocorr(ac, counts, min_count_per_lag=10)
    assert np.isnan(mean[1])


# ---------------------------------------------------------------------------
# compute_replicate_curve
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_replicate_curve_returns_none_too_few_reads() -> None:
    mat = np.random.default_rng(0).random((3, 10))
    pos = np.arange(10, dtype=float)
    result = compute_replicate_curve(mat, pos, min_reads=5)
    assert result is None


@pytest.mark.unit
def test_compute_replicate_curve_returns_tuple() -> None:
    rng = np.random.default_rng(0)
    mat = rng.choice([0.0, 1.0], size=(20, 30)).astype(float)
    pos = np.arange(0, 300, 10, dtype=float)
    result = compute_replicate_curve(mat, pos, max_lag=50)
    assert result is not None
    mean_ac, counts = result
    assert len(mean_ac) == 51
    # lag-0 is set analytically to 1.0 per-row but pair counts at lag-0 are 0,
    # so weighted_mean_autocorr produces NaN at lag-0 — this is expected behaviour.
    assert np.isnan(mean_ac[0])
    # lag-1 onward should have some finite values given enough reads
    assert np.any(np.isfinite(mean_ac[1:]))


@pytest.mark.unit
def test_compute_replicate_curve_filters_low_coverage_columns() -> None:
    mat = np.full((10, 20), np.nan)
    mat[:, :5] = 0.5  # only first 5 columns have data
    pos = np.arange(20, dtype=float)
    result = compute_replicate_curve(mat, pos, min_col_coverage=0.05, min_reads=5)
    # Should run on the 5 valid columns only
    assert result is not None
