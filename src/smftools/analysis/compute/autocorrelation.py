"""
autocorrelation.py — NaN-aware binary autocorrelation over irregularly spaced positions.

Core functions
--------------
binary_autocorrelation_with_spacing  Per-read ACF over gapped binary positions.
weighted_mean_autocorr               Combine per-read ACF curves with lag-count weights.
compute_replicate_curve              Coverage-filter a matrix then compute its mean ACF.

These functions are independent of project constants. Pass positions, matrices, and
threshold values as explicit parameters.
"""

from __future__ import annotations

import numpy as np

# Defaults; override per call as needed.
_DEFAULT_MAX_LAG = 1000
_DEFAULT_MIN_COUNT_PER_LAG = 10
_DEFAULT_MIN_COL_COVERAGE = 0.05
_DEFAULT_MIN_ROW_COVERAGE = 0.05
_DEFAULT_MIN_READS = 5


def binary_autocorrelation_with_spacing(
    row: np.ndarray,
    positions: np.ndarray,
    max_lag: int = _DEFAULT_MAX_LAG,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    NaN-aware autocorrelation over irregularly spaced binary positions.

    Parameters
    ----------
    row       : 1-D float array (values 0/1/NaN); NaN = no coverage at that position.
    positions : 1-D int array of TSS-centred coordinates matching row.
    max_lag   : maximum lag in base pairs to compute.
    return_counts : if True, return (ac, counts) where counts[lag] = number of pairs.

    Returns
    -------
    ac        : float32 array of length (max_lag + 1); NaN where counts == 0.
    counts    : int64 array of length (max_lag + 1)  [only if return_counts=True]
    """
    valid = ~np.isnan(row)
    if valid.sum() < 2:
        ac = np.full(max_lag + 1, np.nan, dtype=np.float32)
        counts = np.zeros(max_lag + 1, dtype=np.int64)
        return (ac, counts) if return_counts else ac

    x = row[valid].astype(np.float64)
    pos = positions[valid].astype(np.int64)
    xc = x - x.mean()
    var = np.sum(xc * xc)
    if var == 0.0:
        ac = np.full(max_lag + 1, np.nan, dtype=np.float32)
        counts = np.zeros(max_lag + 1, dtype=np.int64)
        return (ac, counts) if return_counts else ac

    lag_sums = np.zeros(max_lag + 1, dtype=np.float64)
    lag_counts = np.zeros(max_lag + 1, dtype=np.int64)
    n = len(x)
    j = 1
    for i in range(n - 1):
        if j <= i:
            j = i + 1
        while j < n and (pos[j] - pos[i]) <= max_lag:
            j += 1
        if j - i <= 1:
            continue
        diffs = pos[i + 1 : j] - pos[i]
        contrib = xc[i] * xc[i + 1 : j]
        lag_sums += np.bincount(diffs, weights=contrib, minlength=max_lag + 1)[: max_lag + 1]
        lag_counts += np.bincount(diffs, minlength=max_lag + 1)[: max_lag + 1]

    ac = np.full(max_lag + 1, np.nan, dtype=np.float64)
    nz = lag_counts > 0
    ac[nz] = (lag_sums[nz] / lag_counts[nz]) / (var / len(x))
    ac[0] = 1.0
    if return_counts:
        return ac.astype(np.float32), lag_counts
    return ac.astype(np.float32)


def weighted_mean_autocorr(
    ac_matrix: np.ndarray,
    counts_matrix: np.ndarray,
    min_count_per_lag: int = _DEFAULT_MIN_COUNT_PER_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine read-wise ACF curves using lag-specific pair counts as weights.

    Parameters
    ----------
    ac_matrix     : (n_reads × n_lags) float; NaN where a read had no pairs at that lag.
    counts_matrix : (n_reads × n_lags) int; number of pairs per read per lag.
    min_count_per_lag : lags with fewer total pairs are set to NaN in the output.

    Returns
    -------
    mean_ac      : 1-D float array of length n_lags
    total_counts : 1-D int array of length n_lags
    """
    total_counts = counts_matrix.sum(axis=0)
    filled = np.where(np.isfinite(ac_matrix), ac_matrix, 0.0)
    numerator = (filled * counts_matrix).sum(axis=0)
    mean_ac = np.full(ac_matrix.shape[1], np.nan, dtype=float)
    nz = total_counts > 0
    mean_ac[nz] = numerator[nz] / total_counts[nz]
    mean_ac[total_counts < min_count_per_lag] = np.nan
    return mean_ac, total_counts


def compute_replicate_curve(
    mat: np.ndarray,
    positions: np.ndarray,
    max_lag: int = _DEFAULT_MAX_LAG,
    min_col_coverage: float = _DEFAULT_MIN_COL_COVERAGE,
    min_row_coverage: float = _DEFAULT_MIN_ROW_COVERAGE,
    min_reads: int = _DEFAULT_MIN_READS,
    min_count_per_lag: int = _DEFAULT_MIN_COUNT_PER_LAG,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Coverage-filter a read × position matrix then compute its weighted mean ACF.

    Parameters
    ----------
    mat       : (n_reads × n_positions) float; NaN = no coverage.
    positions : TSS-centred int coordinates matching mat columns.

    Returns (mean_ac, total_counts) or None if too few reads survive filtering.
    """
    col_cov = np.mean(~np.isnan(mat), axis=0)
    keep_cols = (col_cov >= min_col_coverage) & np.isfinite(positions)
    mat = mat[:, keep_cols]
    pos = positions[keep_cols]
    if mat.shape[1] < 2:
        return None

    row_cov = np.mean(~np.isnan(mat), axis=1)
    mat = mat[row_cov >= min_row_coverage, :]
    if mat.shape[0] < min_reads:
        return None

    order = np.argsort(pos)
    mat = mat[:, order]
    pos = pos[order].astype(int)

    ac_rows, count_rows = [], []
    for row in mat:
        ac, counts = binary_autocorrelation_with_spacing(
            row, pos, max_lag=max_lag, return_counts=True
        )
        ac_rows.append(ac)
        count_rows.append(counts)

    return weighted_mean_autocorr(
        np.vstack(ac_rows), np.vstack(count_rows), min_count_per_lag=min_count_per_lag
    )
