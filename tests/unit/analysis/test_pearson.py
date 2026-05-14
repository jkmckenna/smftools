from __future__ import annotations

import numpy as np
import pytest

from smftools.analysis.compute.pearson import make_ticks, nan_pearson_matrix


# ---------------------------------------------------------------------------
# nan_pearson_matrix
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pearson_shape() -> None:
    X = np.ones((5, 4), dtype=float)
    mat = nan_pearson_matrix(X)
    assert mat.shape == (4, 4)


@pytest.mark.unit
def test_pearson_identical_columns_returns_one() -> None:
    col = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    X = np.column_stack([col, col])
    mat = nan_pearson_matrix(X)
    np.testing.assert_allclose(mat[0, 1], 1.0, atol=1e-10)
    np.testing.assert_allclose(mat[1, 0], 1.0, atol=1e-10)


@pytest.mark.unit
def test_pearson_anticorrelated_columns_returns_minus_one() -> None:
    col = np.array([0.0, 1.0, 0.0, 1.0])
    X = np.column_stack([col, 1.0 - col])
    mat = nan_pearson_matrix(X)
    np.testing.assert_allclose(mat[0, 1], -1.0, atol=1e-10)


@pytest.mark.unit
def test_pearson_symmetric() -> None:
    rng = np.random.default_rng(42)
    X = rng.random((20, 6))
    mat = nan_pearson_matrix(X)
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


@pytest.mark.unit
def test_pearson_with_nans() -> None:
    X = np.array([
        [0.0, 1.0, np.nan],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    mat = nan_pearson_matrix(X)
    assert mat.shape == (3, 3)
    # NaN columns should not propagate to the entire matrix
    assert np.isfinite(mat[0, 1])


@pytest.mark.unit
def test_pearson_zero_variance_column_returns_nan() -> None:
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=float)
    mat = nan_pearson_matrix(X)
    # Column 0 is constant — correlation with anything should be NaN or 0
    assert not np.isfinite(mat[0, 1]) or mat[0, 1] == 0.0


@pytest.mark.unit
def test_pearson_single_column() -> None:
    X = np.array([[0.0], [1.0], [0.0]], dtype=float)
    mat = nan_pearson_matrix(X)
    assert mat.shape == (1, 1)


# ---------------------------------------------------------------------------
# make_ticks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_make_ticks_count() -> None:
    coords = np.linspace(-1000, 500, 100)
    idx, labels = make_ticks(coords, n_ticks=10)
    assert len(idx) == 10
    assert len(labels) == 10


@pytest.mark.unit
def test_make_ticks_indices_in_bounds() -> None:
    coords = np.arange(-500, 500)
    idx, _ = make_ticks(coords, n_ticks=8)
    assert idx[0] == 0
    assert idx[-1] == len(coords) - 1


@pytest.mark.unit
def test_make_ticks_labels_are_strings() -> None:
    coords = np.array([-100.0, 0.0, 100.0])
    _, labels = make_ticks(coords, n_ticks=3)
    assert all(isinstance(l, str) for l in labels)
