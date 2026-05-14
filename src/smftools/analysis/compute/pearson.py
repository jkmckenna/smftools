"""
pearson.py — NaN-aware position × position Pearson correlation matrices.

Functions
---------
nan_pearson_matrix   Compute Pearson matrix from a reads × positions float array.
make_ticks           Evenly spaced tick positions and labels for coordinate axes.
"""

from __future__ import annotations

import numpy as np


def nan_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """
    NaN-aware position × position Pearson correlation.

    Parameters
    ----------
    X : (n_reads × n_positions) float array; values typically in {0, 1, NaN}.
        NaN positions are excluded from column means, then zero-filled before
        the matrix multiply (consistent with the reference EMseq implementation).

    Returns
    -------
    mat : (n_positions × n_positions) float array; NaN where denom == 0.
    """
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(X, axis=0)
        Xc  = X - col_mean
        Xc0 = np.nan_to_num(Xc, nan=0.0)
        cov = Xc0.T @ Xc0
        col_norm = np.sqrt((Xc0 ** 2).sum(axis=0))
        denom = col_norm[:, None] * col_norm[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.where(denom != 0.0, cov / denom, np.nan)
    return mat


def make_ticks(
    coords: np.ndarray,
    n_ticks: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """
    Return evenly spaced tick indices and formatted labels for a coordinate array.

    Parameters
    ----------
    coords  : 1-D array of TSS-centred integer coordinates.
    n_ticks : number of ticks to generate.

    Returns
    -------
    indices : integer indices into coords
    labels  : formatted coordinate strings
    """
    idx = np.linspace(0, len(coords) - 1, n_ticks, dtype=int)
    return idx, [f"{coords[i]:.0f}" for i in idx]
