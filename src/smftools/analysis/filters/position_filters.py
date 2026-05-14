"""
position_filters.py — Genomic position selection from AnnData var coordinates.

Functions
---------
build_position_mask   Boolean column mask for a span within adata.var TSS-reindexed coordinates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_position_mask(
    var: pd.DataFrame,
    ref_strand: str,
    span: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a boolean position mask from adata.var for one reference strand.

    Parameters
    ----------
    var        : adata.var DataFrame; must contain column f"{ref_strand}_reindexed".
    ref_strand : e.g. "6B6_top".
    span       : optional (lo, hi) TSS-centred bp span to restrict positions to.

    Returns
    -------
    keep   : bool array of length n_var — True for positions to include.
    coords : float array of TSS-centred coordinates (NaN where not finite).
    """
    reindex_col = f"{ref_strand}_reindexed"
    coords = pd.to_numeric(var[reindex_col], errors="coerce").to_numpy()
    keep = np.isfinite(coords)
    if span is not None:
        keep &= (coords >= span[0]) & (coords <= span[1])
    return keep, coords
