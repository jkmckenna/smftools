"""
Composable obs-level filters for AnnData.

Key functions: :func:`max_cigar_deletion`, :func:`build_obs_mask`.

Example::

    from smftools.analysis.filters.obs_filters import build_obs_mask

    mask = build_obs_mask(
        adata.obs,
        barcode="NB01",
        ref_strand="6B6_top",
        demux_type="double",
        wt_ref_strands=["6B6_top", "6B6_bottom"],
        max_cigar_del=200,
    )
    idx = np.flatnonzero(mask)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def max_cigar_deletion(cigar: str) -> int:
    """Return the largest single deletion length in a CIGAR string, or 0."""
    dels = [int(n) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", str(cigar)) if op == "D"]
    return max(dels) if dels else 0


def build_obs_mask(
    obs: pd.DataFrame,
    barcode: str | None = None,
    ref_strand: str | None = None,
    demux_type: str | None = None,
    wt_ref_strands: list[str] | None = None,
    max_cigar_del: int | None = None,
    extra_eq: dict[str, Any] | None = None,
    barcode_col: str = "Barcode",
    ref_strand_col: str = "Reference_strand",
    demux_col: str = "demux_type",
    cigar_col: str = "CIGAR",
) -> np.ndarray:
    """
    Build a boolean obs mask with composable read filters.

    Parameters
    ----------
    obs            : adata.obs DataFrame.
    barcode        : keep reads with this barcode (obs[barcode_col] == barcode).
    ref_strand     : keep reads mapped to this reference strand.
    demux_type     : keep reads with this demux_type (e.g. "double").
    wt_ref_strands : if ref_strand is in this list, apply the CIGAR deletion filter.
    max_cigar_del : int, optional
        Maximum allowed single deletion in the CIGAR string (bp). Applied only
        when ref_strand is in wt_ref_strands. Use to exclude enh-del allele
        reads misassigned to the WT reference.
    extra_eq       : additional equality filters: {column: value}.
    barcode_col    : obs column for barcode identity.
    ref_strand_col : obs column for reference strand.
    demux_col      : obs column for demultiplexing type.
    cigar_col      : obs column containing CIGAR strings (used for CIGAR filter).

    Returns
    -------
    mask : bool ndarray of length n_obs.
    """
    mask = np.ones(len(obs), dtype=bool)

    if barcode is not None:
        mask &= obs[barcode_col].values == barcode

    if ref_strand is not None:
        mask &= obs[ref_strand_col].values == ref_strand

    if demux_type is not None:
        mask &= obs[demux_col].values == demux_type

    if extra_eq:
        for col, val in extra_eq.items():
            mask &= obs[col].values == val

    # CIGAR deletion filter — only for WT-strand reads where enh-del reads
    # can be misassigned by minimap2 chaining bias (ISSUE-01 in 260406 CLAUDE.md).
    if (
        max_cigar_del is not None
        and ref_strand is not None
        and wt_ref_strands is not None
        and ref_strand in wt_ref_strands
        and cigar_col in obs.columns
    ):
        cigar_vals = obs.loc[mask, cigar_col].apply(max_cigar_deletion).values
        updated = mask.copy()
        updated[mask] = cigar_vals <= max_cigar_del
        mask = updated

    return mask
