"""
hmm_features.py — Interval extraction from binary HMM feature layers.

Functions
---------
extract_intervals_from_row   Sizes and neighbor distances for one read's binary HMM row.

Position masking functions live in smftools.analysis.filters.position_filters.
"""

from __future__ import annotations

import numpy as np


def extract_intervals_from_row(
    row: np.ndarray,
    coords: np.ndarray,
    full_row: np.ndarray | None = None,
    full_coords: np.ndarray | None = None,
    max_mask_overlap_frac: float = 0.5,
) -> tuple[list[int], list[float]]:
    """
    Return (sizes_bp, neighbor_center_distances_bp) for one binary HMM row.

    Parameters
    ----------
    row       : 1-D float/int array within the analysis window (0/1/NaN).
    coords    : TSS-centred bp coordinates matching row.
    full_row  : optional — the full-locus row used to clip partial-boundary features.
    full_coords : coordinates for full_row.
    max_mask_overlap_frac : features where more than this fraction lies outside the
                            analysis window (as determined via full_row) are dropped.

    Returns
    -------
    sizes_bp               : list of interval lengths in bp
    neighbor_center_dists  : list of center-to-center distances between adjacent intervals
    """
    binary = np.asarray(row > 0, dtype=bool)
    if binary.size == 0 or not np.any(binary):
        return [], []

    padded = np.concatenate(([False], binary, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1

    sizes: list[int] = []
    centers: list[float] = []

    for start_idx, end_idx in zip(starts, ends):
        start_bp = float(coords[start_idx])
        end_bp = float(coords[end_idx])

        if full_row is not None and full_coords is not None:
            s_match = np.flatnonzero(full_coords == start_bp)
            e_match = np.flatnonzero(full_coords == end_bp)
            if s_match.size and e_match.size:
                full_binary = np.asarray(full_row > 0, dtype=bool)
                fs, fe = int(s_match[0]), int(e_match[-1])
                while fs > 0 and full_binary[fs - 1]:
                    fs -= 1
                while fe < len(full_binary) - 1 and full_binary[fe + 1]:
                    fe += 1
                clipped_len = end_bp - start_bp + 1.0
                full_len = float(full_coords[fe] - full_coords[fs] + 1.0)
                outside_frac = 1.0 - (clipped_len / full_len) if full_len > 0 else 0.0
                if outside_frac > max_mask_overlap_frac:
                    continue

        sizes.append(int(round(end_bp - start_bp + 1.0)))
        centers.append((start_bp + end_bp) / 2.0)

    neighbor_dists = np.diff(np.asarray(centers, dtype=float)).tolist() if len(centers) > 1 else []
    return sizes, neighbor_dists
