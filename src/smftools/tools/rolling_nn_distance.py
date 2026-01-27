from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def _pack_bool_to_u64(B: np.ndarray) -> np.ndarray:
    """
    Pack a boolean (or 0/1) matrix (n, w) into uint64 blocks (n, ceil(w/64)).
    Safe w.r.t. contiguity/layout.
    """
    B = np.asarray(B, dtype=np.uint8)
    packed_u8 = np.packbits(B, axis=1)  # (n, ceil(w/8)) uint8

    n, nb = packed_u8.shape
    pad = (-nb) % 8
    if pad:
        packed_u8 = np.pad(packed_u8, ((0, 0), (0, pad)), mode="constant", constant_values=0)

    packed_u8 = np.ascontiguousarray(packed_u8)

    # group 8 bytes -> uint64
    packed_u64 = packed_u8.reshape(n, -1, 8).view(np.uint64).reshape(n, -1)
    return packed_u64


def _popcount_u64_matrix(A_u64: np.ndarray) -> np.ndarray:
    """
    Popcount for an array of uint64, vectorized and portable across NumPy versions.

    Returns an integer array with the SAME SHAPE as A_u64.
    """
    A_u64 = np.ascontiguousarray(A_u64)
    # View as bytes; IMPORTANT: reshape to add a trailing byte axis of length 8
    b = A_u64.view(np.uint8).reshape(A_u64.shape + (8,))
    # unpack bits within that byte axis -> (..., 64), then sum
    return np.unpackbits(b, axis=-1).sum(axis=-1)


def rolling_window_nn_distance(
    adata,
    layer: Optional[str] = None,
    window: int = 15,
    step: int = 2,
    min_overlap: int = 10,
    return_fraction: bool = True,
    block_rows: int = 256,
    block_cols: int = 2048,
    store_obsm: Optional[str] = "rolling_nn_dist",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling-window nearest-neighbor distance per read, overlap-aware.

    Distance between reads i,j in a window:
      - use only positions where BOTH are observed (non-NaN)
      - require overlap >= min_overlap
      - mismatch = count(x_i != x_j) over overlapped positions
      - distance = mismatch/overlap (if return_fraction) else mismatch

    Returns
    -------
    out : (n_obs, n_windows) float
        Nearest-neighbor distance per read per window (NaN if no valid neighbor).
    starts : (n_windows,) int
        Window start indices in var-space.
    """
    X = adata.layers[layer] if layer is not None else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    n, p = X.shape
    if window > p:
        raise ValueError(f"window={window} is larger than n_vars={p}")
    if window <= 0:
        raise ValueError("window must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if min_overlap <= 0:
        raise ValueError("min_overlap must be > 0")

    starts = np.arange(0, p - window + 1, step, dtype=int)
    nW = len(starts)
    out = np.full((n, nW), np.nan, dtype=float)

    for wi, s in enumerate(starts):
        wX = X[:, s : s + window]  # (n, window)

        # observed mask; values as 0/1 where observed, 0 elsewhere
        M = ~np.isnan(wX)
        V = np.where(M, wX, 0).astype(np.float32)

        # ensure binary 0/1
        V = (V > 0).astype(np.uint8)

        M64 = _pack_bool_to_u64(M)
        V64 = _pack_bool_to_u64(V.astype(bool))

        best = np.full(n, np.inf, dtype=float)

        for i0 in range(0, n, block_rows):
            i1 = min(n, i0 + block_rows)
            Mi = M64[i0:i1]  # (bi, nb)
            Vi = V64[i0:i1]
            bi = i1 - i0

            local_best = np.full(bi, np.inf, dtype=float)

            for j0 in range(0, n, block_cols):
                j1 = min(n, j0 + block_cols)
                Mj = M64[j0:j1]  # (bj, nb)
                Vj = V64[j0:j1]
                bj = j1 - j0

                overlap_counts = np.zeros((bi, bj), dtype=np.uint16)
                mismatch_counts = np.zeros((bi, bj), dtype=np.uint16)

                for k in range(Mi.shape[1]):
                    ob = (Mi[:, k][:, None] & Mj[:, k][None, :]).astype(np.uint64)
                    overlap_counts += _popcount_u64_matrix(ob).astype(np.uint16)

                    mb = ((Vi[:, k][:, None] ^ Vj[:, k][None, :]) & ob).astype(np.uint64)
                    mismatch_counts += _popcount_u64_matrix(mb).astype(np.uint16)

                ok = overlap_counts >= min_overlap
                if not np.any(ok):
                    continue

                dist = np.full((bi, bj), np.inf, dtype=float)
                if return_fraction:
                    dist[ok] = mismatch_counts[ok] / overlap_counts[ok]
                else:
                    dist[ok] = mismatch_counts[ok].astype(float)

                # exclude self comparisons (diagonal) when blocks overlap
                if (i0 <= j1) and (j0 <= i1):
                    ii = np.arange(i0, i1)
                    jj = ii[(ii >= j0) & (ii < j1)]
                    if jj.size:
                        dist[(jj - i0), (jj - j0)] = np.inf

                local_best = np.minimum(local_best, dist.min(axis=1))

            best[i0:i1] = local_best

        best[~np.isfinite(best)] = np.nan
        out[:, wi] = best

    if store_obsm is not None:
        adata.obsm[store_obsm] = out
        adata.uns[f"{store_obsm}_starts"] = starts
        adata.uns[f"{store_obsm}_window"] = int(window)
        adata.uns[f"{store_obsm}_step"] = int(step)
        adata.uns[f"{store_obsm}_min_overlap"] = int(min_overlap)
        adata.uns[f"{store_obsm}_return_fraction"] = bool(return_fraction)
        adata.uns[f"{store_obsm}_layer"] = layer if layer is not None else "X"

    return out, starts


def assign_rolling_nn_results(
    parent_adata: "ad.AnnData",
    subset_adata: "ad.AnnData",
    values: np.ndarray,
    starts: np.ndarray,
    obsm_key: str,
    window: int,
    step: int,
    min_overlap: int,
    return_fraction: bool,
    layer: Optional[str],
) -> None:
    """
    Assign rolling NN results computed on a subset back onto a parent AnnData.

    Parameters
    ----------
    parent_adata : AnnData
        Parent AnnData that should store the combined results.
    subset_adata : AnnData
        Subset AnnData used to compute `values`.
    values : np.ndarray
        Rolling NN output with shape (n_subset_obs, n_windows).
    starts : np.ndarray
        Window start indices corresponding to `values`.
    obsm_key : str
        Key to store results under in parent_adata.obsm.
    window : int
        Rolling window size (stored in parent_adata.uns).
    step : int
        Rolling window step size (stored in parent_adata.uns).
    min_overlap : int
        Minimum overlap (stored in parent_adata.uns).
    return_fraction : bool
        Whether distances are fractional (stored in parent_adata.uns).
    layer : str | None
        Layer used for calculations (stored in parent_adata.uns).
    """
    n_obs = parent_adata.n_obs
    n_windows = values.shape[1]

    if obsm_key not in parent_adata.obsm:
        parent_adata.obsm[obsm_key] = np.full((n_obs, n_windows), np.nan, dtype=float)
        parent_adata.uns[f"{obsm_key}_starts"] = starts
        parent_adata.uns[f"{obsm_key}_window"] = int(window)
        parent_adata.uns[f"{obsm_key}_step"] = int(step)
        parent_adata.uns[f"{obsm_key}_min_overlap"] = int(min_overlap)
        parent_adata.uns[f"{obsm_key}_return_fraction"] = bool(return_fraction)
        parent_adata.uns[f"{obsm_key}_layer"] = layer if layer is not None else "X"
    else:
        existing = parent_adata.obsm[obsm_key]
        if existing.shape[1] != n_windows:
            raise ValueError(
                f"Existing obsm[{obsm_key!r}] has {existing.shape[1]} windows; "
                f"new values have {n_windows} windows."
            )
        existing_starts = parent_adata.uns.get(f"{obsm_key}_starts")
        if existing_starts is not None and not np.array_equal(existing_starts, starts):
            raise ValueError(
                f"Existing obsm[{obsm_key!r}] has different window starts than new values."
            )

    parent_indexer = parent_adata.obs_names.get_indexer(subset_adata.obs_names)
    if (parent_indexer < 0).any():
        raise ValueError("Subset AnnData contains obs not present in parent AnnData.")

    parent_adata.obsm[obsm_key][parent_indexer, :] = values
