from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from math import floor

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def _window_center_coordinates(adata, starts: np.ndarray, window: int) -> np.ndarray:
    """
    Compute window center coordinates using AnnData var positions.

    If coordinates are numeric, return the mean coordinate per window.
    If not numeric, return the midpoint label for each window.
    """
    coord_source = adata.var_names

    coords = np.asarray(coord_source)
    if coords.size == 0:
        return np.array([], dtype=float)

    try:
        coords_numeric = coords.astype(float)
        return np.array(
            [floor(np.nanmean(coords_numeric[s : s + window])) for s in starts], dtype=float
        )
    except Exception:
        mid = np.clip(starts + (window // 2), 0, coords.size - 1)
        return coords[mid]


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
    collect_zero_pairs: bool = False,
    zero_pairs_uns_key: Optional[str] = None,
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
    centers : (n_windows,) array-like
        Window center coordinates derived from AnnData var positions (stored in ``.uns``).
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

    zero_pairs_by_window = [] if collect_zero_pairs else None

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

        window_pairs = [] if collect_zero_pairs else None

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

                if collect_zero_pairs:
                    zero_mask = ok & (mismatch_counts == 0)
                    if np.any(zero_mask):
                        i_idx, j_idx = np.where(zero_mask)
                        gi = i0 + i_idx
                        gj = j0 + j_idx
                        keep = gi < gj
                        if np.any(keep):
                            window_pairs.append(np.stack([gi[keep], gj[keep]], axis=1))

            best[i0:i1] = local_best

        best[~np.isfinite(best)] = np.nan
        out[:, wi] = best
        if collect_zero_pairs:
            if window_pairs:
                zero_pairs_by_window.append(np.vstack(window_pairs))
            else:
                zero_pairs_by_window.append(np.empty((0, 2), dtype=int))

    if store_obsm is not None:
        adata.obsm[store_obsm] = out
        adata.uns[f"{store_obsm}_starts"] = starts
        adata.uns[f"{store_obsm}_centers"] = _window_center_coordinates(adata, starts, window)
        adata.uns[f"{store_obsm}_window"] = int(window)
        adata.uns[f"{store_obsm}_step"] = int(step)
        adata.uns[f"{store_obsm}_min_overlap"] = int(min_overlap)
        adata.uns[f"{store_obsm}_return_fraction"] = bool(return_fraction)
        adata.uns[f"{store_obsm}_layer"] = layer if layer is not None else "X"
    if collect_zero_pairs:
        if zero_pairs_uns_key is None:
            zero_pairs_uns_key = (
                f"{store_obsm}_zero_pairs" if store_obsm is not None else "rolling_nn_zero_pairs"
            )
        adata.uns[zero_pairs_uns_key] = zero_pairs_by_window
        adata.uns[f"{zero_pairs_uns_key}_starts"] = starts
        adata.uns[f"{zero_pairs_uns_key}_window"] = int(window)
        adata.uns[f"{zero_pairs_uns_key}_step"] = int(step)
        adata.uns[f"{zero_pairs_uns_key}_min_overlap"] = int(min_overlap)
        adata.uns[f"{zero_pairs_uns_key}_return_fraction"] = bool(return_fraction)
        adata.uns[f"{zero_pairs_uns_key}_layer"] = layer if layer is not None else "X"

    return out, starts


def annotate_zero_hamming_segments(
    adata,
    zero_pairs_uns_key: Optional[str] = None,
    output_uns_key: str = "zero_hamming_segments",
    layer: Optional[str] = None,
    min_overlap: Optional[int] = None,
    refine_segments: bool = True,
    binary_layer_key: Optional[str] = None,
    parent_adata: Optional["ad.AnnData"] = None,
) -> list[dict]:
    """
    Merge zero-Hamming windows into maximal segments and annotate onto AnnData.

    Args:
        adata: AnnData containing zero-pair window data in ``.uns``.
        zero_pairs_uns_key: Key for zero-pair window data in ``adata.uns``.
        output_uns_key: Key to store merged/refined segments in ``adata.uns``.
        layer: Layer to use for refinement (defaults to adata.X).
        min_overlap: Minimum overlap required to keep a refined segment.
        refine_segments: Whether to refine merged windows to maximal segments.
        binary_layer_key: Layer key to store a binary span annotation.
        parent_adata: Parent AnnData to receive the binary layer (defaults to adata).

    Returns:
        List of segment records stored in ``adata.uns[output_uns_key]``.
    """
    if zero_pairs_uns_key is None:
        candidate_keys = [key for key in adata.uns if key.endswith("_zero_pairs")]
        if len(candidate_keys) == 1:
            zero_pairs_uns_key = candidate_keys[0]
        elif not candidate_keys:
            raise KeyError("No zero-pair data found in adata.uns.")
        else:
            raise KeyError(
                "Multiple zero-pair keys found in adata.uns; please specify zero_pairs_uns_key."
            )

    if zero_pairs_uns_key not in adata.uns:
        raise KeyError(f"Missing zero-pair data in adata.uns[{zero_pairs_uns_key!r}].")

    zero_pairs_by_window = adata.uns[zero_pairs_uns_key]
    starts = np.asarray(adata.uns.get(f"{zero_pairs_uns_key}_starts"))
    window = int(adata.uns.get(f"{zero_pairs_uns_key}_window", 0))
    if starts.size == 0 or window <= 0:
        raise ValueError("Zero-pair metadata missing starts/window information.")

    if min_overlap is None:
        min_overlap = int(adata.uns.get(f"{zero_pairs_uns_key}_min_overlap", 1))

    X = adata.layers[layer] if layer is not None else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    observed = ~np.isnan(X)
    values = (np.where(observed, X, 0.0) > 0).astype(np.uint8)

    pair_segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for wi, pairs in enumerate(zero_pairs_by_window):
        if pairs is None or len(pairs) == 0:
            continue
        start = int(starts[wi])
        end = start + window
        for i, j in pairs:
            key = (int(i), int(j))
            pair_segments.setdefault(key, []).append((start, end))

    def _merge_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda seg: seg[0])
        merged = [segments[0]]
        for seg_start, seg_end in segments[1:]:
            last_start, last_end = merged[-1]
            if seg_start <= last_end:
                merged[-1] = (last_start, max(last_end, seg_end))
            else:
                merged.append((seg_start, seg_end))
        return merged

    def _refine_segment(
        read_i: int,
        read_j: int,
        start: int,
        end: int,
    ) -> Optional[tuple[int, int]]:
        if not refine_segments:
            return (start, end)
        left = start
        right = end
        while left > 0:
            idx = left - 1
            if observed[read_i, idx] and observed[read_j, idx]:
                if values[read_i, idx] != values[read_j, idx]:
                    break
            left -= 1
        n_vars = values.shape[1]
        while right < n_vars:
            idx = right
            if observed[read_i, idx] and observed[read_j, idx]:
                if values[read_i, idx] != values[read_j, idx]:
                    break
            right += 1
        overlap = np.sum(observed[read_i, left:right] & observed[read_j, left:right])
        if overlap < min_overlap:
            return None
        return (left, right)

    records: list[dict] = []
    obs_names = adata.obs_names
    for (read_i, read_j), segments in pair_segments.items():
        merged = _merge_segments(segments)
        refined_segments = []
        for seg_start, seg_end in merged:
            refined = _refine_segment(read_i, read_j, seg_start, seg_end)
            if refined is not None:
                refined_segments.append(refined)
        if refined_segments:
            records.append(
                {
                    "read_i": read_i,
                    "read_j": read_j,
                    "read_i_name": str(obs_names[read_i]),
                    "read_j_name": str(obs_names[read_j]),
                    "segments": refined_segments,
                }
            )

    adata.uns[output_uns_key] = records
    if binary_layer_key is not None:
        target = parent_adata if parent_adata is not None else adata
        target_layer = np.zeros((target.n_obs, target.n_vars), dtype=np.uint8)
        target_indexer = target.obs_names.get_indexer(obs_names)
        if (target_indexer < 0).any():
            raise ValueError("Provided parent_adata does not contain all subset obs names.")
        var_indexer = target.var_names.get_indexer(adata.var_names)
        if (var_indexer < 0).any():
            raise ValueError("Provided parent_adata does not contain all subset var names.")
        for record in records:
            read_i = int(record["read_i"])
            read_j = int(record["read_j"])
            target_i = target_indexer[read_i]
            target_j = target_indexer[read_j]
            for seg_start, seg_end in record["segments"]:
                seg_slice = var_indexer[seg_start:seg_end]
                target_layer[target_i, seg_slice] = 1
                target_layer[target_j, seg_slice] = 1
        target.layers[binary_layer_key] = target_layer
        target.uns[f"{binary_layer_key}_source"] = output_uns_key
    return records


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
        parent_adata.uns[f"{obsm_key}_centers"] = _window_center_coordinates(
            subset_adata, starts, window
        )
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
        existing_centers = parent_adata.uns.get(f"{obsm_key}_centers")
        if existing_centers is not None:
            expected_centers = _window_center_coordinates(subset_adata, starts, window)
            if not np.array_equal(existing_centers, expected_centers):
                raise ValueError(
                    f"Existing obsm[{obsm_key!r}] has different window centers than new values."
                )

    parent_indexer = parent_adata.obs_names.get_indexer(subset_adata.obs_names)
    if (parent_indexer < 0).any():
        raise ValueError("Subset AnnData contains obs not present in parent AnnData.")

    parent_adata.obsm[obsm_key][parent_indexer, :] = values
