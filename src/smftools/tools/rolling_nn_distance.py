from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def zero_pairs_to_dataframe(adata, zero_pairs_uns_key: str) -> "pd.DataFrame":
    """
    Build a DataFrame of zero-Hamming pairs per window.

    Args:
        adata: AnnData containing zero-pair window data in ``adata.uns``.
        zero_pairs_uns_key: Key for zero-pair window data in ``adata.uns``.

    Returns:
        DataFrame with one row per zero-Hamming pair per window.
    """
    import pandas as pd

    if zero_pairs_uns_key not in adata.uns:
        raise KeyError(f"Missing zero-pair data in adata.uns[{zero_pairs_uns_key!r}].")

    zero_pairs_by_window = adata.uns[zero_pairs_uns_key]
    starts = np.asarray(adata.uns.get(f"{zero_pairs_uns_key}_starts"))
    window = int(adata.uns.get(f"{zero_pairs_uns_key}_window", 0))
    if starts.size == 0 or window <= 0:
        raise ValueError("Zero-pair metadata missing starts/window information.")

    obs_names = np.asarray(adata.obs_names, dtype=object)
    rows = []
    for wi, pairs in enumerate(zero_pairs_by_window):
        if pairs is None or len(pairs) == 0:
            continue
        start = int(starts[wi])
        end = start + window
        for read_i, read_j in pairs:
            read_i = int(read_i)
            read_j = int(read_j)
            rows.append(
                {
                    "window_index": wi,
                    "window_start": start,
                    "window_end": end,
                    "read_i": read_i,
                    "read_j": read_j,
                    "read_i_name": str(obs_names[read_i]),
                    "read_j_name": str(obs_names[read_j]),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "window_index",
            "window_start",
            "window_end",
            "read_i",
            "read_j",
            "read_i_name",
            "read_j_name",
        ],
    )


def zero_hamming_segments_to_dataframe(
    records: list[dict],
    var_names: np.ndarray,
) -> "pd.DataFrame":
    """
    Build a DataFrame of merged/refined zero-Hamming segments.

    Args:
        records: Output records from ``annotate_zero_hamming_segments``.
        var_names: AnnData var names for labeling segment coordinates.

    Returns:
        DataFrame with one row per zero-Hamming segment.
    """
    import pandas as pd

    var_names = np.asarray(var_names, dtype=object)

    def _label_at(idx: int) -> Optional[str]:
        if 0 <= idx < var_names.size:
            return str(var_names[idx])
        return None

    rows = []
    for record in records:
        read_i = int(record["read_i"])
        read_j = int(record["read_j"])
        read_i_name = record.get("read_i_name")
        read_j_name = record.get("read_j_name")
        for seg_start, seg_end in record.get("segments", []):
            seg_start = int(seg_start)
            seg_end = int(seg_end)
            end_inclusive = max(seg_start, seg_end - 1)
            rows.append(
                {
                    "read_i": read_i,
                    "read_j": read_j,
                    "read_i_name": read_i_name,
                    "read_j_name": read_j_name,
                    "segment_start": seg_start,
                    "segment_end_exclusive": seg_end,
                    "segment_end_inclusive": end_inclusive,
                    "segment_start_label": _label_at(seg_start),
                    "segment_end_label": _label_at(end_inclusive),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "read_i",
            "read_j",
            "read_i_name",
            "read_j_name",
            "segment_start",
            "segment_end_exclusive",
            "segment_end_inclusive",
            "segment_start_label",
            "segment_end_label",
        ],
    )


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
    max_nan_run: Optional[int] = None,
    merge_gap: int = 0,
    max_segments_per_read: Optional[int] = None,
    max_segment_overlap: Optional[int] = None,
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
        max_nan_run: Maximum consecutive NaN positions allowed when expanding segments.
            If reached, expansion stops before the NaN run. Set to ``None`` to ignore NaNs.
        merge_gap: Merge segments with gaps of at most this size (in positions).
        max_segments_per_read: Maximum number of segments to retain per read pair.
        max_segment_overlap: Maximum allowed overlap between retained segments (inclusive, in
            var-index coordinates).

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
    max_nan_run = None if max_nan_run is None else max(int(max_nan_run), 1)

    pair_segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for wi, pairs in enumerate(zero_pairs_by_window):
        if pairs is None or len(pairs) == 0:
            continue
        start = int(starts[wi])
        end = start + window
        for i, j in pairs:
            key = (int(i), int(j))
            pair_segments.setdefault(key, []).append((start, end))

    merge_gap = max(int(merge_gap), 0)

    def _merge_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda seg: seg[0])
        merged = [segments[0]]
        for seg_start, seg_end in segments[1:]:
            last_start, last_end = merged[-1]
            if seg_start <= last_end + merge_gap:
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
        nan_run = 0
        while left > 0:
            idx = left - 1
            both_observed = observed[read_i, idx] and observed[read_j, idx]
            if both_observed:
                nan_run = 0
                if values[read_i, idx] != values[read_j, idx]:
                    break
            else:
                nan_run += 1
                if max_nan_run is not None and nan_run >= max_nan_run:
                    break
            left -= 1
        n_vars = values.shape[1]
        nan_run = 0
        while right < n_vars:
            idx = right
            both_observed = observed[read_i, idx] and observed[read_j, idx]
            if both_observed:
                nan_run = 0
                if values[read_i, idx] != values[read_j, idx]:
                    break
            else:
                nan_run += 1
                if max_nan_run is not None and nan_run >= max_nan_run:
                    break
            right += 1
        overlap = np.sum(observed[read_i, left:right] & observed[read_j, left:right])
        if overlap < min_overlap:
            return None
        return (left, right)

    def _segment_length(segment: tuple[int, int]) -> int:
        return int(segment[1]) - int(segment[0])

    def _segment_overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
        return max(0, min(first[1], second[1]) - max(first[0], second[0]))

    def _select_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not segments:
            return []
        if max_segments_per_read is None and max_segment_overlap is None:
            return segments
        ordered = sorted(
            segments,
            key=lambda seg: (_segment_length(seg), -seg[0]),
            reverse=True,
        )
        max_segments = len(ordered) if max_segments_per_read is None else max_segments_per_read
        if max_segment_overlap is None:
            return ordered[:max_segments]
        selected: list[tuple[int, int]] = []
        for segment in ordered:
            if len(selected) >= max_segments:
                break
            if all(_segment_overlap(segment, other) <= max_segment_overlap for other in selected):
                selected.append(segment)
        return selected

    records: list[dict] = []
    obs_names = adata.obs_names
    for (read_i, read_j), segments in pair_segments.items():
        merged = _merge_segments(segments)
        refined_segments = []
        for seg_start, seg_end in merged:
            refined = _refine_segment(read_i, read_j, seg_start, seg_end)
            if refined is not None:
                refined_segments.append(refined)
        refined_segments = _select_segments(refined_segments)
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
    return records


def assign_per_read_segments_layer(
    parent_adata: "ad.AnnData",
    subset_adata: "ad.AnnData",
    per_read_segments: "pd.DataFrame",
    layer_key: str,
) -> None:
    """
    Assign per-read segments into a summed span layer on a parent AnnData.

    Args:
        parent_adata: AnnData that should receive the span layer.
        subset_adata: AnnData used to compute per-read segments.
    per_read_segments: DataFrame with ``read_id``, ``segment_start``, and
            ``segment_end_exclusive`` columns. If ``segment_start_label`` and
            ``segment_end_label`` are present and numeric, they are used to
            map segments using label coordinates.
        layer_key: Name of the layer to store in ``parent_adata.layers``.
    """
    import pandas as pd

    if per_read_segments.empty:
        parent_adata.layers[layer_key] = np.zeros(
            (parent_adata.n_obs, parent_adata.n_vars), dtype=np.uint16
        )
        return
    required_cols = {"read_id", "segment_start", "segment_end_exclusive"}
    missing = required_cols.difference(per_read_segments.columns)
    if missing:
        raise KeyError(f"per_read_segments missing required columns: {sorted(missing)}")

    target_layer = np.zeros((parent_adata.n_obs, parent_adata.n_vars), dtype=np.uint16)

    parent_obs_indexer = parent_adata.obs_names.get_indexer(subset_adata.obs_names)
    if (parent_obs_indexer < 0).any():
        raise ValueError("Subset AnnData contains obs not present in parent AnnData.")
    parent_var_indexer = parent_adata.var_names.get_indexer(subset_adata.var_names)
    if (parent_var_indexer < 0).any():
        raise ValueError("Subset AnnData contains vars not present in parent AnnData.")

    label_indexer = None
    label_columns = {"segment_start_label", "segment_end_label"}
    if label_columns.issubset(per_read_segments.columns):
        try:
            parent_label_values = [int(label) for label in parent_adata.var_names]
            label_indexer = {label: idx for idx, label in enumerate(parent_label_values)}
        except (TypeError, ValueError):
            label_indexer = None

    def _label_to_index(value: object) -> Optional[int]:
        if label_indexer is None or value is None or pd.isna(value):
            return None
        try:
            return label_indexer.get(int(value))
        except (TypeError, ValueError):
            return None

    for row in per_read_segments.itertuples(index=False):
        read_id = int(row.read_id)
        seg_start = int(row.segment_start)
        seg_end = int(row.segment_end_exclusive)
        if seg_start >= seg_end:
            continue
        target_read = parent_obs_indexer[read_id]
        if target_read < 0:
            raise ValueError("Segment read_id not found in parent AnnData.")

        label_start = _label_to_index(getattr(row, "segment_start_label", None))
        label_end = _label_to_index(getattr(row, "segment_end_label", None))
        if label_start is not None and label_end is not None:
            parent_start = min(label_start, label_end)
            parent_end = max(label_start, label_end)
        else:
            parent_positions = parent_var_indexer[seg_start:seg_end]
            if parent_positions.size == 0:
                continue
            parent_start = int(parent_positions.min())
            parent_end = int(parent_positions.max())

        target_layer[target_read, parent_start : parent_end + 1] += 1

    parent_adata.layers[layer_key] = target_layer


def segments_to_per_read_dataframe(
    records: list[dict],
    var_names: np.ndarray,
) -> "pd.DataFrame":
    """
    Build a per-read DataFrame of zero-Hamming segments.

    Args:
        records: Output records from ``annotate_zero_hamming_segments``.
        var_names: AnnData var names for labeling segment coordinates.

    Returns:
        DataFrame with one row per segment per read.
    """
    import pandas as pd

    var_names = np.asarray(var_names, dtype=object)

    def _label_at(idx: int) -> Optional[str]:
        if 0 <= idx < var_names.size:
            return str(var_names[idx])
        return None

    rows = []
    for record in records:
        read_i = int(record["read_i"])
        read_j = int(record["read_j"])
        read_i_name = record.get("read_i_name")
        read_j_name = record.get("read_j_name")
        for seg_start, seg_end in record.get("segments", []):
            seg_start = int(seg_start)
            seg_end = int(seg_end)
            end_inclusive = max(seg_start, seg_end - 1)
            start_label = _label_at(seg_start)
            end_label = _label_at(end_inclusive)
            rows.append(
                {
                    "read_id": read_i,
                    "partner_id": read_j,
                    "read_name": read_i_name,
                    "partner_name": read_j_name,
                    "segment_start": seg_start,
                    "segment_end_exclusive": seg_end,
                    "segment_end_inclusive": end_inclusive,
                    "segment_start_label": start_label,
                    "segment_end_label": end_label,
                }
            )
            rows.append(
                {
                    "read_id": read_j,
                    "partner_id": read_i,
                    "read_name": read_j_name,
                    "partner_name": read_i_name,
                    "segment_start": seg_start,
                    "segment_end_exclusive": seg_end,
                    "segment_end_inclusive": end_inclusive,
                    "segment_start_label": start_label,
                    "segment_end_label": end_label,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "read_id",
            "partner_id",
            "read_name",
            "partner_name",
            "segment_start",
            "segment_end_exclusive",
            "segment_end_inclusive",
            "segment_start_label",
            "segment_end_label",
        ],
    )


def select_top_segments_per_read(
    records: list[dict],
    var_names: np.ndarray,
    max_segments_per_read: Optional[int] = None,
    max_segment_overlap: Optional[int] = None,
    min_span: Optional[float] = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Select top segments per read from distinct partner pairs.

    Args:
        records: Output records from ``annotate_zero_hamming_segments``.
        var_names: AnnData var names for labeling segment coordinates.
        max_segments_per_read: Maximum number of segments to keep per read.
        max_segment_overlap: Maximum allowed overlap between kept segments.
        min_span: Minimum span length to keep (var-name coordinate if numeric, else index span).

    Returns:
        Tuple of (raw per-read segments, filtered per-read segments).
    """
    import pandas as pd

    raw_df = segments_to_per_read_dataframe(records, var_names)
    if raw_df.empty:
        raw_df = raw_df.copy()
        raw_df["segment_length_index"] = pd.Series(dtype=int)
        raw_df["segment_length_label"] = pd.Series(dtype=float)
        return raw_df, raw_df.copy()

    def _span_length(row) -> float:
        try:
            start = float(row["segment_start_label"])
            end = float(row["segment_end_label"])
            return abs(end - start)
        except (TypeError, ValueError):
            return float(row["segment_end_exclusive"] - row["segment_start"])

    raw_df = raw_df.copy()
    raw_df["segment_length_index"] = (
        raw_df["segment_end_exclusive"] - raw_df["segment_start"]
    ).astype(int)
    raw_df["segment_length_label"] = raw_df.apply(_span_length, axis=1)
    if min_span is not None:
        raw_df = raw_df[raw_df["segment_length_label"] >= float(min_span)]

    if raw_df.empty:
        return raw_df, raw_df.copy()

    def _segment_overlap(a, b) -> int:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    filtered_rows = []
    max_segments = max_segments_per_read
    for read_id, read_df in raw_df.groupby("read_id", sort=False):
        per_partner = (
            read_df.sort_values(
                ["segment_length_label", "segment_start"],
                ascending=[False, True],
            )
            .groupby("partner_id", sort=False)
            .head(1)
        )
        ordered = per_partner.sort_values(
            ["segment_length_label", "segment_start"],
            ascending=[False, True],
        ).itertuples(index=False)
        selected = []
        for row in ordered:
            if max_segments is not None and len(selected) >= max_segments:
                break
            seg = (row.segment_start, row.segment_end_exclusive)
            if max_segment_overlap is not None:
                if any(
                    _segment_overlap(seg, (s.segment_start, s.segment_end_exclusive))
                    > max_segment_overlap
                    for s in selected
                ):
                    continue
            selected.append(row)
        for row in selected:
            filtered_rows.append(row._asdict())

    filtered_df = pd.DataFrame(filtered_rows, columns=raw_df.columns)
    if not filtered_df.empty:
        filtered_df["selection_rank"] = (
            filtered_df.sort_values(
                ["read_id", "segment_length_label", "segment_start"],
                ascending=[True, False, True],
            )
            .groupby("read_id", sort=False)
            .cumcount()
            + 1
        )

    return raw_df, filtered_df


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
