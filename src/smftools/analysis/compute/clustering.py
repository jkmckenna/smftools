"""
Row-ordering helpers via hierarchical clustering.

Inputs are plain numpy arrays. No file I/O, no AnnData dependency. Factored
out of what were three independent copies of the same Ward-linkage
block-ordering logic across project driver scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage


def cluster_block_order(
    matrix: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
    nan_fill: float = 0.0,
) -> np.ndarray:
    """
    Row order for one block of a matrix via hierarchical clustering.

    NaNs are filled with ``nan_fill`` before clustering — 0.0 is appropriate
    for signed contribution/attribution matrices, where NaN means "no
    contribution"; pass e.g. 0.5 for a [0, 1]-valued input matrix where NaN
    means "not observed". Falls back to input order when the block has
    <=1 row or is entirely non-finite.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] <= 1:
        return np.arange(matrix.shape[0], dtype=int)
    if not np.isfinite(matrix).any():
        return np.arange(matrix.shape[0], dtype=int)
    matrix_filled = np.nan_to_num(matrix, nan=nan_fill)
    row_link = linkage(matrix_filled, method=method, metric=metric)
    return np.asarray(leaves_list(row_link), dtype=int)


def cluster_row_order_by_labels(
    matrix: np.ndarray,
    label_arrays: list[np.ndarray],
    label_orders: list[list | None] | None = None,
    method: str = "ward",
    metric: str = "euclidean",
    nan_fill: float = 0.0,
) -> np.ndarray:
    """
    Row order for a full matrix: recursively block rows by each array in
    ``label_arrays`` — outermost first — then hierarchically cluster within
    the innermost blocks.

    E.g. ``label_arrays=[leiden_cluster, true_label]`` bins rows by Leiden
    cluster first, then by true label within each cluster, then orders each
    (cluster, label) leaf block by similarity of its ``matrix`` rows.
    Blocking first keeps same-group rows contiguous, so annotation strips
    drawn in this order still read as clean solid blocks/sub-blocks.

    Pass a single-element ``label_arrays`` list for one-level blocking.
    """
    label_arrays = [np.asarray(a) for a in label_arrays]
    if label_orders is None:
        label_orders = [None] * len(label_arrays)

    def _recurse(indices: np.ndarray, depth: int) -> np.ndarray:
        if depth == len(label_arrays):
            local_order = cluster_block_order(
                matrix[indices], method=method, metric=metric, nan_fill=nan_fill
            )
            return indices[local_order]
        arr = label_arrays[depth]
        order = label_orders[depth]
        if order is None:
            order = sorted(np.unique(arr[indices]).tolist())
        ordered_blocks = []
        for value in order:
            sub_idx = indices[arr[indices] == value]
            if sub_idx.size == 0:
                continue
            ordered_blocks.append(_recurse(sub_idx, depth + 1))
        if not ordered_blocks:
            return indices
        return np.concatenate(ordered_blocks)

    if matrix.shape[0] == 0:
        return np.arange(0, dtype=int)
    return _recurse(np.arange(matrix.shape[0], dtype=int), 0)
