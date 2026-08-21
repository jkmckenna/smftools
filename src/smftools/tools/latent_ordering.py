"""Row ordering for latent-clustered molecules (`EGL-28b`).

Rows are binned by their Leiden cluster and hierarchically ordered *within*
each bin, so a clustermap reads as solid blocks with structure inside them
rather than one undifferentiated dendrogram.

`analysis.compute.clustering.cluster_row_order_by_labels` already does
block-then-cluster and is the model for this; two things make it insufficient
here rather than something to call directly:

- **Cluster order is lexicographic** there (`sorted(np.unique(...))`), so
  Leiden's string labels order ``0, 1, 10, 11, 2, ...``. With up to 39 clusters
  measured on a real unit (`EGL-28a`), that is visibly scrambled in a plot whose
  whole point is contiguous blocks.
- **No size guard.** `linkage` builds a condensed distance matrix: 0.1 GB at
  5,000 rows, 1.6 GB at 20,000. The spatial stage already learned this, where an
  uncapped `linkage` turned one PNG into a multi-minute stall
  (`_cap_clustermap_rows`). A Leiden cluster is not bounded by anything except
  the population.

Ordering happens in the *latent* coordinates, not the projected layer -- the
clustermap projects raw data back in an order the latent determined, which is
the point of the figure. Per the lane's self-contained rule, that means the
same embedding the labels came from.
"""

from __future__ import annotations

import numpy as np

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

#: Above this many rows in one block, fall back to a projection order rather
#: than hierarchical linkage. 5,000 rows is ~0.1 GB of condensed distances;
#: the fallback is O(n log n) and still deterministic and structure-aware.
DEFAULT_MAX_LINKAGE_ROWS = 5000


def _numeric_aware_key(label: str):
    """Sort ``"2"`` before ``"10"`` while still ordering non-numeric labels."""
    text = str(label)
    return (0, int(text), "") if text.lstrip("-").isdigit() else (1, 0, text)


def _projection_order(points: np.ndarray) -> np.ndarray:
    """Deterministic order along the block's direction of greatest variance.

    The fallback when a block is too large to link. It is not as good as a
    dendrogram -- it captures one axis rather than a hierarchy -- but it is
    structure-aware rather than arbitrary, and it degrades gracefully instead
    of exhausting memory.
    """
    if points.shape[0] < 2:
        return np.arange(points.shape[0], dtype=int)
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ vt[0]
    except np.linalg.LinAlgError:
        projection = centered[:, 0]
    return np.argsort(projection, kind="stable").astype(int)


def hierarchical_block_order(
    points: np.ndarray,
    *,
    method: str = "average",
    metric: str = "euclidean",
    max_linkage_rows: int = DEFAULT_MAX_LINKAGE_ROWS,
) -> np.ndarray:
    """Order one block's rows by similarity in the latent space."""
    points = np.asarray(points, dtype=float)
    n_rows = points.shape[0]
    if n_rows < 3:
        return np.arange(n_rows, dtype=int)
    if not np.isfinite(points).any():
        return np.arange(n_rows, dtype=int)
    filled = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if max_linkage_rows and n_rows > int(max_linkage_rows):
        logger.info(
            "Block of %d rows exceeds the linkage cap (%d); ordering by projection instead.",
            n_rows,
            int(max_linkage_rows),
        )
        return _projection_order(filled)
    from scipy.cluster.hierarchy import leaves_list, linkage

    try:
        return np.asarray(leaves_list(linkage(filled, method=method, metric=metric)), dtype=int)
    except (ValueError, MemoryError) as exc:
        logger.warning("Linkage failed for a block of %d rows (%s); using projection.", n_rows, exc)
        return _projection_order(filled)


def cluster_display_order(points: np.ndarray, labels: np.ndarray) -> list[str]:
    """Order the clusters themselves, so similar blocks sit next to each other.

    Clusters are linked on their centroids in the latent space. Falling back to
    label order would be deterministic but arbitrary: adjacent blocks in the
    figure would have no relationship, which wastes the one axis a clustermap
    has for showing how clusters relate.
    """
    unique = sorted({str(label) for label in labels}, key=_numeric_aware_key)
    if len(unique) < 3:
        return unique
    labels = np.asarray(labels, dtype=object).astype(str)
    centroids = np.vstack(
        [np.nan_to_num(points[labels == label], nan=0.0).mean(axis=0) for label in unique]
    )
    from scipy.cluster.hierarchy import leaves_list, linkage

    try:
        order = leaves_list(linkage(centroids, method="average", metric="euclidean"))
    except (ValueError, MemoryError):
        return unique
    return [unique[index] for index in order]


def latent_row_order(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "average",
    metric: str = "euclidean",
    max_linkage_rows: int = DEFAULT_MAX_LINKAGE_ROWS,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Bin rows by cluster, order within each bin, and report the block spans.

    Returns ``(row_order, blocks)`` where ``blocks`` is one
    ``(label, start, stop)`` per cluster in display order -- the spans a
    clustermap draws its separators and annotation strips from, so the caller
    never has to re-derive them from the reordered labels.
    """
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=object).astype(str)
    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must describe the same molecules")
    if points.shape[0] == 0:
        return np.arange(0, dtype=int), []

    order_parts: list[np.ndarray] = []
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    for label in cluster_display_order(points, labels):
        member_rows = np.flatnonzero(labels == label)
        if member_rows.size == 0:
            continue
        within = hierarchical_block_order(
            points[member_rows],
            method=method,
            metric=metric,
            max_linkage_rows=max_linkage_rows,
        )
        order_parts.append(member_rows[within])
        blocks.append((label, cursor, cursor + member_rows.size))
        cursor += member_rows.size
    if not order_parts:
        return np.arange(points.shape[0], dtype=int), []
    return np.concatenate(order_parts).astype(int), blocks
