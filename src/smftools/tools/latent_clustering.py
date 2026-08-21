"""Per-embedding KNN + Leiden clustering for the latent stage (`EGL-28a`).

Today the latent stage computes Leiden **once**, from UMAP's internal
``model.graph_``, and reuses those labels for every strategy. So the labels are
not a clustering *of* each representation -- they are a clustering of the UMAP
neighbour graph, borrowed. Two consequences this module removes:

- A PCA embedding and an NMF embedding of the same molecules carry identical
  labels, which makes comparing representations meaningless: any difference the
  clustermaps show is a difference in coordinates, never in structure found.
- The non-fit transfer runs ``_nearest_labels(all_pca, fit_pca, fit_labels)`` --
  UMAP-derived labels assigned using *PCA* distances. That is the
  cross-embedding coupling in its sharpest form, and it is silent.

Each embedding here is self-contained: its graph, its Leiden labels, and its
transfer all live in its own coordinates.

**On the transfer.** Molecules beyond ``latent_max_fit_reads`` are already
embedded -- the stage fits on a subset but transforms everything -- so only the
*labels* stop at the fit boundary. They are filled by a k-neighbour vote among
the fit points, and the vote is recorded: a ``*_label_source`` column marking
each label ``fit`` or ``transferred``, and a ``*_label_confidence`` holding the
fraction of the k neighbours that agreed. A clustermap that renders fit-derived
and proximity-assigned membership identically invites reading the second as
evidence for the first, and on current pilots nothing is transferred at all
(every population is under the ceiling), so the first data this runs on in
anger will be data nobody has inspected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

LABEL_SOURCE_FIT = "fit"
LABEL_SOURCE_TRANSFERRED = "transferred"
UNASSIGNED_LABEL = "unassigned"


def embedding_keys(adata) -> list[str]:
    """Every molecule embedding in ``obsm``, in a stable order."""
    return sorted(str(key) for key in adata.obsm if str(key).startswith("X_"))


def parse_embedding_key(key: str) -> tuple[str, str]:
    """Split ``X_<strategy>_<suffix>`` into its parts."""
    remainder = key.removeprefix("X_")
    strategy, _, suffix = remainder.partition("_")
    return strategy, suffix


def resolve_parameter(cfg, strategy: str, name: str, default):
    """Read a per-strategy override, else the shared knob, else the default.

    The right resolution for a 10-component PCA and a 2-component UMAP are not
    the same number, so per-strategy control is the point of the lane. The
    shared knobs (`latent_knn_neighbors`, `latent_leiden_resolution`) stay
    authoritative when no override is given, so existing configs keep behaving
    as they do now.
    """
    per_strategy = getattr(cfg, f"latent_{name}_by_strategy", None) or {}
    if isinstance(per_strategy, dict) and strategy in per_strategy:
        return per_strategy[strategy]
    return getattr(cfg, f"latent_{name}", default)


def build_knn_connectivities(points: np.ndarray, n_neighbors: int, *, metric: str = "euclidean"):
    """Symmetric, distance-weighted KNN graph over ``points``.

    Built explicitly rather than borrowed from UMAP so that every embedding --
    including ones UMAP never touched (`nmf`, `cp`, `pca`) -- has a graph of its
    own coordinates. Weights are a Gaussian-style similarity on the distance so
    Leiden's resolution behaves comparably to the UMAP-graph case rather than
    treating every edge as equal.
    """
    from sklearn.neighbors import kneighbors_graph

    n_points = int(points.shape[0])
    # kneighbors_graph counts the point itself, and a k above the population is
    # an error rather than a saturated graph.
    k = int(max(1, min(int(n_neighbors), n_points - 1)))
    graph = kneighbors_graph(points, n_neighbors=k, mode="distance", metric=metric)
    graph = graph.maximum(graph.T)  # symmetrize: neighbourhood is mutual here
    distances = graph.data
    finite = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(finite)) if finite.size else 1.0
    if scale <= 0:
        scale = 1.0
    graph.data = np.exp(-((distances / scale) ** 2)).astype(np.float64, copy=False)
    return graph.tocsr()


def transfer_labels(
    points: np.ndarray,
    fit_points: np.ndarray,
    fit_labels: np.ndarray,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign labels to unfit molecules by a k-neighbour vote among fit points.

    Returns ``(labels, confidence)`` where confidence is the fraction of the k
    neighbours that voted for the winning label. A 1-NN assignment -- what the
    stage does today -- cannot express that a molecule sits on a boundary
    between two clusters; the vote fraction can, which is what makes a
    suspicious block of a heatmap checkable without a recompute.
    """
    from sklearn.neighbors import NearestNeighbors

    if fit_points.shape[0] == 0 or fit_labels.size == 0:
        return (
            np.full(points.shape[0], UNASSIGNED_LABEL, dtype=object),
            np.zeros(points.shape[0], dtype=np.float32),
        )
    k = int(max(1, min(int(n_neighbors), fit_points.shape[0])))
    neighbors = NearestNeighbors(n_neighbors=k).fit(fit_points)
    _distances, indices = neighbors.kneighbors(points)
    neighbor_labels = np.asarray(fit_labels, dtype=object)[indices]

    labels = np.empty(points.shape[0], dtype=object)
    confidence = np.empty(points.shape[0], dtype=np.float32)
    for row in range(neighbor_labels.shape[0]):
        values, counts = np.unique(neighbor_labels[row], return_counts=True)
        winner = int(np.argmax(counts))
        labels[row] = values[winner]
        confidence[row] = float(counts[winner]) / float(k)
    return labels, confidence


def cluster_embedding(
    adata,
    key: str,
    *,
    fit_indices: np.ndarray,
    cfg,
) -> dict[str, object] | None:
    """Cluster one embedding in its own coordinates; write labels and provenance.

    Returns the parameters used, for the caller to record, or ``None`` when the
    embedding is too small to cluster.
    """
    import anndata as ad

    from .calculate_leiden import calculate_leiden

    strategy, suffix = parse_embedding_key(key)
    points = np.asarray(adata.obsm[key], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3:
        logger.info("Embedding %s has too few molecules to cluster; skipping.", key)
        return None

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if fit_indices.size == 0:
        fit_indices = np.arange(points.shape[0], dtype=np.int64)
    fit_points = points[fit_indices]
    if fit_points.shape[0] < 3:
        logger.info("Embedding %s has too few fit molecules to cluster; skipping.", key)
        return None

    n_neighbors = int(resolve_parameter(cfg, strategy, "knn_neighbors", 15))
    metric = str(resolve_parameter(cfg, strategy, "knn_metric", "euclidean"))
    resolution = float(resolve_parameter(cfg, strategy, "leiden_resolution", 0.1))

    connectivities = build_knn_connectivities(fit_points, n_neighbors, metric=metric)
    fit_view = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names[fit_indices]))
    fit_view.obsp["connectivities"] = connectivities
    try:
        calculate_leiden(
            fit_view, resolution=resolution, key_added="leiden", connectivities_key="connectivities"
        )
    except Exception as exc:
        # One embedding failing to cluster must not lose the others, but it
        # must be visible rather than silently producing a single-cluster
        # picture that looks like a real result.
        logger.warning("Leiden failed for %s: %s", key, exc)
        return None
    fit_labels = fit_view.obs["leiden"].astype(str).to_numpy()

    labels = np.empty(points.shape[0], dtype=object)
    source = np.full(points.shape[0], LABEL_SOURCE_TRANSFERRED, dtype=object)
    confidence = np.ones(points.shape[0], dtype=np.float32)
    labels[fit_indices] = fit_labels
    source[fit_indices] = LABEL_SOURCE_FIT

    unfit = np.setdiff1d(np.arange(points.shape[0], dtype=np.int64), fit_indices)
    if unfit.size:
        # Transferred *in this embedding's own coordinates*. Using another
        # embedding's distances here is precisely the coupling this lane exists
        # to remove.
        transferred, transferred_confidence = transfer_labels(
            points[unfit], fit_points, fit_labels, n_neighbors=n_neighbors
        )
        labels[unfit] = transferred
        confidence[unfit] = transferred_confidence

    label_key = f"leiden_{strategy}_{suffix}"
    adata.obs[label_key] = pd.Categorical(labels.astype(str))
    adata.obs[f"{label_key}_label_source"] = pd.Categorical(source.astype(str))
    adata.obs[f"{label_key}_label_confidence"] = confidence

    parameters = {
        "embedding": key,
        "strategy": strategy,
        "suffix": suffix,
        "n_neighbors": n_neighbors,
        "metric": metric,
        "resolution": resolution,
        "fit_read_count": int(fit_indices.size),
        "transferred_read_count": int(unfit.size),
        "cluster_count": int(pd.unique(labels.astype(str)).size),
    }
    adata.uns[f"{label_key}_params"] = dict(parameters)
    logger.info(
        "Clustered %s: %d cluster(s) over %d fit + %d transferred molecule(s) "
        "(k=%d, resolution=%.3f)",
        key,
        parameters["cluster_count"],
        parameters["fit_read_count"],
        parameters["transferred_read_count"],
        n_neighbors,
        resolution,
    )
    return parameters


def cluster_all_embeddings(adata, *, fit_indices, cfg) -> list[dict[str, object]]:
    """Cluster every embedding independently. Returns one record per embedding."""
    records = []
    for key in embedding_keys(adata):
        record = cluster_embedding(adata, key, fit_indices=fit_indices, cfg=cfg)
        if record is not None:
            records.append(record)
    return records
