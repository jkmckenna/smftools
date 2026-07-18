"""
dimensionality_reduction.py — PCA → UMAP → KNN → Leiden pipeline for per-read matrices.

Functions
---------
coverage_filter       Drop low-coverage columns and rows from a reads × positions matrix.
make_features_raw     NaN → 0.5 imputation for direct use of modification matrix.
make_features_acf     Per-read ACF features with optional rolling smoothing.
umap_from_pca         UMAP embedding from a cached PCA-space matrix; returns (X_umap, fitted model).
cluster_from_pca      KNN graph + Leiden clustering from a cached PCA-space matrix.
run_pipeline          PCA → UMAP → KNN graph → Leiden clustering (composes the above); returns (X_pca, X_umap, clusters, explained_variance_ratio, fitted PCA model, fitted UMAP model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import umap
    from sklearn.decomposition import PCA

_DEFAULT_MIN_COL_COVERAGE = 0.05
_DEFAULT_MIN_ROW_COVERAGE = 0.80
_DEFAULT_MIN_READS = 10
_DEFAULT_ACF_ROLLING_WINDOW = 5
_DEFAULT_MAX_LAG = 1000


def coverage_filter(
    mat: np.ndarray,
    positions: np.ndarray,
    obs_names: list[str],
    group_labels: list | None = None,
    min_col_coverage: float = _DEFAULT_MIN_COL_COVERAGE,
    min_row_coverage: float = _DEFAULT_MIN_ROW_COVERAGE,
) -> tuple[np.ndarray, np.ndarray, list[str], list | None]:
    """
    Drop low-coverage positions (columns) then low-coverage reads (rows).

    Returns (mat, positions, obs_names, group_labels) after filtering.
    """
    col_cov = np.mean(~np.isnan(mat), axis=0)
    keep_cols = (col_cov >= min_col_coverage) & np.isfinite(positions)
    mat = mat[:, keep_cols]
    positions = positions[keep_cols]

    row_cov = np.mean(~np.isnan(mat), axis=1)
    keep_rows = row_cov >= min_row_coverage
    mat = mat[keep_rows, :]
    obs_names = [o for o, k in zip(obs_names, keep_rows) if k]
    if group_labels is not None:
        group_labels = [g for g, k in zip(group_labels, keep_rows) if k]

    return mat, positions, obs_names, group_labels


def make_features_raw(mat: np.ndarray) -> np.ndarray:
    """NaN → 0.5 imputation for raw modification matrix."""
    return np.where(np.isnan(mat), 0.5, mat)


def make_features_acf(
    mat: np.ndarray,
    pos: np.ndarray,
    window: int = _DEFAULT_ACF_ROLLING_WINDOW,
    max_lag: int = _DEFAULT_MAX_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-read ACF with rolling average smoothing; drop degenerate rows.

    Parameters
    ----------
    mat    : (n_reads × n_positions) float array.
    pos    : TSS-centred int coordinates.
    window : rolling mean window in lags.
    max_lag: maximum ACF lag to compute.

    Returns
    -------
    feat       : (n_valid × (max_lag+1)) float — smoothed ACF per valid read
    valid_mask : bool array of length n_reads (pre-filter)
    """
    from smftools.analysis.compute.autocorrelation import binary_autocorrelation_with_spacing

    ac_rows: list[np.ndarray] = []
    for row in mat:
        ac, _ = binary_autocorrelation_with_spacing(row, pos, max_lag=max_lag, return_counts=True)
        ac_s = pd.Series(ac).rolling(window, min_periods=1, center=True).mean().to_numpy()
        ac_s = np.where(np.isnan(ac_s), 0.0, ac_s)
        ac_rows.append(ac_s)

    feat = np.vstack(ac_rows)
    valid = feat[:, 0] != 0.0
    return feat[valid, :], valid


def umap_from_pca(
    X_pca: np.ndarray,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, "umap.UMAP"]:
    """
    UMAP embedding from a PCA-space matrix.

    Split out of run_pipeline() so a cached X_pca (e.g. ``pca_space.npy``) can
    be re-embedded with different UMAP parameters without recomputing PCA.

    Returns
    -------
    (X_umap, model) : the 2-D embedding and the fitted UMAP transformer. Call
    ``model.transform(new_X_pca)`` to embed additional points into this exact
    space later without refitting (see ``smftools.project.embedding_store`` for
    the project-layer wrapper that does this by default when a set grows).
    """
    try:
        import umap as umap_lib
    except ImportError as e:
        raise ImportError(f"dimensionality_reduction.umap_from_pca requires umap-learn: {e}")

    k = min(n_neighbors, X_pca.shape[0] - 1)
    model = umap_lib.UMAP(
        n_neighbors=k,
        n_components=2,
        random_state=random_state,
        min_dist=0.3,
    )
    X_umap = model.fit_transform(X_pca)
    return X_umap, model


def cluster_from_pca(
    X_pca: np.ndarray,
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    """
    KNN graph + Leiden clustering from a PCA-space matrix.

    Split out of run_pipeline() so a cached X_pca (e.g. ``pca_space.npy``) can
    be re-clustered at a different leiden_resolution (or n_neighbors) without
    recomputing PCA/UMAP.
    """
    try:
        import igraph as ig
        import leidenalg
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError(
            f"dimensionality_reduction.cluster_from_pca requires igraph and leidenalg: {e}"
        )

    n_reads = X_pca.shape[0]
    k = min(n_neighbors, n_reads - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)
    edges = [(i, int(j)) for i, nbrs in enumerate(indices) for j in nbrs[1:]]
    g = ig.Graph(n=n_reads, edges=edges, directed=False)
    g.simplify()
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=leiden_resolution,
        seed=random_state,
    )
    return np.array(partition.membership)


def run_pipeline(
    feat: np.ndarray,
    leiden_resolution: float = 0.5,
    min_reads: int = _DEFAULT_MIN_READS,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "PCA", "umap.UMAP"] | None:
    """
    PCA → UMAP → KNN → Leiden clustering.

    Parameters
    ----------
    feat              : (n_reads × n_features) float feature matrix.
    leiden_resolution : resolution parameter for Leiden community detection.
    min_reads         : return None if fewer reads than this.
    n_neighbors       : number of KNN neighbours for graph construction and UMAP.
    random_state      : seed for reproducibility.

    Returns
    -------
    (X_pca, X_umap, clusters, explained_variance_ratio, pca_model, umap_model) or
    None if too few reads. X_pca contains all computed PCs (up to 50) -- cache it to
    re-derive X_umap/clusters via umap_from_pca()/cluster_from_pca() without
    recomputing PCA. ``pca_model``/``umap_model`` are the fitted transformers: call
    ``pca_model.transform(new_feat)`` then ``umap_model.transform(new_X_pca)`` to
    embed additional reads into this exact space later without refitting (see
    ``smftools.project.embedding_store`` for the project-layer wrapper that does
    this by default when a set grows). Leiden clustering has no equivalent
    incremental transform -- assign new points to the nearest existing point's
    cluster instead (also handled there).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(f"dimensionality_reduction.run_pipeline requires sklearn: {e}")

    n_reads = feat.shape[0]
    if n_reads < min_reads:
        return None

    n_pcs = min(50, n_reads - 1, feat.shape[1])

    pca = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = pca.fit_transform(feat)
    X_umap, umap_model = umap_from_pca(X_pca, n_neighbors=n_neighbors, random_state=random_state)
    clusters = cluster_from_pca(
        X_pca,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    return X_pca, X_umap, clusters, pca.explained_variance_ratio_, pca, umap_model
