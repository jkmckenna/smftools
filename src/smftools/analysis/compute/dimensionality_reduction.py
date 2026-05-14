"""
dimensionality_reduction.py — PCA → UMAP → KNN → Leiden pipeline for per-read matrices.

Functions
---------
coverage_filter       Drop low-coverage columns and rows from a reads × positions matrix.
make_features_raw     NaN → 0.5 imputation for direct use of modification matrix.
make_features_acf     Per-read ACF features with optional rolling smoothing.
run_pipeline          PCA → UMAP → KNN graph → Leiden clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
        ac, _ = binary_autocorrelation_with_spacing(
            row, pos, max_lag=max_lag, return_counts=True
        )
        ac_s = (
            pd.Series(ac)
            .rolling(window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        ac_s = np.where(np.isnan(ac_s), 0.0, ac_s)
        ac_rows.append(ac_s)

    feat = np.vstack(ac_rows)
    valid = feat[:, 0] != 0.0
    return feat[valid, :], valid


def run_pipeline(
    feat: np.ndarray,
    leiden_resolution: float = 0.5,
    min_reads: int = _DEFAULT_MIN_READS,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
    (X_pca, X_umap, clusters) or None if too few reads.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        import umap as umap_lib
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            f"dimensionality_reduction.run_pipeline requires sklearn, umap-learn, "
            f"igraph, and leidenalg: {e}"
        )

    n_reads = feat.shape[0]
    if n_reads < min_reads:
        return None

    n_pcs = min(50, n_reads - 1, feat.shape[1])
    k = min(n_neighbors, n_reads - 1)

    X_pca = PCA(n_components=n_pcs, random_state=random_state).fit_transform(feat)
    X_umap = umap_lib.UMAP(
        n_neighbors=k, n_components=2,
        random_state=random_state, min_dist=0.3,
    ).fit_transform(X_pca)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)
    edges = [(i, int(j)) for i, nbrs in enumerate(indices) for j in nbrs[1:]]
    g = ig.Graph(n=n_reads, edges=edges, directed=False)
    g.simplify()
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=leiden_resolution, seed=random_state,
    )
    clusters = np.array(partition.membership)

    return X_pca, X_umap, clusters
