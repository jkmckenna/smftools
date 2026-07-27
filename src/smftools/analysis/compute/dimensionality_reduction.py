"""
dimensionality_reduction.py — reduction → UMAP → KNN → Leiden pipeline for per-read matrices.

Functions
---------
coverage_filter          Drop low-coverage columns and rows from a reads × positions matrix.
make_features_raw        NaN → 0.5 imputation for direct use of modification matrix.
make_features_acf        Per-read ACF features with optional rolling smoothing.
diffusion_map_embedding  Diffusion-map latent space from a Gaussian-kernel KNN affinity graph.
umap_from_pca            UMAP embedding from a cached latent-space matrix; returns (X_umap, fitted model).
cluster_from_pca         KNN graph + Leiden clustering from a cached latent-space matrix.
run_pipeline             reduction ("pca" or "diffusion_map") → UMAP → KNN graph → Leiden
                         clustering (composes the above); returns (X_latent, X_umap, clusters,
                         variance_info, fitted reduction model or None, fitted UMAP model).
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


def diffusion_map_embedding(
    feat: np.ndarray,
    n_components: int = 16,
    n_neighbors: int = 15,
    diffusion_time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diffusion-map latent space from a Gaussian-kernel KNN affinity graph.

    Builds a symmetrized k-nearest-neighbor affinity graph (Gaussian kernel,
    bandwidth = median KNN distance), row/column-normalizes it into the
    symmetric-normalized graph Laplacian, and takes its leading eigenvectors
    (dropping the trivial constant eigenvector), each scaled by its
    eigenvalue raised to ``diffusion_time`` — the standard diffusion-map
    construction (Coifman & Lafon 2006).

    An alternative to PCA as the ``method`` in :func:`run_pipeline`: PCA
    captures directions of maximal linear variance, while a diffusion map
    captures the manifold's diffusion/random-walk geometry — better suited
    when the read population lies on a curved or branching manifold rather
    than varying along a few linear axes.

    Returns
    -------
    (latent, eigenvalues) : latent has shape (n_reads, n_components);
    eigenvalues are the corresponding non-trivial eigenvalues (descending),
    already folded into ``latent`` via the ``diffusion_time`` scaling —
    kept separate mainly for a scree-style diagnostic plot.
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    n_reads = feat.shape[0]
    if n_reads < 3:
        raise ValueError("diffusion_map_embedding requires at least 3 reads")

    k = min(n_neighbors, n_reads - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(feat)
    distances, indices = nn.kneighbors(feat)

    nz = distances[:, 1:].ravel()
    nz = nz[np.isfinite(nz) & (nz > 0)]
    sigma = float(np.median(nz)) if nz.size else 1.0
    sigma = max(sigma, 1e-6)

    rows = np.repeat(np.arange(n_reads), k)
    cols = indices[:, 1:].ravel()
    vals = np.exp(-(distances[:, 1:].ravel() ** 2) / (2.0 * sigma * sigma))
    affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_reads, n_reads))
    affinity = 0.5 * (affinity + affinity.T)

    degree = np.asarray(affinity.sum(axis=1)).ravel()
    degree[degree <= 0] = 1e-12
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
    normalized = d_inv_sqrt @ affinity @ d_inv_sqrt

    n_wanted = min(max(2, n_components) + 1, n_reads - 1)
    if n_wanted >= n_reads - 1:
        evals, evecs = np.linalg.eigh(normalized.toarray())
    else:
        try:
            evals, evecs = eigsh(normalized, k=n_wanted, which="LA")
        except Exception:
            evals, evecs = np.linalg.eigh(normalized.toarray())
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evals_nontrivial = evals[1:n_wanted]
    evecs_nontrivial = evecs[:, 1:n_wanted]
    scaled = np.power(np.clip(evals_nontrivial, 1e-12, None), diffusion_time)
    latent = evecs_nontrivial * scaled[np.newaxis, :]
    return latent, evals_nontrivial


def run_pipeline(
    feat: np.ndarray,
    leiden_resolution: float = 0.5,
    min_reads: int = _DEFAULT_MIN_READS,
    n_neighbors: int = 15,
    random_state: int = 42,
    method: str = "pca",
    diffusion_time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "PCA | None", "umap.UMAP"] | None:
    """
    Dimensionality reduction → UMAP → KNN → Leiden clustering.

    Parameters
    ----------
    feat              : (n_reads × n_features) float feature matrix.
    leiden_resolution : resolution parameter for Leiden community detection.
    min_reads         : return None if fewer reads than this.
    n_neighbors       : number of KNN neighbours for graph construction and UMAP.
    random_state      : seed for reproducibility.
    method            : ``"pca"`` (default) or ``"diffusion_map"`` — which
        technique produces the latent space that UMAP/Leiden are built from.
        PCA captures directions of maximal linear variance; a diffusion map
        (see :func:`diffusion_map_embedding`) instead captures the
        population's diffusion/random-walk geometry, which can better
        resolve a curved or branching manifold.
    diffusion_time    : only used when ``method="diffusion_map"`` — passed
        through to :func:`diffusion_map_embedding`.

    Returns
    -------
    (X_latent, X_umap, clusters, variance_info, reduction_model, umap_model) or
    None if too few reads. X_latent contains the full computed latent space (up to
    50 dims for PCA) -- cache it to re-derive X_umap/clusters via
    umap_from_pca()/cluster_from_pca() without recomputing the reduction.
    ``variance_info`` is ``pca.explained_variance_ratio_`` for
    ``method="pca"``, or the diffusion map's non-trivial eigenvalues for
    ``method="diffusion_map"`` (same descending-importance shape, different
    units — see :func:`diffusion_map_embedding`).
    ``reduction_model`` is the fitted PCA transformer for ``method="pca"``
    (call ``reduction_model.transform(new_feat)`` to embed additional reads
    into this exact space later without refitting — see
    ``smftools.project.embedding_store`` for the project-layer wrapper that
    does this by default when a set grows) or ``None`` for
    ``method="diffusion_map"``, which has no incremental transform. Leiden
    clustering also has no incremental transform for either method -- assign
    new points to the nearest existing point's cluster instead.
    """
    n_reads = feat.shape[0]
    if n_reads < min_reads:
        return None

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as e:
            raise ImportError(f"dimensionality_reduction.run_pipeline requires sklearn: {e}")

        n_pcs = min(50, n_reads - 1, feat.shape[1])
        pca = PCA(n_components=n_pcs, random_state=random_state)
        X_latent = pca.fit_transform(feat)
        variance_info = pca.explained_variance_ratio_
        reduction_model = pca
    elif method == "diffusion_map":
        X_latent, variance_info = diffusion_map_embedding(
            feat,
            n_components=min(50, n_reads - 2),
            n_neighbors=n_neighbors,
            diffusion_time=diffusion_time,
        )
        reduction_model = None
    else:
        raise ValueError(f"Unsupported method {method!r}; expected 'pca' or 'diffusion_map'")

    X_umap, umap_model = umap_from_pca(X_latent, n_neighbors=n_neighbors, random_state=random_state)
    clusters = cluster_from_pca(
        X_latent,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    return X_latent, X_umap, clusters, variance_info, reduction_model, umap_model
