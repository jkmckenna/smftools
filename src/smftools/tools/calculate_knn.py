from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_knn(
    adata: "ad.AnnData",
    obsm: str = "X_pca",
    knn_neighbors: int = 100,
    overwrite: bool = True,
    threads: int = 8,
    random_state: int | None = 0,
    symmetrize: bool = True,
) -> "ad.AnnData":
    """Compute a KNN distance graph on an embedding in `adata.obsm[obsm]`.

    Stores:
      - adata.obsp[f"knn_distances_{obsm}"] : CSR sparse matrix of distances
      - adata.uns[f"knn_distances_{obsm}"]["params"] : metadata

    Args:
        adata: AnnData object to update.
        obsm: Key in `adata.obsm` to use as the embedding.
        knn_neighbors: Target number of neighbors (will be clipped to n_obs-1).
        overwrite: If False and graph exists, do nothing.
        threads: Parallel jobs for pynndescent.
        random_state: Seed for pynndescent.
        symmetrize: If True, make distance graph symmetric via min(A, A.T).

    Returns:
        Updated AnnData.
    """
    import numpy as np
    import scipy.sparse as sp

    if obsm not in adata.obsm:
        raise KeyError(f"`{obsm}` not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    out_key = f"knn_distances_{obsm}"
    if not overwrite and out_key in adata.obsp:
        logger.info("KNN graph %r already exists and overwrite=False; skipping.", out_key)
        return adata

    data = adata.obsm[obsm]

    if sp.issparse(data):
        # Convert to float32 for pynndescent/numba friendliness if needed
        data = data.astype(np.float32)
        logger.info(
            "Sparse embedding detected (%s). Proceeding without NaN check.", type(data).__name__
        )
    else:
        data = np.asarray(data)
        if np.isnan(data).any():
            logger.warning("NaNs detected in %s; filling NaNs with 0.5 before KNN.", obsm)
            data = np.nan_to_num(data, nan=0.5)
        data = data.astype(np.float32, copy=False)

    pynndescent = require("pynndescent", extra="umap", purpose="KNN graph computation")

    n_obs = data.shape[0]
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations for KNN; got n_obs={n_obs}")

    n_neighbors = min(int(knn_neighbors), n_obs - 1)
    if n_neighbors < 1:
        raise ValueError(f"Computed n_neighbors={n_neighbors}; check knn_neighbors and n_obs.")

    logger.info(
        "Running pynndescent KNN (obsm=%s, n_neighbors=%d, metric=euclidean, n_jobs=%d)",
        obsm,
        n_neighbors,
        threads,
    )

    nn_index = pynndescent.NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=random_state,
        n_jobs=threads,
    )
    knn_indices, knn_dists = nn_index.neighbor_graph  # shapes: (n_obs, n_neighbors)

    rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
    cols = knn_indices.reshape(-1).astype(np.int64, copy=False)
    vals = knn_dists.reshape(-1).astype(np.float32, copy=False)

    distances = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs)).tocsr()

    # Optional: ensure diagonal is 0 and (optionally) symmetrize
    distances.setdiag(0.0)
    distances.eliminate_zeros()

    if symmetrize:
        # Keep the smaller directed distance for each undirected edge
        distances = distances.minimum(distances.T)

    adata.obsp[out_key] = distances
    adata.uns[out_key] = {
        "params": {
            "obsm": obsm,
            "n_neighbors_requested": int(knn_neighbors),
            "n_neighbors_used": int(n_neighbors),
            "method": "pynndescent",
            "metric": "euclidean",
            "random_state": random_state,
            "n_jobs": int(threads),
            "symmetrize": bool(symmetrize),
        }
    }

    return adata
