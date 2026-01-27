from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_umap(
    adata: "ad.AnnData",
    layer: str | None = "nan_half",
    var_filters: Sequence[str] | None = None,
    n_pcs: int = 15,
    knn_neighbors: int = 100,
    overwrite: bool = True,
    threads: int = 8,
    random_state: int | None = 0,
) -> "ad.AnnData":
    """Compute PCA, neighbors, and UMAP embeddings.

    Args:
        adata: AnnData object to update.
        layer: Layer name to use for PCA/UMAP (``None`` uses ``adata.X``).
        var_filters: Optional list of var masks to subset features.
        n_pcs: Number of principal components.
        knn_neighbors: Number of neighbors for the graph.
        overwrite: Whether to recompute embeddings if they exist.
        threads: Number of OMP threads for computation.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    import os

    import numpy as np
    import scipy.linalg as spla
    import scipy.sparse as sp
    umap = require("umap", extra="umap", purpose="UMAP calculation")
    pynndescent = require("pynndescent", extra="umap", purpose="KNN graph computation")

    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Step 1: Apply var filter
    if var_filters:
        subset_mask = np.logical_or.reduce([adata.var[f].values for f in var_filters])
        adata_subset = adata[:, subset_mask].copy()
        logger.info(
            "Subsetting adata: retained %s features based on filters %s",
            adata_subset.shape[1],
            var_filters,
        )
    else:
        adata_subset = adata.copy()
        logger.info("No var filters provided. Using all features.")

    # Step 2: NaN handling inside layer
    if layer:
        data = adata_subset.layers[layer]
        if not sp.issparse(data):
            if np.isnan(data).any():
                logger.warning("NaNs detected, filling with 0.5 before PCA + neighbors.")
                data = np.nan_to_num(data, nan=0.5)
                adata_subset.layers[layer] = data
            else:
                logger.info("No NaNs detected.")
        else:
            logger.info(
                "Sparse matrix detected; skipping NaN check (sparse formats typically do not store NaNs)."
            )

    # Step 3: PCA + neighbors + UMAP on subset
    if "X_umap" not in adata_subset.obsm or overwrite:
        n_pcs = min(adata_subset.shape[1], n_pcs)
        logger.info("Running PCA with n_pcs=%s", n_pcs)

        if layer:
            matrix = adata_subset.layers[layer]
        else:
            matrix = adata_subset.X

        if sp.issparse(matrix):
            logger.warning("Converting sparse matrix to dense for PCA.")
            matrix = matrix.toarray()

        matrix = np.asarray(matrix, dtype=float)
        mean = matrix.mean(axis=0)
        centered = matrix - mean

        if centered.shape[0] == 0 or centered.shape[1] == 0:
            raise ValueError("PCA requires a non-empty matrix.")

        if n_pcs <= 0:
            raise ValueError("n_pcs must be positive.")

        if centered.shape[1] <= n_pcs:
            n_pcs = centered.shape[1]

        if centered.shape[0] < n_pcs:
            n_pcs = centered.shape[0]

        u, s, vt = spla.svd(centered, full_matrices=False)

        u = u[:, :n_pcs]
        s = s[:n_pcs]
        vt = vt[:n_pcs]

        adata_subset.obsm["X_pca"] = u * s
        adata_subset.varm["PCs"] = vt.T

        logger.info("Running neighborhood graph with pynndescent (n_neighbors=%s)", knn_neighbors)
        n_neighbors = min(knn_neighbors, max(1, adata_subset.n_obs - 1))
        nn_index = pynndescent.NNDescent(
            adata_subset.obsm["X_pca"],
            n_neighbors=n_neighbors,
            metric="euclidean",
            random_state=random_state,
            n_jobs=threads,
        )
        knn_indices, knn_dists = nn_index.neighbor_graph

        rows = np.repeat(np.arange(adata_subset.n_obs), n_neighbors)
        cols = knn_indices.reshape(-1)
        distances = sp.coo_matrix(
            (knn_dists.reshape(-1), (rows, cols)),
            shape=(adata_subset.n_obs, adata_subset.n_obs),
        ).tocsr()
        adata_subset.obsp["distances"] = distances

        logger.info("Running UMAP")
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            metric="euclidean",
            random_state=random_state,
        )
        adata_subset.obsm["X_umap"] = umap_model.fit_transform(adata_subset.obsm["X_pca"])

        try:
            from umap.umap_ import fuzzy_simplicial_set

            fuzzy_result = fuzzy_simplicial_set(
                adata_subset.obsm["X_pca"],
                n_neighbors=n_neighbors,
                random_state=random_state,
                metric="euclidean",
                knn_indices=knn_indices,
                knn_dists=knn_dists,
            )
            connectivities = fuzzy_result[0] if isinstance(fuzzy_result, tuple) else fuzzy_result
        except TypeError:
            connectivities = umap_model.graph_

        adata_subset.obsp["connectivities"] = connectivities

    # Step 4: Store results in original adata
    adata.obsm["X_pca"] = adata_subset.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_subset.obsm["X_umap"]
    adata.obsp["distances"] = adata_subset.obsp["distances"]
    adata.obsp["connectivities"] = adata_subset.obsp["connectivities"]
    adata.uns["neighbors"] = {
        "params": {
            "n_neighbors": knn_neighbors,
            "method": "pynndescent",
            "metric": "euclidean",
        }
    }

    # Fix varm["PCs"] shape mismatch
    pc_matrix = np.zeros((adata.shape[1], adata_subset.varm["PCs"].shape[1]))
    if var_filters:
        subset_mask = np.logical_or.reduce([adata.var[f].values for f in var_filters])
        pc_matrix[subset_mask, :] = adata_subset.varm["PCs"]
    else:
        pc_matrix = adata_subset.varm["PCs"]  # No subsetting case

    adata.varm["PCs"] = pc_matrix

    logger.info("Stored: adata.obsm['X_pca'] and adata.obsm['X_umap']")

    return adata
