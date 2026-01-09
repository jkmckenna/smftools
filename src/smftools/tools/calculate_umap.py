from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_umap(
    adata,
    layer="nan_half",
    var_filters=None,
    n_pcs=15,
    knn_neighbors=100,
    overwrite=True,
    threads=8,
):
    import os

    import numpy as np
    import scanpy as sc
    from scipy.sparse import issparse

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
        if not issparse(data):
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
        sc.pp.pca(adata_subset, layer=layer)
        logger.info("Running neighborhood graph")
        sc.pp.neighbors(adata_subset, use_rep="X_pca", n_pcs=n_pcs, n_neighbors=knn_neighbors)
        logger.info("Running UMAP")
        sc.tl.umap(adata_subset)

    # Step 4: Store results in original adata
    adata.obsm["X_pca"] = adata_subset.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_subset.obsm["X_umap"]
    adata.obsp["distances"] = adata_subset.obsp["distances"]
    adata.obsp["connectivities"] = adata_subset.obsp["connectivities"]
    adata.uns["neighbors"] = adata_subset.uns["neighbors"]

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
