def calculate_umap(adata, layer='nan_half', var_filters=None, n_pcs=15, knn_neighbors=100, overwrite=True, threads=8):
    import scanpy as sc
    import numpy as np
    import os
    from scipy.sparse import issparse

    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Step 1: Apply var filter
    if var_filters:
        subset_mask = np.logical_or.reduce([adata.var[f].values for f in var_filters])
        adata_subset = adata[:, subset_mask].copy()
        print(f"🔹 Subsetting adata: Retained {adata_subset.shape[1]} features based on filters {var_filters}")
    else:
        adata_subset = adata.copy()
        print("🔹 No var filters provided. Using all features.")

    # Step 2: NaN handling inside layer
    if layer:
        data = adata_subset.layers[layer]
        if not issparse(data):
            if np.isnan(data).any():
                print("⚠ NaNs detected, filling with 0.5 before PCA + neighbors.")
                data = np.nan_to_num(data, nan=0.5)
                adata_subset.layers[layer] = data
            else:
                print("✅ No NaNs detected.")
        else:
            print("✅ Sparse matrix detected; skipping NaN check (sparse formats typically do not store NaNs).")

    # Step 3: PCA + neighbors + UMAP on subset
    if "X_umap" not in adata_subset.obsm or overwrite:
        n_pcs = min(adata_subset.shape[1], n_pcs)
        print(f"Running PCA with n_pcs={n_pcs}")
        sc.pp.pca(adata_subset, layer=layer)
        print('Running neighborhood graph')
        sc.pp.neighbors(adata_subset, use_rep="X_pca", n_pcs=n_pcs, n_neighbors=knn_neighbors)
        print('Running UMAP')
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


    print(f"✅ Stored: adata.obsm['X_pca'] and adata.obsm['X_umap']")

    return adata