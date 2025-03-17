def calculate_umap(
    adata, 
    layer='nan_half', 
    var_filters=None, 
    clustering_method="leiden", 
    leiden_res=1.0, 
    knn_neighbors=15  # Limit neighbors to avoid full dense matrix
):
    """
    Calculates UMAP based on a subset of the adata object.
    """
    import scanpy as sc
    import numpy as np
    original_adata = adata.copy()

    # 🛑 Apply logical gating (subset features for Jaccard calculation)
    if var_filters:
        subset_mask = np.logical_or.reduce([adata.var[f].values for f in var_filters])
        adata_subset = adata[:, subset_mask].copy()
        print(f"🔹 Subsetting adata: Retained {adata_subset.shape[1]} features based on filters {var_filters}")
    else:
        adata_subset = adata.copy()
        print("🔹 No var filters provided. Using all features.")

    # 🔍 Extract data matrix
    data = adata_subset.layers[layer] if layer else adata_subset.X
    
    if f"{layer}_umap" not in adata.obsm:
        sc.pp.pca(adata, layer=layer)  # Run PCA
        sc.pp.neighbors(adata, n_pcs=40, n_neighbors=knn_neighbors)  # Compute kNN graph
        sc.tl.umap(adata)  # Run UMAP