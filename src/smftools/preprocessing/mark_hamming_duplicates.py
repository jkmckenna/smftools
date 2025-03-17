def mark_hamming_duplicates(adata, var_filters, distance_threshold=0.025):
    """
    Flags rows as 'is_duplicate' if their distance to their previous row (post-sort) is below `distance_threshold`.
    
    - var_filters: dict of var columns to OR together to select relevant columns.
    - distance_threshold: e.g., 0.0 for exact duplicates, or 0.025 for "close enough"
    """
    import torch
    import numpy as np
    import pandas as pd

    # Step 1: OR-combine all conditions (column subset)
    mask = torch.zeros(len(adata.var), dtype=torch.bool)
    for key, value in var_filters.items():
        col_values = adata.var[key].values
        mask |= torch.from_numpy(col_values == value)

    selected_cols = adata.var.index[mask.numpy()].to_list()
    col_indices = [adata.var.index.get_loc(col) for col in selected_cols]

    print(f"Selected {len(col_indices)} columns out of {adata.var.shape[0]} based on provided filters (OR logic).")

    # Step 2: Extract relevant columns from .X
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()  # convert sparse to dense if needed
    X_subset = X[:, col_indices]
    X_tensor = torch.from_numpy(X_subset.astype(np.float32))  # make sure dtype is float for nan handling

    N = X_tensor.shape[0]

    # Step 3: Replace NaNs for sorting (set NaN to -1 so 0,1 come first)
    X_sortable = X_tensor.nan_to_num(-1)

    # Step 4: Perform lexicographic sort on rows
    sort_keys = X_sortable.tolist()
    sorted_idx = sorted(range(N), key=lambda i: sort_keys[i])
    sorted_X = X_tensor[sorted_idx]

    # Step 5: Compute distances & flag
    is_duplicate = torch.zeros(N, dtype=torch.bool)
    for i in range(1, sorted_X.shape[0]):
        row1 = sorted_X[i - 1]
        row2 = sorted_X[i]
        valid_mask = (~torch.isnan(row1)) & (~torch.isnan(row2))

        if valid_mask.sum() == 0:
            dist = float('nan')  # No valid overlap
        else:
            diff = (row1[valid_mask] != row2[valid_mask]).sum()
            dist = diff.item() / valid_mask.sum().item()

        # Flag if distance below threshold
        if dist < distance_threshold:
            is_duplicate[sorted_idx[i]] = True

    # Step 6: Add to obs
    adata.obs['is_duplicate'] = pd.Series(is_duplicate.numpy(), index=adata.obs_names)

    # Step 7: Optional filtering
    adata_unique = adata[~adata.obs['is_duplicate']].copy()
    
    adata_nonunique = adata[adata.obs['is_duplicate']].copy()

    return adata_unique, adata_nonunique, adata 