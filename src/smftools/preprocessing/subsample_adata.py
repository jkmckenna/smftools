def subsample_adata(adata, obs_columns=None, max_samples=2000, random_seed=42):
    """
    Subsamples an AnnData object so that each unique combination of categories 
    in the given `obs_columns` has at most `max_samples` observations.
    If `obs_columns` is None or empty, the function randomly subsamples the entire dataset.
    
    Parameters:
        adata (AnnData): The AnnData object to subsample.
        obs_columns (list of str, optional): List of observation column names to group by.
        max_samples (int): The maximum number of observations per category combination.
        random_seed (int): Random seed for reproducibility.

    Returns:
        AnnData: A new AnnData object with subsampled observations.
    """
    import anndata as ad
    import numpy as np

    np.random.seed(random_seed)  # Ensure reproducibility

    if not obs_columns:  # If no obs columns are given, sample globally
        if adata.shape[0] > max_samples:
            sampled_indices = np.random.choice(adata.obs.index, max_samples, replace=False)
        else:
            sampled_indices = adata.obs.index  # Keep all if fewer than max_samples
        
        return adata[sampled_indices].copy()

    sampled_indices = []

    # Get unique category combinations from all specified obs columns
    unique_combinations = adata.obs[obs_columns].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        # Build filter condition dynamically for multiple columns
        condition = (adata.obs[obs_columns] == row.values).all(axis=1)
        
        # Get indices for the current category combination
        subset_indices = adata.obs[condition].index.to_numpy()

        # Subsample or take all
        if len(subset_indices) > max_samples:
            sampled = np.random.choice(subset_indices, max_samples, replace=False)
        else:
            sampled = subset_indices  # Keep all if fewer than max_samples

        sampled_indices.extend(sampled)

    # ⚠ Handle backed mode detection
    if adata.isbacked:
        print("⚠ Detected backed mode. Subset will be loaded fully into memory.")
        subset = adata[sampled_indices]
        subset = subset.to_memory()
    else:
        subset = adata[sampled_indices]

    # Create a new AnnData object with only the selected indices
    return subset[sampled_indices].copy()
