def binarize_on_Youden(adata, obs_column='Reference'):
    """
    Binarize SMF values based on position thresholds determined by calculate_position_Youden.

    Parameters:
        adata (AnnData): The anndata object to binarize. `calculate_position_Youden` must have been run first.
        obs_column (str): The obs column to stratify on. Needs to match what was passed in `calculate_position_Youden`.

    Modifies:
        Adds a new layer to `adata.layers['binarized_methylation']` containing the binarized methylation matrix.
    """
    import numpy as np
    import anndata as ad    

    # Initialize an empty matrix to store the binarized methylation values
    binarized_methylation = np.full_like(adata.X, np.nan, dtype=float)  # Keeps same shape as adata.X

    # Get unique categories
    categories = adata.obs[obs_column].cat.categories

    for cat in categories:
        # Select subset for this category
        cat_mask = adata.obs[obs_column] == cat
        cat_subset = adata[cat_mask]

        # Extract the probability matrix
        original_matrix = cat_subset.X.copy()

        # Extract the thresholds for each position efficiently
        thresholds = np.array(cat_subset.var[f'{cat}_position_methylation_thresholding_Youden_stats'].apply(lambda x: x[0]))

        # Identify NaN values
        nan_mask = np.isnan(original_matrix)

        # Binarize based on threshold
        binarized_matrix = (original_matrix > thresholds).astype(float)

        # Restore NaN values
        binarized_matrix[nan_mask] = np.nan

        # Assign the binarized values back into the preallocated storage
        binarized_methylation[cat_mask, :] = binarized_matrix

    # Store the binarized matrix in a new layer
    adata.layers['binarized_methylation'] = binarized_methylation