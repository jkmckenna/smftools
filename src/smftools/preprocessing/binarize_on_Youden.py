def binarize_on_Youden(adata, 
                       ref_column='Reference_strand', 
                       output_layer_name='binarized_methylation'):
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
    references = adata.obs[ref_column].cat.categories

    for ref in references:
        # Select subset for this category
        ref_mask = adata.obs[ref_column] == ref
        ref_subset = adata[ref_mask]

        # Extract the probability matrix
        original_matrix = ref_subset.X.copy()

        # Extract the thresholds for each position efficiently
        thresholds = np.array(ref_subset.var[f'{ref}_position_methylation_thresholding_Youden_stats'].apply(lambda x: x[0]))

        # Identify NaN values
        nan_mask = np.isnan(original_matrix)

        # Binarize based on threshold
        binarized_matrix = (original_matrix > thresholds).astype(float)

        # Restore NaN values
        binarized_matrix[nan_mask] = np.nan

        # Assign the binarized values back into the preallocated storage
        binarized_methylation[ref_subset, :] = binarized_matrix

    # Store the binarized matrix in a new layer
    adata.layers[output_layer_name] = binarized_methylation