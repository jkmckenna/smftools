## binarize_on_Youden

def binarize_on_Youden(adata, obs_column='Reference'):
    """
    Add a new layer to the adata object that has binarized SMF values based on the position thresholds determined by calculate_position_Youden

    Parameters:
        adata (AnnData): The anndata object to binarize. pp.calculate_position_Youden function has to be run first.
        obs_column (str): The obs_column to stratify on. Needs to be the same as passed in pp.calculate_position_Youden.
    Input: adata object that has had calculate_position_Youden called on it.
    Output: 
    """
    import numpy as np
    import anndata as ad
    temp_adata = None
    categories = adata.obs[obs_column].cat.categories 
    for cat in categories:
        # Get the category subset
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        # extract the probability matrix for the category subset
        original_matrix = cat_subset.X
        # extract the learned methylation call thresholds for each position in the category.
        thresholds = [cat_subset.var[f'{cat}_position_methylation_thresholding_Youden_stats'][i][0] for i in range(cat_subset.shape[1])]
        # In the original matrix, get all positions that are nan values
        nan_mask = np.isnan(original_matrix)
        # Binarize the matrix on the new thresholds
        binarized_matrix = (original_matrix > thresholds).astype(float)
        # At the original positions that had nan values, replace the values with nans again
        binarized_matrix[nan_mask] = np.nan
        # Make a new layer for the reference that contains the binarized methylation calls
        cat_subset.layers['binarized_methylation'] = binarized_matrix
        if temp_adata:
            # If temp_data already exists, concatenate
            temp_adata = ad.concat([temp_adata, cat_subset], join='outer', index_unique=None).copy()
        else:
            # If temp_adata is still None, initialize temp_adata with reference_subset
            temp_adata = cat_subset.copy()

    # Sort the temp adata on the index names of the primary adata
    temp_adata = temp_adata[adata.obs_names].copy()
    # Pull back the new binarized layers into the original adata object
    adata.layers['binarized_methylation'] = temp_adata.layers['binarized_methylation']