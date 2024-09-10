## calculate_coverage

def calculate_coverage(adata, obs_column='Reference', position_nan_threshold=0.05):
    """
    Append position level metadata regarding whether the position is informative within the given observation category.

    Parameters:
        adata (AnnData): An AnnData object
        obs_column (str): Observation column value to subset on prior to calculating position statistics for that category.
        position_nan_threshold (float): A minimal fractional threshold of coverage within the obs_column category to call the position as valid.

    Returns:
        None
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    categories = adata.obs[obs_column].cat.categories
    n_categories_with_position = np.zeros(adata.shape[1])
    # Loop over categories
    for cat in categories:
        # Look at positional information for each reference
        temp_cat_adata = adata[adata.obs[obs_column] == cat].copy()
        # Look at read coverage on the given category strand
        cat_valid_coverage = np.sum(~np.isnan(temp_cat_adata.X), axis=0)
        cat_invalid_coverage = np.sum(np.isnan(temp_cat_adata.X), axis=0)
        cat_valid_fraction = cat_valid_coverage / (cat_valid_coverage + cat_invalid_coverage)
        # Append metadata for category to the anndata object
        adata.var[f'{cat}_valid_fraction'] = pd.Series(cat_valid_fraction, index=adata.var.index)
        # Characterize if the position is in the given category or not
        conditions = [
            (adata.var[f'{cat}_valid_fraction'] >= position_nan_threshold),
            (adata.var[f'{cat}_valid_fraction'] < position_nan_threshold)
        ]
        choices = [True, False]
        adata.var[f'position_in_{cat}'] = np.select(conditions, choices, default=False)
        n_categories_with_position += np.array(adata.var[f'position_in_{cat}'])

    # Final array with the sum at each position of the number of categories covering that position
    adata.var[f'N_{obs_column}_with_position'] = n_categories_with_position.astype(int)