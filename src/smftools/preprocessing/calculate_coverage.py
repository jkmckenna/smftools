def calculate_coverage(adata, obs_column='Reference_strand', position_nan_threshold=0.05):
    """
    Append position-level metadata regarding whether the position is informative within the given observation category.

    Parameters:
        adata (AnnData): An AnnData object
        obs_column (str): Observation column value to subset on prior to calculating position statistics for that category.
        position_nan_threshold (float): A minimal fractional threshold of coverage within the obs_column category to call the position as valid.

    Modifies:
        - Adds new columns to `adata.var` containing coverage statistics.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    
    categories = adata.obs[obs_column].cat.categories
    n_categories_with_position = np.zeros(adata.shape[1])

    # Loop over categories
    for cat in categories:
        print(f'Assessing positional coverage across samples for {cat} reference')

        # Subset to current category
        cat_mask = adata.obs[obs_column] == cat
        temp_cat_adata = adata[cat_mask]

        # Compute fraction of valid coverage
        cat_valid_coverage = np.sum(~np.isnan(temp_cat_adata.X), axis=0)
        cat_valid_fraction = cat_valid_coverage / temp_cat_adata.shape[0]  # Avoid extra computation

        # Store coverage stats
        adata.var[f'{cat}_valid_fraction'] = pd.Series(cat_valid_fraction, index=adata.var.index)

        # Assign whether the position is covered based on threshold
        adata.var[f'position_in_{cat}'] = cat_valid_fraction >= position_nan_threshold

        # Sum the number of categories covering each position
        n_categories_with_position += adata.var[f'position_in_{cat}'].values

    # Store final category count
    adata.var[f'N_{obs_column}_with_position'] = n_categories_with_position.astype(int)
