## calculate_converted_read_methylation_stats

## Conversion SMF Specific 
# Read methylation QC

def calculate_converted_read_methylation_stats(adata, obs_column='Reference'):
    """
    Adds methylation statistics for each read. Indicates whether the read GpC methylation exceeded other_C methylation (background false positives)

    Parameters:
        adata (AnnData): An AnnData object
        obs_column (str): observation category of interest

    Returns:
        None 
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_site', 'ambiguous_CpG_site', 'other_C']
    categories = adata.obs[obs_column].cat.categories
    for site_type in site_types:
        adata.obs[f'{site_type}_row_methylation_sums'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'{site_type}_row_methylation_means'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
        adata.obs[f'number_valid_{site_type}_in_read'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'fraction_valid_{site_type}_in_range'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    for cat in categories:
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        for site_type in site_types:
            print(f'Iterating over {cat}_{site_type}')
            observation_matrix = cat_subset.obsm[f'{cat}_{site_type}']
            number_valid_positions_in_read = np.nansum(~np.isnan(observation_matrix), axis=1)
            row_methylation_sums = np.nansum(observation_matrix, axis=1)
            number_valid_positions_in_read[number_valid_positions_in_read == 0] = 1
            fraction_valid_positions_in_range = number_valid_positions_in_read / np.max(number_valid_positions_in_read)
            row_methylation_means = np.divide(row_methylation_sums, number_valid_positions_in_read)
            temp_obs_data = pd.DataFrame({f'number_valid_{site_type}_in_read': number_valid_positions_in_read,
                                        f'fraction_valid_{site_type}_in_range': fraction_valid_positions_in_range,
                                        f'{site_type}_row_methylation_sums': row_methylation_sums,
                                        f'{site_type}_row_methylation_means': row_methylation_means}, index=cat_subset.obs.index)
            adata.obs.update(temp_obs_data)
    # Indicate whether the read-level GpC methylation rate exceeds the false methylation rate of the read
    pass_array = np.array(adata.obs[f'GpC_site_row_methylation_means'] > adata.obs[f'other_C_row_methylation_means'])
    adata.obs['GpC_above_other_C'] = pd.Series(pass_array, index=adata.obs.index, dtype=bool)