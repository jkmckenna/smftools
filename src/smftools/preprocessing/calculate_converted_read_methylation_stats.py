## calculate_converted_read_methylation_stats

## Conversion SMF Specific 
# Read methylation QC

def calculate_converted_read_methylation_stats(adata, reference_column, sample_names_col):
    """
    Adds methylation statistics for each read. Indicates whether the read GpC methylation exceeded other_C methylation (background false positives).

    Parameters:
        adata (AnnData): An adata object
        reference_column (str): String representing the name of the Reference column to use
        sample_names_col (str): String representing the name of the sample name column to use

    Returns:
        None 
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    print('Calculating read level methylation statistics')

    references = set(adata.obs[reference_column])
    sample_names = set(adata.obs[sample_names_col])

    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C']

    for site_type in site_types:
        adata.obs[f'{site_type}_row_methylation_sums'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'{site_type}_row_methylation_means'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
        adata.obs[f'number_valid_{site_type}_in_read'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'fraction_valid_{site_type}_in_range'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    for cat in references:
        cat_subset = adata[adata.obs[reference_column] == cat].copy()
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

# Below should be a plotting function
    # adata.uns['methylation_dict'] = {}
    # n_bins = 50
    # site_types_to_analyze = ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C']

    # for reference in references:
    #     reference_adata = adata[adata.obs[reference_column] == reference].copy()
    #     split_reference = reference.split('_')[0][1:]
    #     for sample in sample_names:
    #         sample_adata = reference_adata[reference_adata.obs[sample_names_col] == sample].copy()
    #         for site_type in site_types_to_analyze:
    #             methylation_data = sample_adata.obs[f'{site_type}_row_methylation_means']
    #             max_meth = np.max(sample_adata.obs[f'{site_type}_row_methylation_sums'])
    #             if not np.isnan(max_meth):
    #                 n_bins = int(max_meth // 2)
    #             else:
    #                 n_bins = 1
    #             mean = np.mean(methylation_data)
    #             median = np.median(methylation_data)
    #             stdev = np.std(methylation_data)
    #             adata.uns['methylation_dict'][f'{reference}_{sample}_{site_type}'] = [mean, median, stdev]
    #             if show_methylation_histogram or save_methylation_histogram:
    #                 fig, ax = plt.subplots(figsize=(6, 4))
    #                 count, bins, patches =  plt.hist(methylation_data, bins=n_bins, weights=np.ones(len(methylation_data)) / len(methylation_data), alpha=0.7, color='blue', edgecolor='black')
    #                 plt.axvline(median, color='red', linestyle='dashed', linewidth=1)
    #                 plt.text(median + stdev, max(count)*0.8, f'Median: {median:.2f}', color='red')
    #                 plt.axvline(median - stdev, color='green', linestyle='dashed', linewidth=1, label=f'Stdev: {stdev:.2f}')
    #                 plt.axvline(median + stdev, color='green', linestyle='dashed', linewidth=1)
    #                 plt.text(median + stdev + 0.05, max(count) / 3, f'+1 Stdev: {stdev:.2f}', color='green')
    #                 plt.xlabel('Fraction methylated')
    #                 plt.ylabel('Proportion')
    #                 title = f'Distribution of {methylation_data.shape[0]} read {site_type} methylation means \nfor {sample} sample on {split_reference} after filtering'
    #                 plt.title(title, pad=20)
    #                 plt.xlim(-0.05, 1.05)  # Set x-axis range from 0 to 1
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False)
    #                 save_name = output_directory + f'/{readwrite.date_string()} {title}'
    #                 if save_methylation_histogram:
    #                     plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    #                     plt.close()
    #                 else:
    #                     plt.show()