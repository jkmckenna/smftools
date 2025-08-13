def calculate_read_modification_stats(adata, reference_column, sample_names_col, mod_target_bases):
    """
    Adds methylation/deamination statistics for each read. 
    Indicates the read GpC and CpG methylation ratio to other_C methylation (background false positive metric for Cytosine MTase SMF).

    Parameters:
        adata (AnnData): An adata object
        reference_column (str): String representing the name of the Reference column to use
        sample_names_col (str): String representing the name of the sample name column to use
        mod_target_bases:

    Returns:
        None 
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    print('Calculating read level Modification statistics')

    references = set(adata.obs[reference_column])
    sample_names = set(adata.obs[sample_names_col])
    site_types = []

    if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
        site_types += ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C_site', 'any_C_site']
    
    if 'A' in mod_target_bases:
        site_types += ['A_site']

    for site_type in site_types:
        adata.obs[f'Modified_{site_type}_count'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'Total_{site_type}_in_read'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'Fraction_{site_type}_modified'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)

    for ref in references:
        ref_subset = adata[adata.obs[reference_column] == ref].copy()
        for site_type in site_types:
            print(f'Iterating over {ref}_{site_type}')
            observation_matrix = ref_subset.obsm[f'{ref}_{site_type}']
            total_positions_in_read = np.nansum(~np.isnan(observation_matrix), axis=1)
            number_mods_in_read = np.nansum(observation_matrix, axis=1)
            fraction_modified = number_mods_in_read / total_positions_in_read

            fraction_modified = np.divide(
                number_mods_in_read,
                total_positions_in_read,
                out=np.full_like(number_mods_in_read, np.nan, dtype=float),
                where=total_positions_in_read != 0
            )

            temp_obs_data = pd.DataFrame({f'Total_{site_type}_in_read': total_positions_in_read,
                                        f'Modified_{site_type}_count': number_mods_in_read,
                                        f'Fraction_{site_type}_modified': fraction_modified}, index=ref_subset.obs.index)
            
            adata.obs.update(temp_obs_data)

    if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
        with np.errstate(divide='ignore', invalid='ignore'):
            gpc_to_c_ratio = np.divide(
                adata.obs[f'Fraction_GpC_site_modified'],
                adata.obs[f'Fraction_other_C_site_modified'],
                out=np.full_like(adata.obs[f'Fraction_GpC_site_modified'], np.nan, dtype=float),
                where=adata.obs[f'Fraction_other_C_site_modified'] != 0
            )

            cpg_to_c_ratio = np.divide(
                adata.obs[f'Fraction_CpG_site_modified'],
                adata.obs[f'Fraction_other_C_site_modified'],
                out=np.full_like(adata.obs[f'Fraction_CpG_site_modified'], np.nan, dtype=float),
                where=adata.obs[f'Fraction_other_C_site_modified'] != 0
                )
            
        adata.obs['GpC_to_other_C_mod_ratio'] = gpc_to_c_ratio
        adata.obs['CpG_to_other_C_mod_ratio'] = cpg_to_c_ratio


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