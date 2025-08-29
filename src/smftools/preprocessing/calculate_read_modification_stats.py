def calculate_read_modification_stats(adata, 
                                      reference_column, 
                                      sample_names_col, 
                                      mod_target_bases,
                                      uns_flag="read_modification_stats_calculated",
                                      bypass=False,
                                      force_redo=False
):
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

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

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
        adata.obs[f'Total_{site_type}_in_reference'] = pd.Series(np.nan, index=adata.obs_names, dtype=int)
        adata.obs[f'Valid_{site_type}_in_read_vs_reference'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)


    for ref in references:
        ref_subset = adata[adata.obs[reference_column] == ref]
        for site_type in site_types:
            print(f'Iterating over {ref}_{site_type}')
            observation_matrix = ref_subset.obsm[f'{ref}_{site_type}']
            total_positions_in_read = np.nansum(~np.isnan(observation_matrix), axis=1)
            total_positions_in_reference = observation_matrix.shape[1]
            fraction_valid_positions_in_read_vs_ref = total_positions_in_read / total_positions_in_reference
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
                                        f'Fraction_{site_type}_modified': fraction_modified,
                                        f'Total_{site_type}_in_reference': total_positions_in_reference,
                                        f'Valid_{site_type}_in_read_vs_reference': fraction_valid_positions_in_read_vs_ref}, 
                                        index=ref_subset.obs.index)
            
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

    # mark as done
    adata.uns[uns_flag] = True

    return