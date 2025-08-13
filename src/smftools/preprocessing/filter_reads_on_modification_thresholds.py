def filter_reads_on_modification_thresholds(adata, 
                                          gpc_thresholds=None, 
                                          cpg_thresholds=None,
                                          any_c_thresholds=None,
                                          a_thresholds=None,
                                          use_other_c_as_background=False):
    """
    Filter adata object using minimum thresholds for valid SMF site fraction in read, as well as minimum modification content in read.

    Parameters:
        adata (AnnData): An adata object.
        thresholds (various types) (list of floats): [A minimum read modification fraction, a maximum read modification fraction]. Floats range between 0 and 1.
    Returns:
        Anndata
    """
    import numpy as np
    import anndata as ad
    import pandas as pd


    if gpc_thresholds:
        min_threshold, max_threshold = gpc_thresholds
        if use_other_c_as_background:
            # Keep reads with SMF mod over background mod.
            below_background = (adata.obs['GpC_to_other_C_mod_ratio'] < 1).sum()
            print(f'Removing {below_background} reads that have GpC modification below background other C modification rate')
            adata = adata[adata.obs['GpC_to_other_C_mod_ratio'] > 1].copy()
        # Keep reads over a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_GpC_site_modified'] > min_threshold].copy()
        s1 = adata.shape[0]
        below_threshold = s0 - s1
        print(f'Removing {below_threshold} reads that have GpC modification below a minimum threshold modification rate')
        # Keep reads below a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_GpC_site_modified'] < max_threshold].copy()
        s1 = adata.shape[0]
        above_threshold = s0 - s1
        print(f'Removing {above_threshold} reads that have GpC modification above a maximum threshold modification rate')

    if cpg_thresholds:
        min_threshold, max_threshold = cpg_thresholds
        # Keep reads with SMF mod over background mod.
        if use_other_c_as_background:
            below_background = (adata.obs['CpG_to_other_C_mod_ratio'] < 1).sum()
            print(f'Removing {below_background} reads that have CpG modification below background modification rate')
            adata = adata[adata.obs['CpG_to_other_C_mod_ratio'] > 1].copy()
        # Keep reads over a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_CpG_site_modified'] > min_threshold].copy()
        s1 = adata.shape[0]
        below_threshold = s0 - s1
        print(f'Removing {below_threshold} reads that have CpG modification below a minimum threshold modification rate')
        # Keep reads below a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_CpG_site_modified'] < max_threshold].copy()
        s1 = adata.shape[0]
        above_threshold = s0 - s1
        print(f'Removing {above_threshold} reads that have CpG modification above a maximum threshold modification rate')

    if any_c_thresholds:
        min_threshold, max_threshold = any_c_thresholds
        # Keep reads over a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_any_C_site_modified'] > min_threshold].copy()
        s1 = adata.shape[0]
        below_threshold = s0 - s1
        print(f'Removing {below_threshold} reads that have C modification below a minimum threshold modification rate')
        # Keep reads below a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_any_C_site_modified'] < max_threshold].copy()
        s1 = adata.shape[0]
        above_threshold = s0 - s1
        print(f'Removing {above_threshold} reads that have any C modification above a maximum threshold modification rate')

    if a_thresholds:
        min_threshold, max_threshold = a_thresholds
        # Keep reads over a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_A_site_modified'] > min_threshold].copy()
        s1 = adata.shape[0]
        below_threshold = s0 - s1
        print(f'Removing {below_threshold} reads that have A modification below a minimum threshold modification rate')
        # Keep reads below a defined mod threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['Fraction_A_site_modified'] < max_threshold].copy()
        s1 = adata.shape[0]
        above_threshold = s0 - s1
        print(f'Removing {above_threshold} reads that have any A modification above a maximum threshold modification rate')

    return adata

