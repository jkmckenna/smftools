## filter_converted_reads_on_methylation

## Conversion SMF Specific 
def filter_converted_reads_on_methylation(adata, valid_SMF_site_threshold=0.8, min_SMF_threshold=0.025, max_SMF_threshold=0.975):
    """
    Filter adata object using minimum thresholds for valid SMF site fraction in read, as well as minimum methylation content in read.

    Parameters:
        adata (AnnData): An adata object.
        valid_SMF_site_threshold (float): A minimum proportion of valid SMF sites that must be present in the read. Default is 0.8
        min_SMF_threshold (float): A minimum read methylation level. Default is 0.025
    Returns:
        Anndata
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    if valid_SMF_site_threshold:
        # Keep reads that have over a given valid GpC site content
        adata = adata[adata.obs['fraction_valid_GpC_site_in_range'] > valid_SMF_site_threshold].copy()

    if min_SMF_threshold:
        # Keep reads with SMF methylation over background methylation.
        below_background = (~adata.obs['GpC_above_other_C']).sum()
        print(f'Removing {below_background} reads that have GpC conversion below background conversion rate')
        adata = adata[adata.obs['GpC_above_other_C'] == True].copy()
        # Keep reads over a defined methylation threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['GpC_site_row_methylation_means'] > min_SMF_threshold].copy()
        s1 = adata.shape[0]
        below_threshold = s0 - s1
        print(f'Removing {below_threshold} reads that have GpC conversion below a minimum threshold conversion rate')

    if max_SMF_threshold:
        # Keep reads below a defined methylation threshold
        s0 = adata.shape[0]
        adata = adata[adata.obs['GpC_site_row_methylation_means'] < max_SMF_threshold].copy()
        s1 = adata.shape[0]
        above_threshold = s0 - s1
        print(f'Removing {above_threshold} reads that have GpC conversion above a maximum threshold conversion rate')

    return adata

