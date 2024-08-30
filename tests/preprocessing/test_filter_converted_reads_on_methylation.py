## filter_converted_reads_on_methylation
import numpy as np
import anndata as ad
import pandas as pd

## Conversion SMF Specific 
# Read methylation QC
def filter_converted_reads_on_methylation(adata, valid_SMF_site_threshold=0.8, min_SMF_threshold=0.025):
    """
    Input: Adata object. Minimum thresholds for valid SMF site fraction in read, as well as minimum methylation content in read
    Output: A subset of the adata object
    """
    if valid_SMF_site_threshold:
        # Keep reads that have over a given valid GpC site content
        adata = adata[adata.obs['fraction_valid_GpC_site_in_range'] > valid_SMF_site_threshold].copy()
    if min_SMF_threshold:
        # Keep reads with SMF methylation over background methylation.
        adata = adata[adata.obs['GpC_above_other_C'] == True].copy()
        # Keep reads over a defined methylation threshold
        adata = adata[adata.obs['GpC_site_row_methylation_means'] > min_SMF_threshold].copy()