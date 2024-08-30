## append_C_context
import numpy as np
import anndata as ad
import pandas as pd

## Conversion SMF Specific 
# Read methylation QC
def append_C_context(adata, obs_column='Reference', use_consensus=False):
    """
    Input: An adata object, the obs_column of interst, and whether to use the consensus sequence from the category.
    Output: Adds Cytosine context to the position within the given category. When use_consensus is True, it uses the consensus sequence, otherwise it defaults to the FASTA sequence.
    """
    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_site', 'ambiguous_CpG_site', 'other_C']
    categories = adata.obs[obs_column].cat.categories
    if use_consensus:
        sequence = adata.uns[f'{cat}_consensus_sequence']
    else:
        sequence = adata.uns[f'{cat}_FASTA_sequence']
    for cat in categories:
        boolean_dict = {}
        for site_type in site_types:
            boolean_dict[f'{cat}_{site_type}'] = np.full(len(sequence), False, dtype=bool)
        # Iterate through the sequence and apply the criteria
        for i in range(1, len(sequence) - 1):
            if sequence[i] == 'C':
                if sequence[i - 1] == 'G' and sequence[i + 1] != 'G':
                    boolean_dict[f'{cat}_GpC_site'][i] = True
                elif sequence[i - 1] == 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_ambiguous_GpC_site'][i] = True
                elif sequence[i - 1] != 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_CpG_site'][i] = True
                elif sequence[i - 1] == 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_ambiguous_CpG_site'][i] = True
                elif sequence[i - 1] != 'G' and sequence[i + 1] != 'G':
                    boolean_dict[f'{cat}_other_C'][i] = True
        for site_type in site_types:
            adata.var[f'{cat}_{site_type}'] = boolean_dict[f'{cat}_{site_type}'].astype(bool)
            adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].copy().X

