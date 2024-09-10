## append_C_context

## Conversion SMF Specific 
# Read methylation QC
def append_C_context(adata, obs_column='Reference', use_consensus=False):
    """
    Adds Cytosine context to the position within the given category. When use_consensus is True, it uses the consensus sequence, otherwise it defaults to the FASTA sequence.

    Parameters:
        adata (AnnData): The input adata object.
        obs_column (str): The observation column in which to stratify on. Default is 'Reference', which should not be changed for most purposes.
        use_consensus (bool): A truth statement indicating whether to use the consensus sequence from the reads mapped to the reference. If False, the reference FASTA is used instead.
    Input: An adata object, the obs_column of interst, and whether to use the consensus sequence from the category.
    
    Returns:
        None
    """
    import numpy as np
    import anndata as ad
    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C']
    categories = adata.obs[obs_column].cat.categories
    for cat in categories:
        # Assess if the strand is the top or bottom strand converted
        if 'top' in cat:
            strand = 'top'
        elif 'bottom' in cat:
            strand = 'bottom'

        if use_consensus:
            sequence = adata.uns[f'{cat}_consensus_sequence']
        else:
            # This sequence is the unconverted FASTA sequence of the original input FASTA for the locus
            sequence = adata.uns[f'{cat}_FASTA_sequence']
        # Init a dict keyed by reference site type that points to a bool of whether the position is that site type.    
        boolean_dict = {}
        for site_type in site_types:
            boolean_dict[f'{cat}_{site_type}'] = np.full(len(sequence), False, dtype=bool)
        
        if strand == 'top':
            # Iterate through the sequence and apply the criteria
            for i in range(1, len(sequence) - 1):
                if sequence[i] == 'C':
                    if sequence[i - 1] == 'G' and sequence[i + 1] != 'G':
                        boolean_dict[f'{cat}_GpC_site'][i] = True
                    elif sequence[i - 1] == 'G' and sequence[i + 1] == 'G':
                        boolean_dict[f'{cat}_ambiguous_GpC_CpG_site'][i] = True
                    elif sequence[i - 1] != 'G' and sequence[i + 1] == 'G':
                        boolean_dict[f'{cat}_CpG_site'][i] = True
                    elif sequence[i - 1] != 'G' and sequence[i + 1] != 'G':
                        boolean_dict[f'{cat}_other_C'][i] = True
        elif strand == 'bottom':
            # Iterate through the sequence and apply the criteria
            for i in range(1, len(sequence) - 1):
                if sequence[i] == 'G':
                    if sequence[i + 1] == 'C' and sequence[i - 1] != 'C':
                        boolean_dict[f'{cat}_GpC_site'][i] = True
                    elif sequence[i - 1] == 'C' and sequence[i + 1] == 'C':
                        boolean_dict[f'{cat}_ambiguous_GpC_CpG_site'][i] = True
                    elif sequence[i - 1] == 'C' and sequence[i + 1] != 'C':
                        boolean_dict[f'{cat}_CpG_site'][i] = True
                    elif sequence[i - 1] != 'C' and sequence[i + 1] != 'C':
                        boolean_dict[f'{cat}_other_C'][i] = True
        else:
            print('Error: top or bottom strand of conversion could not be determined. Ensure this value is in the Reference name.')

        for site_type in site_types:
            adata.var[f'{cat}_{site_type}'] = boolean_dict[f'{cat}_{site_type}'].astype(bool)
            adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].copy().X

