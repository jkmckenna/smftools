## append_C_context

## Conversion SMF Specific 
# Read methylation QC
def append_C_context(adata, obs_column='Reference', use_consensus=False, native=False):
    """
    Adds Cytosine context to the position within the given category. When use_consensus is True, it uses the consensus sequence, otherwise it defaults to the FASTA sequence.

    Parameters:
        adata (AnnData): The input adata object.
        obs_column (str): The observation column in which to stratify on. Default is 'Reference', which should not be changed for most purposes.
        use_consensus (bool): A truth statement indicating whether to use the consensus sequence from the reads mapped to the reference. If False, the reference FASTA is used instead.
        native (bool): If False, perform conversion SMF assumptions. If True, perform native SMF assumptions
    
    Returns:
        None
    """
    import numpy as np
    import anndata as ad

    print('Adding Cytosine context based on reference FASTA sequence for sample')
    
    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C', 'any_C_site']
    categories = adata.obs[obs_column].cat.categories
    for cat in categories:
        # Assess if the strand is the top or bottom strand converted
        if 'top' in cat:
            strand = 'top'
        elif 'bottom' in cat:
            strand = 'bottom'

        if native:
            basename = cat.split(f"_{strand}")[0]
            if use_consensus:
                sequence = adata.uns[f'{basename}_consensus_sequence']
            else:
                # This sequence is the unconverted FASTA sequence of the original input FASTA for the locus
                sequence = adata.uns[f'{basename}_FASTA_sequence']
        else:
            basename = cat.split(f"_{strand}")[0]
            if use_consensus:
                sequence = adata.uns[f'{basename}_consensus_sequence']
            else:
                # This sequence is the unconverted FASTA sequence of the original input FASTA for the locus
                sequence = adata.uns[f'{basename}_FASTA_sequence']
        # Init a dict keyed by reference site type that points to a bool of whether the position is that site type.    
        boolean_dict = {}
        for site_type in site_types:
            boolean_dict[f'{cat}_{site_type}'] = np.full(len(sequence), False, dtype=bool)
        
        if strand == 'top':
            # Iterate through the sequence and apply the criteria
            for i in range(1, len(sequence) - 1):
                if sequence[i] == 'C':
                    boolean_dict[f'{cat}_any_C_site'][i] = True
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
                    boolean_dict[f'{cat}_any_C_site'][i] = True
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
            adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].X
