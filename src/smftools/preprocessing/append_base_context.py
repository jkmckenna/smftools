def append_base_context(adata, 
                        obs_column='Reference_strand', 
                        use_consensus=False,
                        native=False, 
                        mod_target_bases=['GpC', 'CpG'],
                        bypass=False,
                        force_redo=False,
                        uns_flag='base_context_added'
):
    """
    Adds nucleobase context to the position within the given category. When use_consensus is True, it uses the consensus sequence, otherwise it defaults to the FASTA sequence.

    Parameters:
        adata (AnnData): The input adata object.
        obs_column (str): The observation column in which to stratify on. Default is 'Reference_strand', which should not be changed for most purposes.
        use_consensus (bool): A truth statement indicating whether to use the consensus sequence from the reads mapped to the reference. If False, the reference FASTA is used instead.
        native (bool): If False, perform conversion SMF assumptions. If True, perform native SMF assumptions
        mod_target_bases (list): Base contexts that may be modified.
    
    Returns:
        None
    """
    import numpy as np
    import anndata as ad

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

    print('Adding base context based on reference FASTA sequence for sample')
    categories = adata.obs[obs_column].cat.categories
    site_types = []
    
    if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
        site_types += ['GpC_site', 'CpG_site', 'ambiguous_GpC_CpG_site', 'other_C_site', 'any_C_site']
    
    if 'A' in mod_target_bases:
        site_types += ['A_site']

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

        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
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
                            boolean_dict[f'{cat}_other_C_site'][i] = True
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
                            boolean_dict[f'{cat}_other_C_site'][i] = True
            else:
                print('Error: top or bottom strand of conversion could not be determined. Ensure this value is in the Reference name.')

        if 'A' in mod_target_bases:
            if strand == 'top':
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == 'A':
                        boolean_dict[f'{cat}_A_site'][i] = True
            elif strand == 'bottom':
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == 'T':
                        boolean_dict[f'{cat}_A_site'][i] = True
            else:
                print('Error: top or bottom strand of conversion could not be determined. Ensure this value is in the Reference name.')

        for site_type in site_types:
            adata.var[f'{cat}_{site_type}'] = boolean_dict[f'{cat}_{site_type}'].astype(bool)
            if native:
                adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].layers['binarized_methylation']
            else:
                adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].X

    # mark as done
    adata.uns[uns_flag] = True

    return None