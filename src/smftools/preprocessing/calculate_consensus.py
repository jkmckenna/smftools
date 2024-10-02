# calculate_consensus

def calculate_consensus(adata, reference, sample=False, reference_column='Reference', sample_column='Sample'):
    """
    Takes an input AnnData object, the reference to subset on, and the sample name to subset on to calculate the consensus sequence of the read set.

    Parameters:
        adata (AnnData): The input adata to append consensus metadata to.
        reference (str): The name of the reference to subset the adata on.
        sample (bool | str): If False, uses all samples. If a string is passed, the adata is further subsetted to only analyze that sample.
        reference_column (str): The name of the reference column (Default is 'Reference')
        sample_column (str): The name of the sample column (Default is 'Sample)

    Returns:
        None
    
    """
    import numpy as np

    # Subset the adata on the refernce of interest. Optionally, subset additionally on a sample of interest.
    record_subset = adata[adata.obs[reference_column] == reference].copy()
    if sample:
        record_subset = record_subset[record_subset.obs[sample_column] == sample].copy()
    else:
        pass

    # Grab layer names from the adata object that correspond to the binary encodings of the read sequences.
    layers = [layer for layer in record_subset.layers if '_binary_' in layer]
    layer_map, layer_counts = {}, []
    for i, layer in enumerate(layers):
        # Gives an integer mapping to access which sequence base the binary layer is encoding
        layer_map[i] = layer.split('_')[0]
        # Get the positional counts from all reads for the given base identity.
        layer_counts.append(np.sum(record_subset.layers[layer], axis=0))
    # Combine the positional counts array derived from each binary base layer into an ndarray
    count_array = np.array(layer_counts)
    # Determine the row index that contains the largest count for each position and store this in an array.
    nucleotide_indexes = np.argmax(count_array, axis=0)
    # Map the base sequence derived from the row index array to attain the consensus sequence in a list.
    consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]

    if sample:
        adata.var[f'{reference}_consensus_from_{sample}'] = consensus_sequence_list
    else:
        adata.var[f'{reference}_consensus_across_samples'] = consensus_sequence_list

    adata.uns[f'{reference}_consensus_sequence'] = consensus_sequence_list