# calculate_consensus

def calculate_consensus(adata, reference_column, sample_column):
    """
    Takes an input AnnData object, the reference to subset on, and the sample name to subset on to calculate the consensus sequence of the read set.

    Parameters:
        adata (AnnData):
        reference_column (str):
        sample_column (str):

    Returns:
        None
    
    """
    # May need to remove the bottom for conversion SMF
    record_subset = adata[adata.obs[reference_column] == record].copy()
    layer_map, layer_counts = {}, []
    for i, layer in enumerate(record_subset.layers):
        layer_map[i] = layer.split('_')[0]
        layer_counts.append(np.sum(record_subset.layers[layer], axis=0))
    count_array = np.array(layer_counts)
    nucleotide_indexes = np.argmax(count_array, axis=0)
    consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
    final_adata.var[f'{record}_consensus_across_samples'] = consensus_sequence_list