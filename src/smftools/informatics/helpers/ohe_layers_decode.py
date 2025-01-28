# ohe_layers_decode

def ohe_layers_decode(adata, obs_names):
    """
    Takes an anndata object and a list of observation names. Returns a list of sequence strings for the reads of interest.
    Parameters:
        adata (AnnData): An anndata object.
        obs_names (list): A list of observation name strings to retrieve sequences for.

    Returns:
        sequences (list of str): List of strings of the one hot encoded array
    """
    import anndata as ad
    import numpy as np
    from .ohe_decode import ohe_decode

    # Define the mapping of one-hot encoded indices to DNA bases
    mapping = ['A', 'C', 'G', 'T', 'N']

    ohe_layers = [f"{base}_binary_encoding" for base in mapping]
    sequences = []

    for obs_name in obs_names:
        obs_subset = adata[obs_name]
        ohe_list = []
        for layer in ohe_layers:
            ohe_list += list(obs_subset.layers[layer])
        ohe_array = np.array(ohe_list)
        sequence = ohe_decode(ohe_array)
        sequences.append(sequence)
        
    return sequences