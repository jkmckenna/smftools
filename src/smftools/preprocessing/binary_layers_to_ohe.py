## binary_layers_to_ohe

## Conversion SMF Specific 
def binary_layers_to_ohe(adata, binary_layers, stack='hstack'):
    """
    Parameters:
        adata (AnnData): Anndata object.
        binary_layers (list): a list of strings. Each string represents a layer in the adata object. The layer should encode a binary matrix. 
        stack (str): Dimension to stack the one-hot-encoding. Options include 'hstack' and 'vstack'. Default is 'hstack', since this is more efficient.
    
    Returns:
        ohe_dict (dict): A dictionary keyed by obs_name that points to a stacked (hstack or vstack) one-hot encoding of the binary layers
    Input: An adata object and a list of layers containing a binary encoding.
    """
    import numpy as np
    import anndata as ad

    # Ensure that the N layer is last!
    # Grab all binary layers that are not encoding N
    ACGT_binary_layers = [layer for layer in binary_layers if 'binary' in layer and layer != 'N_binary_encoding']
    # If there is a binary layer encoding N, hold it in N_binary_layer
    N_binary_layer = [layer for layer in binary_layers if layer == 'N_binary_encoding']
    # Add the N_binary_encoding layer to the end of the list of binary layers
    all_binary_layers = ACGT_binary_layers + N_binary_layer
    print(f'Found {all_binary_layers} layers in adata')

    # Extract the layers
    layers = [adata.layers[layer_name] for layer_name in all_binary_layers]
    n_reads = layers[0].shape[0]
    ohe_dict = {}
    for i in range(n_reads):
        read_ohe = []
        for layer in layers:
            read_ohe.append(layer[i])
        read_name = adata.obs_names[i]
        if stack == 'hstack':
            ohe_dict[read_name] = np.hstack(read_ohe)
        elif stack == 'vstack':
            ohe_dict[read_name] = np.vstack(read_ohe)
    return ohe_dict