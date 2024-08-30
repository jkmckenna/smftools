## binary_layers_to_ohe
import numpy as np
import anndata as ad
import pandas as pd

## Conversion SMF Specific 
def binary_layers_to_ohe(adata, layers, stack='hstack'):
    """
    Input: An adata object and a list of layers containing a binary encoding.
    Output: A dictionary keyed by obs_name that points to a stacked (hstack or vstack) one-hot encoding of the binary layers
    """
    # Extract the layers
    layers = [adata.layers[layer_name] for layer_name in layers]
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