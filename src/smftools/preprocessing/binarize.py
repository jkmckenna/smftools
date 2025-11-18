import numpy as np

def binarize_adata(adata, source="X", target_layer="binary", threshold=0.8):
    """
    Binarize a dense matrix and preserve NaN.
    source: "X" or layer name
    """
    X = adata.X if source == "X" else adata.layers[source]

    # Copy to avoid modifying original in-place
    X_bin = X.copy()

    # Where not NaN: apply threshold
    mask = ~np.isnan(X_bin)
    X_bin[mask] = (X_bin[mask] > threshold).astype(np.int8)

    adata.layers[target_layer] = X_bin
