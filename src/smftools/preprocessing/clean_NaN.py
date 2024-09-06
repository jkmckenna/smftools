## clean_NaN
from ..readwrite import adata_to_df

# NaN handling
def clean_NaN(adata, layer=None):
    """
    Append layers to adata that contain NaN cleaning strategies.

    Parameters:
        adata (AnnData): an adata object
        layer (str): string representing the layer to fill NaN values in

    Returns:
        None
    """
    import numpy as np
    import anndata as ad
    import pandas as pd
    # Fill NaN with closest SMF value
    df = adata_to_df(adata, layer=layer)
    df = df.ffill(axis=1).bfill(axis=1)
    adata.layers['fill_nans_closest'] = df.values

    # Replace NaN values with 0, and 0 with minus 1
    old_value, new_value = [0, -1]
    df = adata_to_df(adata, layer=layer)
    df = df.replace(old_value, new_value)
    old_value, new_value = [np.nan, 0]
    df = df.replace(old_value, new_value)
    adata.layers['nan0_0minus1'] = df.values

    # Replace NaN values with 1, and 1 with 2
    old_value, new_value = [1, 2]
    df = adata_to_df(adata, layer=layer)
    df = df.replace(old_value, new_value)
    old_value, new_value = [np.nan, 1]
    df = df.replace(old_value, new_value)
    adata.layers['nan1_12'] = df.values