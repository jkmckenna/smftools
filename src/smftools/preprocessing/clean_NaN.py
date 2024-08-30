## clean_NaN
import numpy as np
import anndata as ad
import pandas as pd

# NaN handling
def clean_NaN(adata, layer=None):
    """
    Input: An adata object and the layer to fill Nan values of
    Output: Append layers to adata that contain NaN cleaning strategies
    """
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