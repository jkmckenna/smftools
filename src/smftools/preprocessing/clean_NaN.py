def clean_NaN(adata, layer=None):
    """
    Append layers to adata that contain NaN cleaning strategies.

    Parameters:
        adata (AnnData): an anndata object
        layer (str, optional): Name of the layer to fill NaN values in. If None, uses adata.X.

    Modifies:
        - Adds new layers to `adata.layers` with different NaN-filling strategies.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    from ..readwrite import adata_to_df 

    # Ensure the specified layer exists
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Convert to DataFrame
    df = adata_to_df(adata, layer=layer)

    # Fill NaN with closest SMF value (forward then backward fill)
    print('Making layer: fill_nans_closest')
    adata.layers['fill_nans_closest'] = df.ffill(axis=1).bfill(axis=1).values

    # Replace NaN with 0, and 0 with -1
    print('Making layer: nan0_0minus1')
    df_nan0_0minus1 = df.replace(0, -1).fillna(0)
    adata.layers['nan0_0minus1'] = df_nan0_0minus1.values

    # Replace NaN with 1, and 1 with 2
    print('Making layer: nan1_12')
    df_nan1_12 = df.replace(1, 2).fillna(1)
    adata.layers['nan1_12'] = df_nan1_12.values

    # Replace NaN with -1
    print('Making layer: nan_minus_1')
    df_nan_minus_1 = df.fillna(-1)
    adata.layers['nan_minus_1'] = df_nan_minus_1.values

    # Replace NaN with -1
    print('Making layer: nan_half')
    df_nan_half = df.fillna(0.5)
    adata.layers['nan_half'] = df_nan_half.values
