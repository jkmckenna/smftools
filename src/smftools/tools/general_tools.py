def create_nan_mask_from_X(adata, new_layer_name="nan_mask"):
    """
    Generates a nan mask where 1 = NaN in adata.X and 0 = valid value.
    """
    import numpy as np
    nan_mask = np.isnan(adata.X).astype(int)
    adata.layers[new_layer_name] = nan_mask
    print(f"✅ Created '{new_layer_name}' layer based on NaNs in adata.X")
    return adata

def create_nan_or_non_gpc_mask(adata, obs_column, new_layer_name="nan_or_non_gpc_mask"):
    import numpy as np

    nan_mask = np.isnan(adata.X).astype(int)
    combined_mask = np.zeros_like(nan_mask)

    for idx, row in enumerate(adata.obs.itertuples()):
        ref = getattr(row, obs_column)
        gpc_mask = adata.var[f"{ref}_GpC_site"].astype(int).values
        combined_mask[idx, :] = 1 - gpc_mask  # non-GpC is 1

    mask = np.maximum(nan_mask, combined_mask)
    adata.layers[new_layer_name] = mask

    print(f"✅ Created '{new_layer_name}' layer based on NaNs in adata.X and non-GpC regions using {obs_column}")
    return adata

def combine_layers(adata, input_layers, output_layer, negative_mask=None, values=None, binary_mode=False):
    """
    Combines layers into a single layer with specific coding:
        - Background stays 0
        - If binary_mode=True: any overlap = 1
        - If binary_mode=False:
            - Defaults to [1, 2, 3, ...] if values=None
            - Later layers take precedence in overlaps
    
    Parameters:
        adata: AnnData object
        input_layers: list of str
        output_layer: str, name of the output layer
        negative_mask: str (optional), binary mask to enforce 0s
        values: list of ints (optional), values to assign to each input layer
        binary_mode: bool, if True, creates a simple 0/1 mask regardless of values
    
    Returns:
        Updated AnnData with new layer.
    """
    import numpy as np
    combined = np.zeros_like(adata.layers[input_layers[0]])

    if binary_mode:
        for layer in input_layers:
            combined = np.logical_or(combined, adata.layers[layer] > 0)
        combined = combined.astype(int)
    else:
        if values is None:
            values = list(range(1, len(input_layers) + 1))
        for i, layer in enumerate(input_layers):
            arr = adata.layers[layer]
            combined[arr > 0] = values[i]

    if negative_mask:
        mask = adata.layers[negative_mask]
        combined[mask == 0] = 0

    adata.layers[output_layer] = combined
    print(f"✅ Combined layers into {output_layer} {'(binary)' if binary_mode else f'with values {values}'}")

    return adata
