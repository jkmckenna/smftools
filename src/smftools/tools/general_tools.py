def create_nan_mask_from_X(adata, new_layer_name="nan_mask"):
    """
    Generates a nan mask where 1 = NaN in adata.X and 0 = valid value.
    """
    import numpy as np
    nan_mask = np.isnan(adata.X).astype(int)
    adata.layers[new_layer_name] = nan_mask
    print(f"✅ Created '{new_layer_name}' layer based on NaNs in adata.X")
    return adata

def combine_layers(adata, input_layers, output_layer, negative_mask=None, values=None):
    """
    Combines layers into a single layer with specific coding:
        - Background stays 0
        - Each input_layer gets a unique value (1, 2, 3...) or provided in `values`
        - Later layers in the list take precedence at overlapping positions
    
    Parameters:
        adata: AnnData object
        input_layers (list of str): Names of layers to combine
        output_layer (str): Output layer name
        negative_mask (str, optional): If provided, positions with mask==0 will stay 0 in the output
        values (list of int, optional): List of integers to assign to each input layer (must match length of input_layers)
    """
    import numpy as np
    base_shape = adata.layers[input_layers[0]].shape
    
    # Validate shapes
    for layer in input_layers:
        if adata.layers[layer].shape != base_shape:
            raise ValueError(f"Layer {layer} shape {adata.layers[layer].shape} does not match {base_shape}")

    combined = np.zeros(base_shape, dtype=int)

    # Default to 1, 2, 3, ... if no values provided
    if values is None:
        values = list(range(1, len(input_layers) + 1))
    
    for layer, val in zip(input_layers, values):
        arr = adata.layers[layer]
        combined[arr > 0] = val  # Later layers will overwrite
    
    if negative_mask:
        mask = adata.layers[negative_mask]
        combined[mask == 0] = 0

    adata.layers[output_layer] = combined
    print(f"✅ Combined {input_layers} into {output_layer} with values {values}")

    return adata