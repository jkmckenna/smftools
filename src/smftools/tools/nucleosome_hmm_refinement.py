def refine_nucleosome_calls(adata, layer_name, nan_mask_layer, hexamer_size=120, octamer_size=147, max_wiggle=40, device="cpu"):
    import numpy as np

    nucleosome_layer = adata.layers[layer_name]
    nan_mask = adata.layers[nan_mask_layer]  # should be binary mask: 1 = nan region, 0 = valid data

    hexamer_layer = np.zeros_like(nucleosome_layer)
    octamer_layer = np.zeros_like(nucleosome_layer)

    for read_idx, row in enumerate(nucleosome_layer):
        in_patch = False
        start_idx = None

        for pos in range(len(row)):
            if row[pos] == 1 and not in_patch:
                in_patch = True
                start_idx = pos
            if (row[pos] == 0 or pos == len(row) - 1) and in_patch:
                in_patch = False
                end_idx = pos if row[pos] == 0 else pos + 1

                # Expand boundaries into NaNs
                left_expand = 0
                right_expand = 0

                # Left
                for i in range(1, max_wiggle + 1):
                    if start_idx - i >= 0 and nan_mask[read_idx, start_idx - i] == 1:
                        left_expand += 1
                    else:
                        break
                # Right
                for i in range(1, max_wiggle + 1):
                    if end_idx + i < nucleosome_layer.shape[1] and nan_mask[read_idx, end_idx + i] == 1:
                        right_expand += 1
                    else:
                        break

                expanded_start = start_idx - left_expand
                expanded_end = end_idx + right_expand

                available_size = expanded_end - expanded_start
            
                # Octamer placement
                if available_size >= octamer_size:
                    center = (expanded_start + expanded_end) // 2
                    half_oct = octamer_size // 2
                    octamer_layer[read_idx, center - half_oct: center - half_oct + octamer_size] = 1

                # Hexamer placement
                elif available_size >= hexamer_size:
                    center = (expanded_start + expanded_end) // 2
                    half_hex = hexamer_size // 2
                    hexamer_layer[read_idx, center - half_hex: center - half_hex + hexamer_size] = 1

    adata.layers[f"{layer_name}_hexamers"] = hexamer_layer
    adata.layers[f"{layer_name}_octamers"] = octamer_layer

    print(f"✅ Added layers: {layer_name}_hexamers and {layer_name}_octamers")
    return adata

def infer_nucleosomes_in_large_bound(adata, large_bound_layer, combined_nuc_layer, nan_mask_layer, nuc_size=147, linker_size=50, exclusion_buffer=30, device="cpu"):
    import numpy as np

    large_bound = adata.layers[large_bound_layer]
    existing_nucs = adata.layers[combined_nuc_layer]
    nan_mask = adata.layers[nan_mask_layer]

    inferred_layer = np.zeros_like(large_bound)

    for read_idx, row in enumerate(large_bound):
        in_patch = False
        start_idx = None

        for pos in range(len(row)):
            if row[pos] == 1 and not in_patch:
                in_patch = True
                start_idx = pos
            if (row[pos] == 0 or pos == len(row) - 1) and in_patch:
                in_patch = False
                end_idx = pos if row[pos] == 0 else pos + 1

                # Adjust boundaries into flanking NaN regions without getting too close to existing nucleosomes
                left_expand = start_idx
                while left_expand > 0 and nan_mask[read_idx, left_expand - 1] == 1 and np.sum(existing_nucs[read_idx, max(0, left_expand - exclusion_buffer):left_expand]) == 0:
                    left_expand -= 1

                right_expand = end_idx
                while right_expand < row.shape[0] and nan_mask[read_idx, right_expand] == 1 and np.sum(existing_nucs[read_idx, right_expand:min(row.shape[0], right_expand + exclusion_buffer)]) == 0:
                    right_expand += 1

                # Phase nucleosomes with linker spacing only
                region = (left_expand, right_expand)
                pos_cursor = region[0]
                while pos_cursor + nuc_size <= region[1]:
                    if np.all((existing_nucs[read_idx, pos_cursor - exclusion_buffer:pos_cursor + nuc_size + exclusion_buffer] == 0)):
                        inferred_layer[read_idx, pos_cursor:pos_cursor + nuc_size] = 1
                        pos_cursor += nuc_size + linker_size 
                    else:
                        pos_cursor += 1

    adata.layers[f"{large_bound_layer}_phased_nucleosomes"] = inferred_layer
    print(f"✅ Added layer: {large_bound_layer}_phased_nucleosomes")
    return adata