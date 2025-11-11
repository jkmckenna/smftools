def binarize_converted_base_identities(base_identities, strand, modification_type, bam, device='cpu', deaminase_footprinting=False, mismatch_trend_per_read={}, on_missing="nan"):
    """
    Efficiently binarizes conversion SMF data within a sequence string using NumPy arrays.

    Parameters:
        base_identities (dict): A dictionary returned by extract_base_identities. Keyed by read name. Points to a list of base identities.
        strand (str): A string indicating which strand was converted in the experiment (options are 'top' and 'bottom').
        modification_type (str): A string indicating the modification type of interest (options are '5mC' and '6mA').
        bam (str): The bam file path
        deaminase_footprinting (bool): Whether direct deaminase footprinting chemistry was used.
        mismatch_trend_per_read (dict): For deaminase footprinting, indicates the type of conversion relative to the top strand reference for each read. (C->T or G->A if bottom strand was converted)
        on_missing (str): Error handling if a read is missing
        
    Returns:
        dict: A dictionary where 1 represents a methylated site, 0 represents an unmethylated site, and NaN represents a site without methylation info.
        If deaminase_footprinting, 1 represents deaminated sites, while 0 represents non-deaminated sites.
    """
    import numpy as np

    if mismatch_trend_per_read is None:
        mismatch_trend_per_read = {}

    # Fast path
    if modification_type == "unconverted" and not deaminase_footprinting:
        return {k: np.full(len(v), np.nan, dtype=np.float32) for k, v in base_identities.items()}

    out = {}

    if deaminase_footprinting:
        valid_trends = {"C->T", "G->A"}

        for read_id, bases in base_identities.items():
            trend_raw = mismatch_trend_per_read.get(read_id, None)
            if trend_raw is None:
                if on_missing == "error":
                    raise KeyError(f"Missing mismatch trend for read '{read_id}'")
                out[read_id] = np.full(len(bases), np.nan, dtype=np.float32)
                continue

            trend = trend_raw.replace(" ", "").upper()
            if trend not in valid_trends:
                if on_missing == "error":
                    raise KeyError(
                        f"Invalid mismatch trend '{trend_raw}' for read '{read_id}'. "
                        f"Expected one of {sorted(valid_trends)}"
                    )
                out[read_id] = np.full(len(bases), np.nan, dtype=np.float32)
                continue

            arr = np.asarray(bases, dtype="<U1")
            res = np.full(arr.shape, np.nan, dtype=np.float32)

            if trend == "C->T":
                # C (unconverted) -> 0, T (converted) -> 1
                res[arr == "C"] = 0.0
                res[arr == "T"] = 1.0
            else:  # "G->A"
                res[arr == "G"] = 0.0
                res[arr == "A"] = 1.0

            out[read_id] = res

        return out

    # Non-deaminase mapping (bisulfite-style for 5mC; 6mA mapping is protocol dependent)
    bin_maps = {
        ("top", "5mC"):    {"C": 1.0, "T": 0.0},
        ("bottom", "5mC"): {"G": 1.0, "A": 0.0},
        ("top", "6mA"):    {"A": 1.0, "G": 0.0},
        ("bottom", "6mA"): {"T": 1.0, "C": 0.0},
    }
    key = (strand, modification_type)
    if key not in bin_maps:
        raise ValueError(f"Invalid combination of strand='{strand}' and modification_type='{modification_type}'")

    base_map = bin_maps[key]

    for read_id, bases in base_identities.items():
        arr = np.asarray(bases, dtype="<U1")
        res = np.full(arr.shape, np.nan, dtype=np.float32)
        # mask-assign; unknown characters (N, -, etc.) remain NaN
        for b, v in base_map.items():
            res[arr == b] = v
        out[read_id] = res

    return out

    # if mismatch_trend_per_read is None:
    #     mismatch_trend_per_read = {}

    # # If the modification type is 'unconverted', return NaN for all positions if the deaminase_footprinting strategy is not being used.
    # if modification_type == "unconverted" and not deaminase_footprinting:
    #     #print(f"Skipping binarization for unconverted {strand} reads on bam: {bam}.")
    #     return {key: np.full(len(bases), np.nan) for key, bases in base_identities.items()}

    # # Define mappings for binarization based on strand and modification type
    # if deaminase_footprinting:
    #     binarization_maps = {
    #         ('C->T'): {'C': 0, 'T': 1},
    #         ('G->A'): {'G': 0, 'A': 1},
    #     }

    #     binarized_base_identities = {}
    #     for key, bases in base_identities.items():
    #         arr = np.array(bases, dtype='<U1')
    #         # Fetch the appropriate mapping
    #         conversion_type = mismatch_trend_per_read[key]
    #         base_map = binarization_maps.get(conversion_type, None)
    #         binarized = np.vectorize(lambda x: base_map.get(x, np.nan))(arr)  # Apply mapping with fallback to NaN
    #         binarized_base_identities[key] = binarized

    #     return binarized_base_identities
    
    # else:
    #     binarization_maps = {
    #         ('top', '5mC'): {'C': 1, 'T': 0},
    #         ('top', '6mA'): {'A': 1, 'G': 0},
    #         ('bottom', '5mC'): {'G': 1, 'A': 0},
    #         ('bottom', '6mA'): {'T': 1, 'C': 0}
    #     }

    #     if (strand, modification_type) not in binarization_maps:
    #         raise ValueError(f"Invalid combination of strand='{strand}' and modification_type='{modification_type}'")

    #     # Fetch the appropriate mapping
    #     base_map = binarization_maps[(strand, modification_type)]

    #     binarized_base_identities = {}
    #     for key, bases in base_identities.items():
    #         arr = np.array(bases, dtype='<U1')
    #         binarized = np.vectorize(lambda x: base_map.get(x, np.nan))(arr)  # Apply mapping with fallback to NaN
    #         binarized_base_identities[key] = binarized

    #     return binarized_base_identities
    # import torch

    # # If the modification type is 'unconverted', return NaN for all positions
    # if modification_type == "unconverted":
    #     print(f"Skipping binarization for unconverted {strand} reads on bam: {bam}.")
    #     return {key: torch.full((len(bases),), float('nan'), device=device) for key, bases in base_identities.items()}

    # # Define mappings for binarization based on strand and modification type
    # binarization_maps = {
    #     ('top', '5mC'): {'C': 1, 'T': 0},
    #     ('top', '6mA'): {'A': 1, 'G': 0},
    #     ('bottom', '5mC'): {'G': 1, 'A': 0},
    #     ('bottom', '6mA'): {'T': 1, 'C': 0}
    # }

    # if (strand, modification_type) not in binarization_maps:
    #     raise ValueError(f"Invalid combination of strand='{strand}' and modification_type='{modification_type}'")

    # # Fetch the appropriate mapping
    # base_map = binarization_maps[(strand, modification_type)]
    
    # # Convert mapping to tensor
    # base_keys = list(base_map.keys())
    # base_values = torch.tensor(list(base_map.values()), dtype=torch.float32, device=device)

    # # Create a lookup dictionary (ASCII-based for fast mapping)
    # lookup_table = torch.full((256,), float('nan'), dtype=torch.float32, device=device)
    # for k, v in zip(base_keys, base_values):
    #     lookup_table[ord(k)] = v

    # # Process reads
    # binarized_base_identities = {}
    # for key, bases in base_identities.items():
    #     bases_tensor = torch.tensor([ord(c) for c in bases], dtype=torch.uint8, device=device)  # Convert chars to ASCII
    #     binarized = lookup_table[bases_tensor]  # Efficient lookup
    #     binarized_base_identities[key] = binarized

    # return binarized_base_identities
