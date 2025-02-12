def binarize_converted_base_identities(base_identities, strand, modification_type, bam, device='cpu'):
    """
    Efficiently binarizes conversion SMF data within a sequence string using NumPy arrays.

    Parameters:
        base_identities (dict): A dictionary returned by extract_base_identities. Keyed by read name. Points to a list of base identities.
        strand (str): A string indicating which strand was converted in the experiment (options are 'top' and 'bottom').
        modification_type (str): A string indicating the modification type of interest (options are '5mC' and '6mA').
        bam (str): The bam file path

    Returns:
        dict: A dictionary where 1 represents a methylated site, 0 represents an unmethylated site, and NaN represents a site without methylation info.
    """
    import numpy as np

    # If the modification type is 'unconverted', return NaN for all positions
    if modification_type == "unconverted":
        #print(f"Skipping binarization for unconverted {strand} reads on bam: {bam}.")
        return {key: np.full(len(bases), np.nan) for key, bases in base_identities.items()}

    # Define mappings for binarization based on strand and modification type
    binarization_maps = {
        ('top', '5mC'): {'C': 1, 'T': 0},
        ('top', '6mA'): {'A': 1, 'G': 0},
        ('bottom', '5mC'): {'G': 1, 'A': 0},
        ('bottom', '6mA'): {'T': 1, 'C': 0}
    }

    if (strand, modification_type) not in binarization_maps:
        raise ValueError(f"Invalid combination of strand='{strand}' and modification_type='{modification_type}'")

    # Fetch the appropriate mapping
    base_map = binarization_maps[(strand, modification_type)]

    binarized_base_identities = {}
    for key, bases in base_identities.items():
        arr = np.array(bases, dtype='<U1')
        binarized = np.vectorize(lambda x: base_map.get(x, np.nan))(arr)  # Apply mapping with fallback to NaN
        binarized_base_identities[key] = binarized

    return binarized_base_identities
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
