def binarize_converted_base_identities(base_identities, strand, modification_type):
    """
    Efficiently binarizes conversion SMF data within a sequence string using NumPy arrays.

    Parameters:
        base_identities (dict): A dictionary returned by extract_base_identities. Keyed by read name. Points to a list of base identities.
        strand (str): A string indicating which strand was converted in the experiment (options are 'top' and 'bottom').
        modification_type (str): A string indicating the modification type of interest (options are '5mC' and '6mA').

    Returns:
        dict: A dictionary where 1 represents a methylated site, 0 represents an unmethylated site, and NaN represents a site without methylation info.
    """
    import numpy as np

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
