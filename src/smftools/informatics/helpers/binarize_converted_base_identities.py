## binarize_converted_base_identities
# Conversion SMF specific
def binarize_converted_base_identities(base_identities, strand, modification_type):
    """
    Binarizes conversion SMF data within a sequence string

    Parameters:
        base_identities (dict): A dictionary returned by extract_base_identity_at_coordinates.
        strand (str): A string indicating which strand was converted in the experiment (options are 'top' and 'bottom').
        modification_type (str): A string indicating the modification type of interest (options are '5mC' and '6mA').
    
    Returns:
        binarized_base_identities (dict): A binarized dictionary, where 1 represents a methylated site. 0 represents an unmethylated site. NaN represents a site that does not carry methylation information.
    """
    import numpy as np
    binarized_base_identities = {}
    # Iterate over base identity keys to binarize the base identities
    for key in base_identities.keys():
        if strand == 'top':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'C' else 0 if x == 'T' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'A' else 0 if x == 'G' else np.nan for x in base_identities[key]]
        elif strand == 'bottom':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'G' else 0 if x == 'A' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'T' else 0 if x == 'C' else np.nan for x in base_identities[key]]
        else:
            pass
    return binarized_base_identities