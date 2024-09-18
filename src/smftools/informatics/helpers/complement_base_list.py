# complement_base_list

def complement_base_list(sequence):
    """
    Takes a list of DNA base identities and returns their complement.

    Parameters:
        sequence (list): A list of DNA bases (e.g., ['A', 'C', 'G', 'T']).

    Returns:
        complement (list): A list of complementary DNA bases.
    """
    complement_mapping = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
        'N': 'N'  # Handling ambiguous bases like 'N'
    }

    return [complement_mapping[base] for base in sequence]