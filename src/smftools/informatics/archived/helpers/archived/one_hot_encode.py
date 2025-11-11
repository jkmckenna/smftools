# one_hot_encode

def one_hot_encode(sequence, device='auto'):
    """
    One-hot encodes a DNA sequence.

    Parameters:
        sequence (str or list): DNA sequence (e.g., "ACGTN" or ['A', 'C', 'G', 'T', 'N']).

    Returns:
        ndarray: Flattened one-hot encoded representation of the input sequence.
    """
    import numpy as np

    mapping = np.array(['A', 'C', 'G', 'T', 'N'])

    # Ensure input is a list of characters
    if not isinstance(sequence, list):
        sequence = list(sequence)  # Convert string to list of characters

    # Handle empty sequences
    if len(sequence) == 0:
        print("Warning: Empty sequence encountered in one_hot_encode()")
        return np.zeros(len(mapping))  # Return empty encoding instead of failing

    # Convert sequence to NumPy array
    seq_array = np.array(sequence, dtype='<U1')

    # Replace invalid bases with 'N'
    seq_array = np.where(np.isin(seq_array, mapping), seq_array, 'N')

    # Create one-hot encoding matrix
    one_hot_matrix = (seq_array[:, None] == mapping).astype(int)

    # Flatten and return
    return one_hot_matrix.flatten()

    # import torch
    # bases = torch.tensor([ord('A'), ord('C'), ord('G'), ord('T'), ord('N')], dtype=torch.int8, device=device)

    # # Convert input to tensor of character ASCII codes
    # seq_tensor = torch.tensor([ord(c) for c in sequence], dtype=torch.int8, device=device)

    # # Handle empty sequence
    # if seq_tensor.numel() == 0:
    #     print("Warning: Empty sequence encountered in one_hot_encode_torch()")
    #     return torch.zeros(len(bases), device=device)

    # # Replace invalid bases with 'N'
    # is_valid = (seq_tensor[:, None] == bases)  # Compare each base with mapping
    # seq_tensor = torch.where(is_valid.any(dim=1), seq_tensor, ord('N'))

    # # Create one-hot encoding matrix
    # one_hot_matrix = (seq_tensor[:, None] == bases).int()

    # # Flatten and return
    # return one_hot_matrix.flatten()
