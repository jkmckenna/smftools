# one_hot_encode

# String encodings
def one_hot_encode(sequence):
    """
    One hot encodes a sequence list.
    Parameters:
        sequence (list): A list of DNA base sequences.

    Returns:
        flattened (ndarray): A numpy ndarray holding a flattened one hot encoding of the input sequence string.
    """
    import numpy as np

    seq_array = np.array(sequence, dtype='<U1')  # String dtype
    mapping = np.array(['A', 'C', 'G', 'T', 'N'])
    seq_array[~np.isin(seq_array, mapping)] = 'N'
    one_hot_matrix = (seq_array[:, None] == mapping).astype(int)
    flattened = one_hot_matrix.flatten()

    return flattened