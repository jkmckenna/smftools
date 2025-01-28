# one_hot_decode

# String encodings
def one_hot_decode(ohe_array):
    """
    Takes a flattened one hot encoded array and returns the sequence string from that array.
    Parameters:
        ohe_array (np.array): A one hot encoded array

    Returns:
        sequence (str): Sequence string of the one hot encoded array
    """
    import numpy as np
    # Define the mapping of one-hot encoded indices to DNA bases
    mapping = ['A', 'C', 'G', 'T', 'N']
    
    # Reshape the flattened array into a 2D matrix with 5 columns (one for each base)
    one_hot_matrix = ohe_array.reshape(-1, 5)
    
    # Get the index of the maximum value (which will be 1) in each row
    decoded_indices = np.argmax(one_hot_matrix, axis=1)
    
    # Map the indices back to the corresponding bases
    sequence_list = [mapping[i] for i in decoded_indices]
    sequence = ''.join(sequence_list)
    
    return sequence