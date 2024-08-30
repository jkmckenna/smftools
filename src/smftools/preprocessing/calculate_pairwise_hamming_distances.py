## calculate_pairwise_hamming_distances
import numpy as np
import tqdm
from scipy.spatial.distance import hamming

## Conversion SMF Specific 
def calculate_pairwise_hamming_distances(arrays):
    """
    Calculate the pairwise Hamming distances for a list of ndarrays.
    Input: A list of ndarrays
    Output: a 2D array containing the pairwise Hamming distances.
    """
    num_arrays = len(arrays)
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((num_arrays, num_arrays))
    # Calculate pairwise distances with progress bar
    for i in tqdm(range(num_arrays), desc="Calculating Hamming Distances"):
        for j in range(i + 1, num_arrays):
            distance = hamming(arrays[i], arrays[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix