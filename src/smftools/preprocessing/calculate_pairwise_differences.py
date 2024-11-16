# calculate_pairwise_differences

def calculate_pairwise_differences(arrays):
    """
    Calculate the pairwise differences for a list of h-stacked ndarrays. Ignore N-positions

    Parameters:
        arrays (str): A list of ndarrays.

    Returns:
        distance_matrix (ndarray): a 2D array containing the pairwise differences between all arrays.
    """
    import numpy as np
    from tqdm import tqdm

    num_arrays = len(arrays)

    n_rows = 5
    reshaped_arrays = [array.reshape(n_rows, -1) for array in arrays]
    N_masks = [array[-1].astype(bool) for array in reshaped_arrays]
    reshaped_arrays_minus_N = [array[:-1].flatten() for array in reshaped_arrays]

    # Precompute the repeated N masks to avoid repeated computations
    repeated_N_masks = [np.tile(N_mask, (n_rows - 1)) for N_mask in N_masks]

    # Initialize the distance matrix
    distance_matrix = np.zeros((num_arrays, num_arrays), dtype=np.float32)

    # Calculate pairwise distances with progress bar
    for i in tqdm(range(num_arrays), desc="Calculating Pairwise Differences"):
        array_i = reshaped_arrays_minus_N[i]
        N_mask_i = repeated_N_masks[i]

        for j in range(i + 1, num_arrays):
            array_j = reshaped_arrays_minus_N[j]
            N_mask_j = repeated_N_masks[j]

            # Combined mask to ignore N positions
            combined_mask = N_mask_i | N_mask_j

            # Calculate the hamming distance directly with boolean operations
            differences = (array_i != array_j) & ~combined_mask
            distance = np.sum(differences) / np.sum(~combined_mask)
            
            # Store the symmetric distances
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
