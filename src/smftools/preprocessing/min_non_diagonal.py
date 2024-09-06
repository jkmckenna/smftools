## min_non_diagonal

def min_non_diagonal(matrix):
    """
    Takes a matrix and returns the smallest value from each row with the diagonal masked.

    Parameters:
        matrix (ndarray): A 2D ndarray.

    Returns:
        min_values (list): A list of minimum values from each row of the matrix
    """
    import numpy as np

    n = matrix.shape[0]
    min_values = []
    for i in range(n):
        # Mask to exclude the diagonal element
        row_mask = np.ones(n, dtype=bool)
        row_mask[i] = False
        # Extract the row excluding the diagonal element
        row = matrix[i, row_mask]
        # Find the minimum value in the row
        min_values.append(np.min(row))
    return min_values