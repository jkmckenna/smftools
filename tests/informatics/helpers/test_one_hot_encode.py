# one_hot_encode
from .. import readwrite

# String encodings
def one_hot_encode(sequence):
    """
    Input: A sequence string of a read.
    Output: One hot encoding of the sequence string.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot_matrix = np.zeros((len(sequence), 5), dtype=int)
    for i, nucleotide in enumerate(sequence):
        one_hot_matrix[i, mapping[nucleotide]] = 1
    return one_hot_matrix