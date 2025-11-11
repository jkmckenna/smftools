import numpy as np
import anndata as ad

import os
import concurrent.futures

def one_hot_encode(sequence, device='auto'):
    """
    One-hot encodes a DNA sequence.

    Parameters:
        sequence (str or list): DNA sequence (e.g., "ACGTN" or ['A', 'C', 'G', 'T', 'N']).

    Returns:
        ndarray: Flattened one-hot encoded representation of the input sequence.
    """
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

def one_hot_decode(ohe_array):
    """
    Takes a flattened one hot encoded array and returns the sequence string from that array.
    Parameters:
        ohe_array (np.array): A one hot encoded array

    Returns:
        sequence (str): Sequence string of the one hot encoded array
    """
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

def ohe_layers_decode(adata, obs_names):
    """
    Takes an anndata object and a list of observation names. Returns a list of sequence strings for the reads of interest.
    Parameters:
        adata (AnnData): An anndata object.
        obs_names (list): A list of observation name strings to retrieve sequences for.

    Returns:
        sequences (list of str): List of strings of the one hot encoded array
    """
    # Define the mapping of one-hot encoded indices to DNA bases
    mapping = ['A', 'C', 'G', 'T', 'N']

    ohe_layers = [f"{base}_binary_encoding" for base in mapping]
    sequences = []

    for obs_name in obs_names:
        obs_subset = adata[obs_name]
        ohe_list = []
        for layer in ohe_layers:
            ohe_list += list(obs_subset.layers[layer])
        ohe_array = np.array(ohe_list)
        sequence = one_hot_decode(ohe_array)
        sequences.append(sequence)
        
    return sequences

def _encode_sequence(args):
    """Parallel helper function for one-hot encoding."""
    read_name, seq, device = args
    try:
        one_hot_matrix = one_hot_encode(seq, device)
        return read_name, one_hot_matrix
    except Exception:
        return None  # Skip invalid sequences

def _encode_and_save_batch(batch_data, tmp_dir, prefix, record, batch_number):
    """Encodes a batch and writes to disk immediately."""
    batch = {read_name: matrix for read_name, matrix in batch_data if matrix is not None}

    if batch:
        save_name = os.path.join(tmp_dir, f'tmp_{prefix}_{record}_{batch_number}.h5ad')
        tmp_ad = ad.AnnData(X=np.zeros((1, 1)), uns=batch)  # Placeholder X
        tmp_ad.write_h5ad(save_name)
        return save_name
    return None

def ohe_batching(base_identities, tmp_dir, record, prefix='', batch_size=100000, progress_bar=None, device='auto', threads=None):
    """
    Efficient version of ohe_batching: one-hot encodes sequences in parallel and writes batches immediately.

    Parameters:
        base_identities (dict): Dictionary mapping read names to sequences.
        tmp_dir (str): Directory for storing temporary files.
        record (str): Record name.
        prefix (str): Prefix for file naming.
        batch_size (int): Number of reads per batch.
        progress_bar (tqdm instance, optional): Shared progress bar.
        device (str): Device for encoding.
        threads (int, optional): Number of parallel workers.

    Returns:
        list: List of valid H5AD file paths.
    """
    threads = threads or os.cpu_count()  # Default to max available CPU cores
    batch_data = []
    batch_number = 0
    file_names = []

    # Step 1: Prepare Data for Parallel Encoding
    encoding_args = [(read_name, seq, device) for read_name, seq in base_identities.items() if seq is not None]

    # Step 2: Parallel One-Hot Encoding using threads (to avoid nested processes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for result in executor.map(_encode_sequence, encoding_args):
            if result:
                batch_data.append(result)

                if len(batch_data) >= batch_size:
                    # Step 3: Process and Write Batch Immediately
                    file_name = _encode_and_save_batch(batch_data.copy(), tmp_dir, prefix, record, batch_number)
                    if file_name:
                        file_names.append(file_name)

                    batch_data.clear()
                    batch_number += 1

                if progress_bar:
                    progress_bar.update(1)

    # Step 4: Process Remaining Batch
    if batch_data:
        file_name = _encode_and_save_batch(batch_data, tmp_dir, prefix, record, batch_number)
        if file_name:
            file_names.append(file_name)

    return file_names