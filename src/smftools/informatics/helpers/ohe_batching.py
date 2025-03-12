import os
import anndata as ad
import numpy as np
import concurrent.futures
from .one_hot_encode import one_hot_encode

def encode_sequence(args):
    """Parallel helper function for one-hot encoding."""
    read_name, seq, device = args
    try:
        one_hot_matrix = one_hot_encode(seq, device)
        return read_name, one_hot_matrix
    except Exception:
        return None  # Skip invalid sequences

def encode_and_save_batch(batch_data, tmp_dir, prefix, record, batch_number):
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
        for result in executor.map(encode_sequence, encoding_args):
            if result:
                batch_data.append(result)

                if len(batch_data) >= batch_size:
                    # Step 3: Process and Write Batch Immediately
                    file_name = encode_and_save_batch(batch_data.copy(), tmp_dir, prefix, record, batch_number)
                    if file_name:
                        file_names.append(file_name)

                    batch_data.clear()
                    batch_number += 1

                if progress_bar:
                    progress_bar.update(1)

    # Step 4: Process Remaining Batch
    if batch_data:
        file_name = encode_and_save_batch(batch_data, tmp_dir, prefix, record, batch_number)
        if file_name:
            file_names.append(file_name)

    return file_names