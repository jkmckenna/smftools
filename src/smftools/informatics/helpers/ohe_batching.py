# ohe_batching

def ohe_batching(base_identities, tmp_dir, record, batch_size=10000):
    """
    Processes base identities to one-hot encoded matrices and writes to a pickle file in batches.

    Parameters:
        base_identities (dict): A dictionary of read names and sequences.
        tmp_dir (str): Path to directory where the files will be saved.
        record (str): Name of the record.
        batch_size (int): Number of reads to process in each batch.

    Returns:
        ohe_file (list): list of output file names
    """
    import os
    import numpy as np
    from tqdm import tqdm
    from .one_hot_encode import one_hot_encode

    batch = {}
    count = 0
    batch_number = 0
    total_reads = len(base_identities)
    file_names = []
    
    for read_name, seq in tqdm(base_identities.items(), desc="Encoding and writing one hot encoded reads", total=total_reads):
        one_hot_matrix = one_hot_encode(seq)
        batch[read_name] = one_hot_matrix
        count += 1
        # If the batch size is reached, write out the batch and reset
        if count >= batch_size:
            save_name = os.path.join(tmp_dir, f'tmp_{record}_{batch_number}.npz')
            np.savez(save_name, **batch)
            file_names.append(save_name)
            batch.clear()
            count = 0
            batch_number += 1

    # Write out any remaining reads in the final batch
    if batch:
        save_name = os.path.join(tmp_dir, f'tmp_{record}_{batch_number}.npz')
        np.savez(save_name, **batch)
        file_names.append(save_name)

    return file_names