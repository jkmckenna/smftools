# ohe_batching

def ohe_batching(base_identities, tmp_dir, record, prefix='', batch_size=100000):
    """
    Processes base identities to one-hot encoded matrices and writes to a h5ad file in batches.

    Parameters:
        base_identities (dict): A dictionary of read names and sequences.
        tmp_dir (str): Path to directory where the files will be saved.
        record (str): Name of the record.
        prefix (str): Prefix to add to the output file name
        batch_size (int): Number of reads to process in each batch.

    Returns:
        ohe_file (list): list of output file names
    """
    import os
    import anndata as ad
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
            save_name = os.path.join(tmp_dir, f'tmp_{prefix}_{record}_{batch_number}.h5ad.gz')
            X = np.random.rand(1, 1)
            tmp_ad = ad.AnnData(X=X, uns=batch) 
            tmp_ad.write_h5ad(save_name, compression='gzip')
            file_names.append(save_name)
            batch.clear()
            count = 0
            batch_number += 1

    # Write out any remaining reads in the final batch
    if batch:
        save_name = os.path.join(tmp_dir, f'tmp_{prefix}_{record}_{batch_number}.h5ad.gz')
        X = np.random.rand(1, 1)
        tmp_ad = ad.AnnData(X=X, uns=batch) 
        tmp_ad.write_h5ad(save_name, compression='gzip')
        file_names.append(save_name)

    return file_names