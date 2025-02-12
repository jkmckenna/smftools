def ohe_batching(base_identities, tmp_dir, record, prefix='', batch_size=100000, progress_bar=None, device='auto'):
    """
    Processes base identities to one-hot encoded matrices and writes to h5ad files in batches.

    Parameters:
        base_identities (dict): Dictionary mapping read names to sequences.
        tmp_dir (str): Directory for storing temporary files.
        record (str): Record name.
        prefix (str): Prefix for file naming.
        batch_size (int): Number of reads per batch.
        progress_bar (tqdm instance, optional): Shared progress bar from parent function.

    Returns:
        list: List of valid H5AD file paths.
    """
    import os
    import anndata as ad
    import numpy as np
    from .one_hot_encode import one_hot_encode

    batch = {}
    count = 0
    batch_number = 0
    file_names = []

    for read_name, seq in base_identities.items():
        if seq is None or not isinstance(seq, (str, list, np.ndarray)):
            continue  # Skip invalid sequence

        try:
            one_hot_matrix = one_hot_encode(seq, device)
        except Exception:
            continue  # Skip on encoding failure

        batch[read_name] = one_hot_matrix # may need to convert to numpy
        count += 1

        # Save batch when reaching batch_size
        if count >= batch_size:
            save_name = os.path.join(tmp_dir, f'tmp_{prefix}_{record}_{batch_number}.h5ad')
            if any(v.size > 0 for v in batch.values()):  # Ensure non-empty data
                tmp_ad = ad.AnnData(X=np.random.rand(1, 1), uns=batch)
                tmp_ad.write_h5ad(save_name)
                file_names.append(save_name)

            batch.clear()
            count = 0
            batch_number += 1

        # Update shared progress bar
        if progress_bar:
            progress_bar.update(1)

    # Save remaining batch
    if batch and any(v.size > 0 for v in batch.values()):
        save_name = os.path.join(tmp_dir, f'tmp_{prefix}_{record}_{batch_number}.h5ad')
        tmp_ad = ad.AnnData(X=np.random.rand(1, 1), uns=batch)
        tmp_ad.write_h5ad(save_name)
        file_names.append(save_name)

    return file_names