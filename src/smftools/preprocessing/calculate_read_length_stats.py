## calculate_read_length_stats

# Read length QC
def calculate_read_length_stats(adata):
    """
    Append first valid position in a read and last valid position in the read. From this determine and append the read length. 

    Parameters:
        adata (AnnData): An adata object
    
    Returns:
        upper_bound (int): last valid position in the dataset
        lower_bound (int): first valid position in the dataset
    """
    import numpy as np
    import anndata as ad
    import pandas as pd
    ## Add basic observation-level (read-level) metadata to the object: first valid position in a read and last valid position in the read. From this determine the read length. Save two new variable which hold the first and last valid positions in the entire dataset

    # Add some basic observation-level (read-level) metadata to the anndata object
    read_first_valid_position = np.array([int(adata.var_names[i]) for i in np.argmax(~np.isnan(adata.X), axis=1)])
    read_last_valid_position = np.array([int(adata.var_names[i]) for i in (adata.X.shape[1] - 1 - np.argmax(~np.isnan(adata.X[:, ::-1]), axis=1))])
    read_length = read_last_valid_position - read_first_valid_position + np.ones(len(read_first_valid_position))

    adata.obs['first_valid_position'] = pd.Series(read_first_valid_position, index=adata.obs.index, dtype=int)
    adata.obs['last_valid_position'] = pd.Series(read_last_valid_position, index=adata.obs.index, dtype=int)
    adata.obs['read_length'] = pd.Series(read_length, index=adata.obs.index, dtype=int)

    # Define variables to hold the first and last valid position in the dataset
    upper_bound = int(np.nanmax(adata.obs['last_valid_position']))
    lower_bound = int(np.nanmin(adata.obs['first_valid_position']))
    return upper_bound, lower_bound