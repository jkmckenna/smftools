## filter_reads_on_length

def filter_reads_on_length(adata, filter_on_coordinates=False, min_read_length=2700, max_read_length=3200):
    """
    Filters the adata object to keep a defined coordinate window, as well as reads that are over a minimum threshold in length.

    Parameters:
        adata (AnnData): An adata object.
        filter_on_coordinates (bool | list): If False, skips filtering. Otherwise, provide a list containing integers representing the lower and upper bound coordinates to filter on. Default is False.
        min_read_length (int): The minimum read length to keep in the filtered dataset. Default is 2700.
        max_read_length (int): The maximum query read length to keep in the filtered dataset. Default is 3200.

    Returns:
        adata
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    if filter_on_coordinates:
        lower_bound, upper_bound = filter_on_coordinates
        # Extract the position information from the adata object as an np array
        var_names_arr = adata.var_names.astype(int).to_numpy()
        # Find the upper bound coordinate that is closest to the specified value
        closest_end_index = np.argmin(np.abs(var_names_arr - upper_bound))
        upper_bound = int(adata.var_names[closest_end_index])
        # Find the lower bound coordinate that is closest to the specified value
        closest_start_index = np.argmin(np.abs(var_names_arr - lower_bound))
        lower_bound = int(adata.var_names[closest_start_index])
        # Get a list of positional indexes that encompass the lower and upper bounds of the dataset
        position_list = list(range(lower_bound, upper_bound + 1))
        position_list = [str(pos) for pos in position_list]
        position_set = set(position_list)
        print(f'Subsetting adata to keep data between coordinates {lower_bound} and {upper_bound}')
        adata = adata[:, adata.var_names.isin(position_set)].copy()

    if min_read_length:
        print(f'Subsetting adata to keep reads longer than {min_read_length}')
        s0 = adata.shape[0]
        adata = adata[adata.obs['read_length'] > min_read_length].copy()
        s1 = adata.shape[0]
        print(f'Removed {s0-s1} reads less than {min_read_length} basepairs in length')

    if max_read_length:
        print(f'Subsetting adata to keep reads shorter than {max_read_length}')
        s0 = adata.shape[0]
        adata = adata[adata.obs['read_length'] < max_read_length].copy()
        s1 = adata.shape[0]
        print(f'Removed {s0-s1} reads greater than {max_read_length} basepairs in length')

    return adata