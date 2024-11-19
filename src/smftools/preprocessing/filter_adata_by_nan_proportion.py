## filter_adata_by_nan_proportion

def filter_adata_by_nan_proportion(adata, threshold, axis='obs'):
    """
    Filters an anndata object on a nan proportion threshold in a given matrix axis.

    Parameters:
        adata (AnnData):
        threshold (float): The max np.nan content to allow in the given axis.
        axis (str): Whether to filter the adata based on obs or var np.nan content
    Returns:
        filtered_adata
    """
    import numpy as np
    import anndata as ad

    if axis == 'obs':
        # Calculate the proportion of NaN values in each read
        nan_proportion = np.isnan(adata.X).mean(axis=1)
        # Filter reads to keep reads with less than a certain NaN proportion
        filtered_indices = np.where(nan_proportion <= threshold)[0]
        filtered_adata = adata[filtered_indices, :].copy()
    elif axis == 'var':
        # Calculate the proportion of NaN values at a given position
        nan_proportion = np.isnan(adata.X).mean(axis=0)
        # Filter positions to keep positions with less than a certain NaN proportion
        filtered_indices = np.where(nan_proportion <= threshold)[0]
        filtered_adata = adata[:, filtered_indices].copy()
    else:
        raise ValueError("Axis must be either 'obs' or 'var'")
    return filtered_adata