# subsample_adata

def subsample_adata(adata, n_subsamples):
    """
    Subsample the adata on an integer number of observations.

    Parameters:
        adata (AnnData): The AnnData object to use as input.
        n_subsamples (int): How many obs to use.

    Returns:
        subsampled_adata (AnnData): The subsampled AnnData object.
    """
    import anndata as ad
    import numpy as np

    # Get the total number of rows (observations)
    n_obs = adata.n_obs

    if n_subsamples < n_obs:
        # Generate random indices
        random_indices = np.random.choice(n_obs, n_subsamples, replace=False)
        # Subset the AnnData object by selecting the random indices
        subsampled_adata = adata[random_indices, :].copy()
        return subsampled_adata
    else:
        print(f"Number of observations {n_obs} is less than the requested number of subsamples {n_subsamples}")
        return adata