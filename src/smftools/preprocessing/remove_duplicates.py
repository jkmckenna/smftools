# remove_duplicates

def remove_duplicates(adata):
    """
    Remove duplicates from the adata object

    Parameters:
        adata (Anndata): An adata object.

    Returns:
        None 
    """
    import anndata as ad

    initial_size = adata.shape[0]
    adata = adata[adata.obs['Unique_in_final_read_set'] == True].copy()
    final_size = adata.shape[0]
    print(f'Removed {initial_size-final_size} reads from the dataset')