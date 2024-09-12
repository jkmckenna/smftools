# remove_duplicates

def remove_duplicates(adata):
    """
    Remove duplicates from the adata object

    Parameters:
        adata (Anndata): An adata object.

    Returns:
        filtered_adata (AnnData): An AnnData object of the filtered reads 
        duplicates (AnnData): An AnnData object of the duplicate reads 
    """
    import anndata as ad

    initial_size = adata.shape[0]
    filtered_adata = adata[adata.obs['Unique_in_final_read_set'] == True].copy()
    final_size = filtered_adata.shape[0]
    print(f'Removed {initial_size-final_size} reads from the dataset')
    duplicates = adata[adata.obs['Unique_in_final_read_set'] == False].copy()
    return filtered_adata, duplicates