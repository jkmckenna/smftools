# remove_duplicates
import anndata as ad

def remove_duplicates(adata):
    """
    Input: adata object with marked duplicates
    Output: Remove duplicates from the adata object
    """
    initial_size = adata.shape[0]
    adata = adata[adata.obs['Unique_in_final_read_set'] == True].copy()
    final_size = adata.shape[0]
    print(f'Removed {initial_size-final_size} reads from the dataset')