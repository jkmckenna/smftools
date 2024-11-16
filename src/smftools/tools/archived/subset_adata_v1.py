# subset_adata

def subset_adata(adata, obs_columns):
    """
    Subsets an AnnData object based on categorical values in specified `.obs` columns.
    
    Parameters:
        adata (AnnData): The AnnData object to subset.
        obs_columns (list of str): List of `.obs` column names to subset by. The order matters.

    Returns:
        dict: A dictionary where keys are tuples of category values and values are corresponding AnnData subsets.
    """

    def subset_recursive(adata_subset, columns):
        if not columns:
            return {(): adata_subset}
        
        current_column = columns[0]
        categories = adata_subset.obs[current_column].cat.categories
        
        subsets = {}
        for cat in categories:
            subset = adata_subset[adata_subset.obs[current_column] == cat]
            subsets.update(subset_recursive(subset, columns[1:]))
        
        return subsets
    
    # Start the recursive subset process
    subsets_dict = subset_recursive(adata, obs_columns)
    
    return subsets_dict