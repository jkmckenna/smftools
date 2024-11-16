# subset_adata

def subset_adata(adata, columns, cat_type='obs'):
    """
    Subsets an AnnData object based on categorical values in specified .obs or .var columns.
    
    Parameters:
        adata (AnnData): The AnnData object to subset.
        columns (list of str): List of .obs or .var column names to subset by. The order matters.
        cat_type (str): obs or var. Default is obs

    Returns:
        dict: A dictionary where keys are tuples of category values and values are corresponding AnnData subsets.
    """

    def subset_recursive(adata_subset, columns, cat_type, key_prefix=()):
        # Returns when the bottom of the stack is reached
        if not columns:
            # If there's only one column, return the key as a single value, not a tuple
            if len(key_prefix) == 1:
                return {key_prefix[0]: adata_subset}
            return {key_prefix: adata_subset}
        
        current_column = columns[0]
        subsets = {}

        if 'obs' in cat_type:
            categories = adata_subset.obs[current_column].cat.categories
            for cat in categories:
                subset = adata_subset[adata_subset.obs[current_column] == cat].copy()
                new_key = key_prefix + (cat,)
                subsets.update(subset_recursive(subset, columns[1:], cat_type, new_key))

        elif 'var' in cat_type:
            categories = adata_subset.var[current_column].cat.categories
            for cat in categories:
                subset = adata_subset[:, adata_subset.var[current_column] == cat].copy()
                new_key = key_prefix + (cat,)
                subsets.update(subset_recursive(subset, columns[1:], cat_type, new_key))            
        
        return subsets
    
    # Start the recursive subset process
    subsets_dict = subset_recursive(adata, columns, cat_type)
    
    return subsets_dict