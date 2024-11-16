# subset_adata

def subset_adata(adata, columns, cat_type='obs'):
    """
    Adds subset metadata to an AnnData object based on categorical values in specified .obs or .var columns.
    
    Parameters:
        adata (AnnData): The AnnData object to add subset metadata to.
        columns (list of str): List of .obs or .var column names to subset by. The order matters.
        cat_type (str): obs or var. Default is obs

    Returns:
        None
    """
    import pandas as pd
    import anndata as ad

    subgroup_name = '_'.join(columns)
    if 'obs' in cat_type:
        df = adata.obs[columns]
        adata.obs[subgroup_name] = df.apply(lambda row: '_'.join(row.astype(str)), axis=1)
        adata.obs[subgroup_name] = adata.obs[subgroup_name].astype('category')
    elif 'var' in cat_type:    
        df = adata.var[columns]
        adata.var[subgroup_name] = df.apply(lambda row: '_'.join(row.astype(str)), axis=1)
        adata.var[subgroup_name] = adata.var[subgroup_name].astype('category')

    return None