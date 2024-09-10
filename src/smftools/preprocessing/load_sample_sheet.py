# load_sample_sheet

def load_sample_sheet(adata, sample_sheet_path, mapping_key_column):
    """
    Loads a sample sheet csv and uses one of the columns to map sample information into the AnnData object.

    Parameters:
        adata (AnnData): The Anndata object to append sample information to.
        sample_sheet_path (str):
        mapping_key_column (str):

    Returns:
        None
    """
    import pandas as pd
    import anndata as ad
    df = pd.read_csv(sample_sheet_path)
    key_column = mapping_key_column
    df[key_column] = df[key_column].astype(str)
    value_columns = [column for column in df.columns if column != key_column]
    mapping_dict = df.set_index(key_column)[value_columns].to_dict(orient='index')
    for column in value_columns:
        column_map = {key: value[column] for key, value in mapping_dict.items()}
        adata.obs[column] = adata.obs[key_column].map(column_map)