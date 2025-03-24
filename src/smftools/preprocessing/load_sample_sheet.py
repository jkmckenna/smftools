def load_sample_sheet(adata, sample_sheet_path, mapping_key_column='obs_names', as_category=True):
    """
    Loads a sample sheet CSV and maps metadata into the AnnData object as categorical columns.

    Parameters:
        adata (AnnData): The AnnData object to append sample information to.
        sample_sheet_path (str): Path to the CSV file.
        mapping_key_column (str): Column name in the CSV to map against adata.obs_names or an existing obs column.
        as_category (bool): If True, added columns will be cast as pandas Categorical.

    Returns:
        AnnData: Updated AnnData object.
    """
    import pandas as pd

    print('🔹 Loading sample sheet...')
    df = pd.read_csv(sample_sheet_path)
    df[mapping_key_column] = df[mapping_key_column].astype(str)
    
    # If matching against obs_names directly
    if mapping_key_column == 'obs_names':
        key_series = adata.obs_names.astype(str)
    else:
        key_series = adata.obs[mapping_key_column].astype(str)

    value_columns = [col for col in df.columns if col != mapping_key_column]
    
    print(f'🔹 Appending metadata columns: {value_columns}')
    df = df.set_index(mapping_key_column)

    for col in value_columns:
        mapped = key_series.map(df[col])
        if as_category:
            mapped = mapped.astype('category')
        adata.obs[col] = mapped

    print('✅ Sample sheet metadata successfully added as categories.' if as_category else '✅ Metadata added.')
    return adata
