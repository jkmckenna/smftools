## readwrite ##

######################################################################################################
## Datetime functionality
def date_string():
    """
    Each time this is called, it returns the current date string
    """
    from datetime import datetime
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    date_string = date_string[2:]
    return date_string

def time_string():
    """
    Each time this is called, it returns the current time string
    """
    from datetime import datetime
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")
######################################################################################################

######################################################################################################
## Numpy, Pandas, Anndata functionality

def adata_to_df(adata, layer=None):
    """
    Convert an AnnData object into a Pandas DataFrame.

    Parameters:
        adata (AnnData): The input AnnData object.
        layer (str, optional): The layer to extract. If None, uses adata.X.

    Returns:
        pd.DataFrame: A DataFrame where rows are observations and columns are positions.
    """
    import pandas as pd
    import anndata as ad
    import numpy as np

    # Validate that the requested layer exists
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Extract the data matrix
    data_matrix = adata.layers.get(layer, adata.X)

    # Ensure matrix is dense (handle sparse formats)
    if hasattr(data_matrix, "toarray"):  
        data_matrix = data_matrix.toarray()

    # Ensure obs and var have unique indices
    if adata.obs.index.duplicated().any():
        raise ValueError("Duplicate values found in `adata.obs.index`. Ensure unique observation indices.")
    
    if adata.var.index.duplicated().any():
        raise ValueError("Duplicate values found in `adata.var.index`. Ensure unique variable indices.")

    # Convert to DataFrame
    df = pd.DataFrame(data_matrix, index=adata.obs.index, columns=adata.var.index)

    return df


def save_matrix(matrix, save_name):
    """
    Input: A numpy matrix and a save_name
    Output: A txt file representation of the data matrix
    """
    import numpy as np
    np.savetxt(f'{save_name}.txt', matrix)

def concatenate_h5ads(output_file, file_suffix='h5ad.gz', delete_inputs=True):
    """
    Concatenate all h5ad files in a directory and delete them after the final adata is written out.
    Input: an output file path relative to the directory in which the function is called
    """
    import os
    import anndata as ad
    # Runtime warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='anndata')
    warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')
    
    # List all files in the directory
    files = os.listdir(os.getcwd())
    # get current working directory
    cwd = os.getcwd()
    suffix = file_suffix
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if suffix in hdf]
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    # Iterate over all of the hdf5 files and concatenate them.
    final_adata = None
    for hdf in hdfs:
        print('{0}: Reading in {1} hdf5 file'.format(time_string(), hdf))
        temp_adata = ad.read_h5ad(hdf)
        if final_adata:
            print('{0}: Concatenating final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
        else:
            print('{0}: Initializing final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = temp_adata
    print('{0}: Writing final concatenated hdf5 file'.format(time_string()))
    final_adata.write_h5ad(output_file, compression='gzip')

    # Delete the individual h5ad files and only keep the final concatenated file
    if delete_inputs:
        files = os.listdir(os.getcwd())
        hdfs = [hdf for hdf in files if suffix in hdf]
        if output_file in hdfs:
            hdfs.remove(output_file)
            # Iterate over the files and delete them
            for hdf in hdfs:
                try:
                    os.remove(hdf)
                    print(f"Deleted file: {hdf}")
                except OSError as e:
                    print(f"Error deleting file {hdf}: {e}")
    else:
        print('Keeping input files')

def safe_write_h5ad(adata, path, compression="gzip", backup=False, backup_dir="./"):
    """
    Saves an AnnData object safely by omitting problematic columns from .obs and .var.

    Parameters:
        adata (AnnData): The AnnData object to save.
        path (str): Output .h5ad file path.
        compression (str): Compression method for h5ad file.
        backup (bool): If True, saves problematic columns to CSV files.
        backup_dir (str): Directory to store backups if backup=True.
    """
    import anndata as ad
    import pandas as pd
    import os

    os.makedirs(backup_dir, exist_ok=True)

    def filter_df(df, df_name):
        bad_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                if not df[col].apply(lambda x: isinstance(x, (str, type(None)))).all():
                    bad_cols.append(col)
        if bad_cols:
            print(f"⚠️ Skipping columns from {df_name}: {bad_cols}")
            if backup:
                df[bad_cols].to_csv(os.path.join(backup_dir, f"{df_name}_skipped_columns.csv"))
                print(f"📝 Backed up skipped columns to {backup_dir}/{df_name}_skipped_columns.csv")
        return df.drop(columns=bad_cols)

    # Clean obs and var
    obs_clean = filter_df(adata.obs, "obs")
    var_clean = filter_df(adata.var, "var")

    # Save clean version
    adata_copy = ad.AnnData(
        X=adata.X,
        obs=obs_clean,
        var=var_clean,
        layers=adata.layers,
        uns=adata.uns,
        obsm=adata.obsm,
        varm=adata.varm
    )
    adata_copy.write_h5ad(path, compression=compression)
    print(f"✅ Saved safely to {path}")

def merge_barcoded_anndatas(adata_single, adata_double):
    import numpy as np
    import anndata as ad

    # Step 1: Identify overlap
    overlap = np.intersect1d(adata_single.obs_names, adata_double.obs_names)

    # Step 2: Filter out overlaps from adata_single
    adata_single_filtered = adata_single[~adata_single.obs_names.isin(overlap)].copy()

    # Step 3: Add source tag
    adata_single_filtered.obs['source'] = 'single_barcode'
    adata_double.obs['source'] = 'double_barcode'

    # Step 4: Concatenate all components
    adata_merged = ad.concat([
        adata_single_filtered,
        adata_double
    ], join='outer', merge='same')  # merge='same' preserves matching layers, obsm, etc.

    # Step 5: Merge `.uns`
    adata_merged.uns = {**adata_single.uns, **adata_double.uns}

    return adata_merged
    
######################################################################################################