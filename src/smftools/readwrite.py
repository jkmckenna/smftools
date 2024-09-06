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
    Input: An adata object with a specified layer.
    Output: A dataframe for the specific layer.
    """
    import pandas as pd
    import anndata as ad

    # Extract the data matrix from the given layer
    if layer:
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X
    # Extract observation (read) annotations
    obs_df = adata.obs
    # Extract variable (position) annotations
    var_df = adata.var
    # Convert data matrix and annotations to pandas DataFrames
    df = pd.DataFrame(data_matrix, index=obs_df.index, columns=var_df.index)
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
######################################################################################################