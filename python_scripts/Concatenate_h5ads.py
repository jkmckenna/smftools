######################################################################################################
import pandas as pd
import anndata as ad
import argparse
import os
import gc
import math
import numpy as np
from datetime import datetime
######################################################################################################

# Get the current date
current_date = datetime.now()
# Format the date as a string
date_string = current_date.strftime("%Y%m%d")
date_string = date_string[2:]
output_file = '{0}_final_concatenated_SMF_binarized_sample_hdf5.h5ad.gz'.format(date_string)
######################################################################################################
def time_string():
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")
######################################################################################################

######################################################################################################
if __name__ == "__main__":
    ### Parse Inputs ###
    parser = argparse.ArgumentParser(description="Concatenate h5ad files into a single file and delete the individual files")
    parser.add_argument("file_suffix", help="file type to find")
    parser.add_argument("delete_inputs", help="Indicate True of False if you want to delete the input h5ad files")
    args = parser.parse_args()
    ####################
  
    ###################################################
    ### Get input tsv file names into a sorted list ###
    # List all files in the directory
    files = os.listdir(os.getcwd())
    # get current working directory
    cwd = os.getcwd()
    suffix = args.file_suffix
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if suffix in hdf]
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    ###################################################

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
    if args.delete_inputs == 'True':
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


