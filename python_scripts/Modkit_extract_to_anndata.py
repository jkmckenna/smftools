######################################################################################################
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse
import matplotlib.pyplot as plt
import os
import gc
import math
import numpy as np
from Bio import SeqIO
from pympler import asizeof, muppy
from datetime import datetime
######################################################################################################

# Get the current date
current_date = datetime.now()
# Format the date as a string
date_string = current_date.strftime("%Y%m%d")
date_string = date_string[2:]
# #columns = ['read_id','forward_read_position','ref_position','chrom','mod_strand','ref_strand','ref_mod_strand','fw_soft_clipped_start','fw_soft_clipped_end','read_length','call_prob','call_code','base_qual','ref_kmer','query_kmer','canonical_base','modified_primary_base','fail','inferred','within_alignment','flag']

######################################################################################################
def time_string():
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")

def extract_chromosome_name(fasta_file):
    with open(fasta_file, 'r') as file:
        # Parse the FASTA file
        for record in SeqIO.parse(file, 'fasta'):
            # Extract the chromosome name from the record description
            # The chromosome name is often found in the description after the first whitespace
            chromosome_name = record.description.split()[0]
            sequence = str(record.seq)
            return chromosome_name, len(sequence) - 1

def find_GpC(fasta_file, start_coordinate=0):
    """
    A function to find genomic coordinates in a FASTA file of C in the GpC and not GpCpG context.
    Needs the start coordinate specified.
    Returns lists of tuples containing the chromosome number as an integer and the zero indexed genomic coordinate as an integer for the top and bottom strands (in relation to the FASTA)
    """
    top_coordinates = []
    bottom_coordinates = []
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequence = str(record.seq).upper()
            for i in range(1, len(sequence)):
                if sequence[i] == 'C' and sequence[i-1] == 'G' and (i == len(sequence) - 1 or sequence[i+1] != 'G'):
                    top_coordinates.append((record.id, i + start_coordinate))  # 0-indexed coordinate
                if sequence[i] == 'G' and sequence[i+1] == 'C' and sequence[i-1] != 'C':
                    bottom_coordinates.append((record.id, i + start_coordinate))  # 0-indexed coordinate                  
    return top_coordinates, bottom_coordinates

def find_A(fasta_file, start_coordinate=0):
    """
    A function to find genomic coordinates in a FASTA file of A.
    Needs the start coordinate specified.
    Returns lists of tuples containing the chromosome number as an integer and the zero indexed genomic coordinate as an integer for the top and bottom strands (in relation to the FASTA)
    """
    top_coordinates = []
    bottom_coordinates = []
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequence = str(record.seq).upper()
            for i in range(1, len(sequence)):
                if sequence[i] == 'A':
                    top_coordinates.append((record.id, i + start_coordinate))  # 0-indexed coordinate
                if sequence[i] == 'T':
                    bottom_coordinates.append((record.id, i + start_coordinate))  # 0-indexed coordinate                  
    return top_coordinates, bottom_coordinates
######################################################################################################

######################################################################################################
if __name__ == "__main__":
    ### Parse Inputs ###
    parser = argparse.ArgumentParser(description="Reformat single molecule modification data from a set of tsv files to stranded modification dictionaries")
    parser.add_argument("mods", help="list of modifications to plot. Available mods [6mA, 5mC]")
    parser.add_argument("fasta", help="a FASTA file to extract positions of interest from.")
    parser.add_argument("batch_size", help="Number of sample TSV files to process per batch")
    parser.add_argument("mod_threshold", help="A lower probability threshold for a modifcation call to be passed.")
    parser.add_argument("experiment_name", help="An experiment name to add to the final anndata object")
    args = parser.parse_args()
    ####################
  
    ###################################################
    ### Get input tsv file names into a sorted list ###
    # List all files in the directory
    files = os.listdir(os.getcwd())
    # get current working directory
    cwd = os.getcwd()
    # Filter file names that contain the search string in their filename and keep them in a list
    tsvs = [tsv for tsv in files if 'extract.tsv' in tsv]
    # Sort file list by names and print the list of file names
    tsvs.sort()
    print('{0} sample files found: {1}'.format(len(tsvs), tsvs))
    chromosome_name, end_position = extract_chromosome_name(args.fasta)
    print('{0}: chromosome to keep {1} '.format(time_string(), chromosome_name))
    batch_size = int(args.batch_size) # Number of TSVs to maximally process in a batch
    batches = math.ceil(len(tsvs) / batch_size) # Number of batches to process
    print('{0}: Processing input tsvs in {1} batches of {2} tsvs '.format(time_string(), batches, batch_size))
    ###################################################

    ###################################################
    # Note the gpc coordinates extracted are 0-indexed, and the tsv file is also 0-indexed on the start position.
    # get a list of the chromosome and coordinates of cytosines of interest in the top and bottom strands
    gpc_top, gpc_bottom = find_GpC(args.fasta, start_coordinate=0)
    A_top, A_bottom,  = find_A(args.fasta, start_coordinate=0)
    # Extract the coordinate positions into stranded sets
    gpc_top_set = set([position[1] for position in gpc_top])
    gpc_bottom_set = set([position[1] for position in gpc_bottom])
    A_top_set = set([position[1] for position in A_top])
    A_bottom_set = set([position[1] for position in A_bottom])
    ###################################################

    ###################################################
    # Begin iterating over batches
    for batch in range(batches):
        print('{0}: Prcoessing tsvs for batch {1} '.format(time_string(), batch))
        # For the final batch, just take the remaining tsv files
        if batch == batches - 1:
            tsv_batch = tsvs
        # For all other batches, take the next batch of tsvs out of the file queue.    
        else:
            tsv_batch = tsvs[:batch_size]
            tsvs = tsvs[batch_size:]
        print('{0}: tsvs in batch {1} '.format(time_string(), tsv_batch))
        print('{0}: Calculating session memory usage'.format(time_string()))
        print('{0}: Total size of objects in current session > {1}Mb'.format(time_string(), muppy.get_size(muppy.get_objects()) / 1e6))
    ###################################################

        ###################################################
        ### Add the tsvs as dataframes to a dictionary (dict_total) keyed by integer index. Also make modification specific dictionaries and strand specific dictionaries.
        # Initialize dictionaries and place them in a list
        dict_total, dict_a, dict_a_bottom, dict_a_top, dict_c, dict_c_bottom, dict_c_top, dict_combined_bottom, dict_combined_top = {},{},{},{},{},{},{},{},{}
        dict_list = [dict_total, dict_a, dict_a_bottom, dict_a_top, dict_c, dict_c_bottom, dict_c_top, dict_combined_bottom, dict_combined_top]

        # Give names to represent each dictionary in the list
        sample_types = ['total', 'm6A', 'm6A_bottom_strand', 'm6A_top_strand', '5mC', '5mC_bottom_strand', '5mC_top_strand', 'combined_bottom_strand', 'combined_top_strand']

        # Give indices of dictionaries to skip for analysis and final dictionary saving.
        dict_to_skip = [0, 1, 4]
        combined_dicts = [7, 8]

        # Initialize a max_position variable to hold the maximum coordinate value in the experiment
        max_position = 0

        # Load the dict_total dictionary with all of the tsv files as dataframes.
        for i, tsv in enumerate(tsv_batch):
            # load the tsv as a dataframe into the dictionary
            print('{0}: Loading sample tsv {1} into dictionary'.format(time_string(), tsv))
            dict_total[i] = pd.read_csv(tsv, sep='\t', header=0)

            # Only keep the reads aligned to the chromosome of interest
            print('{0}: Filtering sample dataframe to keep chromosome of interest'.format(time_string()))
            dict_total[i] = dict_total[i][dict_total[i]['chrom'] == chromosome_name]

            # Only keep the read positions that fall within the region of interest
            print('{0}: Filtering sample dataframe to keep positions falling within region of interest'.format(time_string()))
            dict_total[i] = dict_total[i][(end_position >= dict_total[i]['ref_position']) & (dict_total[i]['ref_position']>= 0)]

            # Update max position in the dataset
            if dict_total[i]['ref_position'].max() > max_position:
                max_position = dict_total[i]['ref_position'].max()
        print('Max positional coordinate in experiment: ' + str(max_position))

        # Iterate over dict_total of all the tsv files and extract the modification specific and strand specific dataframes into dictionaries
        for i in dict_total.keys():
            if '6mA' in args.mods:
                # get a dictionary of dataframes that only contain methylated adenine positions
                dict_a[i] = dict_total[i][dict_total[i]['modified_primary_base'] == 'A']
                print('{}: Successfully created a methyl-adenine dictionary for '.format(time_string()) + str(i))
                # Stratify the adenine dictionary into two strand specific dictionaries.
                dict_a_bottom[i] = dict_a[i][dict_a[i]['ref_strand'] == '-']
                dict_a_bottom[i] = dict_a_bottom[i][dict_a_bottom[i]['ref_position'].isin(A_bottom_set)] 
                print('{}: Successfully created a minus strand methyl-adenine dictionary for '.format(time_string()) + str(i))
                dict_a_top[i] = dict_a[i][dict_a[i]['ref_strand'] == '+']
                dict_a_top[i] = dict_a_top[i][dict_a_top[i]['ref_position'].isin(A_top_set)] 
                print('{}: Successfully created a plus strand methyl-adenine dictionary for '.format(time_string()) + str(i))

            if '5mC' in args.mods:
                # get a dictionary of dataframes that only contain methylated cytosine positions
                dict_c[i] = dict_total[i][dict_total[i]['modified_primary_base'] == 'C']
                print('{}: Successfully created a methyl-cytosine dictionary for '.format(time_string()) + str(i))
                # Stratify the cytosine dictionary into two strand specific dictionaries.
                dict_c_bottom[i] = dict_c[i][dict_c[i]['ref_strand'] == '-']
                print('{}: Successfully created a minus strand methyl-cytosine dictionary for '.format(time_string()) + str(i))
                dict_c_top[i] = dict_c[i][dict_c[i]['ref_strand'] == '+']
                print('{}: Successfully created a plus strand methyl-cytosine dictionary for '.format(time_string()) + str(i))
                # In the strand specific dictionaries, only keep positions that are informative for GpC SMF
                dict_c_bottom[i] = dict_c_bottom[i][dict_c_bottom[i]['ref_position'].isin(gpc_bottom_set)]   
                dict_c_top[i] = dict_c_top[i][dict_c_top[i]['ref_position'].isin(gpc_top_set)] 

            # Initialize the sample keys for the combined dictionaries
            print('{}: Successfully created a minus strand combined methylation dictionary for '.format(time_string()) + str(i))
            dict_combined_bottom[i] = []
            print('{}: Successfully created a plus strand combined methylation dictionary for '.format(time_string()) + str(i))
            dict_combined_top[i] = []

        # Iterate over the stranded modification dictionaries and replace the dataframes with a dictionary of read names pointing to a list of values from the dataframe
        for i, dict in enumerate(dict_list):
            # Only iterate over stranded dictionaries
            if i not in dict_to_skip:
                print('{0}: Binarizing methylation states for {1} dictionary'.format(time_string(), sample_types[i]))
                # For each sample in a stranded dictionary
                for sample in dict.keys():
                    # Load the combined bottom strand dictionary after all the individual dictionaries have been made for the sample
                    if i == 7:
                        print('{0}: Binarizing {1} dictionary for sample {2}'.format(time_string(), sample_types[i], sample))
                        # Load the minus strand dictionaries for each sample into temporary variables
                        temp_a_dict = dict_list[2][sample].copy()
                        temp_c_dict = dict_list[5][sample].copy()
                        dict[sample] = {}
                        # Iterate over the reads present in the merge of both dictionaries
                        for read in set(temp_a_dict) | set(temp_c_dict):
                            # Add the arrays element-wise if the read is present in both dictionaries
                            if read in temp_a_dict and read in temp_c_dict:
                                dict[sample][read] = np.nansum([temp_a_dict[read], temp_c_dict[read]], axis=0)
                            # If the read is present in only one dictionary, copy its value
                            elif read in temp_a_dict:
                                dict[sample][read] = temp_a_dict[read]
                            else:
                                dict[sample][read] = temp_c_dict[read]
                   # Load the combined top strand dictionary after all the individual dictionaries have been made for the sample
                    elif i == 8:
                        print('{0}: Binarizing {1} dictionary for sample {2}'.format(time_string(), sample_types[i], sample))
                      # Load the plus strand dictionaries for each sample into temporary variables
                        temp_a_dict = dict_list[3][sample].copy()
                        temp_c_dict = dict_list[6][sample].copy()
                        dict[sample] = {}
                        # Iterate over the reads present in the merge of both dictionaries
                        for read in set(temp_a_dict) | set(temp_c_dict):
                            # Add the arrays element-wise if the read is present in both dictionaries
                            if read in temp_a_dict and read in temp_c_dict:
                                dict[sample][read] = np.nansum([temp_a_dict[read], temp_c_dict[read]], axis=0)
                            # If the read is present in only one dictionary, copy its value
                            elif read in temp_a_dict:
                                dict[sample][read] = temp_a_dict[read]
                            else:
                                dict[sample][read] = temp_c_dict[read]
                    # For all other dictionaries
                    else:
                        print('{0}: Binarizing {1} dictionary for sample {2}'.format(time_string(), sample_types[i], sample))
                        # extract the dataframe from the dictionary into a temporary variable
                        temp_df = dict[sample]
                        # reassign the dictionary pointer to a nested dictionary.
                        dict[sample] = {}
                        # # Iterate through rows in the temp DataFrame
                        for index, row in temp_df.iterrows():
                            read = row['read_id'] # read name
                            position = row['ref_position']  # positional coordinate
                            probability = row['call_prob'] # Get the probability of the given call
                            prob_thresh = float(args.mod_threshold) # set a lower threshold for the call probability
                            methylated = np.nan # initialize the methylated_variable
                            # if the call_code is modified and was called above the probability threshold, change methylated value to 1
                            if (row['call_code'] in ['a', 'h', 'm']) and probability > prob_thresh: 
                                methylated = 1
                            # If the call code is canonical and was called above the probability threshold, change the methylated value to 0
                            elif probability > prob_thresh:
                                methylated = 0
                            # If the current read is not in the dictionary yet, initalize the dictionary with a nan filled numpy array of proper size.
                            if read not in dict[sample]:
                                dict[sample][read] = np.full(max_position, np.nan) 
                            # add the positional methylation state to the numpy array
                            dict[sample][read][position-1] = methylated

        # Save the sample files in the batch as gzipped hdf5 files
        print('{0}: Converting batch {1} dictionaries to anndata objects'.format(time_string(), batch))
        for i, dict in enumerate(dict_list):
            if i not in dict_to_skip:
                # Initialize an hdf5 file for the current modified strand
                adata = None
                print('{0}: Converting {1} dictionary to an anndata object'.format(time_string(), sample_types[i]))
                for sample in dict.keys():
                    print('{0}: Converting {1} dictionary for sample {2} to an anndata object'.format(time_string(), sample_types[i], sample))
                    sample = int(sample)
                    final_sample_index = sample + (batch * batch_size)
                    print('{0}: Final sample index for sample: {1}'.format(time_string(), final_sample_index))
                    print('{0}: Converting {1} dictionary for sample {2} to a dataframe'.format(time_string(), sample_types[i], final_sample_index))
                    temp_df = pd.DataFrame.from_dict(dict[sample], orient='index')
                    X = temp_df.values
                    print('{0}: Loading {1} dataframe for sample {2} into a temp anndata object'.format(time_string(), sample_types[i], final_sample_index))
                    temp_adata = sc.AnnData(X, dtype=X.dtype)
                    print('{0}: Adding read names and position ids to {1} anndata for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
                    temp_adata.obs_names = temp_df.index
                    temp_adata.obs_names = temp_adata.obs_names.astype(str)
                    temp_adata.var_names = temp_df.columns
                    temp_adata.var_names = temp_adata.var_names.astype(str)
                    print('{0}: Adding final sample id to {1} anndata for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
                    temp_adata.obs['Sample'] = [str(final_sample_index)] * len(temp_adata)
                    temp_adata.obs['Dataset'] = [sample_types[i]] * len(temp_adata)
                    # If final adata object already has a sample loaded, concatenate the current sample into the existing adata object 
                    if adata:
                        print('{0}: Concatenating {1} anndata object for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
                        adata = ad.concat([adata, temp_adata], join='outer', index_unique=None)
                    else:
                        print('{0}: Initializing {1} anndata object for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
                        adata = temp_adata
                print('{0}: Writing {1} anndata out as a gzipped hdf5 file'.format(time_string(), sample_types[i]))
                adata.write_h5ad('{0}_{1}_{2}_SMF_binarized_sample_hdf5.h5ad.gz'.format(date_string, batch, sample_types[i]), compression='gzip')

        # Delete the batch dictionaries from memory
        del dict_list
        gc.collect()
    
    # Iterate over all of the batched hdf5 files and concatenate them.
    files = os.listdir(os.getcwd())
    # Name the final output file
    final_hdf = '{0}_{1}_final_concatenated_SMF_binarized_sample_hdf5.h5ad.gz'.format(date_string, args.experiment_name)
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    final_adata = None
    for hdf in hdfs:
        print('{0}: Reading in {1} hdf5 file'.format(time_string(), hdf))
        temp_adata = sc.read_h5ad(hdf)
        if final_adata:
            print('{0}: Concatenating final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
        else:
            print('{0}: Initializing final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = temp_adata
    print('{0}: Writing final concatenated hdf5 file'.format(time_string()))
    final_adata.write_h5ad(final_hdf, compression='gzip')

    # Delete the individual h5ad files and only keep the final concatenated file
    files = os.listdir(os.getcwd())
    hdfs_to_delete = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
    # Iterate over the files and delete them
    for hdf in hdfs_to_delete:
        try:
            os.remove(hdf)
            print(f"Deleted file: {hdf}")
        except OSError as e:
            print(f"Error deleting file {hdf}: {e}")

    ##################################################
######################################################################################################
