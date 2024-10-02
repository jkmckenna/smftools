## modkit_extract_to_adata

def modkit_extract_to_adata(fasta, bam_dir, mapping_threshold, experiment_name, mods, batch_size, mod_tsv_dir, delete_batch_hdfs=False):
    """
    Takes modkit extract outputs and organizes it into an adata object

    Parameters:
        fasta (str): File path to the reference genome to align to.
        bam_dir (str): File path to the directory containing the aligned_sorted split modified BAM files
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        mods (list): A list of strings of the modification types to use in the analysis.
        batch_size (int): An integer number of TSV files to analyze in memory at once while loading the final adata object.
        mod_tsv_dir (str): String representing the path to the mod TSV directory
        delete_batch_hdfs (bool): Whether to delete the batch hdfs after writing out the final concatenated hdf. Default is False

    Returns:
        None
    """
    ###################################################
    # Package imports
    from .. import readwrite
    from .get_native_references import get_native_references
    from .count_aligned_reads import count_aligned_reads
    from .extract_base_identities import extract_base_identities
    from .one_hot_encode import one_hot_encode
    from .ohe_batching import ohe_batching
    import pandas as pd
    import anndata as ad
    import os
    import gc
    import math
    import numpy as np
    from Bio.Seq import Seq
    from tqdm import tqdm
    import h5py
    from .make_dirs import make_dirs
    ###################################################

    ################## Get input tsv and bam file names into a sorted list ################
    # List all files in the directory
    tsv_files = os.listdir(mod_tsv_dir)
    bam_files = os.listdir(bam_dir)
    # get current working directory
    parent_dir = os.path.dirname(mod_tsv_dir)
    # Make output dirs
    h5_dir = os.path.join(parent_dir, 'h5ads')
    tmp_dir = os.path.join(parent_dir, 'tmp')
    make_dirs([h5_dir, tmp_dir])
    # Filter file names that contain the search string in their filename and keep them in a list
    tsvs = [tsv for tsv in tsv_files if 'extract.tsv' in tsv]
    bams = [bam for bam in bam_files if '.bam' in bam and '.bai' not in bam]
    # Sort file list by names and print the list of file names
    tsvs.sort()
    tsv_path_list = [os.path.join(mod_tsv_dir, tsv) for tsv in tsvs]
    bams.sort()
    bam_path_list = [os.path.join(bam_dir, bam) for bam in bams]
    print(f'{len(tsvs)} sample tsv files found: {tsvs}')
    print(f'{len(bams)} sample bams found: {bams}')
    ##########################################################################################

    ######### Get Record names that have over a passed threshold of mapped reads #############
    # get all records that are above a certain mapping threshold in at least one sample bam
    records_to_analyze = []
    for bami, bam in enumerate(bam_path_list):
        aligned_reads_count, unaligned_reads_count, record_counts_dict = count_aligned_reads(bam)
        percent_aligned = aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)
        print(f'{percent_aligned} percent of reads in {bams[bami]} aligned successfully')
        # Iterate over references and decide which to use in the analysis based on the mapping_threshold
        for record in record_counts_dict:
            print('{0} reads mapped to reference record {1}. This is {2} percent of all mapped reads in {3}'.format(record_counts_dict[record][0], record, record_counts_dict[record][1]*100, bams[bami]))
            if record_counts_dict[record][1] >= mapping_threshold:
                records_to_analyze.append(record)
    records_to_analyze = set(records_to_analyze)
    print(f'Records to analyze: {records_to_analyze}')
    ##########################################################################################

    ########### Determine the maximum record length to analyze in the dataset ################
    # Get all references within the FASTA and indicate the length and identity of the record sequence
    max_reference_length = 0
    reference_dict = get_native_references(fasta) # returns a dict keyed by record name. Points to a tuple of (reference length, reference sequence)
    # Get the max record length in the dataset.
    for record in records_to_analyze:
        if reference_dict[record][0] > max_reference_length:
            max_reference_length = reference_dict[record][0]
    print(f'{readwrite.time_string()}: Max reference length in dataset: {max_reference_length}')
    batches = math.ceil(len(tsvs) / batch_size) # Number of batches to process
    print('{0}: Processing input tsvs in {1} batches of {2} tsvs '.format(readwrite.time_string(), batches, batch_size))
    ##########################################################################################

    ##########################################################################################
    # One hot encode read sequences and write them out into the tmp_dir as h5ad files. 
    # Save the file paths in the bam_record_ohe_files dict.
    bam_record_ohe_files = {}
    bam_record_save = os.path.join(tmp_dir, 'tmp_file_dict.h5ad.gz')
    fwd_mapped_reads = set()
    rev_mapped_reads = set()
    # If this step has already been performed, read in the tmp_dile_dict
    if os.path.exists(bam_record_save):
        bam_record_ohe_files = ad.read_h5ad(bam_record_save).uns
        print('Found existing OHE reads, using these')
    else:
        # Iterate over split bams
        for bami, bam in enumerate(bam_path_list):
            # Iterate over references to process
            for record in records_to_analyze:
                current_reference_length = reference_dict[record][0]
                positions = range(current_reference_length)
                # Extract the base identities of reads aligned to the record
                fwd_base_identities, rev_base_identities = extract_base_identities(bam, record, positions, max_reference_length)
                # Store read names of fwd and rev mapped reads
                fwd_mapped_reads.update(fwd_base_identities.keys())
                rev_mapped_reads.update(rev_base_identities.keys())
                # One hot encode the sequence string of the reads
                fwd_ohe_files = ohe_batching(fwd_base_identities, tmp_dir, record, f"{bami}_fwd",batch_size=100000)
                rev_ohe_files = ohe_batching(rev_base_identities, tmp_dir, record, f"{bami}_rev",batch_size=100000)
                bam_record_ohe_files[f'{bami}_{record}'] = fwd_ohe_files + rev_ohe_files
                del fwd_base_identities, rev_base_identities
        # Save out the ohe file paths
        X = np.random.rand(1, 1)
        tmp_ad = ad.AnnData(X=X, uns=bam_record_ohe_files) 
        tmp_ad.write_h5ad(bam_record_save, compression='gzip')
    ##########################################################################################

    ##########################################################################################
    # Iterate over records to analyze and return a dictionary keyed by the reference name that points to a tuple containing the top strand sequence and the complement
    record_seq_dict = {}
    for record in records_to_analyze:
        current_reference_length = reference_dict[record][0]
        delta_max_length = max_reference_length - current_reference_length
        sequence = reference_dict[record][1] + 'N'*delta_max_length
        complement = str(Seq(reference_dict[record][1]).complement()).upper() + 'N'*delta_max_length
        record_seq_dict[record] = (sequence, complement)
    ##########################################################################################

    ###################################################
    existing_h5s =  os.listdir(h5_dir)
    existing_h5s = [h5 for h5 in existing_h5s if '.h5ad.gz' in h5]
    final_hdf = f'{experiment_name}_final_experiment_hdf5.h5ad.gz'
    final_hdf_already_exists = final_hdf in existing_h5s

    if final_hdf_already_exists:
        print(f'{final_hdf} has already been made. Skipping processing.')
    else:
        # Begin iterating over batches
        for batch in range(batches):
            print('{0}: Processing tsvs for batch {1} '.format(readwrite.time_string(), batch))
            # For the final batch, just take the remaining tsv and bam files
            if batch == batches - 1:
                tsv_batch = tsv_path_list
                bam_batch = bam_path_list
            # For all other batches, take the next batch of tsvs and bams out of the file queue.    
            else:
                tsv_batch = tsv_path_list[:batch_size]
                bam_batch = bam_path_list[:batch_size]
                tsv_path_list = tsv_path_list[batch_size:]
                bam_path_list = bam_path_list[batch_size:]
            print('{0}: tsvs in batch {1} '.format(readwrite.time_string(), tsv_batch))

            batch_already_processed = sum([1 for h5 in existing_h5s if f'_{batch}_' in h5])
        ###################################################
            if batch_already_processed:
                print(f'Batch {batch} has already been processed into h5ads. Skipping batch and using existing files')
            else:
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
                A_stranded_dicts = [2, 3]
                C_stranded_dicts = [5, 6]
                dict_to_skip = dict_to_skip + combined_dicts + A_stranded_dicts + C_stranded_dicts
                dict_to_skip = set(dict_to_skip)

                # Load the dict_total dictionary with all of the tsv files as dataframes.
                for sample_index, tsv in tqdm(enumerate(tsv_batch), desc=f'Loading TSVs into dataframes and filtering on chromosome/position for batch {batch}', total=batch_size):
                    #print('{0}: Loading sample tsv {1} into dataframe'.format(readwrite.time_string(), tsv))
                    temp_df = pd.read_csv(tsv, sep='\t', header=0)
                    for record in records_to_analyze:
                        if record not in dict_total.keys():
                            dict_total[record] = {}
                        # Only keep the reads aligned to the chromosomes of interest
                        #print('{0}: Filtering sample dataframe to keep chromosome of interest'.format(readwrite.time_string()))
                        dict_total[record][sample_index] = temp_df[temp_df['chrom'] == record]
                        # Only keep the read positions that fall within the region of interest
                        #print('{0}: Filtering sample dataframe to keep positions falling within region of interest'.format(readwrite.time_string()))
                        current_reference_length = reference_dict[record][0]
                        dict_total[record][sample_index] = dict_total[record][sample_index][(current_reference_length > dict_total[record][sample_index]['ref_position']) & (dict_total[record][sample_index]['ref_position']>= 0)]

                # Iterate over dict_total of all the tsv files and extract the modification specific and strand specific dataframes into dictionaries
                for record in dict_total.keys():
                    for sample_index in dict_total[record].keys():
                        if '6mA' in mods:
                            # Remove Adenine stranded dicts from the dicts to skip set
                            dict_to_skip.difference_update(set(A_stranded_dicts))

                            if record not in dict_a.keys() and record not in dict_a_bottom.keys() and record not in dict_a_top.keys():
                                dict_a[record], dict_a_bottom[record], dict_a_top[record] = {}, {}, {}

                            # get a dictionary of dataframes that only contain methylated adenine positions
                            dict_a[record][sample_index] = dict_total[record][sample_index][dict_total[record][sample_index]['modified_primary_base'] == 'A']
                            print('{}: Successfully loaded a methyl-adenine dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            # Stratify the adenine dictionary into two strand specific dictionaries.
                            dict_a_bottom[record][sample_index] = dict_a[record][sample_index][dict_a[record][sample_index]['ref_strand'] == '-']
                            print('{}: Successfully loaded a minus strand methyl-adenine dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            dict_a_top[record][sample_index] = dict_a[record][sample_index][dict_a[record][sample_index]['ref_strand'] == '+']
                            print('{}: Successfully loaded a plus strand methyl-adenine dictionary for '.format(readwrite.time_string()) + str(sample_index))

                            # Reassign pointer for dict_a to None and delete the original value that it pointed to in order to decrease memory usage.
                            dict_a[record][sample_index] = None
                            gc.collect()

                        if '5mC' in mods:
                            # Remove Cytosine stranded dicts from the dicts to skip set
                            dict_to_skip.difference_update(set(C_stranded_dicts))

                            if record not in dict_c.keys() and record not in dict_c_bottom.keys() and record not in dict_c_top.keys():
                                dict_c[record], dict_c_bottom[record], dict_c_top[record] = {}, {}, {}

                            # get a dictionary of dataframes that only contain methylated cytosine positions
                            dict_c[record][sample_index] = dict_total[record][sample_index][dict_total[record][sample_index]['modified_primary_base'] == 'C']
                            print('{}: Successfully loaded a methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            # Stratify the cytosine dictionary into two strand specific dictionaries.
                            dict_c_bottom[record][sample_index] = dict_c[record][sample_index][dict_c[record][sample_index]['ref_strand'] == '-']
                            print('{}: Successfully loaded a minus strand methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            dict_c_top[record][sample_index] = dict_c[record][sample_index][dict_c[record][sample_index]['ref_strand'] == '+']
                            print('{}: Successfully loaded a plus strand methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            # In the strand specific dictionaries, only keep positions that are informative for GpC SMF

                            # Reassign pointer for dict_c to None and delete the original value that it pointed to in order to decrease memory usage.
                            dict_c[record][sample_index] = None
                            gc.collect()
                        
                        if '6mA' in mods and '5mC' in mods:
                            # Remove combined stranded dicts from the dicts to skip set
                            dict_to_skip.difference_update(set(combined_dicts))                
                            # Initialize the sample keys for the combined dictionaries

                            if record not in dict_combined_bottom.keys() and record not in dict_combined_top.keys():
                                dict_combined_bottom[record], dict_combined_top[record]= {}, {}

                            print('{}: Successfully created a minus strand combined methylation dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            dict_combined_bottom[record][sample_index] = []
                            print('{}: Successfully created a plus strand combined methylation dictionary for '.format(readwrite.time_string()) + str(sample_index))
                            dict_combined_top[record][sample_index] = []

                        # Reassign pointer for dict_total to None and delete the original value that it pointed to in order to decrease memory usage.
                        dict_total[record][sample_index] = None
                        gc.collect()

                # Iterate over the stranded modification dictionaries and replace the dataframes with a dictionary of read names pointing to a list of values from the dataframe
                for dict_index, dict_type in enumerate(dict_list):
                    # Only iterate over stranded dictionaries
                    if dict_index not in dict_to_skip:
                        print('{0}: Extracting methylation states for {1} dictionary'.format(readwrite.time_string(), sample_types[dict_index]))
                        for record in dict_type.keys():
                            # Get the dictionary for the modification type of interest from the reference mapping of interest
                            mod_strand_record_sample_dict = dict_type[record]
                            print('{0}: Extracting methylation states for {1} dictionary'.format(readwrite.time_string(), record))
                            # For each sample in a stranded dictionary
                            n_samples = len(mod_strand_record_sample_dict.keys())
                            for sample in tqdm(mod_strand_record_sample_dict.keys(), desc=f'Extracting {sample_types[dict_index]} dictionary from record {record} for sample', total=n_samples):
                                # Load the combined bottom strand dictionary after all the individual dictionaries have been made for the sample
                                if dict_index == 7:
                                    # Load the minus strand dictionaries for each sample into temporary variables
                                    temp_a_dict = dict_list[2][record][sample].copy()
                                    temp_c_dict = dict_list[5][record][sample].copy()
                                    mod_strand_record_sample_dict[sample] = {}
                                    # Iterate over the reads present in the merge of both dictionaries
                                    for read in set(temp_a_dict) | set(temp_c_dict):
                                        # Add the arrays element-wise if the read is present in both dictionaries
                                        if read in temp_a_dict and read in temp_c_dict:
                                            mod_strand_record_sample_dict[sample][read] = np.nansum([temp_a_dict[read], temp_c_dict[read]], axis=0)
                                        # If the read is present in only one dictionary, copy its value
                                        elif read in temp_a_dict:
                                            mod_strand_record_sample_dict[sample][read] = temp_a_dict[read]
                                        elif read in temp_c_dict:
                                            mod_strand_record_sample_dict[sample][read] = temp_c_dict[read]
                                    del temp_a_dict, temp_c_dict
                            # Load the combined top strand dictionary after all the individual dictionaries have been made for the sample
                                elif dict_index == 8:
                                # Load the plus strand dictionaries for each sample into temporary variables
                                    temp_a_dict = dict_list[3][record][sample].copy()
                                    temp_c_dict = dict_list[6][record][sample].copy()
                                    mod_strand_record_sample_dict[sample] = {}
                                    # Iterate over the reads present in the merge of both dictionaries
                                    for read in set(temp_a_dict) | set(temp_c_dict):
                                        # Add the arrays element-wise if the read is present in both dictionaries
                                        if read in temp_a_dict and read in temp_c_dict:
                                            mod_strand_record_sample_dict[sample][read] = np.nansum([temp_a_dict[read], temp_c_dict[read]], axis=0)
                                        # If the read is present in only one dictionary, copy its value
                                        elif read in temp_a_dict:
                                            mod_strand_record_sample_dict[sample][read] = temp_a_dict[read]
                                        elif read in temp_c_dict:
                                            mod_strand_record_sample_dict[sample][read] = temp_c_dict[read]
                                    del temp_a_dict, temp_c_dict
                                # For all other dictionaries
                                else:
                                    # use temp_df to point to the dataframe held in mod_strand_record_sample_dict[sample]
                                    temp_df = mod_strand_record_sample_dict[sample]
                                    # reassign the dictionary pointer to a nested dictionary.
                                    mod_strand_record_sample_dict[sample] = {}
                                    # # Iterate through rows in the temp DataFrame
                                    for index, row in temp_df.iterrows():
                                        read = row['read_id'] # read name
                                        position = row['ref_position']  # 1-indexed positional coordinate
                                        probability = row['call_prob'] # Get the probability of the given call
                                        # if the call_code is modified change methylated value to the probability of methylation
                                        if (row['call_code'] in ['a', 'h', 'm']): 
                                            methylated = probability
                                        # If the call code is canonical, change the methylated value to 1 - the probability of canonical
                                        elif (row['call_code'] in ['-']):
                                            methylated = 1 - probability

                                        # If the current read is not in the dictionary yet, initalize the dictionary with a nan filled numpy array of proper size.
                                        if read not in mod_strand_record_sample_dict[sample]:
                                            mod_strand_record_sample_dict[sample][read] = np.full(max_reference_length, np.nan) 

                                        # add the positional methylation state to the numpy array
                                        mod_strand_record_sample_dict[sample][read][position-1] = methylated

                # Save the sample files in the batch as gzipped hdf5 files
                os.chdir(h5_dir)
                print('{0}: Converting batch {1} dictionaries to anndata objects'.format(readwrite.time_string(), batch))
                for dict_index, dict_type in enumerate(dict_list):
                    if dict_index not in dict_to_skip:
                        # Initialize an hdf5 file for the current modified strand
                        adata = None
                        print('{0}: Converting {1} dictionary to an anndata object'.format(readwrite.time_string(), sample_types[dict_index]))
                        for record in dict_type.keys():
                            # Get the dictionary for the modification type of interest from the reference mapping of interest
                            mod_strand_record_sample_dict = dict_type[record]
                            for sample in mod_strand_record_sample_dict.keys():
                                print('{0}: Converting {1} dictionary for sample {2} to an anndata object'.format(readwrite.time_string(), sample_types[dict_index], sample))
                                sample = int(sample)
                                final_sample_index = sample + (batch * batch_size)
                                print('{0}: Final sample index for sample: {1}'.format(readwrite.time_string(), final_sample_index))
                                print('{0}: Converting {1} dictionary for sample {2} to a dataframe'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                temp_df = pd.DataFrame.from_dict(mod_strand_record_sample_dict[sample], orient='index')
                                mod_strand_record_sample_dict[sample] = None # reassign pointer to facilitate memory usage
                                sorted_index = sorted(temp_df.index)
                                temp_df = temp_df.reindex(sorted_index)
                                X = temp_df.values

                                print('{0}: Loading {1} dataframe for sample {2} into a temp anndata object'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                temp_adata = ad.AnnData(X, dtype=X.dtype)
                                if temp_adata.shape[0] > 0:
                                    print('{0}: Adding read names and position ids to {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                    temp_adata.obs_names = temp_df.index
                                    temp_adata.obs_names = temp_adata.obs_names.astype(str)
                                    temp_adata.var_names = temp_df.columns
                                    temp_adata.var_names = temp_adata.var_names.astype(str)
                                    print('{0}: Adding {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                    temp_adata.obs['Sample'] = [str(final_sample_index)] * len(temp_adata)
                                    dataset, strand = sample_types[dict_index].split('_')[:2]
                                    temp_adata.obs['Strand'] = [strand] * len(temp_adata)
                                    temp_adata.obs['Dataset'] = [dataset] * len(temp_adata)
                                    temp_adata.obs['Reference'] = [f'{record}_{dataset}_{strand}'] * len(temp_adata)
                                    temp_adata.obs['Reference_chromosome'] = [f'{record}'] * len(temp_adata)

                                    # Load in the one hot encoded reads from the current sample and record
                                    one_hot_reads = {}
                                    n_rows_OHE = 5
                                    ohe_files = bam_record_ohe_files[f'{final_sample_index}_{record}']
                                    print(f'Loading OHEs from {ohe_files}')
                                    fwd_mapped_reads = set()
                                    rev_mapped_reads = set()
                                    for ohe_file in ohe_files:
                                        tmp_ohe_dict = ad.read_h5ad(ohe_file).uns
                                        one_hot_reads.update(tmp_ohe_dict)
                                        if '_fwd_' in ohe_file:
                                            fwd_mapped_reads.update(tmp_ohe_dict.keys())
                                        elif '_rev_' in ohe_file:
                                            rev_mapped_reads.update(tmp_ohe_dict.keys())
                                        del tmp_ohe_dict

                                    read_names = list(one_hot_reads.keys())

                                    read_mapping_direction = []
                                    for read_id in temp_adata.obs_names:
                                        if read_id in fwd_mapped_reads:
                                            read_mapping_direction.append('fwd')
                                        elif read_id in rev_mapped_reads:
                                            read_mapping_direction.append('rev')
                                        else:
                                            read_mapping_direction.append('unk')

                                    temp_adata.obs['Read_mapping_direction'] = read_mapping_direction

                                    del temp_df
                                    
                                    dict_A, dict_C, dict_G, dict_T, dict_N = {}, {}, {}, {}, {}
                                    sequence_length = one_hot_reads[read_names[0]].reshape(n_rows_OHE, -1).shape[1]
                                    df_A = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                                    df_C = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                                    df_G = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                                    df_T = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                                    df_N = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))

                                    for read_name, one_hot_array in one_hot_reads.items():
                                        one_hot_array = one_hot_array.reshape(n_rows_OHE, -1)
                                        dict_A[read_name] = one_hot_array[0, :]
                                        dict_C[read_name] = one_hot_array[1, :]
                                        dict_G[read_name] = one_hot_array[2, :]
                                        dict_T[read_name] = one_hot_array[3, :]
                                        dict_N[read_name] = one_hot_array[4, :]

                                    del one_hot_reads
                                    gc.collect()

                                    for j, read_name in tqdm(enumerate(sorted_index), desc='Loading dataframes of OHE reads', total=len(sorted_index)):
                                        df_A.iloc[j] = dict_A[read_name]
                                        df_C.iloc[j] = dict_C[read_name]
                                        df_G.iloc[j] = dict_G[read_name]
                                        df_T.iloc[j] = dict_T[read_name]
                                        df_N.iloc[j] = dict_N[read_name]

                                    del dict_A, dict_C, dict_G, dict_T, dict_N
                                    gc.collect()

                                    ohe_df_map = {0: df_A, 1: df_C, 2: df_G, 3: df_T, 4: df_N}

                                    for j, base in enumerate(['A', 'C', 'G', 'T', 'N']):
                                        temp_adata.layers[f'{base}_binary_encoding'] = ohe_df_map[j].values
                                        ohe_df_map[j] = None # Reassign pointer for memory usage purposes

                                    # If final adata object already has a sample loaded, concatenate the current sample into the existing adata object 
                                    if adata:
                                        if temp_adata.shape[0] > 0:
                                            print('{0}: Concatenating {1} anndata object for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                            adata = ad.concat([adata, temp_adata], join='outer', index_unique=None)
                                            del temp_adata
                                        else:
                                            print(f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata")
                                    else:
                                        if temp_adata.shape[0] > 0:
                                            print('{0}: Initializing {1} anndata object for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                            adata = temp_adata
                                        else:
                                            print(f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata")

                                    gc.collect()
                                else:
                                    print(f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata. Skipping sample.")

                        print('{0}: Writing {1} anndata out as a gzipped hdf5 file'.format(readwrite.time_string(), sample_types[dict_index]))
                        adata.write_h5ad('{0}_{1}_{2}_SMF_binarized_sample_hdf5.h5ad.gz'.format(readwrite.date_string(), batch, sample_types[dict_index]), compression='gzip')

                # Delete the batch dictionaries from memory
                del dict_list, adata
                gc.collect()

        # Iterate over all of the batched hdf5 files and concatenate them.
        os.chdir(h5_dir)
        files = os.listdir(h5_dir)        
        # Filter file names that contain the search string in their filename and keep them in a list
        hdfs = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
        # Sort file list by names and print the list of file names
        hdfs.sort()
        print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
        hdf_paths = [os.path.join(h5_dir, hd5) for hd5 in hdfs]
        final_adata = None
        for hdf_index, hdf in enumerate(hdf_paths):
            print('{0}: Reading in {1} hdf5 file'.format(readwrite.time_string(), hdfs[hdf_index]))
            temp_adata = ad.read_h5ad(hdf)
            if final_adata:
                print('{0}: Concatenating final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdf[hdf_index]))
                final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
            else:
                print('{0}: Initializing final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdf[hdf_index]))
                final_adata = temp_adata
            del temp_adata

        # Set obs columns to type 'category'
        for col in final_adata.obs.columns:
            final_adata.obs[col] = final_adata.obs[col].astype('category')

        for record in records_to_analyze:
            # Add FASTA sequence to the object
            sequence = record_seq_dict[record][0]
            complement = record_seq_dict[record][1]
            final_adata.var[f'{record}_top_strand_FASTA_base_at_coordinate'] = list(sequence)
            final_adata.var[f'{record}_bottom_strand_FASTA_base_at_coordinate'] = list(complement)
            final_adata.uns[f'{record}_FASTA_sequence'] = sequence
            # Add consensus sequence of samples mapped to the record to the object
            record_subset = final_adata[final_adata.obs['Reference_chromosome'] == record].copy()
            for strand in record_subset.obs['Strand'].cat.categories:
                strand_subset = record_subset[record_subset.obs['Strand'] == strand].copy()
                for mapping_dir in strand_subset.obs['Read_mapping_direction'].cat.categories:
                    mapping_dir_subset = strand_subset[strand_subset.obs['Read_mapping_direction'] == mapping_dir].copy()
                    layer_map, layer_counts = {}, []
                    for i, layer in enumerate(mapping_dir_subset.layers):
                        layer_map[i] = layer.split('_')[0]
                        layer_counts.append(np.sum(mapping_dir_subset.layers[layer], axis=0))
                    count_array = np.array(layer_counts)
                    nucleotide_indexes = np.argmax(count_array, axis=0)
                    consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
                    final_adata.var[f'{record}_{strand}_strand_{mapping_dir}_mapping_dir_consensus_from_all_samples'] = consensus_sequence_list

        final_adata.write_h5ad(os.path.join(h5_dir, final_hdf), compression='gzip')

        # Delete the individual h5ad files and only keep the final concatenated file
        if delete_batch_hdfs:
            files = os.listdir(h5_dir)
            hdfs_to_delete = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
            hdf_paths_to_delete = [os.path.join(h5_dir, hdf) for hdf in hdfs_to_delete]
            # Iterate over the files and delete them
            for hdf in hdf_paths_to_delete:
                try:
                    os.remove(hdf)
                    print(f"Deleted file: {hdf}")
                except OSError as e:
                    print(f"Error deleting file {hdf}: {e}")