## modkit_extract_to_adata

def modkit_extract_to_adata(fasta, bam_dir, mapping_threshold, experiment_name, mods, batch_size, mod_tsv_dir):
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

    Returns:
        None
    """
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

    ### Get input tsv file names into a sorted list ###
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

    # Get all references within the FASTA and indicate the length and identity of the record sequence
    max_reference_length = 0
    reference_dict = get_native_references(fasta) # returns a dict keyed by record name. Points to a tuple of (reference length, reference sequence)
    # Get the max record length in the dataset.
    for record in reference_dict.keys():
        if reference_dict[record][0] > max_reference_length:
            max_reference_length = reference_dict[record][0]

    print(f'{readwrite.time_string()}: Max reference length in dataset: {max_reference_length}')
    batches = math.ceil(len(tsvs) / batch_size) # Number of batches to process
    print('{0}: Processing input tsvs in {1} batches of {2} tsvs '.format(readwrite.time_string(), batches, batch_size))

    # look at aligned read proportions in the bams and get all records that are above a certain mapping threshold.
    records_to_analyze = []
    for bami, bam in enumerate(bam_path_list):
        aligned_reads_count, unaligned_reads_count, record_counts = count_aligned_reads(bam)
        percent_aligned = aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)
        print(f'{percent_aligned} percent of reads in {bams[bami]} aligned successfully')
        # Iterate over references and decide which to use in the analysis based on the mapping_threshold
        for record in record_counts:
            print('{0} reads mapped to reference record {1}. This is {2} percent of all mapped reads in {3}'.format(record_counts[record][0], record, record_counts[record][1]*100, bams[bami]))
            if record_counts[record][1] >= mapping_threshold:
                records_to_analyze.append(record)

    records_to_analyze = set(records_to_analyze)
    print(f'Records to analyze: {records_to_analyze}')

    # Iterate over records to analyze and return a dictionary keyed by the reference name that points to a tuple containing the top strand sequence and the complement
    record_seq_dict = {}
    for record in records_to_analyze:
        current_reference_length = reference_dict[record][0]
        delta_max_length = max_reference_length - current_reference_length
        sequence = reference_dict[record][1] + 'N'*delta_max_length
        complement = str(Seq(reference_dict[record][1]).complement()).upper() + 'N'*delta_max_length
        # Get a dictionary of positional base identities keyed by read id
        positions = range(current_reference_length)
        record_seq_dict[record] = (sequence, complement)

        # if os.path.isdir(tmp_dir):
        #     tmp_files = os.listdir(tmp_dir)
        #     tmp_check = sum([1 for file in tmp_files if f'tmp_{record}' in file and '.npz' in file])
        # if tmp_check > 0:
        #     print("Using existing OHE read encodings")
        #     ohe_files = [os.path.join(tmp_dir, file) for file in tmp_files if f'tmp_{record}' in file and '.npz' in file]
        # else:
        #     base_identities = extract_base_identities(bam, record, positions, max_reference_length)
        #     # One hot encode the sequence string of the reads
        #     ohe_files = ohe_batching(base_identities, tmp_dir, record, batch_size=1000)
        #     del base_identities

    # Tuple contains another dictionary keyed by read names that point to an OHE of the read sequence. Also contains the top strand sequence, and the complement sequence.
        # one_hot_reads = {}
        # n_rows_OHE = 5
        # for ohe_file in tqdm(ohe_files, desc="Reading in OHE reads"):
        #     one_hot_reads.update(np.load(ohe_file))
        # OHE_record_seq_dict[record] = one_hot_reads

    ###################################################

    ###################################################
    # Begin iterating over batches
    for batch in range(batches):
        print('{0}: Processing tsvs for batch {1} '.format(readwrite.time_string(), batch))
        # For the final batch, just take the remaining tsv files
        if batch == batches - 1:
            tsv_batch = tsv_path_list
        # For all other batches, take the next batch of tsvs out of the file queue.    
        else:
            tsv_batch = tsv_path_list[:batch_size]
            tsv_path_list = tsv_path_list[batch_size:]
        print('{0}: tsvs in batch {1} '.format(readwrite.time_string(), tsv_batch))
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
        A_stranded_dicts = [2, 3]
        C_stranded_dicts = [5, 6]
        dict_to_skip = dict_to_skip + combined_dicts + A_stranded_dicts + C_stranded_dicts
        dict_to_skip = set(dict_to_skip)

        # Load the dict_total dictionary with all of the tsv files as dataframes.
        for i, tsv in tqdm(enumerate(tsv_batch), desc=f'Loading TSVs into dataframes and filtering on chromosome/position for {batch}', total=batch_size):
            print('{0}: Loading sample tsv {1} into dataframe'.format(readwrite.time_string(), tsv))
            temp_df = pd.read_csv(tsv, sep='\t', header=0)
            for record in records_to_analyze:
                if record not in dict_total.keys():
                    dict_total[record] = {}
                # Only keep the reads aligned to the chromosomes of interest
                #print('{0}: Filtering sample dataframe to keep chromosome of interest'.format(readwrite.time_string()))
                dict_total[record][i] = temp_df[temp_df['chrom'] == record]
                # Only keep the read positions that fall within the region of interest
                #print('{0}: Filtering sample dataframe to keep positions falling within region of interest'.format(readwrite.time_string()))
                current_reference_length = reference_dict[record][0]
                dict_total[record][i] = dict_total[record][i][(current_reference_length > dict_total[record][i]['ref_position']) & (dict_total[record][i]['ref_position']>= 0)]

        # Iterate over dict_total of all the tsv files and extract the modification specific and strand specific dataframes into dictionaries
        for record in tqdm(dict_total.keys(), desc=f'Extracting strand specific and modification specific dictionaries for batch {batch}', total=len(dict_total)):
            for i in dict_total[record].keys():
                if '6mA' in mods:
                    # Remove Adenine stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(A_stranded_dicts)

                    if record not in dict_a.keys() and record not in dict_a_bottom.keys() and record not in dict_a_top.keys():
                        dict_a[record], dict_a_bottom[record], dict_a_top[record] = {}, {}, {}

                    # get a dictionary of dataframes that only contain methylated adenine positions
                    dict_a[record][i] = dict_total[record][i][dict_total[record][i]['modified_primary_base'] == 'A']
                    print('{}: Successfully created a methyl-adenine dictionary for '.format(readwrite.time_string()) + str(i))
                    # Stratify the adenine dictionary into two strand specific dictionaries.
                    dict_a_bottom[record][i] = dict_a[record][i][dict_a[record][i]['ref_strand'] == '-']
                    print('{}: Successfully created a minus strand methyl-adenine dictionary for '.format(readwrite.time_string()) + str(i))
                    dict_a_top[record][i] = dict_a[record][i][dict_a[record][i]['ref_strand'] == '+']
                    print('{}: Successfully created a plus strand methyl-adenine dictionary for '.format(readwrite.time_string()) + str(i))

                if '5mC' in mods:
                    # Remove Cytosine stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(C_stranded_dicts)

                    if record not in dict_c.keys() and record not in dict_c_bottom.keys() and record not in dict_c_top.keys():
                        dict_c[record], dict_c_bottom[record], dict_c_top[record] = {}, {}, {}

                    # get a dictionary of dataframes that only contain methylated cytosine positions
                    dict_c[record][i] = dict_total[record][i][dict_total[record][i]['modified_primary_base'] == 'C']
                    print('{}: Successfully created a methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(i))
                    # Stratify the cytosine dictionary into two strand specific dictionaries.
                    dict_c_bottom[record][i] = dict_c[record][i][dict_c[record][i]['ref_strand'] == '-']
                    print('{}: Successfully created a minus strand methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(i))
                    dict_c_top[record][i] = dict_c[record][i][dict_c[record][i]['ref_strand'] == '+']
                    print('{}: Successfully created a plus strand methyl-cytosine dictionary for '.format(readwrite.time_string()) + str(i))
                    # In the strand specific dictionaries, only keep positions that are informative for GpC SMF
                
                if '6mA' in mods and '5mC' in mods:
                    # Remove combined stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(combined_dicts)                
                    # Initialize the sample keys for the combined dictionaries

                    if record not in dict_combined_bottom.keys() and record not in dict_combined_top.keys():
                        dict_combined_bottom[record], dict_combined_top[record]= {}, {}

                    print('{}: Successfully created a minus strand combined methylation dictionary for '.format(readwrite.time_string()) + str(i))
                    dict_combined_bottom[record][i] = []
                    print('{}: Successfully created a plus strand combined methylation dictionary for '.format(readwrite.time_string()) + str(i))
                    dict_combined_top[record][i] = []

        # Iterate over the stranded modification dictionaries and replace the dataframes with a dictionary of read names pointing to a list of values from the dataframe
        for i, dict_type in tqdm(enumerate(dict_list), desc=f'Reformatting modified dictionaries in batch {batch}', total=len(dict_list)):
            # Only iterate over stranded dictionaries
            if i not in dict_to_skip:
                print('{0}: Extracting methylation states for {1} dictionary'.format(readwrite.time_string(), sample_types[i]))
                for record in dict_type.keys():
                    # Get the dictionary for the modification type of interest from the reference mapping of interest
                    dict = dict_type[record]
                    print('{0}: Extracting methylation states for {1} dictionary'.format(readwrite.time_string(), record))
                    # For each sample in a stranded dictionary
                    for sample in dict.keys():
                        print('{0}: Extracting {1} dictionary from record {2} for sample {3}'.format(readwrite.time_string(), sample_types[i], record, sample))
                        # Load the combined bottom strand dictionary after all the individual dictionaries have been made for the sample
                        if i == 7:
                            # Load the minus strand dictionaries for each sample into temporary variables
                            temp_a_dict = dict_list[2][record][sample].copy()
                            temp_c_dict = dict_list[5][record][sample].copy()
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
                        # Load the plus strand dictionaries for each sample into temporary variables
                            temp_a_dict = dict_list[3][record][sample].copy()
                            temp_c_dict = dict_list[6][record][sample].copy()
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
                            # extract the dataframe from the dictionary into a temporary variable
                            temp_df = dict[sample]
                            # reassign the dictionary pointer to a nested dictionary.
                            dict[sample] = {}
                            # # Iterate through rows in the temp DataFrame
                            for index, row in temp_df.iterrows():
                                read = row['read_id'] # read name
                                position = row['ref_position']  # positional coordinate
                                probability = row['call_prob'] # Get the probability of the given call
                                # if the call_code is modified change methylated value to the probability of methylation
                                if (row['call_code'] in ['a', 'h', 'm']): 
                                    methylated = probability
                                # If the call code is canonical, change the methylated value to 1 - the probability of canonical
                                elif (row['call_code'] in ['-']):
                                    methylated = 1 - probability

                                # If the current read is not in the dictionary yet, initalize the dictionary with a nan filled numpy array of proper size.
                                if read not in dict[sample]:
                                    dict[sample][read] = np.full(max_reference_length, np.nan) 
                                else:
                                    pass
                                # add the positional methylation state to the numpy array
                                dict[sample][read][position-1] = methylated

        # Save the sample files in the batch as gzipped hdf5 files
        os.chdir(h5_dir)
        print('{0}: Converting batch {1} dictionaries to anndata objects'.format(readwrite.time_string(), batch))
        for i, dict_type in tqdm(enumerate(dict_list), desc=f'Loading AnnDatas for batch {batch}', total=len(dict_list)):
            if i not in dict_to_skip:
                # Initialize an hdf5 file for the current modified strand
                adata = None
                print('{0}: Converting {1} dictionary to an anndata object'.format(readwrite.time_string(), sample_types[i]))
                for record in dict_type.keys():
                    # Get the dictionary for the modification type of interest from the reference mapping of interest
                    dict = dict_type[record]
                    for sample in dict.keys():
                        print('{0}: Converting {1} dictionary for sample {2} to an anndata object'.format(readwrite.time_string(), sample_types[i], sample))
                        sample = int(sample)
                        final_sample_index = sample + (batch * batch_size)
                        print('{0}: Final sample index for sample: {1}'.format(readwrite.time_string(), final_sample_index))
                        print('{0}: Converting {1} dictionary for sample {2} to a dataframe'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                        temp_df = pd.DataFrame.from_dict(dict[sample], orient='index')
                        sorted_index = sorted(temp_df.index)
                        temp_df = temp_df.reindex(sorted_index)
                        X = temp_df.values


                        one_hot_encodings = record_seq_dict[record][0]
                        read_names = list(one_hot_encodings.keys())
                        ohe_length = one_hot_encodings[read_names[0]].shape[0]
                        dict_A, dict_C, dict_G, dict_T, dict_N = {}, {}, {}, {}, {}
                        # Loop through each read name and its corresponding one-hot array
                        print('{0}: Extracting one hot encodings into dictionaries'.format(readwrite.time_string()))
                        n_encodings = len(one_hot_encodings)
                        # Initialize empty DataFrames for each base
                        read_names = list(one_hot_reads.keys())
                        sequence_length = one_hot_reads[read_names[0]].reshape(n_rows_OHE, -1).shape[0]
                        df_A = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                        df_C = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                        df_G = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                        df_T = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
                        df_N = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))

                        # Iterate through the dictionary and populate the DataFrames
                        for read_name, one_hot_array in one_hot_reads.items():
                            one_hot_array = one_hot_array.reshape(n_rows_OHE, -1)
                            df_A.loc[read_name] = one_hot_array[:, 0]
                            df_C.loc[read_name] = one_hot_array[:, 1]
                            df_G.loc[read_name] = one_hot_array[:, 2]
                            df_T.loc[read_name] = one_hot_array[:, 3]
                            df_N.loc[read_name] = one_hot_array[:, 4]
                        # for read_name, one_hot_array in tqdm(one_hot_encodings.items('Extracting one hot encodings into dictionaries'), desc='', total=n_encodings):
                        #     one_hot_array = one_hot_array.reshape(n_rows_OHE, -1)
                        #     dict_A[read_name] = one_hot_array[:, 0]
                        #     dict_C[read_name] = one_hot_array[:, 1]
                        #     dict_G[read_name] = one_hot_array[:, 2]
                        #     dict_T[read_name] = one_hot_array[:, 3]
                        #     dict_N[read_name] = one_hot_array[:, 4]
                        # # Load dfs with data from the dictionaries
                        # print('{0}: Loading dataframes from one hot encoded dictionaries'.format(readwrite.time_string()))
                        # df_A = pd.DataFrame.from_dict(dict_A, orient='index').reindex(sorted_index)
                        # df_C = pd.DataFrame.from_dict(dict_C, orient='index').reindex(sorted_index)
                        # df_G = pd.DataFrame.from_dict(dict_G, orient='index').reindex(sorted_index)
                        # df_T = pd.DataFrame.from_dict(dict_T, orient='index').reindex(sorted_index)
                        # df_N = pd.DataFrame.from_dict(dict_N, orient='index').reindex(sorted_index)

                        ohe_df_map = {0: df_A, 1: df_C, 2: df_G, 3: df_T, 4: df_N}

                        print('{0}: Loading {1} dataframe for sample {2} into a temp anndata object'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                        temp_adata = ad.AnnData(X, dtype=X.dtype)
                        print('{0}: Adding read names and position ids to {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                        temp_adata.obs_names = temp_df.index
                        temp_adata.obs_names = temp_adata.obs_names.astype(str)
                        temp_adata.var_names = temp_df.columns
                        temp_adata.var_names = temp_adata.var_names.astype(str)
                        print('{0}: Adding final sample id to {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                        temp_adata.obs['Sample'] = [str(final_sample_index)] * len(temp_adata)
                        dataset, strand = sample_types[i].split('_')[:2]
                        temp_adata.obs['Strand'] = [strand] * len(temp_adata)
                        temp_adata.obs['Dataset'] = [dataset] * len(temp_adata)
                        temp_adata.obs['Reference'] = [f'{record}_{dataset}_{strand}'] * len(temp_adata)
                        temp_adata.obs['Reference_chromosome'] = [f'{record}'] * len(temp_adata)

                        for j, base in enumerate(['A', 'C', 'G', 'T', 'N']):
                            temp_adata.layers[f'{base}_binary_encoding'] = ohe_df_map[j].values

                        # If final adata object already has a sample loaded, concatenate the current sample into the existing adata object 
                        if adata:
                            if temp_adata.shape[0] > 0:
                                print('{0}: Concatenating {1} anndata object for sample {2}'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                                adata = ad.concat([adata, temp_adata], join='outer', index_unique=None)
                            else:
                                print(f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata")
                        else:
                            if temp_adata.shape[0] > 0:
                                print('{0}: Initializing {1} anndata object for sample {2}'.format(readwrite.time_string(), sample_types[i], final_sample_index))
                                adata = temp_adata
                            else:
                                print(f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata")

                print('{0}: Writing {1} anndata out as a gzipped hdf5 file'.format(readwrite.time_string(), sample_types[i]))
                adata.write_h5ad('{0}_{1}_{2}_SMF_binarized_sample_hdf5.h5ad.gz'.format(readwrite.date_string(), batch, sample_types[i]), compression='gzip')

        # Delete the batch dictionaries from memory
        del dict_list
        gc.collect()
    
    # Iterate over all of the batched hdf5 files and concatenate them.
    os.chdir(h5_dir)
    files = os.listdir(os.getcwd())
    # Name the final output file
    final_hdf = '{0}_{1}_final_experiment_hdf5.h5ad.gz'.format(readwrite.date_string(), experiment_name)
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    final_adata = None
    for hdf in hdfs:
        print('{0}: Reading in {1} hdf5 file'.format(readwrite.time_string(), hdf))
        temp_adata = ad.read_h5ad(hdf)
        if final_adata:
            print('{0}: Concatenating final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdf))
            final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
        else:
            print('{0}: Initializing final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdf))
            final_adata = temp_adata
    print('{0}: Writing final concatenated hdf5 file'.format(readwrite.time_string()))

    # Set obs columns to type 'category'
    for col in final_adata.obs.columns:
        final_adata.obs[col] = final_adata.obs[col].astype('category')

    for record in records_to_analyze:
        # Add FASTA sequence to the object
        sequence = record_seq_dict[record][1]
        complement = record_seq_dict[record][2]
        final_adata.var[f'{record}_top_strand_FASTA_sequence'] = list(sequence)
        final_adata.var[f'{record}_bottom_strand_FASTA_sequence'] = list(complement)
        # Add consensus sequence of samples mapped to the record to the object
        record_subset = final_adata[final_adata.obs['Reference_chromosome'] == record].copy()
        for strand in adata.obs['Strand'].cat.categories:
            strand_subset = record_subset[record_subset.obs['Strand'] == strand].copy()
            layer_map, layer_counts = {}, []
            for i, layer in enumerate(strand_subset.layers):
                layer_map[i] = layer.split('_')[0]
                layer_counts.append(np.sum(strand_subset.layers[layer], axis=0))
            count_array = np.array(layer_counts)
            nucleotide_indexes = np.argmax(count_array, axis=0)
            consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
            final_adata.var[f'{record}_{strand}_strand_consensus_from_all_samples'] = consensus_sequence_list

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