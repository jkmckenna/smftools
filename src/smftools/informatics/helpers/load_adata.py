# load_adata
######################################################################################################
import .utils
# File I/O
import subprocess
import gc

# bioinformatic operations
import .informatics_module

# User interface
from tqdm import tqdm

######################################################################################################
# Conversion SMF
def converted_BAM_to_adata(converted_fasta_file, bam_directory, mapping_threshold, experiment_name, modification_types=['5mC', '6mA'], strands=['top', 'bottom']):
    """
    Inputs:
        "converted_fasta_file", help="converted FASTA file"
        "bam_directory", help="Directory containing input BAMs to binarize"
        "mapping_threshold", help="Minimal threshold of mapped reads to a reference chromosome to allow"
        "experiment_name", help="String to append to the output h5ad file"
        "modification_types", help=" a list of modifications to detect. Options are 5mC and 6mA"
        "strands", help="A list of strands to include in the analysis. Options are top and bottom"
    Outputs:
        Takes a directory of BAM files from conversion SMF and generates a gzipped h5ad file.
    """
    mapping_threshold = float(args.mapping_threshold)
    bam_suffix = '.bam'
    # Get all of the input BAM files
    files = os.listdir(bam_directory)
    # Change directory to the BAM directory
    os.chdir(bam_directory)
    # Filter file names that contain the search string in their filename and keep them in a list
    bams = [bam for bam in files if bam_suffix in bam and '.bai' not in bam]
    # Sort file list by names and print the list of file names
    bams.sort()
    print(f'Found the following BAMS: {bams}')
     # Options include 6mA, 5mC
     # Options include top and bottom
    final_adata = None

    # Make a dictionary, keyed by modification type, that points to another dictionary of unconverted_record_ids. This points to a list of: 1) record length, 2) top strand conversion coordinates, 3) bottom strand conversion coordinates, 4) record sequence
    modification_dict = {}
    # While populating the dictionary, also extract the longest sequence record in the input references
    max_reference_length = 0
    for modification_type in modification_types:
        modification_dict[modification_type] = find_coordinates(converted_fasta_file, modification_type)
        for record in modification_dict[modification_type].keys():
            if modification_dict[modification_type][record][0] > max_reference_length:
                max_reference_length = modification_dict[modification_type][record][0]
    # Iterate over the experiment BAM files
    for bam_index, bam in enumerate(bams):
        # Give each bam a sample name
        sample = bam.split(sep=bam_suffix)[0]
        # look at aligned read proportions in the bam
        aligned_reads_count, unaligned_reads_count, record_counts = count_aligned_reads(bam)
        percent_aligned = aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)
        print(f'{percent_aligned} percent of total reads in {bam} aligned successfully')

        records_to_analyze = []
        # Iterate over converted reference strands and decide which to use in the analysis based on the mapping_threshold
        for record in record_counts:
            print(f'{record_counts[record][0]} reads mapped to reference record {record}. This is {record_counts[record][1]*100} percent of all mapped reads in the sample.')
            if record_counts[record][1] >= mapping_threshold:
                records_to_analyze.append(record)
        print(f'Records to analyze: {records_to_analyze}')

        # Iterate over records to analyze (ie all conversions detected)
        record_FASTA_dict = {}
        for record in records_to_analyze:
            mod_type, strand = record.split('_')[-2:]
            if strand == 'top':
                strand_index = 1
            elif strand == 'bottom':
                strand_index = 2

            chromosome = record.split('_{0}_{1}'.format(mod_type, strand))[0]
            unconverted_chromosome_name = chromosome + '_unconverted_top'
            positions = modification_dict[mod_type][unconverted_chromosome_name][strand_index]
            current_reference_length = modification_dict[mod_type][unconverted_chromosome_name][0]
            delta_max_length = max_reference_length - current_reference_length
            sequence = modification_dict[mod_type][unconverted_chromosome_name][3] + 'N'*delta_max_length
            record_FASTA_dict[f'{record}'] = sequence
            print(f'Chromosome: {chromosome}\nUnconverted Sequence: {sequence}')

            # Get a dictionary of positional identities keyed by read id
            print(f'Extracting base identities of target positions')
            target_base_identities = extract_base_identity_at_coordinates(bam, record, positions, max_reference_length) 
            # binarize the dictionary of positional identities
            print(f'Binarizing base identities of target positions')
            binarized_base_identities = binarize_base_identities(target_base_identities, strand, mod_type) 
            # converts the base identity dictionary to a dataframe.
            binarized_base_identities_df = pd.DataFrame.from_dict(binarized_base_identities, orient='index') 
            sorted_index = sorted(binarized_base_identities_df.index)
            binarized_base_identities_df = binarized_base_identities_df.reindex(sorted_index)
            # Get the sequence string of every read
            print(f'Extracting base identities of all positions in each read')
            all_base_identities = extract_base_identity_at_coordinates(bam, record, range(current_reference_length), max_reference_length)
            # One hot encode the sequence string of the reads
            print(f'One hot encoding base identities of all positions in each read')
            one_hot_reads = {read_name: one_hot_encode(seq) for read_name, seq in all_base_identities.items()}

            # Initialize empty DataFrames for each base
            read_names = list(one_hot_reads.keys())
            sequence_length = one_hot_reads[read_names[0]].shape[0]
            df_A = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
            df_C = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
            df_G = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
            df_T = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))
            df_N = pd.DataFrame(0, index=sorted_index, columns=range(sequence_length))

            # Iterate through the dictionary and populate the DataFrames
            for read_name, one_hot_array in one_hot_reads.items():
                df_A.loc[read_name] = one_hot_array[:, 0]
                df_C.loc[read_name] = one_hot_array[:, 1]
                df_G.loc[read_name] = one_hot_array[:, 2]
                df_T.loc[read_name] = one_hot_array[:, 3]
                df_N.loc[read_name] = one_hot_array[:, 4]

            ohe_df_map = {0: df_A, 1: df_C, 2: df_G, 3: df_T, 4: df_N}   

            # Load an anndata object with the sample data
            X = binarized_base_identities_df.values
            adata = ad.AnnData(X, dtype=X.dtype)
            adata.obs_names = binarized_base_identities_df.index
            adata.obs_names = adata.obs_names.astype(str)
            adata.var_names = binarized_base_identities_df.columns
            adata.var_names = adata.var_names.astype(str)
            adata.obs['Sample'] = [sample] * len(adata)
            adata.obs['Strand'] = [strand] * len(adata)
            adata.obs['Dataset'] = [mod_type] * len(adata)
            adata.obs['Reference'] = [record] * len(adata)
            adata.obs['Reference_chromosome'] = [chromosome] * len(adata)
            
            for j, base in enumerate(['A', 'C', 'G', 'T', 'N']):
                adata.layers[f'{base}_binary_encoding'] = ohe_df_map[j].values 

            if final_adata:
                final_adata = ad.concat([final_adata, adata], join='outer', index_unique=None)
            else:
                final_adata = adata

    for record in record_FASTA_dict.keys():
        chromosome = record.split('_')[0]
        sequence = record_FASTA_dict[record]
        final_adata.uns[f'{record}_FASTA_sequence'] = sequence
        final_adata.var[f'{record}_FASTA_sequence'] = list(sequence)
        record_subset = final_adata[final_adata.obs['Reference'] == record].copy()
        layer_map, layer_counts = {}, []
        for i, layer in enumerate(record_subset.layers):
            layer_map[i] = layer.split('_')[0]
            layer_counts.append(np.sum(record_subset.layers[layer], axis=0))
        count_array = np.array(layer_counts)
        nucleotide_indexes = np.argmax(count_array, axis=0)
        consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
        final_adata.var[f'{record}_consensus_across_samples'] = consensus_sequence_list
        final_adata.uns[f'{record}_consensus_sequence'] = ''.join(consensus_sequence_list)
    
    ## Export the final adata object
    final_adata.write_h5ad('{0}_{1}.h5ad.gz'.format(date_string, experiment_name), compression='gzip')

# Direct detection SMF
def modkit_extract_to_adata(fasta, bam, mapping_threshold, experiment_name, mods, batch_size):
    """
    Inputs:
    "mods", help="list of modifications to analayze. Available mods [6mA, 5mC]"
    "fasta", help="a FASTA file to extract positions of interest from."
    "bam", help="a bam file to extract read-level sequence identities."
    "mapping_threshold", help="Minimal threshold of mapped reads to a reference chromosome to allow"
    "batch_size", help="Number of sample TSV files to process per batch"
    "experiment_name", help="An experiment name to add to the final anndata object"
    Output:
        Take modkit extract sample tsv files and the experiment level BAM file to generate an anndata object
    """
    mapping_threshold = float(mapping_threshold)
    
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
    print(f'{len(tsvs)} sample tsv files found: {tsvs}')
    print(f'sample bam file found: {bam}')

    # Get all references within the FASTA and indicate the length and identity of the record sequence
    max_reference_length = 0
    reference_dict = get_references(fasta)
    for record in reference_dict.keys():
        if reference_dict[record][0] > max_reference_length:
            max_reference_length = reference_dict[record][0]

    print(f'{time_string()}: Max reference length in dataset: {max_reference_length}')
    batch_size = int(batch_size) # Number of TSVs to maximally process in a batch
    batches = math.ceil(len(tsvs) / batch_size) # Number of batches to process
    print('{0}: Processing input tsvs in {1} batches of {2} tsvs '.format(time_string(), batches, batch_size))

    # look at aligned read proportions in the bam
    aligned_reads_count, unaligned_reads_count, record_counts = count_aligned_reads(bam)
    print('{} percent of reads in bam aligned successfully'.format(aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)))
    records_to_analyze = []
    # Iterate over references and decide which to use in the analysis based on the mapping_threshold
    for record in record_counts:
        print('{0} reads mapped to reference record {1}. This is {2} percent of all mapped reads'.format(record_counts[record][0], record, record_counts[record][1]*100))
        if record_counts[record][1] >= mapping_threshold:
            records_to_analyze.append(record)
    print(f'Records to analyze: {records_to_analyze}')
    # Iterate over records to analyze and return a dictionary keyed by the reference name that points to another dictionary keyed by read names that map to that reference. This internal dictionary points to a one-hot encoding of the mapped read
    record_seq_dict = {}
    for record in records_to_analyze:
        current_reference_length = reference_dict[record][0]
        delta_max_length = max_reference_length - current_reference_length
        sequence = reference_dict[record][1] + 'N'*delta_max_length
        # Get a dictionary of positional base identities keyed by read id
        base_identities = extract_base_identity_at_coordinates(bam, record, current_reference_length, max_reference_length)
        # One hot encode the sequence string of the reads
        one_hot_reads = {read_name: one_hot_encode(seq) for read_name, seq in base_identities.items()}
        record_seq_dict[record] = (one_hot_reads, sequence)

    ###################################################

    ###################################################
    # Begin iterating over batches
    for batch in range(batches):
        print('{0}: Processing tsvs for batch {1} '.format(time_string(), batch))
        # For the final batch, just take the remaining tsv files
        if batch == batches - 1:
            tsv_batch = tsvs
        # For all other batches, take the next batch of tsvs out of the file queue.    
        else:
            tsv_batch = tsvs[:batch_size]
            tsvs = tsvs[batch_size:]
        print('{0}: tsvs in batch {1} '.format(time_string(), tsv_batch))
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
        for i, tsv in enumerate(tsv_batch):
            print('{0}: Loading sample tsv {1} into dataframe'.format(time_string(), tsv))
            temp_df = pd.read_csv(tsv, sep='\t', header=0)
            for record in records_to_analyze:
                if record not in dict_total.keys():
                    dict_total[record] = {}
                # Only keep the reads aligned to the chromosomes of interest
                print('{0}: Filtering sample dataframe to keep chromosome of interest'.format(time_string()))
                dict_total[record][i] = temp_df[temp_df['chrom'] == record]
                # Only keep the read positions that fall within the region of interest
                print('{0}: Filtering sample dataframe to keep positions falling within region of interest'.format(time_string()))
                current_reference_length = reference_dict[record][0]
                dict_total[record][i] = dict_total[record][i][(current_reference_length > dict_total[record][i]['ref_position']) & (dict_total[record][i]['ref_position']>= 0)]

        # Iterate over dict_total of all the tsv files and extract the modification specific and strand specific dataframes into dictionaries
        for record in dict_total.keys():
            for i in dict_total[record].keys():
                if '6mA' in mods:
                    # Remove Adenine stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(A_stranded_dicts)

                    if record not in dict_a.keys() and record not in dict_a_bottom.keys() and record not in dict_a_top.keys():
                        dict_a[record], dict_a_bottom[record], dict_a_top[record] = {}, {}, {}

                    # get a dictionary of dataframes that only contain methylated adenine positions
                    dict_a[record][i] = dict_total[record][i][dict_total[record][i]['modified_primary_base'] == 'A']
                    print('{}: Successfully created a methyl-adenine dictionary for '.format(time_string()) + str(i))
                    # Stratify the adenine dictionary into two strand specific dictionaries.
                    dict_a_bottom[record][i] = dict_a[record][i][dict_a[record][i]['ref_strand'] == '-']
                    print('{}: Successfully created a minus strand methyl-adenine dictionary for '.format(time_string()) + str(i))
                    dict_a_top[record][i] = dict_a[record][i][dict_a[record][i]['ref_strand'] == '+']
                    print('{}: Successfully created a plus strand methyl-adenine dictionary for '.format(time_string()) + str(i))

                if '5mC' in mods:
                    # Remove Cytosine stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(C_stranded_dicts)

                    if record not in dict_c.keys() and record not in dict_c_bottom.keys() and record not in dict_c_top.keys():
                        dict_c[record], dict_c_bottom[record], dict_c_top[record] = {}, {}, {}

                    # get a dictionary of dataframes that only contain methylated cytosine positions
                    dict_c[record][i] = dict_total[record][i][dict_total[record][i]['modified_primary_base'] == 'C']
                    print('{}: Successfully created a methyl-cytosine dictionary for '.format(time_string()) + str(i))
                    # Stratify the cytosine dictionary into two strand specific dictionaries.
                    dict_c_bottom[record][i] = dict_c[record][i][dict_c[record][i]['ref_strand'] == '-']
                    print('{}: Successfully created a minus strand methyl-cytosine dictionary for '.format(time_string()) + str(i))
                    dict_c_top[record][i] = dict_c[record][i][dict_c[record][i]['ref_strand'] == '+']
                    print('{}: Successfully created a plus strand methyl-cytosine dictionary for '.format(time_string()) + str(i))
                    # In the strand specific dictionaries, only keep positions that are informative for GpC SMF
                
                if '6mA' in mods and '5mC' in mods:
                    # Remove combined stranded dicts from the dicts to skip set
                    dict_to_skip.difference_update(combined_dicts)                
                    # Initialize the sample keys for the combined dictionaries

                    if record not in dict_combined_bottom.keys() and record not in dict_combined_top.keys():
                        dict_combined_bottom[record], dict_combined_top[record]= {}, {}

                    print('{}: Successfully created a minus strand combined methylation dictionary for '.format(time_string()) + str(i))
                    dict_combined_bottom[record][i] = []
                    print('{}: Successfully created a plus strand combined methylation dictionary for '.format(time_string()) + str(i))
                    dict_combined_top[record][i] = []

        # Iterate over the stranded modification dictionaries and replace the dataframes with a dictionary of read names pointing to a list of values from the dataframe
        for i, dict_type in enumerate(dict_list):
            # Only iterate over stranded dictionaries
            if i not in dict_to_skip:
                print('{0}: Extracting methylation states for {1} dictionary'.format(time_string(), sample_types[i]))
                for record in dict_type.keys():
                    # Get the dictionary for the modification type of interest from the reference mapping of interest
                    dict = dict_type[record]
                    print('{0}: Extracting methylation states for {1} dictionary'.format(time_string(), record))
                    # For each sample in a stranded dictionary
                    for sample in dict.keys():
                        print('{0}: Extracting {1} dictionary from record {2} for sample {3}'.format(time_string(), sample_types[i], record, sample))
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
        print('{0}: Converting batch {1} dictionaries to anndata objects'.format(time_string(), batch))
        for i, dict_type in enumerate(dict_list):
            if i not in dict_to_skip:
                # Initialize an hdf5 file for the current modified strand
                adata = None
                print('{0}: Converting {1} dictionary to an anndata object'.format(time_string(), sample_types[i]))
                for record in dict_type.keys():
                    # Get the dictionary for the modification type of interest from the reference mapping of interest
                    dict = dict_type[record]
                    for sample in dict.keys():
                        print('{0}: Converting {1} dictionary for sample {2} to an anndata object'.format(time_string(), sample_types[i], sample))
                        sample = int(sample)
                        final_sample_index = sample + (batch * batch_size)
                        print('{0}: Final sample index for sample: {1}'.format(time_string(), final_sample_index))
                        print('{0}: Converting {1} dictionary for sample {2} to a dataframe'.format(time_string(), sample_types[i], final_sample_index))
                        temp_df = pd.DataFrame.from_dict(dict[sample], orient='index')
                        sorted_index = sorted(temp_df.index)
                        temp_df = temp_df.reindex(sorted_index)
                        X = temp_df.values
                        one_hot_encodings = record_seq_dict[record][0]
                        # Initialize empty DataFrames for each base
                        read_names = list(one_hot_encodings.keys())
                        sequence_length = one_hot_encodings[read_names[0]].shape[0]
                        df_A = pd.DataFrame(np.nan, index=sorted_index, columns=range(sequence_length))
                        df_C = pd.DataFrame(np.nan, index=sorted_index, columns=range(sequence_length))
                        df_G = pd.DataFrame(np.nan, index=sorted_index, columns=range(sequence_length))
                        df_T = pd.DataFrame(np.nan, index=sorted_index, columns=range(sequence_length))
                        df_N = pd.DataFrame(np.nan, index=sorted_index, columns=range(sequence_length))

                        # Iterate through the dictionary and populate the DataFrames
                        for read_name, one_hot_array in one_hot_encodings.items():
                            df_A.loc[read_name] = one_hot_array[:, 0]
                            df_C.loc[read_name] = one_hot_array[:, 1]
                            df_G.loc[read_name] = one_hot_array[:, 2]
                            df_T.loc[read_name] = one_hot_array[:, 3]
                            df_N.loc[read_name] = one_hot_array[:, 4]

                        ohe_df_map = {0: df_A, 1: df_C, 2: df_G, 3: df_T, 4: df_N}

                        print('{0}: Loading {1} dataframe for sample {2} into a temp anndata object'.format(time_string(), sample_types[i], final_sample_index))
                        temp_adata = sc.AnnData(X, dtype=X.dtype)
                        print('{0}: Adding read names and position ids to {1} anndata for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
                        temp_adata.obs_names = temp_df.index
                        temp_adata.obs_names = temp_adata.obs_names.astype(str)
                        temp_adata.var_names = temp_df.columns
                        temp_adata.var_names = temp_adata.var_names.astype(str)
                        print('{0}: Adding final sample id to {1} anndata for sample {2}'.format(time_string(), sample_types[i], final_sample_index))
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
    final_hdf = '{0}_{1}_final_experiment_hdf5.h5ad.gz'.format(date_string, args.experiment_name)
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

    for record in records_to_analyze:
        # Add FASTA sequence to the object
        sequence = record_seq_dict[record][1]
        final_adata.uns[f'{record}_FASTA_sequence'] = sequence
        final_adata.var[f'{record}_FASTA_sequence_base'] = list(sequence)

        # Add consensus sequence of samples mapped to the record to the object
        record_subset = final_adata[final_adata.obs['Reference_chromosome'] == record].copy()
        layer_map, layer_counts = {}, []
        for i, layer in enumerate(record_subset.layers):
            layer_map[i] = layer.split('_')[0]
            layer_counts.append(np.sum(record_subset.layers[layer], axis=0))
        count_array = np.array(layer_counts)
        nucleotide_indexes = np.argmax(count_array, axis=0)
        consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
        final_adata.var[f'{record}_consensus_across_samples'] = consensus_sequence_list
        final_adata.uns[f'{record}_consensus_sequence'] = ''.join(consensus_sequence_list)

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
######################################################################################################