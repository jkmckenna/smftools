## converted_BAM_to_adata

def converted_BAM_to_adata(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix):
    """
    A wrapper function to take converted aligned_sorted_split BAM files and format the data into an anndata object.

    Parameters:
        converted_FASTA (str): A string representing the file path to the converted FASTA reference.
        split_dir (str): A string representing the file path to the directory containing the converted aligned_sorted_split BAM files.
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        conversion_types (list): A list of strings of the conversion types to use in the analysis.
        bam_suffix (str): The suffix to use for the BAM file.
    
    Returns:
        None
        Outputs a single gzipped adata object for the experiment.
    """
    from .. import readwrite
    from .binarize_converted_base_identities import binarize_converted_base_identities
    from .find_conversion_sites import find_conversion_sites
    from .count_aligned_reads import count_aligned_reads
    from .extract_base_identities import extract_base_identities
    from .make_dirs import make_dirs
    from .ohe_batching import ohe_batching
    import pandas as pd
    import numpy as np
    import anndata as ad
    import os
    from tqdm import tqdm
    import gc
    
    ##########################################################################################
    ## Get file paths and make necessary directories. ##
    # Get all of the input BAM files
    files = os.listdir(split_dir)
    # Make output dir
    parent_dir = os.path.dirname(split_dir)
    h5_dir = os.path.join(parent_dir, 'h5ads')
    tmp_dir = os.path.join(parent_dir, 'tmp')
    make_dirs([h5_dir, tmp_dir])
    # Filter file names that contain the search string in their filename and keep them in a list
    bams = [bam for bam in files if bam_suffix in bam and '.bai' not in bam]
    # Sort file list by names and print the list of file names
    bams.sort()
    bam_path_list = [os.path.join(split_dir, bam) for bam in bams]
    print(f'Found the following BAMS: {bams}')
    final_adata = None
    ##########################################################################################

    ##########################################################################################

    ## need to fix this section
    # Make a dictionary, keyed by modification type, that points to another dictionary of unconverted_record_ids. This points to a list of: 1) record length, 2) top strand conversion coordinates, 3) bottom strand conversion coordinates, 4) sequence string unconverted , 5) Complement sequence unconverted
    modification_dict = {}
    # Init a dict to be keyed by FASTA record that points to the sequence string of the unconverted record
    record_FASTA_dict = {}
    # While populating the dictionary, also extract the longest sequence record in the input references
    max_reference_length = 0
    for conversion_type in conversion_types:
        # Points to a list containing: 1) sequence length of the record, 2) top strand coordinate list, 3) bottom strand coorinate list, 4) sequence string unconverted , 5) Complement sequence unconverted
        modification_dict[conversion_type] = find_conversion_sites(converted_FASTA, conversion_type, conversion_types)
        # Get the max reference length
        for record in modification_dict[conversion_type].keys():
            if modification_dict[conversion_type][record][0] > max_reference_length:
                max_reference_length = modification_dict[conversion_type][record][0]

            mod_type, strand = record.split('_')[-2:]

            chromosome = record.split('_{0}_{1}'.format(mod_type, strand))[0]
            unconverted_chromosome_name = f'{chromosome}_{conversion_types[0]}_top'
            current_reference_length = modification_dict[mod_type][unconverted_chromosome_name][0]
            delta_max_length = max_reference_length - current_reference_length
            sequence = modification_dict[mod_type][unconverted_chromosome_name][3] + 'N'*delta_max_length
            complement = modification_dict[mod_type][unconverted_chromosome_name][4] + 'N'*delta_max_length
            record_FASTA_dict[record] = [sequence, complement, chromosome, unconverted_chromosome_name, current_reference_length, delta_max_length, conversion_type, strand]
    ##########################################################################################

    ##########################################################################################
    bam_alignment_stats_dict = {}
    records_to_analyze = []
    for bam_index, bam in enumerate(bam_path_list):
        bam_alignment_stats_dict[bam_index] = {}
        # look at aligned read proportions in the bam
        aligned_reads_count, unaligned_reads_count, record_counts = count_aligned_reads(bam)
        percent_aligned = aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)
        print(f'{percent_aligned} percent of total reads in {bams[bam_index]} aligned successfully')
        bam_alignment_stats_dict[bam_index]['Total'] = (aligned_reads_count, percent_aligned)
        # Iterate over converted reference strands and decide which to use in the analysis based on the mapping_threshold
        for record in record_counts:
            print(f'{record_counts[record][0]} reads mapped to reference record {record}. This is {record_counts[record][1]*100} percent of all mapped reads in the sample.')
            if record_counts[record][1] >= mapping_threshold:
                records_to_analyze.append(record)
                bam_alignment_stats_dict[bam_index]
                bam_alignment_stats_dict[bam_index][record] = (record_counts[record][0], record_counts[record][1]*100)
    records_to_analyze = set(records_to_analyze)
    ##########################################################################################

    ##########################################################################################
    # One hot encode read sequences and write them out into the tmp_dir as h5ad files. 
    # Save the file paths in the bam_record_ohe_files dict.
    bam_record_ohe_files = {}

    # Iterate over split bams
    for bam_index, bam in enumerate(bam_path_list):
        # Iterate over references to process
        for record in records_to_analyze:
            unconverted_record_name = "_".join(record.split('_')[:-2]) + '_unconverted_top'
            sample = bams[bam_index].split(sep=bam_suffix)[0]
            chromosome = record_FASTA_dict[unconverted_record_name][2]
            current_reference_length = record_FASTA_dict[unconverted_record_name][4]
            mod_type = record_FASTA_dict[unconverted_record_name][6]
            strand = record_FASTA_dict[unconverted_record_name][7]
            
            # Extract the base identities of reads aligned to the record
            fwd_base_identities, rev_base_identities = extract_base_identities(bam, record, range(current_reference_length), max_reference_length)

            # binarize the dictionary of positional identities
            print(f'Binarizing base identities')
            fwd_binarized_base_identities = binarize_converted_base_identities(fwd_base_identities, strand, mod_type) 
            rev_binarized_base_identities = binarize_converted_base_identities(rev_base_identities, strand, mod_type)
            merged_binarized_base_identities = {**fwd_binarized_base_identities, **rev_binarized_base_identities}
            # converts the base identity dictionary to a dataframe.
            binarized_base_identities_df = pd.DataFrame.from_dict(merged_binarized_base_identities, orient='index') 
            sorted_index = sorted(binarized_base_identities_df.index)
            binarized_base_identities_df = binarized_base_identities_df.reindex(sorted_index)

            # Load an anndata object with the sample data
            X = binarized_base_identities_df.values
            adata = ad.AnnData(X, dtype=X.dtype)
            if adata.shape[0] > 0:
                adata.obs_names = binarized_base_identities_df.index.astype(str)
                adata.var_names = binarized_base_identities_df.columns.astype(str)
                adata.obs['Sample'] = [sample] * len(adata)
                adata.obs['Strand'] = [strand] * len(adata)
                adata.obs['Dataset'] = [mod_type] * len(adata)
                adata.obs['Reference'] = [record] * len(adata)
                adata.obs['Reference_chromosome'] = [chromosome] * len(adata)

                read_mapping_direction = []
                for read_id in adata.obs_names:
                    if read_id in fwd_base_identities.keys():
                        read_mapping_direction.append('fwd')
                    elif read_id in rev_base_identities.keys():
                        read_mapping_direction.append('rev')
                    else:
                        read_mapping_direction.append('unk')

                adata.obs['Read_mapping_direction'] = read_mapping_direction

                # One hot encode the sequence string of the reads
                fwd_ohe_files = ohe_batching(fwd_base_identities, tmp_dir, record, f"{bam_index}_fwd",batch_size=100000)
                rev_ohe_files = ohe_batching(rev_base_identities, tmp_dir, record, f"{bam_index}_rev",batch_size=100000)
                bam_record_ohe_files[f'{bam_index}_{record}'] = fwd_ohe_files + rev_ohe_files
                del fwd_base_identities, rev_base_identities

                one_hot_reads = {}
                n_rows_OHE = 5
                for ohe_file in tqdm(bam_record_ohe_files[f'{bam_index}_{record}'], desc="Reading in OHE reads"):
                    tmp_ohe_dict = ad.read_h5ad(ohe_file).uns
                    one_hot_reads.update(tmp_ohe_dict)
                    del tmp_ohe_dict

                read_names = list(one_hot_reads.keys())
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
                    adata.layers[f'{base}_binary_encoding'] = ohe_df_map[j].values 
                    ohe_df_map[j] = None # Reassign pointer for memory usage purposes

                if final_adata:
                    if adata.shape[0] > 0:
                        final_adata = ad.concat([final_adata, adata], join='outer', index_unique=None)
                    else:
                        print(f"{sample} did not have any mapped reads on {record}, omiting from final adata")
                else:
                    if adata.shape[0] > 0:
                        final_adata = adata
                    else:
                        print(f"{sample} did not have any mapped reads on {record}, omiting from final adata")

            else:
                print(f"{sample} did not have any mapped reads on {record}, omiting from final adata")

    # Set obs columns to type 'category'
    for col in final_adata.obs.columns:
        final_adata.obs[col] = final_adata.obs[col].astype('category')

    for record in records_to_analyze:
        unconverted_record_name = "_".join(record.split('_')[:-2]) + '_unconverted_top'
        sequence = record_FASTA_dict[unconverted_record_name][0]
        complement = record_FASTA_dict[unconverted_record_name][1]
        chromosome = record_FASTA_dict[unconverted_record_name][2]
        final_adata.var[f'{chromosome}_unconverted_top_strand_FASTA_base'] = list(sequence)
        final_adata.var[f'{chromosome}_unconverted_bottom_strand_FASTA_base'] = list(complement)
        final_adata.uns[f'{record}_FASTA_sequence'] = sequence

    ######################################################################################################

    ######################################################################################################
    ## Export the final adata object
    final_output = os.path.join(h5_dir, f'{readwrite.date_string()}_{experiment_name}.h5ad.gz')
    final_adata.write_h5ad(final_output, compression='gzip')