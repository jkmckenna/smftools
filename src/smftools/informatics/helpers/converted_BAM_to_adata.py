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
    from .one_hot_encode import one_hot_encode
    from .ohe_batching import ohe_batching
    import pandas as pd
    import numpy as np
    import anndata as ad
    import os
    from tqdm import tqdm
    
    # Get all of the input BAM files
    files = os.listdir(split_dir)
    # Make output dir
    parent_dir = os.path.dirname(split_dir)
    h5_dir = os.path.join(parent_dir, 'h5ads')
    tmp_dir = os.path.join(parent_dir, 'tmp')
    make_dirs([h5_dir, tmp_dir])
    # Change directory to the BAM directory
    os.chdir(split_dir)
    # Filter file names that contain the search string in their filename and keep them in a list
    bams = [bam for bam in files if bam_suffix in bam and '.bai' not in bam]
    # Sort file list by names and print the list of file names
    bams.sort()
    print(f'Found the following BAMS: {bams}')
    final_adata = None

    # Make a dictionary, keyed by modification type, that points to another dictionary of unconverted_record_ids. This points to a list of: 1) record length, 2) top strand conversion coordinates, 3) bottom strand conversion coordinates, 4) record sequence
    modification_dict = {}
    # While populating the dictionary, also extract the longest sequence record in the input references
    max_reference_length = 0
    for conversion_type in conversion_types:
        # Points to a list containing: 1) sequence length of the record, 2) top strand coordinate list, 3) bottom strand coorinate list, 4) sequence string unconverted , 5) Complement sequence unconverted
        modification_dict[conversion_type] = find_conversion_sites(converted_FASTA, conversion_type, conversion_types)
        # Get the max reference length
        for record in modification_dict[conversion_type].keys():
            if modification_dict[conversion_type][record][0] > max_reference_length:
                max_reference_length = modification_dict[conversion_type][record][0]

    # Init a dict to be keyed by FASTA record that points to the sequence string of the unconverted record
    record_FASTA_dict = {}

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
        for record in records_to_analyze:
            mod_type, strand = record.split('_')[-2:]
            if strand == 'top':
                strand_index = 1
            elif strand == 'bottom':
                strand_index = 2

            chromosome = record.split('_{0}_{1}'.format(mod_type, strand))[0]
            unconverted_chromosome_name = f'{chromosome}_{conversion_types[0]}_top'
            positions = modification_dict[mod_type][unconverted_chromosome_name][strand_index]
            current_reference_length = modification_dict[mod_type][unconverted_chromosome_name][0]
            delta_max_length = max_reference_length - current_reference_length
            sequence = modification_dict[mod_type][unconverted_chromosome_name][3] + 'N'*delta_max_length
            complement = modification_dict[mod_type][unconverted_chromosome_name][4] + 'N'*delta_max_length
            record_FASTA_dict[f'{record}'] = [sequence, complement]
            #print(f'Chromosome: {chromosome}\nUnconverted Sequence: {sequence}')

            # Get a dictionary of positional identities keyed by read id
            print(f'Extracting base identities of target positions')
            target_base_identities = extract_base_identities(bam, record, positions, max_reference_length) 
            # binarize the dictionary of positional identities
            print(f'Binarizing base identities of target positions')
            binarized_base_identities = binarize_converted_base_identities(target_base_identities, strand, mod_type) 
            # converts the base identity dictionary to a dataframe.
            binarized_base_identities_df = pd.DataFrame.from_dict(binarized_base_identities, orient='index') 
            sorted_index = sorted(binarized_base_identities_df.index)
            binarized_base_identities_df = binarized_base_identities_df.reindex(sorted_index)
            # Get the sequence string of every read
            print(f'Extracting base identities of all positions in each read')
            all_base_identities = extract_base_identities(bam, record, range(current_reference_length), max_reference_length)
            # One hot encode the sequence string of the reads
            print(f'One hot encoding base identities of all positions in each read')
            # One hot encode the sequence string of the reads
            ohe_files = ohe_batching(all_base_identities, tmp_dir, record, batch_size=1000)
            del all_base_identities
            one_hot_reads = {}
            n_rows_OHE = 5
            for ohe_file in tqdm(ohe_files, desc="Reading in OHE reads"):
                one_hot_reads.update(np.load(ohe_file))

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
                if adata.shape[0] > 0:
                    final_adata = ad.concat([final_adata, adata], join='outer', index_unique=None)
                else:
                    print(f"{sample} did not have any mapped reads on {record}, omiting from final adata")
            else:
                if adata.shape[0] > 0:
                    final_adata = adata
                else:
                    print(f"{sample} did not have any mapped reads on {record}, omiting from final adata")

    # Set obs columns to type 'category'
    for col in final_adata.obs.columns:
        final_adata.obs[col] = final_adata.obs[col].astype('category')

    for record in record_FASTA_dict.keys():
        chromosome = record.split('_')[0]
        sequence = record_FASTA_dict[record][0]
        complement = record_FASTA_dict[record][1]
        final_adata.var[f'{chromosome}_unconverted_top_strand_FASTA_sequence'] = list(sequence)
        final_adata.var[f'{chromosome}_unconverted_bottom_strand_FASTA_sequence'] = list(complement)

    ######################################################################################################

    ######################################################################################################
    ## Export the final adata object
    final_adata.write_h5ad('{0}_{1}.h5ad.gz'.format(readwrite.date_string(), experiment_name), compression='gzip')