## modkit_extract_to_adata

import concurrent.futures
import gc
from .count_aligned_reads import count_aligned_reads
import pandas as pd
from tqdm import tqdm
import numpy as np

def filter_bam_records(bam, mapping_threshold):
    """Processes a single BAM file, counts reads, and determines records to analyze."""
    aligned_reads_count, unaligned_reads_count, record_counts_dict = count_aligned_reads(bam)
    
    total_reads = aligned_reads_count + unaligned_reads_count
    percent_aligned = (aligned_reads_count * 100 / total_reads) if total_reads > 0 else 0
    print(f'{percent_aligned:.2f}% of reads in {bam} aligned successfully')

    records = []
    for record, (count, percentage) in record_counts_dict.items():
        print(f'{count} reads mapped to reference {record}. This is {percentage*100:.2f}% of all mapped reads in {bam}')
        if percentage >= mapping_threshold:
            records.append(record)
    
    return set(records)

def parallel_filter_bams(bam_path_list, mapping_threshold):
    """Parallel processing for multiple BAM files."""
    records_to_analyze = set()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(filter_bam_records, bam_path_list, [mapping_threshold] * len(bam_path_list))

    # Aggregate results
    for result in results:
        records_to_analyze.update(result)

    print(f'Records to analyze: {records_to_analyze}')
    return records_to_analyze

def process_tsv(tsv, records_to_analyze, reference_dict, sample_index):
    """
    Loads and filters a single TSV file based on chromosome and position criteria.
    """
    temp_df = pd.read_csv(tsv, sep='\t', header=0)
    filtered_records = {}

    for record in records_to_analyze:
        if record not in reference_dict:
            continue
        
        ref_length = reference_dict[record][0]
        filtered_df = temp_df[(temp_df['chrom'] == record) & 
                              (temp_df['ref_position'] >= 0) & 
                              (temp_df['ref_position'] < ref_length)]

        if not filtered_df.empty:
            filtered_records[record] = {sample_index: filtered_df}

    return filtered_records

def parallel_load_tsvs(tsv_batch, records_to_analyze, reference_dict, batch, batch_size, threads=4):
    """
    Loads and filters TSV files in parallel.

    Parameters:
        tsv_batch (list): List of TSV file paths.
        records_to_analyze (list): Chromosome records to analyze.
        reference_dict (dict): Dictionary containing reference lengths.
        batch (int): Current batch number.
        batch_size (int): Total files in the batch.
        threads (int): Number of parallel workers.

    Returns:
        dict: Processed `dict_total` dictionary.
    """
    dict_total = {record: {} for record in records_to_analyze}

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(process_tsv, tsv, records_to_analyze, reference_dict, sample_index): sample_index
            for sample_index, tsv in enumerate(tsv_batch)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), desc=f'Processing batch {batch}', total=batch_size):
            result = future.result()
            for record, sample_data in result.items():
                dict_total[record].update(sample_data)

    return dict_total

def update_dict_to_skip(dict_to_skip, detected_modifications):
    """
    Updates the dict_to_skip set based on the detected modifications.
    
    Parameters:
        dict_to_skip (set): The initial set of dictionary indices to skip.
        detected_modifications (list or set): The modifications (e.g. ['6mA', '5mC']) present.
    
    Returns:
        set: The updated dict_to_skip set.
    """
    # Define which indices correspond to modification-specific or strand-specific dictionaries
    A_stranded_dicts = {2, 3}       # m6A bottom and top strand dictionaries
    C_stranded_dicts = {5, 6}       # 5mC bottom and top strand dictionaries
    combined_dicts   = {7, 8}       # Combined strand dictionaries

    # If '6mA' is present, remove the A_stranded indices from the skip set
    if '6mA' in detected_modifications:
        dict_to_skip -= A_stranded_dicts
    # If '5mC' is present, remove the C_stranded indices from the skip set
    if '5mC' in detected_modifications:
        dict_to_skip -= C_stranded_dicts
    # If both modifications are present, remove the combined indices from the skip set
    if '6mA' in detected_modifications and '5mC' in detected_modifications:
        dict_to_skip -= combined_dicts

    return dict_to_skip

def process_modifications_for_sample(args):
    """
    Processes a single (record, sample) pair to extract modification-specific data.
    
    Parameters:
        args: (record, sample_index, sample_df, mods, max_reference_length)
    
    Returns:
        (record, sample_index, result) where result is a dict with keys:
          'm6A', 'm6A_minus', 'm6A_plus', '5mC', '5mC_minus', '5mC_plus', and
          optionally 'combined_minus' and 'combined_plus' (initialized as empty lists).
    """
    record, sample_index, sample_df, mods, max_reference_length = args
    result = {}
    if '6mA' in mods:
        m6a_df = sample_df[sample_df['modified_primary_base'] == 'A']
        result['m6A'] = m6a_df
        result['m6A_minus'] = m6a_df[m6a_df['ref_strand'] == '-']
        result['m6A_plus'] = m6a_df[m6a_df['ref_strand'] == '+']
        m6a_df = None
        gc.collect()
    if '5mC' in mods:
        m5c_df = sample_df[sample_df['modified_primary_base'] == 'C']
        result['5mC'] = m5c_df
        result['5mC_minus'] = m5c_df[m5c_df['ref_strand'] == '-']
        result['5mC_plus'] = m5c_df[m5c_df['ref_strand'] == '+']
        m5c_df = None
        gc.collect()
    if '6mA' in mods and '5mC' in mods:
        result['combined_minus'] = []
        result['combined_plus'] = []
    return record, sample_index, result

def parallel_process_modifications(dict_total, mods, max_reference_length, threads=4):
    """
    Processes each (record, sample) pair in dict_total in parallel to extract modification-specific data.
    
    Returns:
        processed_results: Dict keyed by record, with sub-dict keyed by sample index and the processed results.
    """
    tasks = []
    for record, sample_dict in dict_total.items():
        for sample_index, sample_df in sample_dict.items():
            tasks.append((record, sample_index, sample_df, mods, max_reference_length))
    processed_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        for record, sample_index, result in tqdm(
                executor.map(process_modifications_for_sample, tasks),
                total=len(tasks),
                desc="Processing modifications"):
            if record not in processed_results:
                processed_results[record] = {}
            processed_results[record][sample_index] = result
    return processed_results

def merge_modification_results(processed_results, mods):
    """
    Merges individual sample results into global dictionaries.
    
    Returns:
        A tuple: (m6A_dict, m6A_minus, m6A_plus, c5m_dict, c5m_minus, c5m_plus, combined_minus, combined_plus)
    """
    m6A_dict = {}
    m6A_minus = {}
    m6A_plus = {}
    c5m_dict = {}
    c5m_minus = {}
    c5m_plus = {}
    combined_minus = {}
    combined_plus = {}
    for record, sample_results in processed_results.items():
        for sample_index, res in sample_results.items():
            if '6mA' in mods:
                if record not in m6A_dict:
                    m6A_dict[record], m6A_minus[record], m6A_plus[record] = {}, {}, {}
                m6A_dict[record][sample_index] = res.get('m6A', pd.DataFrame())
                m6A_minus[record][sample_index] = res.get('m6A_minus', pd.DataFrame())
                m6A_plus[record][sample_index] = res.get('m6A_plus', pd.DataFrame())
            if '5mC' in mods:
                if record not in c5m_dict:
                    c5m_dict[record], c5m_minus[record], c5m_plus[record] = {}, {}, {}
                c5m_dict[record][sample_index] = res.get('5mC', pd.DataFrame())
                c5m_minus[record][sample_index] = res.get('5mC_minus', pd.DataFrame())
                c5m_plus[record][sample_index] = res.get('5mC_plus', pd.DataFrame())
            if '6mA' in mods and '5mC' in mods:
                if record not in combined_minus:
                    combined_minus[record], combined_plus[record] = {}, {}
                combined_minus[record][sample_index] = res.get('combined_minus', [])
                combined_plus[record][sample_index] = res.get('combined_plus', [])
    return (m6A_dict, m6A_minus, m6A_plus,
            c5m_dict, c5m_minus, c5m_plus,
            combined_minus, combined_plus)

def process_stranded_methylation(args):
    """
    Processes a single (dict_index, record, sample) task.
    
    For combined dictionaries (indices 7 or 8), it merges the corresponding A-stranded and C-stranded data.
    For other dictionaries, it converts the DataFrame into a nested dictionary mapping read names to a 
    NumPy methylation array (of float type). Non-numeric values (e.g. '-') are coerced to NaN.
    
    Parameters:
        args: (dict_index, record, sample, dict_list, max_reference_length)
    
    Returns:
        (dict_index, record, sample, processed_data)
    """
    dict_index, record, sample, dict_list, max_reference_length = args
    processed_data = {}
    
    # For combined bottom strand (index 7)
    if dict_index == 7:
        temp_a = dict_list[2][record].get(sample, {}).copy()
        temp_c = dict_list[5][record].get(sample, {}).copy()
        processed_data = {}
        for read in set(temp_a.keys()) | set(temp_c.keys()):
            if read in temp_a:
                # Convert using pd.to_numeric with errors='coerce'
                value_a = pd.to_numeric(np.array(temp_a[read]), errors='coerce')
            else:
                value_a = None
            if read in temp_c:
                value_c = pd.to_numeric(np.array(temp_c[read]), errors='coerce')
            else:
                value_c = None
            if value_a is not None and value_c is not None:
                processed_data[read] = np.where(
                    np.isnan(value_a) & np.isnan(value_c),
                    np.nan,
                    np.nan_to_num(value_a) + np.nan_to_num(value_c)
                )
            elif value_a is not None:
                processed_data[read] = value_a
            elif value_c is not None:
                processed_data[read] = value_c
        del temp_a, temp_c

    # For combined top strand (index 8)
    elif dict_index == 8:
        temp_a = dict_list[3][record].get(sample, {}).copy()
        temp_c = dict_list[6][record].get(sample, {}).copy()
        processed_data = {}
        for read in set(temp_a.keys()) | set(temp_c.keys()):
            if read in temp_a:
                value_a = pd.to_numeric(np.array(temp_a[read]), errors='coerce')
            else:
                value_a = None
            if read in temp_c:
                value_c = pd.to_numeric(np.array(temp_c[read]), errors='coerce')
            else:
                value_c = None
            if value_a is not None and value_c is not None:
                processed_data[read] = np.where(
                    np.isnan(value_a) & np.isnan(value_c),
                    np.nan,
                    np.nan_to_num(value_a) + np.nan_to_num(value_c)
                )
            elif value_a is not None:
                processed_data[read] = value_a
            elif value_c is not None:
                processed_data[read] = value_c
        del temp_a, temp_c

    # For all other dictionaries
    else:
        # current_data is a DataFrame
        temp_df = dict_list[dict_index][record][sample]
        processed_data = {}
        # Extract columns and convert probabilities to float (coercing errors)
        read_ids = temp_df['read_id'].values
        positions = temp_df['ref_position'].values
        call_codes = temp_df['call_code'].values
        probabilities = pd.to_numeric(temp_df['call_prob'].values, errors='coerce')
        
        modified_codes = {'a', 'h', 'm'}
        canonical_codes = {'-'}
        
        # Compute methylation probabilities (vectorized)
        methylation_prob = np.full(probabilities.shape, np.nan, dtype=float)
        methylation_prob[np.isin(call_codes, list(modified_codes))] = probabilities[np.isin(call_codes, list(modified_codes))]
        methylation_prob[np.isin(call_codes, list(canonical_codes))] = 1 - probabilities[np.isin(call_codes, list(canonical_codes))]
        
        # Preallocate storage for each unique read
        unique_reads = np.unique(read_ids)
        for read in unique_reads:
            processed_data[read] = np.full(max_reference_length, np.nan, dtype=float)
        
        # Assign values efficiently
        for i in range(len(read_ids)):
            read = read_ids[i]
            pos = positions[i]
            prob = methylation_prob[i]
            processed_data[read][pos] = prob

    gc.collect()
    return dict_index, record, sample, processed_data

def parallel_extract_stranded_methylation(dict_list, dict_to_skip, max_reference_length, threads=4):
    """
    Processes all (dict_index, record, sample) tasks in dict_list (excluding indices in dict_to_skip) in parallel.
    
    Returns:
        Updated dict_list with processed (nested) dictionaries.
    """
    tasks = []
    for dict_index, current_dict in enumerate(dict_list):
        if dict_index not in dict_to_skip:
            for record in current_dict.keys():
                for sample in current_dict[record].keys():
                    tasks.append((dict_index, record, sample, dict_list, max_reference_length))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        for dict_index, record, sample, processed_data in tqdm(
            executor.map(process_stranded_methylation, tasks),
            total=len(tasks),
            desc="Extracting stranded methylation states"
        ):
            dict_list[dict_index][record][sample] = processed_data
    return dict_list

def modkit_extract_to_adata(fasta, bam_dir, mapping_threshold, experiment_name, mods, batch_size, mod_tsv_dir, delete_batch_hdfs=False, threads=None):
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
        final_adata_path (str): Path to the final adata
    """
    ###################################################
    # Package imports
    from .. import readwrite
    from .get_native_references import get_native_references
    from .extract_base_identities import extract_base_identities
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
    existing_h5s =  os.listdir(h5_dir)
    existing_h5s = [h5 for h5 in existing_h5s if '.h5ad.gz' in h5]
    final_hdf = f'{experiment_name}_final_experiment_hdf5.h5ad'    
    final_adata_path = os.path.join(h5_dir, final_hdf)

    if os.path.exists(f"{final_adata_path}.gz"):
        print(f'{final_adata_path}.gz already exists. Using existing adata')
        return f"{final_adata_path}.gz"
    
    elif os.path.exists(f"{final_adata_path}"):
        print(f'{final_adata_path} already exists. Using existing adata')
        return final_adata_path
    
    # Filter file names that contain the search string in their filename and keep them in a list
    tsvs = [tsv for tsv in tsv_files if 'extract.tsv' in tsv and 'unclassified' not in tsv]
    bams = [bam for bam in bam_files if '.bam' in bam and '.bai' not in bam and 'unclassified' not in bam]
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
    records_to_analyze = parallel_filter_bams(bam_path_list, mapping_threshold)

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
    bam_record_save = os.path.join(tmp_dir, 'tmp_file_dict.h5ad')
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
                fwd_ohe_files = ohe_batching(fwd_base_identities, tmp_dir, record, f"{bami}_fwd",batch_size=100000, threads=threads)
                rev_ohe_files = ohe_batching(rev_base_identities, tmp_dir, record, f"{bami}_rev",batch_size=100000, threads=threads)
                bam_record_ohe_files[f'{bami}_{record}'] = fwd_ohe_files + rev_ohe_files
                del fwd_base_identities, rev_base_identities
        # Save out the ohe file paths
        X = np.random.rand(1, 1)
        tmp_ad = ad.AnnData(X=X, uns=bam_record_ohe_files) 
        tmp_ad.write_h5ad(bam_record_save)
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
            # # Initialize dictionaries and place them in a list
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

            # # Step 1):Load the dict_total dictionary with all of the batch tsv files as dataframes.
            dict_total = parallel_load_tsvs(tsv_batch, records_to_analyze, reference_dict, batch, batch_size=len(tsv_batch), threads=threads)

            # # Step 2: Extract modification-specific data (per (record,sample)) in parallel
            # processed_mod_results = parallel_process_modifications(dict_total, mods, max_reference_length, threads=threads or 4)
            # (m6A_dict, m6A_minus_strand, m6A_plus_strand,
            # c5m_dict, c5m_minus_strand, c5m_plus_strand,
            # combined_minus_strand, combined_plus_strand) = merge_modification_results(processed_mod_results, mods)

            # # Create dict_list with the desired ordering:
            # # 0: dict_total, 1: m6A, 2: m6A_minus, 3: m6A_plus, 4: 5mC, 5: 5mC_minus, 6: 5mC_plus, 7: combined_minus, 8: combined_plus
            # dict_list = [dict_total, m6A_dict, m6A_minus_strand, m6A_plus_strand,
            #             c5m_dict, c5m_minus_strand, c5m_plus_strand,
            #             combined_minus_strand, combined_plus_strand]

            # # Initialize dict_to_skip (default skip all mod-specific indices)
            # dict_to_skip = set([0, 1, 4, 7, 8, 2, 3, 5, 6])
            # # Update dict_to_skip based on modifications present in mods
            # dict_to_skip = update_dict_to_skip(dict_to_skip, mods)

            # # Step 3: Process stranded methylation data in parallel
            # dict_list = parallel_extract_stranded_methylation(dict_list, dict_to_skip, max_reference_length, threads=threads or 4)

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
                                        mod_strand_record_sample_dict[sample][read] = np.where(np.isnan(temp_a_dict[read]) & np.isnan(temp_c_dict[read]), np.nan, np.nan_to_num(temp_a_dict[read]) + np.nan_to_num(temp_c_dict[read]))
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
                                        mod_strand_record_sample_dict[sample][read] = np.where(np.isnan(temp_a_dict[read]) & np.isnan(temp_c_dict[read]), np.nan, np.nan_to_num(temp_a_dict[read]) + np.nan_to_num(temp_c_dict[read]))
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

                                # Get relevant columns as NumPy arrays
                                read_ids = temp_df['read_id'].values
                                positions = temp_df['ref_position'].values
                                call_codes = temp_df['call_code'].values
                                probabilities = temp_df['call_prob'].values

                                # Define valid call code categories
                                modified_codes = {'a', 'h', 'm'}
                                canonical_codes = {'-'}

                                # Vectorized methylation calculation with NaN for other codes
                                methylation_prob = np.full_like(probabilities, np.nan)  # Default all to NaN
                                methylation_prob[np.isin(call_codes, list(modified_codes))] = probabilities[np.isin(call_codes, list(modified_codes))]
                                methylation_prob[np.isin(call_codes, list(canonical_codes))] = 1 - probabilities[np.isin(call_codes, list(canonical_codes))]

                                # Find unique reads
                                unique_reads = np.unique(read_ids)
                                # Preallocate storage for each read
                                for read in unique_reads:
                                    mod_strand_record_sample_dict[sample][read] = np.full(max_reference_length, np.nan)

                                # Efficient NumPy indexing to assign values
                                for i in range(len(read_ids)):
                                    read = read_ids[i]
                                    pos = positions[i]
                                    prob = methylation_prob[i]
                                    
                                    # Assign methylation probability
                                    mod_strand_record_sample_dict[sample][read][pos] = prob


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
                            dataset, strand = sample_types[dict_index].split('_')[:2]

                            print('{0}: Loading {1} dataframe for sample {2} into a temp anndata object'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                            temp_adata = ad.AnnData(X)
                            if temp_adata.shape[0] > 0:
                                print('{0}: Adding read names and position ids to {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                temp_adata.obs_names = temp_df.index
                                temp_adata.obs_names = temp_adata.obs_names.astype(str)
                                temp_adata.var_names = temp_df.columns
                                temp_adata.var_names = temp_adata.var_names.astype(str)
                                print('{0}: Adding {1} anndata for sample {2}'.format(readwrite.time_string(), sample_types[dict_index], final_sample_index))
                                temp_adata.obs['Sample'] = [str(final_sample_index)] * len(temp_adata)
                                temp_adata.obs['Reference'] = [f'{record}'] * len(temp_adata)
                                temp_adata.obs['Strand'] = [strand] * len(temp_adata)
                                temp_adata.obs['Dataset'] = [dataset] * len(temp_adata)
                                temp_adata.obs['Reference_dataset_strand'] = [f'{record}_{dataset}_{strand}'] * len(temp_adata)
                                temp_adata.obs['Reference_strand'] = [f'{record}_{strand}'] * len(temp_adata)
                                
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
                                
                                # Initialize NumPy arrays
                                sequence_length = one_hot_reads[read_names[0]].reshape(n_rows_OHE, -1).shape[1]
                                df_A = np.zeros((len(sorted_index), sequence_length), dtype=int)
                                df_C = np.zeros((len(sorted_index), sequence_length), dtype=int)
                                df_G = np.zeros((len(sorted_index), sequence_length), dtype=int)
                                df_T = np.zeros((len(sorted_index), sequence_length), dtype=int)
                                df_N = np.zeros((len(sorted_index), sequence_length), dtype=int)

                                # Process one-hot data into dictionaries
                                dict_A, dict_C, dict_G, dict_T, dict_N = {}, {}, {}, {}, {}
                                for read_name, one_hot_array in one_hot_reads.items():
                                    one_hot_array = one_hot_array.reshape(n_rows_OHE, -1)
                                    dict_A[read_name] = one_hot_array[0, :]
                                    dict_C[read_name] = one_hot_array[1, :]
                                    dict_G[read_name] = one_hot_array[2, :]
                                    dict_T[read_name] = one_hot_array[3, :]
                                    dict_N[read_name] = one_hot_array[4, :]

                                del one_hot_reads
                                gc.collect()

                                # Fill the arrays
                                for j, read_name in tqdm(enumerate(sorted_index), desc='Loading dataframes of OHE reads', total=len(sorted_index)):
                                    df_A[j, :] = dict_A[read_name]
                                    df_C[j, :] = dict_C[read_name]
                                    df_G[j, :] = dict_G[read_name]
                                    df_T[j, :] = dict_T[read_name]
                                    df_N[j, :] = dict_N[read_name]

                                del dict_A, dict_C, dict_G, dict_T, dict_N
                                gc.collect()

                                # Store the results in AnnData layers
                                ohe_df_map = {0: df_A, 1: df_C, 2: df_G, 3: df_T, 4: df_N}
                                for j, base in enumerate(['A', 'C', 'G', 'T', 'N']):
                                    temp_adata.layers[f'{base}_binary_encoding'] = ohe_df_map[j]
                                    ohe_df_map[j] = None  # Reassign pointer for memory usage purposes

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

                    try:
                        print('{0}: Writing {1} anndata out as a hdf5 file'.format(readwrite.time_string(), sample_types[dict_index]))
                        adata.write_h5ad('{0}_{1}_{2}_SMF_binarized_sample_hdf5.h5ad.gz'.format(readwrite.date_string(), batch, sample_types[dict_index]), compression='gzip')
                    except:
                        print(f"Skipping writing anndata for sample")

            # Delete the batch dictionaries from memory
            del dict_list, adata
            gc.collect()

    # Iterate over all of the batched hdf5 files and concatenate them.
    os.chdir(h5_dir)
    files = os.listdir(h5_dir)        
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if 'hdf5.h5ad' in hdf and hdf != final_hdf]
    combined_hdfs = [hdf for hdf in hdfs if "combined" in hdf]
    if len(combined_hdfs) > 0:
        hdfs = combined_hdfs
    else:
        pass
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    hdf_paths = [os.path.join(h5_dir, hd5) for hd5 in hdfs]
    final_adata = None
    for hdf_index, hdf in enumerate(hdf_paths):
        print('{0}: Reading in {1} hdf5 file'.format(readwrite.time_string(), hdfs[hdf_index]))
        temp_adata = ad.read_h5ad(hdf)
        if final_adata:
            print('{0}: Concatenating final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdfs[hdf_index]))
            final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
        else:
            print('{0}: Initializing final adata object with {1} hdf5 file'.format(readwrite.time_string(), hdfs[hdf_index]))
            final_adata = temp_adata
        del temp_adata

    # Set obs columns to type 'category'
    for col in final_adata.obs.columns:
        final_adata.obs[col] = final_adata.obs[col].astype('category')

    ohe_bases = ['A', 'C', 'G', 'T'] # ignore N bases for consensus
    ohe_layers = [f"{ohe_base}_binary_encoding" for ohe_base in ohe_bases]
    for record in records_to_analyze:
        # Add FASTA sequence to the object
        sequence = record_seq_dict[record][0]
        complement = record_seq_dict[record][1]
        final_adata.var[f'{record}_top_strand_FASTA_base'] = list(sequence)
        final_adata.var[f'{record}_bottom_strand_FASTA_base'] = list(complement)
        final_adata.uns[f'{record}_FASTA_sequence'] = sequence
        # Add consensus sequence of samples mapped to the record to the object
        record_subset = final_adata[final_adata.obs['Reference'] == record]
        for strand in record_subset.obs['Strand'].cat.categories:
            strand_subset = record_subset[record_subset.obs['Strand'] == strand]
            for mapping_dir in strand_subset.obs['Read_mapping_direction'].cat.categories:
                mapping_dir_subset = strand_subset[strand_subset.obs['Read_mapping_direction'] == mapping_dir]
                layer_map, layer_counts = {}, []
                for i, layer in enumerate(ohe_layers):
                    layer_map[i] = layer.split('_')[0]
                    layer_counts.append(np.sum(mapping_dir_subset.layers[layer], axis=0))
                count_array = np.array(layer_counts)
                nucleotide_indexes = np.argmax(count_array, axis=0)
                consensus_sequence_list = [layer_map[i] for i in nucleotide_indexes]
                final_adata.var[f'{record}_{strand}_{mapping_dir}_consensus_sequence_from_all_samples'] = consensus_sequence_list

    #final_adata.write_h5ad(final_adata_path)

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

    return final_adata, final_adata_path