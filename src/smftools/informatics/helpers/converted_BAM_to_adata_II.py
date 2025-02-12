import numpy as np
import time
import os
import gc
import pandas as pd
import anndata as ad
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager, Lock, current_process, Pool
import traceback
import gzip
import torch

from .. import readwrite
from .binarize_converted_base_identities import binarize_converted_base_identities
from .find_conversion_sites import find_conversion_sites
from .count_aligned_reads import count_aligned_reads
from .extract_base_identities import extract_base_identities
from .make_dirs import make_dirs
from .ohe_batching import ohe_batching

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)

def converted_BAM_to_adata_II(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix, device='cpu', num_threads=8):
    """
    Converts BAM files into an AnnData object by binarizing modified base identities.

    Parameters:
        converted_FASTA (str): Path to the converted FASTA reference.
        split_dir (str): Directory containing converted BAM files.
        mapping_threshold (float): Minimum fraction of aligned reads required for inclusion.
        experiment_name (str): Name for the output AnnData object.
        conversion_types (list): List of modification types (e.g., ['unconverted', '5mC', '6mA']).
        bam_suffix (str): File suffix for BAM files.
        num_threads (int): Number of parallel processing threads.

    Returns:
        str: Path to the final AnnData object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    ## Set Up Directories and File Paths
    parent_dir = os.path.dirname(split_dir)
    h5_dir = os.path.join(parent_dir, 'h5ads')
    tmp_dir = os.path.join(parent_dir, 'tmp')
    final_adata_path = os.path.join(h5_dir, f'{experiment_name}_{os.path.basename(split_dir)}.h5ad.gz')

    if os.path.exists(final_adata_path):
        print(f"{final_adata_path} already exists. Using existing AnnData object.")
        return final_adata_path

    make_dirs([h5_dir, tmp_dir])

    ## Get BAM Files ##
    bam_files = [f for f in os.listdir(split_dir) if f.endswith(bam_suffix) and not f.endswith('.bai') and 'unclassified' not in f]
    bam_files.sort()
    bam_path_list = [os.path.join(split_dir, f) for f in bam_files]
    print(f"Found {len(bam_files)} BAM files: {bam_files}")

    ## Process Conversion Sites
    max_reference_length, record_FASTA_dict, chromosome_FASTA_dict = process_conversion_sites(converted_FASTA, conversion_types)

    ## Filter BAM Files by Mapping Threshold
    records_to_analyze = filter_bams_by_mapping_threshold(bam_path_list, bam_files, mapping_threshold)

    ## Process BAMs in Parallel
    final_adata = process_bams_parallel(bam_path_list, records_to_analyze, record_FASTA_dict, tmp_dir, h5_dir, num_threads, max_reference_length, device)

    for chromosome, [seq, comp] in chromosome_FASTA_dict.items():
        final_adata.var[f'{chromosome}_top_strand_FASTA_base'] = list(seq)
        final_adata.var[f'{chromosome}_bottom_strand_FASTA_base'] = list(comp)
        final_adata.uns[f'{chromosome}_FASTA_sequence'] = seq

    ## Save Final AnnData
    # print(f"Saving AnnData to {final_adata_path}")
    # final_adata.write_h5ad(final_adata_path, compression='gzip')
    return final_adata, final_adata_path


def process_conversion_sites(converted_FASTA, conversion_types):
    """
    Extracts conversion sites and determines the max reference length.

    Parameters:
        converted_FASTA (str): Path to the converted reference FASTA.
        conversion_types (list): List of modification types (e.g., ['unconverted', '5mC', '6mA']).

    Returns:
        max_reference_length (int): The length of the longest sequence.
        record_FASTA_dict (dict): Dictionary of sequence information for **both converted & unconverted** records.
    """
    modification_dict = {}
    record_FASTA_dict = {}
    chromosome_FASTA_dict = {}
    max_reference_length = 0
    unconverted = conversion_types[0]
    conversions = conversion_types[1:]

    # Process the unconverted sequence once
    modification_dict[unconverted] = find_conversion_sites(converted_FASTA, unconverted, conversion_types)
    # Above points to record_dict[record.id] = [sequence_length, [], [], sequence, complement] with only unconverted record.id keys

    # Get **max sequence length** from unconverted records
    max_reference_length = max(values[0] for values in modification_dict[unconverted].values())

    # Add **unconverted records** to `record_FASTA_dict`
    for record, values in modification_dict[unconverted].items():
        sequence_length, top_coords, bottom_coords, sequence, complement = values
        chromosome = record.replace(f"_{unconverted}_top", "")

        # Store **original sequence**
        record_FASTA_dict[record] = [
            sequence + "N" * (max_reference_length - sequence_length),
            complement + "N" * (max_reference_length - sequence_length),
            chromosome, record, sequence_length, max_reference_length - sequence_length, unconverted, "top"
        ]

        if chromosome not in chromosome_FASTA_dict:
            chromosome_FASTA_dict[chromosome] = [sequence + "N" * (max_reference_length - sequence_length), complement + "N" * (max_reference_length - sequence_length)]

    # Process converted records
    for conversion in conversions:
        modification_dict[conversion] = find_conversion_sites(converted_FASTA, conversion, conversion_types)
        # Above points to record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence, complement] with only unconverted record.id keys

        for record, values in modification_dict[conversion].items():
            sequence_length, top_coords, bottom_coords, sequence, complement = values
            chromosome = record.split(f"_{unconverted}_")[0]  # Extract chromosome name

            # Add **both strands** for converted records
            for strand in ["top", "bottom"]:
                converted_name = f"{chromosome}_{conversion}_{strand}"
                unconverted_name = f"{chromosome}_{unconverted}_top"

                record_FASTA_dict[converted_name] = [
                    sequence + "N" * (max_reference_length - sequence_length),
                    complement + "N" * (max_reference_length - sequence_length),
                    chromosome, unconverted_name, sequence_length, 
                    max_reference_length - sequence_length, conversion, strand
                ]

    print("Updated record_FASTA_dict Keys:", list(record_FASTA_dict.keys()))
    return max_reference_length, record_FASTA_dict, chromosome_FASTA_dict


def filter_bams_by_mapping_threshold(bam_path_list, bam_files, mapping_threshold):
    """Filters BAM files based on mapping threshold."""
    records_to_analyze = set()

    for i, bam in enumerate(bam_path_list):
        aligned_reads, unaligned_reads, record_counts = count_aligned_reads(bam)
        aligned_percent = aligned_reads * 100 / (aligned_reads + unaligned_reads)
        print(f"{aligned_percent:.2f}% of reads in {bam_files[i]} aligned successfully.")

        for record, (count, percent) in record_counts.items():
            if percent >= mapping_threshold:
                records_to_analyze.add(record)

    print(f"Analyzing the following FASTA records: {records_to_analyze}")
    return records_to_analyze


def process_single_bam(bam_index, bam, records_to_analyze, record_FASTA_dict, tmp_dir, max_reference_length, device):
    """Worker function to process a single BAM file (must be at top-level for multiprocessing)."""
    adata_list = []

    for record in records_to_analyze:
        sample = os.path.basename(bam).split(sep=".bam")[0]
        chromosome = record_FASTA_dict[record][2]
        current_length = record_FASTA_dict[record][4]
        mod_type, strand = record_FASTA_dict[record][6], record_FASTA_dict[record][7]

        # Extract Base Identities
        fwd_bases, rev_bases = extract_base_identities(bam, record, range(current_length), max_reference_length)

        # Skip processing if both forward and reverse base identities are empty
        if not fwd_bases and not rev_bases:
            print(f"{timestamp()} [Worker {current_process().pid}] Skipping {sample} - No valid base identities for {record}.")
            continue

        merged_bin = {}

        # Binarize the Base Identities if they exist
        if fwd_bases:
            fwd_bin = binarize_converted_base_identities(fwd_bases, strand, mod_type, bam, device)
            merged_bin.update(fwd_bin)

        if rev_bases:
            rev_bin = binarize_converted_base_identities(rev_bases, strand, mod_type, bam, device)
            merged_bin.update(rev_bin)

        # Skip if merged_bin is empty (no valid binarized data)
        if not merged_bin:
            print(f"{timestamp()} [Worker {current_process().pid}] Skipping {sample} - No valid binarized data for {record}.")
            continue

        # Convert to DataFrame
        # for key in merged_bin:
        #     merged_bin[key] = merged_bin[key].cpu().numpy()  # Move to CPU & convert to NumPy
        bin_df = pd.DataFrame.from_dict(merged_bin, orient='index')
        sorted_index = sorted(bin_df.index)
        bin_df = bin_df.reindex(sorted_index)

        # One-Hot Encode Reads if there is valid data
        one_hot_reads = {}

        if fwd_bases:
            fwd_ohe_files = ohe_batching(fwd_bases, tmp_dir, record, f"{bam_index}_fwd", batch_size=100000)
            for ohe_file in fwd_ohe_files:
                tmp_ohe_dict = ad.read_h5ad(ohe_file).uns
                one_hot_reads.update(tmp_ohe_dict)
                del tmp_ohe_dict

        if rev_bases:
            rev_ohe_files = ohe_batching(rev_bases, tmp_dir, record, f"{bam_index}_rev", batch_size=100000)
            for ohe_file in rev_ohe_files:
                tmp_ohe_dict = ad.read_h5ad(ohe_file).uns
                one_hot_reads.update(tmp_ohe_dict)
                del tmp_ohe_dict

        # Skip if one_hot_reads is empty
        if not one_hot_reads:
            print(f"{timestamp()} [Worker {current_process().pid}] Skipping {sample} - No valid one-hot encoded data for {record}.")
            continue

        gc.collect()

        # Convert One-Hot Encodings to Numpy Arrays
        n_rows_OHE = 5
        read_names = list(one_hot_reads.keys())

        # Skip if no read names exist
        if not read_names:
            print(f"{timestamp()} [Worker {current_process().pid}] Skipping {sample} - No reads found in one-hot encoded data for {record}.")
            continue

        sequence_length = one_hot_reads[read_names[0]].reshape(n_rows_OHE, -1).shape[1]
        df_A, df_C, df_G, df_T, df_N = [np.zeros((len(sorted_index), sequence_length), dtype=int) for _ in range(5)]

        # Populate One-Hot Arrays
        for j, read_name in enumerate(sorted_index):
            if read_name in one_hot_reads:
                one_hot_array = one_hot_reads[read_name].reshape(n_rows_OHE, -1)
                df_A[j], df_C[j], df_G[j], df_T[j], df_N[j] = one_hot_array

        # Convert to AnnData
        X = bin_df.values.astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs_names = bin_df.index.astype(str)
        adata.var_names = bin_df.columns.astype(str)
        adata.obs["Sample"] = [sample] * len(adata)
        adata.obs["Reference"] = [chromosome] * len(adata)
        adata.obs["Strand"] = [strand] * len(adata)
        adata.obs["Dataset"] = [mod_type] * len(adata)
        adata.obs["Reference_dataset_strand"] = [f"{chromosome}_{mod_type}_{strand}"] * len(adata)
        adata.obs["Reference_strand"] = [f"{chromosome}_{strand}"] * len(adata)

        # Attach One-Hot Encodings to Layers
        adata.layers["A_binary_encoding"] = df_A
        adata.layers["C_binary_encoding"] = df_C
        adata.layers["G_binary_encoding"] = df_G
        adata.layers["T_binary_encoding"] = df_T
        adata.layers["N_binary_encoding"] = df_N

        adata_list.append(adata)

    return ad.concat(adata_list, join="outer") if adata_list else None

def timestamp():
    """Returns a formatted timestamp for logging."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def worker_function(bam_index, bam, records_to_analyze, shared_record_FASTA_dict, tmp_dir, h5_dir, max_reference_length, device, progress_queue):
    """Worker function that processes a single BAM and writes the output to an H5AD file."""
    worker_id = current_process().pid  # Get worker process ID
    sample = os.path.basename(bam).split(sep=".bam")[0]

    try:
        print(f"{timestamp()} [Worker {worker_id}] Processing BAM: {sample}")

        h5ad_path = os.path.join(h5_dir, f"{sample}.h5ad")
        if os.path.exists(h5ad_path):
            print(f"{timestamp()} [Worker {worker_id}] Skipping {sample}: Already processed.")
            progress_queue.put(sample)
            return

        # Filter records specific to this BAM
        bam_records_to_analyze = {record for record in records_to_analyze if record in shared_record_FASTA_dict}

        if not bam_records_to_analyze:
            print(f"{timestamp()} [Worker {worker_id}] No valid records to analyze for {sample}. Skipping.")
            progress_queue.put(sample)
            return

        # Process BAM
        adata = process_single_bam(bam_index, bam, bam_records_to_analyze, shared_record_FASTA_dict, tmp_dir, max_reference_length, device)

        if adata is not None:
            adata.write_h5ad(h5ad_path)
            print(f"{timestamp()} [Worker {worker_id}] Completed processing for BAM: {sample}")

            # Free memory
            del adata
            gc.collect()

        progress_queue.put(sample)

    except Exception as e:
        print(f"{timestamp()} [Worker {worker_id}] ERROR while processing {sample}:\n{traceback.format_exc()}")
        progress_queue.put(sample)  # Still signal completion to prevent deadlock

def process_bams_parallel(bam_path_list, records_to_analyze, record_FASTA_dict, tmp_dir, h5_dir, num_threads, max_reference_length, device):
    """Processes BAM files in parallel, writes each H5AD to disk, and concatenates them at the end."""
    os.makedirs(h5_dir, exist_ok=True)  # Ensure h5_dir exists

    print(f"{timestamp()} Starting parallel BAM processing with {num_threads} threads...")

    # Ensure macOS uses forkserver to avoid spawning issues
    try:
        import multiprocessing
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        print(f"{timestamp()} [WARNING] Multiprocessing context already set. Skipping set_start_method.")

    with Manager() as manager:
        progress_queue = manager.Queue()
        shared_record_FASTA_dict = manager.dict(record_FASTA_dict)

        with Pool(processes=num_threads) as pool:
            results = [
                pool.apply_async(worker_function, (i, bam, records_to_analyze, shared_record_FASTA_dict, tmp_dir, h5_dir, max_reference_length, device, progress_queue))
                for i, bam in enumerate(bam_path_list)
            ]

            print(f"{timestamp()} Submitted {len(bam_path_list)} BAMs for processing.")

            # Track completed BAMs
            completed_bams = set()
            while len(completed_bams) < len(bam_path_list):
                try:
                    processed_bam = progress_queue.get(timeout=2400)  # Wait for a finished BAM
                    completed_bams.add(processed_bam)
                except Exception as e:
                    print(f"{timestamp()} [ERROR] Timeout waiting for worker process. Possible crash? {e}")

            pool.close()
            pool.join()  # Ensure all workers finish

    # Final Concatenation Step
    h5ad_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5ad")]

    if not h5ad_files:
        print(f"{timestamp()} No valid H5AD files generated. Exiting.")
        return None

    print(f"{timestamp()} Concatenating {len(h5ad_files)} H5AD files into final output...")
    final_adata = ad.concat([ad.read_h5ad(f) for f in h5ad_files], join="outer")

    print(f"{timestamp()} Successfully generated final AnnData object.")
    return final_adata