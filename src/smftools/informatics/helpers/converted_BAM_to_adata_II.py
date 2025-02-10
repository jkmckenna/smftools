import numpy as np
import os
import gc
import pandas as pd
import anndata as ad
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gzip

from .. import readwrite
from .binarize_converted_base_identities import binarize_converted_base_identities
from .find_conversion_sites import find_conversion_sites
from .count_aligned_reads import count_aligned_reads
from .extract_base_identities import extract_base_identities
from .make_dirs import make_dirs
from .ohe_batching import ohe_batching


def converted_BAM_to_adata_II(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix, num_threads=4):
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
    bam_files = [f for f in os.listdir(split_dir) if f.endswith(bam_suffix) and not f.endswith('.bai')]
    bam_files.sort()
    bam_path_list = [os.path.join(split_dir, f) for f in bam_files]
    print(f"Found {len(bam_files)} BAM files: {bam_files}")

    ## 2️⃣ **Process Conversion Sites**
    max_reference_length, record_FASTA_dict = process_conversion_sites(converted_FASTA, conversion_types)

    ## 3️⃣ **Filter BAM Files by Mapping Threshold**
    records_to_analyze = filter_bams_by_mapping_threshold(bam_path_list, bam_files, mapping_threshold)

    ## 4️⃣ **Process BAMs in Parallel**
    final_adata = process_bams_parallel(bam_path_list, records_to_analyze, record_FASTA_dict, tmp_dir, num_threads)

    ## 5️⃣ **Save Final AnnData**
    print(f"Saving AnnData to {final_adata_path}")
    final_adata.write_h5ad(final_adata_path, compression='gzip')
    return final_adata_path


def process_conversion_sites(converted_FASTA, conversion_types):
    """Extracts conversion sites and determines the max reference length."""
    modification_dict = {}
    record_FASTA_dict = {}
    max_reference_length = 0
    conversions = conversion_types[1:]  # Skip unconverted type

    for conversion in conversions:
        modification_dict[conversion] = find_conversion_sites(converted_FASTA, conversion, conversion_types)

        for record, values in modification_dict[conversion].items():
            seq_length, sequence, complement = values[0], values[3], values[4]
            if seq_length > max_reference_length:
                max_reference_length = seq_length

            mod_type, strand = record.split('_')[-2:]
            chromosome = record.split(f"_{mod_type}_{strand}")[0]
            unconverted_name = f"{chromosome}_{conversion_types[0]}_top"

            record_FASTA_dict[record] = [
                sequence + "N" * (max_reference_length - seq_length),
                complement + "N" * (max_reference_length - seq_length),
                chromosome, unconverted_name, seq_length, max_reference_length - seq_length, conversion, strand
            ]

    return max_reference_length, record_FASTA_dict


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

    return records_to_analyze


def process_single_bam(args):
    """Worker function to process a single BAM file (must be at top-level for multiprocessing)."""
    bam_index, bam, records_to_analyze, record_FASTA_dict, tmp_dir = args
    adata_list = []

    for record in records_to_analyze:
        sample = os.path.basename(bam).split(sep=".bam")[0]
        chromosome = record_FASTA_dict[record][2]
        current_length = record_FASTA_dict[record][4]
        mod_type, strand = record_FASTA_dict[record][6], record_FASTA_dict[record][7]

        # **Extract Base Identities**
        fwd_bases, rev_bases = extract_base_identities(bam, record, range(current_length), current_length)

        # **Binarize the Base Identities**
        fwd_bin = binarize_converted_base_identities(fwd_bases, strand, mod_type)
        rev_bin = binarize_converted_base_identities(rev_bases, strand, mod_type)
        merged_bin = {**fwd_bin, **rev_bin}

        # **Convert to DataFrame**
        bin_df = pd.DataFrame.from_dict(merged_bin, orient='index').fillna(0)
        sorted_index = sorted(bin_df.index)
        bin_df = bin_df.reindex(sorted_index)

        # **One-Hot Encode Reads**
        fwd_ohe_files = ohe_batching(fwd_bases, tmp_dir, record, f"{bam_index}_fwd", batch_size=100000)
        rev_ohe_files = ohe_batching(rev_bases, tmp_dir, record, f"{bam_index}_rev", batch_size=100000)
        one_hot_reads = {}
        ohe_files = fwd_ohe_files + rev_ohe_files

        for ohe_file in tqdm(ohe_files, desc=f"Loading OHE for {sample}"):
            tmp_ohe_dict = ad.read_h5ad(ohe_file).uns
            one_hot_reads.update(tmp_ohe_dict)
            del tmp_ohe_dict
        gc.collect()

        # **Convert One-Hot Encodings to Numpy Arrays**
        n_rows_OHE = 5
        read_names = list(one_hot_reads.keys())
        sequence_length = one_hot_reads[read_names[0]].reshape(n_rows_OHE, -1).shape[1]
        df_A, df_C, df_G, df_T, df_N = [np.zeros((len(sorted_index), sequence_length), dtype=int) for _ in range(5)]

        # **Populate One-Hot Arrays**
        for j, read_name in enumerate(sorted_index):
            if read_name in one_hot_reads:
                one_hot_array = one_hot_reads[read_name].reshape(n_rows_OHE, -1)
                df_A[j], df_C[j], df_G[j], df_T[j], df_N[j] = one_hot_array

        # **Convert to AnnData**
        X = bin_df.values.astype(np.float32)
        adata = ad.AnnData(X, dtype=np.float32)
        adata.obs_names = bin_df.index.astype(str)
        adata.var_names = bin_df.columns.astype(str)
        adata.obs["Sample"] = [sample] * len(adata)
        adata.obs["Reference"] = [chromosome] * len(adata)
        adata.obs["Strand"] = [strand] * len(adata)
        adata.obs["Dataset"] = [mod_type] * len(adata)
        adata.obs["Reference_dataset_strand"] = [f"{chromosome}_{mod_type}_{strand}"] * len(adata)
        adata.obs["Reference_strand"] = [record] * len(adata)

        # **Attach One-Hot Encodings to Layers**
        adata.layers["A_binary_encoding"] = df_A
        adata.layers["C_binary_encoding"] = df_C
        adata.layers["G_binary_encoding"] = df_G
        adata.layers["T_binary_encoding"] = df_T
        adata.layers["N_binary_encoding"] = df_N

        adata_list.append(adata)

    return ad.concat(adata_list, join="outer") if adata_list else None


def process_bams_parallel(bam_path_list, records_to_analyze, record_FASTA_dict, tmp_dir, num_threads):
    """Processes BAM files in parallel and constructs the AnnData object, including one-hot encoding."""
    final_adata = None

    # **Prepare arguments for parallel execution**
    args_list = [(i, bam, records_to_analyze, record_FASTA_dict, tmp_dir) for i, bam in enumerate(bam_path_list)]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(process_single_bam, args_list)

    for adata in results:
        if adata is not None:
            final_adata = ad.concat([final_adata, adata], join="outer") if final_adata else adata

    return final_adata
