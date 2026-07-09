from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from smftools.constants import (
    BARCODE,
    BASE_QUALITY_SCORES,
    DATASET,
    DEMUX_TYPE,
    H5_DIR,
    MISMATCH_INTEGER_ENCODING,
    MODKIT_EXTRACT_CALL_CODE_CANONICAL,
    MODKIT_EXTRACT_CALL_CODE_MODIFIED,
    MODKIT_EXTRACT_MODIFIED_BASE_A,
    MODKIT_EXTRACT_MODIFIED_BASE_C,
    MODKIT_EXTRACT_REF_STRAND_MINUS,
    MODKIT_EXTRACT_REF_STRAND_PLUS,
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    MODKIT_EXTRACT_SEQUENCE_BASES,
    MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE,
    MODKIT_EXTRACT_SEQUENCE_PADDING_BASE,
    MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE,
    MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB,
    MODKIT_EXTRACT_TSV_COLUMN_CHROM,
    MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE,
    MODKIT_EXTRACT_TSV_COLUMN_READ_ID,
    MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION,
    MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND,
    READ_MAPPING_DIRECTION,
    READ_SPAN_MASK,
    REFERENCE,
    REFERENCE_DATASET_STRAND,
    REFERENCE_STRAND,
    SAMPLE,
    SEQUENCE_INTEGER_DECODING,
    SEQUENCE_INTEGER_ENCODING,
    STRAND,
)
from smftools.logging_utils import get_logger

from .bam_functions import count_aligned_reads

logger = get_logger(__name__)

# Backstop for AsyncResult.get() in the parallel batch dispatch loop below, not a
# performance SLA: a vanilla multiprocessing.Pool is not guaranteed to raise
# promptly for a result whose worker was killed out from under it (e.g. by the
# memory watchdog in smftools.memory_guard) -- confirmed in production: after a
# run where the watchdog killed every worker, the pipeline sat idle rather than
# promptly raising the aggregate "N of M batches failed" error. 30 minutes is a
# large multiple of real observed batch durations (seconds to a couple of
# minutes) without leaving an operator waiting hours to find out everything failed.
_BATCH_RESULT_TIMEOUT_SECONDS = 30 * 60


@dataclass
class ModkitBatchDictionaries:
    """Container for per-batch modification dictionaries.

    Attributes:
        dict_total: Raw TSV DataFrames keyed by record and sample index.
        dict_a: Adenine modification DataFrames.
        dict_a_bottom: Adenine minus-strand DataFrames.
        dict_a_top: Adenine plus-strand DataFrames.
        dict_c: Cytosine modification DataFrames.
        dict_c_bottom: Cytosine minus-strand DataFrames.
        dict_c_top: Cytosine plus-strand DataFrames.
        dict_combined_bottom: Combined minus-strand methylation arrays.
        dict_combined_top: Combined plus-strand methylation arrays.
    """

    dict_total: dict = field(default_factory=dict)
    dict_a: dict = field(default_factory=dict)
    dict_a_bottom: dict = field(default_factory=dict)
    dict_a_top: dict = field(default_factory=dict)
    dict_c: dict = field(default_factory=dict)
    dict_c_bottom: dict = field(default_factory=dict)
    dict_c_top: dict = field(default_factory=dict)
    dict_combined_bottom: dict = field(default_factory=dict)
    dict_combined_top: dict = field(default_factory=dict)

    @property
    def sample_types(self) -> list[str]:
        """Return ordered labels for the dictionary list."""
        return [
            "total",
            "m6A",
            "m6A_bottom_strand",
            "m6A_top_strand",
            "5mC",
            "5mC_bottom_strand",
            "5mC_top_strand",
            "combined_bottom_strand",
            "combined_top_strand",
        ]

    def as_list(self) -> list[dict]:
        """Return the dictionaries in the expected list ordering."""
        return [
            self.dict_total,
            self.dict_a,
            self.dict_a_bottom,
            self.dict_a_top,
            self.dict_c,
            self.dict_c_bottom,
            self.dict_c_top,
            self.dict_combined_bottom,
            self.dict_combined_top,
        ]


def filter_bam_records(bam, mapping_threshold, samtools_backend: str | None = "auto"):
    """Identify reference records that exceed a mapping threshold in one BAM.

    Args:
        bam (Path | str): BAM file to inspect.
        mapping_threshold (float): Minimum fraction of mapped reads required to keep a record.
        samtools_backend (str | None): Samtools backend selection.

    Returns:
        set[str]: Record names that pass the mapping threshold.

    Processing Steps:
        1. Count aligned/unaligned reads per record.
        2. Compute percent aligned and per-record mapping percentages.
        3. Return records whose mapping fraction meets the threshold.
    """
    aligned_reads_count, unaligned_reads_count, record_counts_dict = count_aligned_reads(
        bam, samtools_backend
    )

    total_reads = aligned_reads_count + unaligned_reads_count
    percent_aligned = (aligned_reads_count * 100 / total_reads) if total_reads > 0 else 0
    logger.info(f"{percent_aligned:.2f}% of reads in {bam} aligned successfully")

    records = []
    for record, (count, percentage) in record_counts_dict.items():
        logger.info(
            f"{count} reads mapped to reference {record}. This is {percentage * 100:.2f}% of all mapped reads in {bam}"
        )
        if percentage >= mapping_threshold:
            records.append(record)

    return set(records)


def parallel_filter_bams(bam_path_list, mapping_threshold, samtools_backend: str | None = "auto"):
    """Aggregate mapping-threshold records across BAM files in parallel.

    Args:
        bam_path_list (list[Path | str]): BAM files to scan.
        mapping_threshold (float): Minimum fraction of mapped reads required to keep a record.
        samtools_backend (str | None): Samtools backend selection.

    Returns:
        set[str]: Union of all record names passing the threshold in any BAM.

    Processing Steps:
        1. Spawn workers to compute passing records per BAM.
        2. Merge all passing records into a single set.
        3. Log the final record set.
    """
    records_to_analyze = set()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            filter_bam_records,
            bam_path_list,
            [mapping_threshold] * len(bam_path_list),
            [samtools_backend] * len(bam_path_list),
        )

    # Aggregate results
    for result in results:
        records_to_analyze.update(result)

    logger.info(f"Records to analyze: {records_to_analyze}")
    return records_to_analyze


def process_tsv(tsv, records_to_analyze, reference_dict, sample_index):
    """Load and filter a modkit TSV file for relevant records and positions.

    Args:
        tsv (Path | str): TSV file produced by modkit extract.
        records_to_analyze (Iterable[str]): Record names to keep.
        reference_dict (dict[str, tuple[int, str]]): Mapping of record to (length, sequence).
        sample_index (int): Sample index to attach to the filtered results.

    Returns:
        dict[str, dict[int, pd.DataFrame]]: Filtered data keyed by record and sample index.

    Processing Steps:
        1. Read the TSV into a DataFrame.
        2. Filter rows for each record to valid reference positions.
        3. Emit per-record DataFrames keyed by the provided sample index.
    """
    temp_df = pd.read_csv(tsv, sep="\t", header=0)
    filtered_records = {}

    for record in records_to_analyze:
        if record not in reference_dict:
            continue

        ref_length = reference_dict[record][0]
        filtered_df = temp_df[
            (temp_df[MODKIT_EXTRACT_TSV_COLUMN_CHROM] == record)
            & (temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION] >= 0)
            & (temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION] < ref_length)
        ]

        if not filtered_df.empty:
            filtered_records[record] = {sample_index: filtered_df}

    return filtered_records


def _split_read_set(read_names: set[str], n_chunks: int) -> list[set[str]]:
    """Split read names into roughly equal subsets."""
    read_list = sorted(read_names)
    n_chunks = min(n_chunks, len(read_list)) or 1
    chunk_size, remainder = divmod(len(read_list), n_chunks)
    chunks: list[set[str]] = []
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(set(read_list[start:end]))
        start = end
    return chunks


def _split_extract_tsv_by_read_filters(
    tsv_path: Path,
    read_name_filters: Mapping[int, set[str]],
    tmp_dir: Path,
) -> dict[int, Path]:
    """Split one modkit-extract TSV into per-chunk filtered files, once.

    Non-split mode's pseudo-sample batches all read from the exact same
    underlying TSV. Before this, each of the N batches re-parsed the *entire*
    (unfiltered) file from scratch inside `_process_one_batch` and only
    filtered down to its own read subset afterward -- so every batch paid the
    full parse cost of the whole dataset just to use its own ~1/N slice, which
    was the dominant contributor to per-batch memory blowups (roughly constant
    ~66-83 GiB peaks regardless of batch, since the "per batch" work wasn't
    actually scoped to the batch). Reading and filtering once here, up front,
    means each batch's worker only ever reads its own already-small chunk file.

    Resumable: skips writing a chunk file that already exists, so a
    killed/retried run doesn't redo this work.
    """
    chunk_paths: dict[int, Path] = {}
    missing_chunks: list[int] = []
    for chunk_idx in read_name_filters:
        chunk_path = tmp_dir / f"tmp_extract_chunk_{chunk_idx}.tsv.gz"
        chunk_paths[chunk_idx] = chunk_path
        if not chunk_path.exists():
            missing_chunks.append(chunk_idx)

    if not missing_chunks:
        logger.debug(
            "All %d per-chunk extract TSVs already cached; skipping split", len(chunk_paths)
        )
        return chunk_paths

    logger.info(
        "Splitting %s into %d per-chunk filtered TSVs (%d already cached)",
        tsv_path,
        len(missing_chunks),
        len(chunk_paths) - len(missing_chunks),
    )
    full_df = pd.read_csv(tsv_path, sep="\t", header=0)
    try:
        read_id_col = full_df[MODKIT_EXTRACT_TSV_COLUMN_READ_ID]
        for chunk_idx in missing_chunks:
            chunk_df = full_df[read_id_col.isin(read_name_filters[chunk_idx])]
            chunk_df.to_csv(
                chunk_paths[chunk_idx], sep="\t", header=True, index=False, compression="gzip"
            )
    finally:
        del full_df
        gc.collect()

    return chunk_paths


def process_tsv_with_read_filter(
    tsv,
    records_to_analyze,
    reference_dict,
    sample_index,
    read_name_filter: set[str] | None = None,
):
    """Load and filter a modkit TSV file, optionally restricting to a read-name subset."""
    temp_df = pd.read_csv(tsv, sep="\t", header=0)
    if read_name_filter is not None:
        temp_df = temp_df[temp_df[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].isin(read_name_filter)]
    filtered_records = {}

    for record in records_to_analyze:
        if record not in reference_dict:
            continue

        ref_length = reference_dict[record][0]
        filtered_df = temp_df[
            (temp_df[MODKIT_EXTRACT_TSV_COLUMN_CHROM] == record)
            & (temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION] >= 0)
            & (temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION] < ref_length)
        ]

        if not filtered_df.empty:
            filtered_records[record] = {sample_index: filtered_df}

    return filtered_records


def parallel_load_tsvs(
    tsv_batch,
    records_to_analyze,
    reference_dict,
    batch,
    batch_size,
    threads=4,
    sample_indices: list[int] | None = None,
    read_name_filters: dict[int, set[str]] | None = None,
):
    """Load and filter a batch of TSVs in parallel.

    Args:
        tsv_batch (list[Path | str]): TSV file paths for the batch.
        records_to_analyze (Iterable[str]): Record names to keep.
        reference_dict (dict[str, tuple[int, str]]): Mapping of record to (length, sequence).
        batch (int): Batch number for progress logging.
        batch_size (int): Number of TSVs in the batch.
        threads (int): Parallel worker count.

    Returns:
        dict[str, dict[int, pd.DataFrame]]: Per-record DataFrames keyed by sample index.

    Processing Steps:
        1. Submit each TSV to a worker via `process_tsv`.
        2. Merge per-record outputs into a single dictionary.
        3. Return the aggregated per-record dictionary for the batch.
    """
    dict_total = {record: {} for record in records_to_analyze}

    if sample_indices is None:
        sample_indices = list(range(len(tsv_batch)))

    # Run in-process, with no nested process pool, whenever there's only one TSV to
    # load or the caller explicitly asked for no parallelism here (threads<=1). This
    # also makes it safe to call from inside a multiprocessing.Pool worker (e.g. the
    # per-batch parallel dispatch in modkit_extract_to_adata) -- Pool workers are
    # daemonic processes, and daemonic processes are not allowed to spawn their own
    # child processes, so unconditionally spawning a ProcessPoolExecutor here would
    # crash with "daemonic processes are not allowed to have children".
    if threads is None or threads <= 1 or len(tsv_batch) <= 1:
        for sample_index, tsv in tqdm(
            zip(sample_indices, tsv_batch),
            desc=f"Processing batch {batch}",
            total=batch_size,
        ):
            result = process_tsv_with_read_filter(
                tsv,
                records_to_analyze,
                reference_dict,
                sample_index,
                read_name_filters.get(sample_index) if read_name_filters else None,
            )
            for record, sample_data in result.items():
                dict_total[record].update(sample_data)
        return dict_total

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(
                process_tsv_with_read_filter,
                tsv,
                records_to_analyze,
                reference_dict,
                sample_index,
                read_name_filters.get(sample_index) if read_name_filters else None,
            ): sample_index
            for sample_index, tsv in zip(sample_indices, tsv_batch)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc=f"Processing batch {batch}",
            total=batch_size,
        ):
            result = future.result()
            for record, sample_data in result.items():
                dict_total[record].update(sample_data)

    return dict_total


def update_dict_to_skip(dict_to_skip, detected_modifications):
    """Update dictionary skip indices based on modifications in the batch.

    Args:
        dict_to_skip (set[int]): Initial set of dictionary indices to skip.
        detected_modifications (Iterable[str]): Modification labels present (e.g., ["6mA", "5mC"]).

    Returns:
        set[int]: Updated skip set after considering present modifications.

    Processing Steps:
        1. Define indices for A- and C-stranded dictionaries.
        2. Remove indices for modifications that are present.
        3. Return the updated skip set.
    """
    # Define which indices correspond to modification-specific or strand-specific dictionaries
    A_stranded_dicts = {2, 3}  # m6A bottom and top strand dictionaries
    C_stranded_dicts = {5, 6}  # 5mC bottom and top strand dictionaries
    combined_dicts = {7, 8}  # Combined strand dictionaries

    # If '6mA' is present, remove the A_stranded indices from the skip set
    if "6mA" in detected_modifications:
        dict_to_skip -= A_stranded_dicts
    # If '5mC' is present, remove the C_stranded indices from the skip set
    if "5mC" in detected_modifications:
        dict_to_skip -= C_stranded_dicts
    # If both modifications are present, remove the combined indices from the skip set
    if "6mA" in detected_modifications and "5mC" in detected_modifications:
        dict_to_skip -= combined_dicts

    return dict_to_skip


def process_modifications_for_sample(args):
    """Extract modification-specific subsets for one record/sample pair.

    Args:
        args (tuple): (record, sample_index, sample_df, mods, max_reference_length).

    Returns:
        tuple[str, int, dict[str, pd.DataFrame | list]]:
            Record, sample index, and a dict of modification-specific DataFrames
            (with optional combined placeholders).

    Processing Steps:
        1. Filter by modified base (A/C) when requested.
        2. Split filtered rows by strand where needed.
        3. Add empty combined placeholders when both modifications are present.
    """
    record, sample_index, sample_df, mods, max_reference_length = args
    result = {}
    if "6mA" in mods:
        m6a_df = sample_df[
            sample_df[MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE]
            == MODKIT_EXTRACT_MODIFIED_BASE_A
        ]
        result["m6A"] = m6a_df
        result["m6A_minus"] = m6a_df[
            m6a_df[MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND] == MODKIT_EXTRACT_REF_STRAND_MINUS
        ]
        result["m6A_plus"] = m6a_df[
            m6a_df[MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND] == MODKIT_EXTRACT_REF_STRAND_PLUS
        ]
        m6a_df = None
        gc.collect()
    if "5mC" in mods:
        m5c_df = sample_df[
            sample_df[MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE]
            == MODKIT_EXTRACT_MODIFIED_BASE_C
        ]
        result["5mC"] = m5c_df
        result["5mC_minus"] = m5c_df[
            m5c_df[MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND] == MODKIT_EXTRACT_REF_STRAND_MINUS
        ]
        result["5mC_plus"] = m5c_df[
            m5c_df[MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND] == MODKIT_EXTRACT_REF_STRAND_PLUS
        ]
        m5c_df = None
        gc.collect()
    if "6mA" in mods and "5mC" in mods:
        result["combined_minus"] = []
        result["combined_plus"] = []
    return record, sample_index, result


def parallel_process_modifications(dict_total, mods, max_reference_length, threads=4):
    """Parallelize modification extraction across records and samples.

    Args:
        dict_total (dict[str, dict[int, pd.DataFrame]]): Raw TSV DataFrames per record/sample.
        mods (list[str]): Modification labels to process.
        max_reference_length (int): Maximum reference length in the dataset.
        threads (int): Parallel worker count.

    Returns:
        dict[str, dict[int, dict[str, pd.DataFrame | list]]]: Processed results keyed by
        record and sample index.

    Processing Steps:
        1. Build a task list of (record, sample) pairs.
        2. Submit tasks to a process pool.
        3. Collect and store results in a nested dictionary.
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
            desc="Processing modifications",
        ):
            if record not in processed_results:
                processed_results[record] = {}
            processed_results[record][sample_index] = result
    return processed_results


def merge_modification_results(processed_results, mods):
    """Merge per-sample modification outputs into global dictionaries.

    Args:
        processed_results (dict[str, dict[int, dict]]): Output of parallel modification extraction.
        mods (list[str]): Modification labels to include.

    Returns:
        tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
            Global dictionaries for each modification/strand combination.

    Processing Steps:
        1. Initialize empty output dictionaries per modification category.
        2. Populate each dictionary using the processed sample results.
        3. Return the ordered tuple for downstream processing.
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
            if "6mA" in mods:
                if record not in m6A_dict:
                    m6A_dict[record], m6A_minus[record], m6A_plus[record] = {}, {}, {}
                m6A_dict[record][sample_index] = res.get("m6A", pd.DataFrame())
                m6A_minus[record][sample_index] = res.get("m6A_minus", pd.DataFrame())
                m6A_plus[record][sample_index] = res.get("m6A_plus", pd.DataFrame())
            if "5mC" in mods:
                if record not in c5m_dict:
                    c5m_dict[record], c5m_minus[record], c5m_plus[record] = {}, {}, {}
                c5m_dict[record][sample_index] = res.get("5mC", pd.DataFrame())
                c5m_minus[record][sample_index] = res.get("5mC_minus", pd.DataFrame())
                c5m_plus[record][sample_index] = res.get("5mC_plus", pd.DataFrame())
            if "6mA" in mods and "5mC" in mods:
                if record not in combined_minus:
                    combined_minus[record], combined_plus[record] = {}, {}
                combined_minus[record][sample_index] = res.get("combined_minus", [])
                combined_plus[record][sample_index] = res.get("combined_plus", [])
    return (
        m6A_dict,
        m6A_minus,
        m6A_plus,
        c5m_dict,
        c5m_minus,
        c5m_plus,
        combined_minus,
        combined_plus,
    )


def process_stranded_methylation(args):
    """Convert modification DataFrames into per-read methylation arrays.

    Args:
        args (tuple): (dict_index, record, sample, dict_list, max_reference_length).

    Returns:
        tuple[int, str, int, dict[str, np.ndarray]]: Updated dictionary entries for the task.

    Processing Steps:
        1. For combined dictionaries (indices 7/8), merge A- and C-strand arrays.
        2. For other dictionaries, compute methylation probabilities per read/position.
        3. Return per-read arrays keyed by read name.
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
                value_a = pd.to_numeric(np.array(temp_a[read]), errors="coerce")
            else:
                value_a = None
            if read in temp_c:
                value_c = pd.to_numeric(np.array(temp_c[read]), errors="coerce")
            else:
                value_c = None
            if value_a is not None and value_c is not None:
                processed_data[read] = np.where(
                    np.isnan(value_a) & np.isnan(value_c),
                    np.nan,
                    np.nan_to_num(value_a) + np.nan_to_num(value_c),
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
                value_a = pd.to_numeric(np.array(temp_a[read]), errors="coerce")
            else:
                value_a = None
            if read in temp_c:
                value_c = pd.to_numeric(np.array(temp_c[read]), errors="coerce")
            else:
                value_c = None
            if value_a is not None and value_c is not None:
                processed_data[read] = np.where(
                    np.isnan(value_a) & np.isnan(value_c),
                    np.nan,
                    np.nan_to_num(value_a) + np.nan_to_num(value_c),
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
        read_ids = temp_df[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].values
        positions = temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION].values
        call_codes = temp_df[MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE].values
        probabilities = pd.to_numeric(
            temp_df[MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB].values, errors="coerce"
        )

        modified_codes = MODKIT_EXTRACT_CALL_CODE_MODIFIED
        canonical_codes = MODKIT_EXTRACT_CALL_CODE_CANONICAL

        # Compute methylation probabilities (vectorized)
        methylation_prob = np.full(probabilities.shape, np.nan, dtype=float)
        methylation_prob[np.isin(call_codes, list(modified_codes))] = probabilities[
            np.isin(call_codes, list(modified_codes))
        ]
        methylation_prob[np.isin(call_codes, list(canonical_codes))] = (
            1 - probabilities[np.isin(call_codes, list(canonical_codes))]
        )

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
    """Parallelize per-read methylation extraction over all dictionary entries.

    Args:
        dict_list (list[dict]): List of modification/strand dictionaries.
        dict_to_skip (set[int]): Dictionary indices to exclude from processing.
        max_reference_length (int): Maximum reference length for array sizing.
        threads (int): Parallel worker count.

    Returns:
        list[dict]: Updated dictionary list with per-read methylation arrays.

    Processing Steps:
        1. Build tasks for every (dict_index, record, sample) to process.
        2. Execute tasks in a process pool.
        3. Replace DataFrames with per-read arrays in-place.
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
            desc="Extracting stranded methylation states",
        ):
            dict_list[dict_index][record][sample] = processed_data
    return dict_list


def delete_intermediate_h5ads_and_tmpdir(
    h5_dir: Union[str, Path, Iterable[str], None],
    tmp_dir: Optional[Union[str, Path]] = None,
    *,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Delete intermediate .h5ad files and optionally a temporary directory.

    Args:
        h5_dir (str | Path | Iterable[str] | None): Directory or iterable of h5ad paths.
        tmp_dir (str | Path | None): Temporary directory to remove recursively.
        dry_run (bool): If True, log deletions without performing them.
        verbose (bool): If True, log progress and warnings.

    Returns:
        None: This function performs deletions in-place.

    Processing Steps:
        1. Iterate over .h5ad file candidates and delete them (if not dry-run).
        2. Remove the temporary directory tree if requested.
    """

    # Helper: remove a single file path (Path-like or string)
    def _maybe_unlink(p: Path):
        """Remove a file path if it exists and is a file."""
        if not p.exists():
            if verbose:
                logger.debug(f"[skip] not found: {p}")
            return
        if not p.is_file():
            if verbose:
                logger.debug(f"[skip] not a file: {p}")
            return
        if dry_run:
            logger.debug(f"[dry-run] would remove file: {p}")
            return
        try:
            p.unlink()
            if verbose:
                logger.info(f"Removed file: {p}")
        except Exception as e:
            logger.warning(f"[error] failed to remove file {p}: {e}")

    # Handle h5_dir input (directory OR iterable of file paths)
    if h5_dir is not None:
        # If it's a path to a directory, iterate its children
        if isinstance(h5_dir, (str, Path)) and Path(h5_dir).is_dir():
            dpath = Path(h5_dir)
            for p in dpath.iterdir():
                # only target top-level files (not recursing); require '.h5ad' suffix and exclude gz
                name = p.name.lower()
                if "h5ad" in name:
                    _maybe_unlink(p)
                else:
                    if verbose:
                        # optional: comment this out if too noisy
                        logger.debug(f"[skip] not matching pattern: {p.name}")
        else:
            # treat as iterable of file paths
            for f in h5_dir:
                p = Path(f)
                name = p.name.lower()
                if name.endswith(".h5ad") and not name.endswith(".gz"):
                    _maybe_unlink(p)
                else:
                    if verbose:
                        logger.debug(f"[skip] not matching pattern or not a file: {p}")

    # Remove tmp_dir recursively (if provided)
    if tmp_dir is not None:
        td = Path(tmp_dir)
        if not td.exists():
            if verbose:
                logger.debug(f"[skip] tmp_dir not found: {td}")
        else:
            if not td.is_dir():
                if verbose:
                    logger.debug(f"[skip] tmp_dir is not a directory: {td}")
            else:
                if dry_run:
                    logger.debug(f"[dry-run] would remove directory tree: {td}")
                else:
                    try:
                        shutil.rmtree(td)
                        if verbose:
                            logger.info(f"Removed directory tree: {td}")
                    except Exception as e:
                        logger.warning(f"[error] failed to remove tmp dir {td}: {e}")


def _collect_input_paths(mod_tsv_dir: Path, bam_dir: Path) -> tuple[list[Path], list[Path]]:
    """Collect sorted TSV and BAM paths for processing.

    Args:
        mod_tsv_dir (Path): Directory containing modkit extract TSVs.
        bam_dir (Path): Directory containing aligned BAM files.

    Returns:
        tuple[list[Path], list[Path]]: Sorted TSV paths and BAM paths.

    Processing Steps:
        1. Filter TSVs for extract outputs and exclude unclassified entries.
        2. Filter BAMs for aligned files and exclude indexes/unclassified entries.
        3. Sort both lists for deterministic processing.
    """
    tsvs = sorted(
        p
        for p in mod_tsv_dir.iterdir()
        if p.is_file() and "unclassified" not in p.name and "extract.tsv" in p.name
    )
    bams = sorted(
        p
        for p in bam_dir.iterdir()
        if p.is_file()
        and p.suffix == ".bam"
        and "unclassified" not in p.name
        and ".bai" not in p.name
    )
    return tsvs, bams


def _build_sample_maps(bam_path_list: list[Path]) -> tuple[dict[int, str], dict[int, str]]:
    """Build sample name and barcode maps from BAM filenames.

    Args:
        bam_path_list (list[Path]): Paths to BAM files in sample order.

    Returns:
        tuple[dict[int, str], dict[int, str]]: Maps of sample index to sample name and barcode.

    Processing Steps:
        1. Parse the BAM stem for barcode suffixes.
        2. Build a standardized sample name with barcode suffix.
        3. Store mappings for downstream metadata annotations.
    """
    sample_name_map: dict[int, str] = {}
    barcode_map: dict[int, str] = {}

    for idx, bam_path in enumerate(bam_path_list):
        stem = bam_path.stem
        m = re.search(r"^(.*?)[_\-\.]?(barcode[0-9A-Za-z\-]+)$", stem)
        if m:
            sample_name = m.group(1) or stem
            barcode = m.group(2)
        else:
            sample_name = stem
            barcode = stem

        sample_name = f"{sample_name}_{barcode}"
        if barcode.lower().startswith("barcode"):
            barcode_id = barcode[len("barcode") :]
        else:
            barcode_id = barcode

        sample_name_map[idx] = sample_name
        barcode_map[idx] = str(barcode_id)

    return sample_name_map, barcode_map


def _encode_sequence_array(
    read_sequence: np.ndarray,
    valid_length: int,
    base_to_int: Mapping[str, int],
    padding_value: int,
) -> np.ndarray:
    """Convert a base-identity array into integer encoding with padding.

    Args:
        read_sequence (np.ndarray): Array of base calls (dtype "<U1").
        valid_length (int): Number of valid reference positions for this record.
        base_to_int (Mapping[str, int]): Base-to-integer mapping for A/C/G/T/N/PAD.
        padding_value (int): Integer value to use for padding.

    Returns:
        np.ndarray: Integer-encoded sequence with padding applied.

    Processing Steps:
        1. Initialize an integer array filled with the N value.
        2. Overwrite values for known bases (A/C/G/T/N).
        3. Replace positions beyond valid_length with padding.
    """
    read_sequence = np.asarray(read_sequence, dtype="<U1")
    encoded = np.full(read_sequence.shape, base_to_int["N"], dtype=np.int16)
    for base in MODKIT_EXTRACT_SEQUENCE_BASES:
        encoded[read_sequence == base] = base_to_int[base]
    if valid_length < encoded.size:
        encoded[valid_length:] = padding_value
    return encoded


def _filter_reads_by_names(
    mapping: Mapping[str, np.ndarray], read_names: set[str]
) -> dict[str, np.ndarray]:
    """Return the subset of `mapping` whose keys are in `read_names`."""
    return {read_name: value for read_name, value in mapping.items() if read_name in read_names}


def _write_sequence_batches(
    base_identities: Mapping[str, np.ndarray],
    tmp_dir: Path,
    record: str,
    prefix: str,
    base_to_int: Mapping[str, int],
    valid_length: int,
    batch_size: int,
) -> list[str]:
    """Encode base identities into integer arrays and write batched H5AD files.

    Args:
        base_identities (Mapping[str, np.ndarray]): Read name to base identity arrays.
        tmp_dir (Path): Directory for temporary H5AD files.
        record (str): Reference record identifier.
        prefix (str): Prefix used to name batch files.
        base_to_int (Mapping[str, int]): Base-to-integer mapping.
        valid_length (int): Valid reference length for padding determination.
        batch_size (int): Number of reads per H5AD batch file.

    Returns:
        list[str]: Paths to written H5AD batch files.

    Processing Steps:
        1. Encode each read sequence to integer values.
        2. Accumulate encoded reads into batches.
        3. Persist each batch as an H5AD with the dictionary stored in `.uns`.
    """
    import anndata as ad

    padding_value = base_to_int[MODKIT_EXTRACT_SEQUENCE_PADDING_BASE]
    batch_files: list[str] = []
    batch: dict[str, np.ndarray] = {}
    batch_number = 0

    for read_name, sequence in base_identities.items():
        if sequence is None:
            continue
        batch[read_name] = _encode_sequence_array(
            sequence, valid_length, base_to_int, padding_value
        )
        if len(batch) >= batch_size:
            save_name = tmp_dir / f"tmp_{prefix}_{record}_{batch_number}.h5ad"
            ad.AnnData(X=np.zeros((1, 1)), uns=batch).write_h5ad(save_name)
            batch_files.append(str(save_name))
            batch = {}
            batch_number += 1

    if batch:
        save_name = tmp_dir / f"tmp_{prefix}_{record}_{batch_number}.h5ad"
        ad.AnnData(X=np.zeros((1, 1)), uns=batch).write_h5ad(save_name)
        batch_files.append(str(save_name))

    return batch_files


def _write_integer_batches(
    sequences: Mapping[str, np.ndarray],
    tmp_dir: Path,
    record: str,
    prefix: str,
    batch_size: int,
) -> list[str]:
    """Write integer-encoded sequences into batched H5AD files.

    Args:
        sequences (Mapping[str, np.ndarray]): Read name to integer arrays.
        tmp_dir (Path): Directory for temporary H5AD files.
        record (str): Reference record identifier.
        prefix (str): Prefix used to name batch files.
        batch_size (int): Number of reads per H5AD batch file.

    Returns:
        list[str]: Paths to written H5AD batch files.

    Processing Steps:
        1. Accumulate integer arrays into batches.
        2. Persist each batch as an H5AD with the dictionary stored in `.uns`.
    """
    import anndata as ad

    batch_files: list[str] = []
    batch: dict[str, np.ndarray] = {}
    batch_number = 0

    for read_name, sequence in sequences.items():
        if sequence is None:
            continue
        batch[read_name] = np.asarray(sequence, dtype=np.int16)
        if len(batch) >= batch_size:
            save_name = tmp_dir / f"tmp_{prefix}_{record}_{batch_number}.h5ad"
            ad.AnnData(X=np.zeros((1, 1)), uns=batch).write_h5ad(save_name)
            batch_files.append(str(save_name))
            batch = {}
            batch_number += 1

    if batch:
        save_name = tmp_dir / f"tmp_{prefix}_{record}_{batch_number}.h5ad"
        ad.AnnData(X=np.zeros((1, 1)), uns=batch).write_h5ad(save_name)
        batch_files.append(str(save_name))

    return batch_files


def _load_sequence_batches(
    batch_files: list[Path | str],
) -> tuple[dict[str, np.ndarray], set[str], set[str]]:
    """Load integer-encoded sequence batches from H5AD files.

    Args:
        batch_files (list[Path | str]): H5AD paths containing encoded sequences in `.uns`.

    Returns:
        tuple[dict[str, np.ndarray], set[str], set[str]]:
            Read-to-sequence mapping and sets of forward/reverse mapped reads.

    Processing Steps:
        1. Read each H5AD file.
        2. Merge `.uns` dictionaries into a single mapping.
        3. Track forward/reverse read IDs based on the filename marker.
    """
    import anndata as ad

    sequences: dict[str, np.ndarray] = {}
    fwd_reads: set[str] = set()
    rev_reads: set[str] = set()
    for batch_file in batch_files:
        batch_path = Path(batch_file)
        batch_sequences = ad.read_h5ad(batch_path).uns
        sequences.update(batch_sequences)
        if "_fwd_" in batch_path.name:
            fwd_reads.update(batch_sequences.keys())
        elif "_rev_" in batch_path.name:
            rev_reads.update(batch_sequences.keys())
    return sequences, fwd_reads, rev_reads


def _load_integer_batches(batch_files: list[Path | str]) -> dict[str, np.ndarray]:
    """Load integer arrays from batched H5AD files.

    Args:
        batch_files (list[Path | str]): H5AD paths containing arrays in `.uns`.

    Returns:
        dict[str, np.ndarray]: Read-to-array mapping.

    Processing Steps:
        1. Read each H5AD file.
        2. Merge `.uns` dictionaries into a single mapping.
    """
    import anndata as ad

    sequences: dict[str, np.ndarray] = {}
    for batch_file in batch_files:
        batch_path = Path(batch_file)
        sequences.update(ad.read_h5ad(batch_path).uns)
    return sequences


def _resolve_demux_type_annotation_mode(
    input_already_demuxed: bool,
    demux_backend: str | None,
    double_barcoded_path,
) -> str:
    """Decide how (or whether) to annotate `obs[DEMUX_TYPE]` on the final AnnData.

    Regression guard for a crash where `double_barcoded_path` is `None` -- which
    happens whenever `skip_bam_split=True` was used (no physical BAM splitting, so no
    dorado `barcoding_summary.txt` was ever produced) -- but the dorado-backend branch
    unconditionally did `double_barcoded_path / "barcoding_summary.txt"`. There is
    nothing to derive per-end BM scoring from in that case regardless of backend, so it
    must be treated the same as the smftools-backend "skip annotation" branch instead of
    crashing.

    Returns one of: "already", "skip_smftools", "dorado_barcoding_summary",
    "skip_no_double_barcoded_path".
    """
    if input_already_demuxed:
        return "already"
    if demux_backend and demux_backend.lower() == "smftools":
        return "skip_smftools"
    if double_barcoded_path is not None:
        return "dorado_barcoding_summary"
    return "skip_no_double_barcoded_path"


def _individual_mod_dicts_superseded_by_combined(mods: list[str]) -> set[int]:
    """Dict-type indices whose full AnnData construction can be skipped in Loop B.

    When both "6mA" and "5mC" are requested, the final cross-batch assembly only ever
    reads back the combined-strand dict types (indices 7, 8) -- see the `combined_hdfs`
    filter later in this module. The individual-modality dict types (m6A bottom/top = 2,
    3; 5mC bottom/top = 5, 6) are still required upstream, in-memory, to compute the
    combined per-read arrays, but building/writing full AnnData objects for them is
    otherwise pure waste. Returns an empty set for single-modification runs, where the
    individual dict types are the only output and must still be built.
    """
    if "6mA" in mods and "5mC" in mods:
        return {2, 3, 5, 6}
    return set()


def _load_sample_record_batches_cached(
    cache: dict[str, tuple],
    cache_key: str,
    sequence_files: list[Path | str],
    mismatch_files: list[Path | str],
    quality_files: list[Path | str],
    read_span_files: list[Path | str],
) -> tuple[dict[str, np.ndarray], set[str], set[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load (or reuse from `cache`) the encoded-sequence/mismatch/quality/read-span batch
    files for one (sample, record) pair.

    The same `cache_key` (`f"{final_sample_index}_{record}"`) is looked up once per
    modality/strand "dict type" in the caller's loop, but the underlying tmp H5AD files
    are write-once and never mutated across that loop, so it is safe -- and up to Nx
    cheaper in disk I/O for N dict types -- to read them from disk only the first time
    a given key is seen per batch, and reuse the same in-memory dicts for every
    subsequent dict-type iteration. `cache` should be reset once per outer processing
    batch so its memory footprint stays bounded to one batch's data, not the whole run.

    Callers must treat the returned dicts as read-only: anything written back into
    `cache` is shared across all dict-type iterations for this batch.
    """
    if cache_key in cache:
        return cache[cache_key]

    encoded_reads, fwd_mapped_reads, rev_mapped_reads = _load_sequence_batches(sequence_files)
    mismatch_reads: dict[str, np.ndarray] = {}
    if mismatch_files:
        mismatch_reads, _mismatch_fwd_reads, _mismatch_rev_reads = _load_sequence_batches(
            mismatch_files
        )
    quality_reads: dict[str, np.ndarray] = {}
    if quality_files:
        quality_reads = _load_integer_batches(quality_files)
    read_span_reads: dict[str, np.ndarray] = {}
    if read_span_files:
        read_span_reads = _load_integer_batches(read_span_files)

    result = (
        encoded_reads,
        fwd_mapped_reads,
        rev_mapped_reads,
        mismatch_reads,
        quality_reads,
        read_span_reads,
    )
    cache[cache_key] = result
    return result


def _normalize_sequence_batch_files(batch_files: object) -> list[Path]:
    """Normalize cached batch file entries into a list of Paths.

    Args:
        batch_files (object): Cached batch file entry from AnnData `.uns`.

    Returns:
        list[Path]: Paths to batch files, filtered to non-empty values.

    Processing Steps:
        1. Convert numpy arrays and scalars into Python lists.
        2. Filter out empty/placeholder values.
        3. Cast remaining entries to Path objects.
    """
    if batch_files is None:
        return []
    if isinstance(batch_files, np.ndarray):
        batch_files = batch_files.tolist()
    if isinstance(batch_files, (str, Path)):
        batch_files = [batch_files]
    if not isinstance(batch_files, list):
        batch_files = list(batch_files)
    normalized: list[Path] = []
    for entry in batch_files:
        if entry is None:
            continue
        entry_str = str(entry).strip()
        if not entry_str or entry_str == ".":
            continue
        normalized.append(Path(entry_str))
    return normalized


def _build_modification_dicts(
    dict_total: dict,
    mods: list[str],
) -> tuple[ModkitBatchDictionaries, set[int]]:
    """Build modification/strand dictionaries from the raw TSV batch dictionary.

    Args:
        dict_total (dict): Raw TSV DataFrames keyed by record and sample index.
        mods (list[str]): Modification labels to include (e.g., ["6mA", "5mC"]).

    Returns:
        tuple[ModkitBatchDictionaries, set[int]]: Batch dictionaries and indices to skip.

    Processing Steps:
        1. Initialize modification dictionaries and skip-set.
        2. Filter TSV rows per record/sample into modification and strand subsets.
        3. Populate combined dict placeholders when both modifications are present.
    """
    batch_dicts = ModkitBatchDictionaries(dict_total=dict_total)
    dict_to_skip = {0, 1, 4}
    combined_dicts = {7, 8}
    A_stranded_dicts = {2, 3}
    C_stranded_dicts = {5, 6}
    dict_to_skip.update(combined_dicts | A_stranded_dicts | C_stranded_dicts)

    for record in dict_total.keys():
        for sample_index in dict_total[record].keys():
            if "6mA" in mods:
                dict_to_skip.difference_update(A_stranded_dicts)
                if (
                    record not in batch_dicts.dict_a.keys()
                    and record not in batch_dicts.dict_a_bottom.keys()
                    and record not in batch_dicts.dict_a_top.keys()
                ):
                    (
                        batch_dicts.dict_a[record],
                        batch_dicts.dict_a_bottom[record],
                        batch_dicts.dict_a_top[record],
                    ) = ({}, {}, {})

                batch_dicts.dict_a[record][sample_index] = dict_total[record][sample_index][
                    dict_total[record][sample_index][
                        MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE
                    ]
                    == MODKIT_EXTRACT_MODIFIED_BASE_A
                ]
                logger.debug(
                    "Successfully loaded a methyl-adenine dictionary for {}".format(
                        str(sample_index)
                    )
                )

                batch_dicts.dict_a_bottom[record][sample_index] = batch_dicts.dict_a[record][
                    sample_index
                ][
                    batch_dicts.dict_a[record][sample_index][MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND]
                    == MODKIT_EXTRACT_REF_STRAND_MINUS
                ]
                logger.debug(
                    "Successfully loaded a minus strand methyl-adenine dictionary for {}".format(
                        str(sample_index)
                    )
                )
                batch_dicts.dict_a_top[record][sample_index] = batch_dicts.dict_a[record][
                    sample_index
                ][
                    batch_dicts.dict_a[record][sample_index][MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND]
                    == MODKIT_EXTRACT_REF_STRAND_PLUS
                ]
                logger.debug(
                    "Successfully loaded a plus strand methyl-adenine dictionary for ".format(
                        str(sample_index)
                    )
                )

                batch_dicts.dict_a[record][sample_index] = None
                gc.collect()

            if "5mC" in mods:
                dict_to_skip.difference_update(C_stranded_dicts)
                if (
                    record not in batch_dicts.dict_c.keys()
                    and record not in batch_dicts.dict_c_bottom.keys()
                    and record not in batch_dicts.dict_c_top.keys()
                ):
                    (
                        batch_dicts.dict_c[record],
                        batch_dicts.dict_c_bottom[record],
                        batch_dicts.dict_c_top[record],
                    ) = ({}, {}, {})

                batch_dicts.dict_c[record][sample_index] = dict_total[record][sample_index][
                    dict_total[record][sample_index][
                        MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE
                    ]
                    == MODKIT_EXTRACT_MODIFIED_BASE_C
                ]
                logger.debug(
                    "Successfully loaded a methyl-cytosine dictionary for {}".format(
                        str(sample_index)
                    )
                )

                batch_dicts.dict_c_bottom[record][sample_index] = batch_dicts.dict_c[record][
                    sample_index
                ][
                    batch_dicts.dict_c[record][sample_index][MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND]
                    == MODKIT_EXTRACT_REF_STRAND_MINUS
                ]
                logger.debug(
                    "Successfully loaded a minus strand methyl-cytosine dictionary for {}".format(
                        str(sample_index)
                    )
                )
                batch_dicts.dict_c_top[record][sample_index] = batch_dicts.dict_c[record][
                    sample_index
                ][
                    batch_dicts.dict_c[record][sample_index][MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND]
                    == MODKIT_EXTRACT_REF_STRAND_PLUS
                ]
                logger.debug(
                    "Successfully loaded a plus strand methyl-cytosine dictionary for {}".format(
                        str(sample_index)
                    )
                )

                batch_dicts.dict_c[record][sample_index] = None
                gc.collect()

            if "6mA" in mods and "5mC" in mods:
                dict_to_skip.difference_update(combined_dicts)
                if (
                    record not in batch_dicts.dict_combined_bottom.keys()
                    and record not in batch_dicts.dict_combined_top.keys()
                ):
                    (
                        batch_dicts.dict_combined_bottom[record],
                        batch_dicts.dict_combined_top[record],
                    ) = ({}, {})

                logger.debug(
                    "Successfully created a minus strand combined methylation dictionary for {}".format(
                        str(sample_index)
                    )
                )
                batch_dicts.dict_combined_bottom[record][sample_index] = []
                logger.debug(
                    "Successfully created a plus strand combined methylation dictionary for {}".format(
                        str(sample_index)
                    )
                )
                batch_dicts.dict_combined_top[record][sample_index] = []

            dict_total[record][sample_index] = None
            gc.collect()

    return batch_dicts, dict_to_skip


def _count_active_mod_dict_types(mods: list[str]) -> int:
    """Count the per-read float32 dictionaries `_build_modification_dicts` will populate.

    Mirrors its `"6mA" in mods` / `"5mC" in mods` membership checks exactly: each
    active channel contributes 3 dicts (unstranded + top-strand + bottom-strand),
    plus 2 more (combined top/bottom strand) when both channels are active --
    i.e. 0, 3, or 8, matching `ModkitBatchDictionaries.sample_types`.
    """
    has_a = "6mA" in mods
    has_c = "5mC" in mods
    count = 3 * (int(has_a) + int(has_c))
    if has_a and has_c:
        count += 2
    return count


def _estimate_worker_peak_bytes(
    sequence_batch_files: dict,
    mismatch_batch_files: dict,
    quality_batch_files: dict,
    read_span_batch_files: dict,
    mem_multiplier: float = 4.0,
    min_worker_budget_gb: float = 2.0,
    keys_per_batch: int = 1,
    reads_per_batch: int = 0,
    max_reference_length: int = 0,
    n_mod_dict_types: int = 0,
    mod_array_dtype_bytes: int = 4,
    overall_safety_multiplier: float = 3.0,
) -> int:
    """Estimate one worker's peak memory footprint for one batch, in bytes.

    Used both by `_estimate_max_workers` (to size the worker pool) and by
    `MemoryGuard`-style per-worker caps (to know what a single worker should be
    killed for exceeding) -- both need the same number, just for different
    purposes, so it's computed once here.

    Has two independent components, both approximated conservatively because
    neither is directly observable before a batch runs:

    1. On-disk batch-file footprint: `_process_one_batch` loads the sequence,
       mismatch, quality, and read-span batch files for the same (bam_index,
       record) key together (see `_load_sample_record_batches_cached`), and
       `sample_record_batch_cache` is reset only at batch boundaries, so up to
       `keys_per_batch` distinct keys can be resident at once. This component
       sums (not maxes) the four dict types per key, then sums the
       `keys_per_batch` largest keys, then applies `mem_multiplier` to account
       for decoded-array/AnnData overhead a raw file size does not capture.
    2. Modkit-extract TSV / modification-dict footprint: independently of (1),
       the same worker builds `dict_total` (raw per-call-row DataFrames) and,
       for every active modification/strand dictionary
       (`_build_modification_dicts`), a full `max_reference_length` float32 NaN
       array per read. This is approximated directly from read/position/dict
       counts as `reads_per_batch * max_reference_length *
       mod_array_dtype_bytes * n_mod_dict_types` when those are known (e.g. in
       non-split mode, where `reads_per_batch` can be read exactly off
       `read_name_filters`); it is omitted (0) when unknown rather than guessed
       from an indirect proxy, so callers that can supply it should.

    The combined total is then scaled by `overall_safety_multiplier` and
    floored at `min_worker_budget_gb`. This extra margin matters most for
    component (2): unlike the on-disk component's `mem_multiplier` (which
    exists specifically to cover decode/AnnData overhead a raw file size
    doesn't capture), (2) is a literal analytical byte count of arrays that
    will genuinely be allocated, with no fudge factor of its own -- so if it
    (or anything else about the run, e.g. per-sample AnnData/DataFrame
    construction overhead that accumulates across a batch's several
    surviving dict-type writes) is off by even a modest amount, the combined
    estimate has no headroom left to absorb that.

    In production this under-margin was observed directly, four times, each
    with less slack than the last but never quite zero: with no multiplier,
    every worker landed 3-16% over the raw estimate; at 1.25x, still up to
    ~15% over; at 1.6x (and after halving the dominant term's own footprint
    by switching its arrays from float64 to float32), down to 0.3-2.3% over;
    at 2.0x, most batches (8/30) finally cleared it, but the batches with
    genuinely more reads than the rest (real per-batch variance -- this
    estimate already uses the run's worst observed on-disk key size and read
    count, not a per-batch value, so it's already "sized for the worst
    batch" in principle) still landed 2.5-4.6% over and failed. The pattern
    across all four attempts -- closing in on, but never comfortably
    clearing, the threshold as the multiplier grows -- is consistent with a
    small fixed per-worker overhead a purely multiplicative margin can't
    fully absorb, compounded by real batch-to-batch variance this
    single-flat-budget model doesn't capture. 3.0x is deliberately generous
    rather than another tight increment, since this machine has ample
    headroom to spare (even accounting for the resulting drop in worker
    count) and another near-miss costs an entire wasted batch-processing
    pass to discover. Every failure was harmless to the machine itself (each
    worker was only a few GiB), but cost a partial-to-total batch failure
    each time. The worst incident was worse than "just retry": several workers were killed
    *after* writing some but not all of
    a batch's per-dict-type output files, and since resumability used to be
    judged by "does any output file for this batch exist", a naive retry
    would have treated those batches as done and permanently skipped the
    dict types that hadn't been written yet -- see `_process_one_batch`'s
    completion-marker docstring for the fix to that half of the problem.
    `overall_safety_multiplier` is the other half: it absorbs exactly this
    kind of small, real-world slop that a purely analytical model can't
    predict, on top of (not instead of) `mem_multiplier`.

    Returns 0 (rather than the floor) only when both components are 0, i.e.
    nothing at all is known -- callers should treat that as "no estimate
    available", not "zero memory needed".
    """
    per_key_totals: dict[str, int] = {}
    for key_dict in (
        sequence_batch_files,
        mismatch_batch_files,
        quality_batch_files,
        read_span_batch_files,
    ):
        for key, raw_files in key_dict.items():
            total = 0
            for p in _normalize_sequence_batch_files(raw_files):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
            per_key_totals[key] = per_key_totals.get(key, 0) + total

    if per_key_totals:
        sorted_totals = sorted(per_key_totals.values(), reverse=True)
        on_disk_worst_case_bytes = sum(sorted_totals[: max(1, keys_per_batch)])
    else:
        on_disk_worst_case_bytes = 0

    mod_dict_bytes = 0
    if reads_per_batch > 0 and max_reference_length > 0 and n_mod_dict_types > 0:
        mod_dict_bytes = (
            reads_per_batch * max_reference_length * mod_array_dtype_bytes * n_mod_dict_types
        )

    if on_disk_worst_case_bytes <= 0 and mod_dict_bytes <= 0:
        return 0

    return int(
        max(
            (on_disk_worst_case_bytes * mem_multiplier + mod_dict_bytes)
            * overall_safety_multiplier,
            min_worker_budget_gb * (1024**3),
        )
    )


def _estimate_max_workers(
    sequence_batch_files: dict,
    mismatch_batch_files: dict,
    quality_batch_files: dict,
    read_span_batch_files: dict,
    threads,
    mem_multiplier: float = 4.0,
    mem_safety_fraction: float = 0.7,
    min_worker_budget_gb: float = 2.0,
    keys_per_batch: int = 1,
    reads_per_batch: int = 0,
    max_reference_length: int = 0,
    n_mod_dict_types: int = 0,
    mod_array_dtype_bytes: int = 4,
) -> int:
    """Estimate a safe worker count from available RAM and on-disk batch-file sizes.

    Per-worker peak memory is estimated by `_estimate_worker_peak_bytes` (see its
    docstring for what it accounts for). Available memory is read from the OS and
    only `mem_safety_fraction` of it is budgeted, leaving headroom for the OS,
    page cache, and other processes.

    Falls back to `threads` (or `os.cpu_count()`) alone, ignoring the memory
    estimate, if file sizes or system memory cannot be determined (e.g. on a
    platform without `os.sysconf`).
    """
    import os

    cpu_cap = int(threads) if threads else (os.cpu_count() or 4)

    try:
        est_worker_peak_bytes = _estimate_worker_peak_bytes(
            sequence_batch_files,
            mismatch_batch_files,
            quality_batch_files,
            read_span_batch_files,
            mem_multiplier=mem_multiplier,
            min_worker_budget_gb=min_worker_budget_gb,
            keys_per_batch=keys_per_batch,
            reads_per_batch=reads_per_batch,
            max_reference_length=max_reference_length,
            n_mod_dict_types=n_mod_dict_types,
            mod_array_dtype_bytes=mod_array_dtype_bytes,
        )
        if est_worker_peak_bytes <= 0:
            return max(1, cpu_cap)

        total_mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_cap = max(1, int((total_mem_bytes * mem_safety_fraction) // est_worker_peak_bytes))
        return max(1, min(cpu_cap, mem_cap))
    except Exception:
        logger.debug("Could not estimate memory-based worker cap; falling back to CPU count")
        return max(1, cpu_cap)


def _resolve_max_workers(
    max_workers: int | str | None,
    n_batches: int,
    threads,
    sequence_batch_files: dict | None = None,
    mismatch_batch_files: dict | None = None,
    quality_batch_files: dict | None = None,
    read_span_batch_files: dict | None = None,
    keys_per_batch: int = 1,
    reads_per_batch: int = 0,
    max_reference_length: int = 0,
    n_mod_dict_types: int = 0,
) -> int:
    """Resolve the `max_workers` argument into a concrete, safe worker count.

    - `None` -> 1 (serial; unchanged default behavior from before this parameter existed).
    - `"auto"` -> memory-and-CPU-aware estimate via `_estimate_max_workers`.
    - A positive int -> that value, additionally capped by the same memory
      estimate and by `n_batches` (no point spawning more workers than batches).

    `keys_per_batch`, `reads_per_batch`, `max_reference_length`, and
    `n_mod_dict_types` are passed straight through to `_estimate_max_workers`;
    see its docstring for what each accounts for.
    """
    if max_workers is None:
        return 1
    if isinstance(max_workers, str):
        if max_workers.lower() != "auto":
            raise ValueError(f"Unrecognized max_workers value: {max_workers!r}")
        estimated = _estimate_max_workers(
            sequence_batch_files or {},
            mismatch_batch_files or {},
            quality_batch_files or {},
            read_span_batch_files or {},
            threads,
            keys_per_batch=keys_per_batch,
            reads_per_batch=reads_per_batch,
            max_reference_length=max_reference_length,
            n_mod_dict_types=n_mod_dict_types,
        )
        return max(1, min(estimated, n_batches))
    requested = int(max_workers)
    if requested <= 1:
        return 1
    estimated = _estimate_max_workers(
        sequence_batch_files or {},
        mismatch_batch_files or {},
        quality_batch_files or {},
        read_span_batch_files or {},
        threads,
        keys_per_batch=keys_per_batch,
        reads_per_batch=reads_per_batch,
        max_reference_length=max_reference_length,
        n_mod_dict_types=n_mod_dict_types,
    )
    return max(1, min(requested, estimated, n_batches))


def _process_one_batch(
    batch: int,
    tsv_batch: list,
    sample_indices_batch,
    records_to_analyze,
    reference_dict,
    max_reference_length: int,
    mods: list[str],
    batch_size: int,
    threads,
    read_name_filters,
    use_global_sample_indices: bool,
    sample_name_map: dict,
    barcode_map: dict,
    read_to_barcode,
    h5_dir,
    sequence_batch_files: dict,
    mismatch_batch_files: dict,
    quality_batch_files: dict,
    read_span_batch_files: dict,
) -> None:
    """Process one batch of TSVs into per-dict-type H5AD files under `h5_dir`.

    This is the exact per-batch body that `modkit_extract_to_adata`'s serial loop
    used to run inline; it was extracted, unchanged, so it could also be dispatched
    to a `multiprocessing.Pool` worker for the parallel path (see `max_workers` on
    `modkit_extract_to_adata`) without duplicating or reimplementing any of its
    (already fixed/tested) numeric logic. Every argument is picklable (paths,
    already-computed small dicts of file paths/metadata) -- none of the large
    per-read dictionaries this function builds internally are ever passed in or
    returned, so dispatching this function across processes does not incur the
    "ship a huge dict per task" cost that made the older orphaned
    `parallel_process_modifications`/`parallel_extract_stranded_methylation` helpers
    unsafe to use under this codebase's `forkserver` multiprocessing start method.

    Side effects only: writes `<date>_<batch>_<dict_type>_SMF_binarized_sample_hdf5.h5ad.gz`
    files into `h5_dir`, one per surviving dict type that has at least one mapped read,
    then a `_batch_{batch}_complete.marker` file once every dict type has been written.

    Resumability note: completeness for a batch is judged solely by the presence of
    that marker file, not by whether any `*.h5ad.gz` output for the batch exists.
    A batch can legitimately write some (but not all) of its per-dict-type files
    before being killed (e.g. by the memory watchdog in smftools.memory_guard) --
    checking "does any output file exist" would then treat a genuinely incomplete,
    partially-written batch as done and skip it forever, silently dropping whichever
    dict types hadn't been written yet. The marker is only written after the full
    per-dict-type write loop below completes without being interrupted.
    """
    import anndata as ad

    from .. import readwrite

    batch_complete_marker = h5_dir / f"_batch_{batch}_complete.marker"
    if batch_complete_marker.exists():
        logger.debug(
            f"Batch {batch} has already been fully processed (completion marker found). "
            "Skipping batch and using existing files"
        )
        return

    ###################################################
    ### Add the tsvs as dataframes to a dictionary (dict_total) keyed by integer index. Also make modification specific dictionaries and strand specific dictionaries.
    # # Initialize dictionaries and place them in a list
    batch_dicts = ModkitBatchDictionaries()
    dict_list = batch_dicts.as_list()
    sample_types = batch_dicts.sample_types
    # Cache of (sample, record) -> loaded sequence/mismatch/quality/read-span batches,
    # reset every batch so its memory is bounded to one batch. See
    # _load_sample_record_batches_cached for why this reuse is safe.
    sample_record_batch_cache: dict[str, tuple] = {}

    # Pool workers are daemonic processes, which are not allowed to spawn their own
    # child processes. When this function is itself running inside a
    # multiprocessing.Pool worker (the parallel per-batch dispatch path), force the
    # inner TSV load to run without its own nested process pool -- parallel_load_tsvs
    # already falls back to a serial in-process loop for threads<=1.
    import multiprocessing as _mp

    tsv_load_threads = 1 if _mp.current_process().daemon else threads

    # # Step 1):Load the dict_total dictionary with all of the batch tsv files as dataframes.
    dict_total = parallel_load_tsvs(
        tsv_batch,
        records_to_analyze,
        reference_dict,
        batch,
        batch_size=len(tsv_batch),
        threads=tsv_load_threads,
        sample_indices=sample_indices_batch,
        read_name_filters=read_name_filters,
    )

    batch_dicts, dict_to_skip = _build_modification_dicts(dict_total, mods)
    dict_list = batch_dicts.as_list()
    sample_types = batch_dicts.sample_types

    # Iterate over the stranded modification dictionaries and replace the dataframes with a dictionary of read names pointing to a list of values from the dataframe
    for dict_index, dict_type in enumerate(dict_list):
        # Only iterate over stranded dictionaries
        if dict_index not in dict_to_skip:
            logger.debug(
                "Extracting methylation states for {} dictionary".format(
                    sample_types[dict_index]
                )
            )
            for record in dict_type.keys():
                # Get the dictionary for the modification type of interest from the reference mapping of interest
                mod_strand_record_sample_dict = dict_type[record]
                logger.debug(
                    "Extracting methylation states for {} dictionary".format(record)
                )
                # For each sample in a stranded dictionary
                n_samples = len(mod_strand_record_sample_dict.keys())
                for sample in tqdm(
                    mod_strand_record_sample_dict.keys(),
                    desc=f"Extracting {sample_types[dict_index]} dictionary from record {record} for sample",
                    total=n_samples,
                ):
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
                                mod_strand_record_sample_dict[sample][read] = np.where(
                                    np.isnan(temp_a_dict[read])
                                    & np.isnan(temp_c_dict[read]),
                                    np.nan,
                                    np.nan_to_num(temp_a_dict[read])
                                    + np.nan_to_num(temp_c_dict[read]),
                                )
                            # If the read is present in only one dictionary, copy its value
                            elif read in temp_a_dict:
                                mod_strand_record_sample_dict[sample][read] = temp_a_dict[
                                    read
                                ]
                            elif read in temp_c_dict:
                                mod_strand_record_sample_dict[sample][read] = temp_c_dict[
                                    read
                                ]
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
                                mod_strand_record_sample_dict[sample][read] = np.where(
                                    np.isnan(temp_a_dict[read])
                                    & np.isnan(temp_c_dict[read]),
                                    np.nan,
                                    np.nan_to_num(temp_a_dict[read])
                                    + np.nan_to_num(temp_c_dict[read]),
                                )
                            # If the read is present in only one dictionary, copy its value
                            elif read in temp_a_dict:
                                mod_strand_record_sample_dict[sample][read] = temp_a_dict[
                                    read
                                ]
                            elif read in temp_c_dict:
                                mod_strand_record_sample_dict[sample][read] = temp_c_dict[
                                    read
                                ]
                        del temp_a_dict, temp_c_dict
                    # For all other dictionaries
                    else:
                        # use temp_df to point to the dataframe held in mod_strand_record_sample_dict[sample]
                        temp_df = mod_strand_record_sample_dict[sample]
                        # reassign the dictionary pointer to a nested dictionary.
                        mod_strand_record_sample_dict[sample] = {}

                        # Get relevant columns as NumPy arrays
                        read_ids = temp_df[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].values
                        positions = temp_df[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION].values
                        call_codes = temp_df[MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE].values
                        probabilities = temp_df[MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB].values

                        # Define valid call code categories
                        modified_codes = MODKIT_EXTRACT_CALL_CODE_MODIFIED
                        canonical_codes = MODKIT_EXTRACT_CALL_CODE_CANONICAL

                        # Vectorized methylation calculation with NaN for other codes.
                        # float32 (not float64): these are probabilities in [0, 1] or NaN,
                        # far more precision than needed, and this array is later broadcast
                        # across the full reference length for every read -- at production
                        # scale (hundreds of thousands of reads x thousands of positions)
                        # the float64/float32 difference alone is several GB of peak memory.
                        methylation_prob = np.full_like(
                            probabilities, np.nan, dtype=np.float32
                        )  # Default all to NaN
                        methylation_prob[np.isin(call_codes, list(modified_codes))] = (
                            probabilities[np.isin(call_codes, list(modified_codes))]
                        )
                        methylation_prob[np.isin(call_codes, list(canonical_codes))] = (
                            1 - probabilities[np.isin(call_codes, list(canonical_codes))]
                        )

                        # Find unique reads
                        unique_reads = np.unique(read_ids)
                        # Preallocate storage for each read
                        for read in unique_reads:
                            mod_strand_record_sample_dict[sample][read] = np.full(
                                max_reference_length, np.nan, dtype=np.float32
                            )

                        # Efficient NumPy indexing to assign values
                        for i in range(len(read_ids)):
                            read = read_ids[i]
                            pos = positions[i]
                            prob = methylation_prob[i]

                            # Assign methylation probability
                            mod_strand_record_sample_dict[sample][read][pos] = prob

    # Save the sample files in the batch as gzipped hdf5 files
    logger.info("Converting batch {} dictionaries to anndata objects".format(batch))
    # See _individual_mod_dicts_superseded_by_combined docstring: skips AnnData
    # construction only (not the upstream array computation which dict_to_skip,
    # used above in Loop A, already governs correctly).
    loop_b_skip = dict_to_skip | _individual_mod_dicts_superseded_by_combined(mods)
    for dict_index, dict_type in enumerate(dict_list):
        if dict_index not in loop_b_skip:
            # Collect one AnnData per sample and concatenate once at the end
            # (instead of concatenating incrementally per sample, which is O(n^2)).
            adata_list: list[ad.AnnData] = []
            logger.info(
                "Converting {} dictionary to an anndata object".format(
                    sample_types[dict_index]
                )
            )
            for record in dict_type.keys():
                # Get the dictionary for the modification type of interest from the reference mapping of interest
                mod_strand_record_sample_dict = dict_type[record]
                for sample in mod_strand_record_sample_dict.keys():
                    logger.info(
                        "Converting {0} dictionary for sample {1} to an anndata object".format(
                            sample_types[dict_index], sample
                        )
                    )
                    sample = int(sample)
                    if use_global_sample_indices:
                        final_sample_index = sample
                    else:
                        final_sample_index = sample + (batch * batch_size)
                    logger.info(
                        "Final sample index for sample: {}".format(final_sample_index)
                    )
                    logger.debug(
                        "Converting {0} dictionary for sample {1} to a dataframe".format(
                            sample_types[dict_index],
                            final_sample_index,
                        )
                    )
                    temp_df = pd.DataFrame.from_dict(
                        mod_strand_record_sample_dict[sample], orient="index"
                    )
                    mod_strand_record_sample_dict[sample] = (
                        None  # reassign pointer to facilitate memory usage
                    )
                    sorted_index = sorted(temp_df.index)
                    temp_df = temp_df.reindex(sorted_index)
                    # Safety net: guarantee float32 X regardless of any dtype drift
                    # upstream (e.g. pandas DataFrame construction) -- see the
                    # np.full_like/np.full float32 comments above for why this matters
                    # at production scale.
                    X = temp_df.values.astype(np.float32, copy=False)
                    dataset, strand = sample_types[dict_index].split("_")[:2]

                    logger.info(
                        "Loading {0} dataframe for sample {1} into a temp anndata object".format(
                            sample_types[dict_index],
                            final_sample_index,
                        )
                    )
                    temp_adata = ad.AnnData(X)
                    if temp_adata.shape[0] > 0:
                        logger.info(
                            "Adding read names and position ids to {0} anndata for sample {1}".format(
                                sample_types[dict_index],
                                final_sample_index,
                            )
                        )
                        temp_adata.obs_names = temp_df.index
                        temp_adata.obs_names = temp_adata.obs_names.astype(str)
                        temp_adata.var_names = temp_df.columns
                        temp_adata.var_names = temp_adata.var_names.astype(str)
                        logger.info(
                            "Adding {0} anndata for sample {1}".format(
                                sample_types[dict_index],
                                final_sample_index,
                            )
                        )
                        temp_adata.obs[SAMPLE] = [
                            sample_name_map[final_sample_index]
                        ] * len(temp_adata)
                        if read_to_barcode is not None:
                            temp_adata.obs[BARCODE] = [
                                read_to_barcode.get(rn, "unknown")
                                for rn in temp_adata.obs_names
                            ]
                        else:
                            temp_adata.obs[BARCODE] = [
                                barcode_map[final_sample_index]
                            ] * len(temp_adata)
                        temp_adata.obs[REFERENCE] = [f"{record}"] * len(temp_adata)
                        temp_adata.obs[STRAND] = [strand] * len(temp_adata)
                        temp_adata.obs[DATASET] = [dataset] * len(temp_adata)
                        temp_adata.obs[REFERENCE_DATASET_STRAND] = [
                            f"{record}_{dataset}_{strand}"
                        ] * len(temp_adata)
                        temp_adata.obs[REFERENCE_STRAND] = [f"{record}_{strand}"] * len(
                            temp_adata
                        )

                        # Load integer-encoded reads for the current sample/record
                        sequence_files = _normalize_sequence_batch_files(
                            sequence_batch_files.get(f"{final_sample_index}_{record}", [])
                        )
                        mismatch_files = _normalize_sequence_batch_files(
                            mismatch_batch_files.get(f"{final_sample_index}_{record}", [])
                        )
                        quality_files = _normalize_sequence_batch_files(
                            quality_batch_files.get(f"{final_sample_index}_{record}", [])
                        )
                        read_span_files = _normalize_sequence_batch_files(
                            read_span_batch_files.get(f"{final_sample_index}_{record}", [])
                        )
                        if not sequence_files:
                            logger.warning(
                                "No encoded sequence batches found for sample %s record %s",
                                final_sample_index,
                                record,
                            )
                            continue
                        cache_key = f"{final_sample_index}_{record}"
                        if cache_key in sample_record_batch_cache:
                            logger.info(
                                f"Reusing cached encoded sequences for {cache_key}"
                            )
                        else:
                            logger.info(
                                f"Loading encoded sequences from {sequence_files}"
                            )
                        (
                            encoded_reads,
                            fwd_mapped_reads,
                            rev_mapped_reads,
                            mismatch_reads,
                            quality_reads,
                            read_span_reads,
                        ) = _load_sample_record_batches_cached(
                            sample_record_batch_cache,
                            cache_key,
                            sequence_files,
                            mismatch_files,
                            quality_files,
                            read_span_files,
                        )

                        read_names = list(encoded_reads.keys())

                        read_mapping_direction = []
                        for read_id in temp_adata.obs_names:
                            if read_id in fwd_mapped_reads:
                                read_mapping_direction.append("fwd")
                            elif read_id in rev_mapped_reads:
                                read_mapping_direction.append("rev")
                            else:
                                read_mapping_direction.append("unk")

                        temp_adata.obs[READ_MAPPING_DIRECTION] = read_mapping_direction

                        del temp_df

                        padding_value = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[
                            MODKIT_EXTRACT_SEQUENCE_PADDING_BASE
                        ]
                        sequence_length = encoded_reads[read_names[0]].shape[0]
                        encoded_matrix = np.full(
                            (len(sorted_index), sequence_length),
                            padding_value,
                            dtype=np.int16,
                        )

                        for j, read_name in tqdm(
                            enumerate(sorted_index),
                            desc="Loading integer-encoded reads",
                            total=len(sorted_index),
                        ):
                            encoded_matrix[j, :] = encoded_reads[read_name]

                        del encoded_reads
                        gc.collect()

                        temp_adata.layers[SEQUENCE_INTEGER_ENCODING] = encoded_matrix
                        if mismatch_reads:
                            current_reference_length = reference_dict[record][0]
                            default_mismatch_sequence = np.full(
                                sequence_length,
                                MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
                                dtype=np.int16,
                            )
                            if current_reference_length < sequence_length:
                                default_mismatch_sequence[current_reference_length:] = (
                                    padding_value
                                )
                            mismatch_matrix = np.vstack(
                                [
                                    mismatch_reads.get(read_name, default_mismatch_sequence)
                                    for read_name in sorted_index
                                ]
                            )
                            temp_adata.layers[MISMATCH_INTEGER_ENCODING] = mismatch_matrix
                        if quality_reads:
                            default_quality_sequence = np.full(
                                sequence_length, -1, dtype=np.int16
                            )
                            quality_matrix = np.vstack(
                                [
                                    quality_reads.get(read_name, default_quality_sequence)
                                    for read_name in sorted_index
                                ]
                            )
                            temp_adata.layers[BASE_QUALITY_SCORES] = quality_matrix
                        if read_span_reads:
                            default_read_span = np.zeros(sequence_length, dtype=np.int16)
                            read_span_matrix = np.vstack(
                                [
                                    read_span_reads.get(read_name, default_read_span)
                                    for read_name in sorted_index
                                ]
                            )
                            temp_adata.layers[READ_SPAN_MASK] = read_span_matrix

                        # Queue this sample's AnnData; concatenated once, after the sample loop.
                        if temp_adata.shape[0] > 0:
                            logger.info(
                                "Queuing {0} anndata object for sample {1}".format(
                                    sample_types[dict_index],
                                    final_sample_index,
                                )
                            )
                            adata_list.append(temp_adata)
                        else:
                            logger.warning(
                                f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata"
                            )
                        del temp_adata

                        gc.collect()
                    else:
                        logger.warning(
                            f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata. Skipping sample."
                        )

            if adata_list:
                logger.info(
                    "Concatenating {0} anndata objects for {1}".format(
                        len(adata_list), sample_types[dict_index]
                    )
                )
                adata = ad.concat(adata_list, join="outer", index_unique=None)
                del adata_list
                gc.collect()
                try:
                    logger.info(
                        "Writing {0} anndata out as a hdf5 file".format(
                            sample_types[dict_index]
                        )
                    )
                    adata.write_h5ad(
                        h5_dir
                        / "{0}_{1}_{2}_SMF_binarized_sample_hdf5.h5ad.gz".format(
                            readwrite.date_string(), batch, sample_types[dict_index]
                        ),
                        compression="gzip",
                    )
                except Exception:
                    logger.debug("Skipping writing anndata for sample")
            else:
                adata = None
                logger.debug(
                    "No samples had mapped reads for {0}; skipping write".format(
                        sample_types[dict_index]
                    )
                )

    # Every per-dict-type file for this batch has now been written (or correctly
    # skipped for lack of mapped reads) -- only now is it safe to mark the batch
    # complete for the resumability check at the top of this function.
    batch_complete_marker.touch()

    try:
        # Delete the batch dictionaries from memory
        del dict_list, adata
    except Exception:
        pass
    gc.collect()


def modkit_extract_to_adata(
    fasta,
    bam_dir,
    out_dir,
    input_already_demuxed,
    mapping_threshold,
    experiment_name,
    mods,
    batch_size,
    mod_tsv_dir,
    delete_batch_hdfs=False,
    threads=None,
    double_barcoded_path=None,
    samtools_backend: str | None = "auto",
    demux_backend: str | None = None,
    single_bam=None,
    barcode_sidecar=None,
    max_workers: int | str | None = None,
):
    """Convert modkit extract TSVs and BAMs into an AnnData object.

    Args:
        fasta (Path): Reference FASTA path.
        bam_dir (Path): Directory with aligned BAM files (ignored when single_bam is set).
        out_dir (Path): Output directory for intermediate and final H5ADs.
        input_already_demuxed (bool): Whether reads were already demultiplexed.
        mapping_threshold (float): Minimum fraction of mapped reads to keep a record.
        experiment_name (str): Experiment name used in output file naming.
        mods (list[str]): Modification labels to analyze (e.g., ["6mA", "5mC"]).
        batch_size (int): Number of TSVs to process per batch.
        mod_tsv_dir (Path): Directory containing modkit extract TSVs.
        delete_batch_hdfs (bool): Remove batch H5ADs after concatenation.
        threads (int | None): Thread count for parallel operations.
        double_barcoded_path (Path | None): Dorado demux summary directory for double barcodes.
        samtools_backend (str | None): Samtools backend selection.
        demux_backend (str | None): Demux backend used ("smftools" or "dorado"). If "smftools",
            demux_type annotation is skipped here and derived from BM tag later.
        single_bam: When set, use this single BAM instead of bam_dir (non-split mode).
        barcode_sidecar: Path to barcode sidecar parquet for read-to-barcode lookup in non-split mode.
        max_workers (int | str | None): If None (default), batches are processed serially
            in-process -- the same behavior as before this parameter existed. If a positive
            int, up to that many batches are processed concurrently via
            `multiprocessing.Pool`, using `batch_size` to control how many TSVs/samples
            each worker task covers (set `batch_size=1` for one worker task per sample,
            the finest available granularity). If "auto", a worker count is chosen from
            available CPU count and estimated per-batch memory footprint (see
            `_estimate_max_workers`).

    Returns:
        tuple[ad.AnnData | None, Path]: The final AnnData (if created) and its H5AD path.

    Processing Steps:
        1. Discover input TSV/BAM files and derive sample metadata.
        2. Identify records that pass mapping thresholds and build reference metadata.
        3. Encode read sequences into integer arrays and cache them.
        4. Process TSV batches into per-read methylation matrices.
        5. Concatenate batch H5ADs into a final AnnData with consensus sequences.
    """
    ###################################################
    # Package imports
    import gc
    import math

    import anndata as ad
    import numpy as np
    import pandas as pd
    from Bio.Seq import Seq
    from tqdm import tqdm

    from .. import readwrite
    from ..readwrite import make_dirs
    from .bam_functions import extract_base_identities
    from .fasta_functions import get_native_references
    ###################################################

    ################## Get input tsv and bam file names into a sorted list ################
    # Make output dirs
    h5_dir = out_dir / H5_DIR
    tmp_dir = out_dir / "tmp"
    make_dirs([h5_dir, tmp_dir])

    final_hdf = f"{experiment_name}.h5ad.gz"
    final_adata_path = h5_dir / final_hdf
    final_adata = None

    if final_adata_path.exists():
        logger.debug(f"{final_adata_path} already exists. Using existing adata")
        return final_adata, final_adata_path

    nonsplit_mode = single_bam is not None
    read_to_barcode: dict[str, str] | None = None
    nonsplit_chunk_count = 1
    use_global_sample_indices = False
    read_name_filters: dict[int, set[str]] | None = None

    if nonsplit_mode:
        from .bam_functions import build_classified_read_set

        _classified, read_to_barcode = build_classified_read_set(
            barcode_sidecar=barcode_sidecar,
            bam_path=single_bam,
        )
        # In non-split mode, TSVs come from running extract_mods on the single BAM
        base_tsvs = sorted(
            p for p in mod_tsv_dir.iterdir() if p.is_file() and "extract.tsv" in p.name
        )
        if not base_tsvs:
            raise ValueError(
                "Non-split mode expects at least one modkit extract TSV in mod_tsv_dir."
            )
        read_names = set(read_to_barcode.keys()) if read_to_barcode is not None else set()
        if read_names:
            target_chunks = int(threads) if threads else 1
            target_chunks = max(1, target_chunks)
            read_chunks = _split_read_set(read_names, target_chunks)
        else:
            read_chunks = [set()]
        nonsplit_chunk_count = len(read_chunks)
        read_name_filters = {idx: chunk for idx, chunk in enumerate(read_chunks)}
        if nonsplit_chunk_count > 1:
            # Pre-split into per-chunk filtered TSVs once, rather than pointing
            # every chunk at the same full-corpus file and re-parsing +
            # filtering it from scratch inside every batch (see
            # _split_extract_tsv_by_read_filters docstring).
            chunk_tsv_paths = _split_extract_tsv_by_read_filters(
                base_tsvs[0], read_name_filters, tmp_dir
            )
            tsvs = [chunk_tsv_paths[idx] for idx in range(nonsplit_chunk_count)]
        else:
            tsvs = [base_tsvs[0]]
        use_global_sample_indices = True
        bam_path_list = [Path(single_bam)]
        tsv_path_list = list(tsvs)
        # Build sample/barcode maps from unique barcodes in sidecar
        unique_barcodes = sorted(set(read_to_barcode.values()))
        sample_name_map = {idx: Path(single_bam).stem for idx in range(nonsplit_chunk_count)}
        barcode_map = {idx: "all" for idx in range(nonsplit_chunk_count)}
        logger.info(
            f"Non-split mode: {len(tsvs)} chunked TSV tasks from {base_tsvs[0].name}, "
            f"single BAM {single_bam}, {len(_classified)} classified reads across "
            f"{len(unique_barcodes)} barcodes in {nonsplit_chunk_count} read chunks"
        )
    else:
        tsvs, bams = _collect_input_paths(mod_tsv_dir, bam_dir)
        tsv_path_list = list(tsvs)
        bam_path_list = list(bams)
        # Map global sample index (bami / final_sample_index) -> sample name / barcode
        sample_name_map, barcode_map = _build_sample_maps(bam_path_list)
    logger.info(f"{len(tsv_path_list)} sample tsv files found: {tsv_path_list}")
    logger.info(f"{len(bam_path_list)} sample bams found: {bam_path_list}")
    ##########################################################################################

    ######### Get Record names that have over a passed threshold of mapped reads #############
    # get all records that are above a certain mapping threshold in at least one sample bam
    records_to_analyze = parallel_filter_bams(bam_path_list, mapping_threshold, samtools_backend)

    ##########################################################################################

    ########### Determine the maximum record length to analyze in the dataset ################
    # Get all references within the FASTA and indicate the length and identity of the record sequence
    max_reference_length = 0
    reference_dict = get_native_references(
        str(fasta)
    )  # returns a dict keyed by record name. Points to a tuple of (reference length, reference sequence)
    # Get the max record length in the dataset.
    for record in records_to_analyze:
        if reference_dict[record][0] > max_reference_length:
            max_reference_length = reference_dict[record][0]
    logger.info(f"Max reference length in dataset: {max_reference_length}")
    batches = math.ceil(len(tsvs) / batch_size)  # Number of batches to process
    logger.info("Processing input tsvs in {0} batches of {1} tsvs ".format(batches, batch_size))
    ##########################################################################################

    ##########################################################################################
    # Encode read sequences into integer arrays and cache in tmp_dir.
    sequence_batch_files: dict[str, list[str]] = {}
    mismatch_batch_files: dict[str, list[str]] = {}
    quality_batch_files: dict[str, list[str]] = {}
    read_span_batch_files: dict[str, list[str]] = {}
    sequence_cache_path = tmp_dir / "tmp_sequence_int_file_dict.h5ad"
    cache_needs_rebuild = True
    if sequence_cache_path.exists():
        cached_uns = ad.read_h5ad(sequence_cache_path).uns
        if "sequence_batch_files" in cached_uns:
            sequence_batch_files = cached_uns.get("sequence_batch_files", {})
            mismatch_batch_files = cached_uns.get("mismatch_batch_files", {})
            quality_batch_files = cached_uns.get("quality_batch_files", {})
            read_span_batch_files = cached_uns.get("read_span_batch_files", {})
            cache_needs_rebuild = not (
                quality_batch_files and read_span_batch_files and sequence_batch_files
            )
        else:
            sequence_batch_files = cached_uns
            cache_needs_rebuild = True
        if cache_needs_rebuild:
            logger.info(
                "Cached sequence batches missing quality or read-span data; rebuilding cache."
            )
        else:
            logger.debug("Found existing integer-encoded reads, using these")
    if cache_needs_rebuild:
        for bami, bam in enumerate(bam_path_list):
            logger.info(
                f"Extracting base level sequences, qualities, reference spans, and mismatches per read for bam {bami}"
            )
            for record in records_to_analyze:
                current_reference_length = reference_dict[record][0]
                positions = range(current_reference_length)
                ref_seq = reference_dict[record][1]
                (
                    fwd_base_identities,
                    rev_base_identities,
                    _mismatch_counts_per_read,
                    _mismatch_trend_per_read,
                    mismatch_base_identities,
                    base_quality_scores,
                    read_span_masks,
                ) = extract_base_identities(
                    bam,
                    record,
                    positions,
                    max_reference_length,
                    ref_seq,
                    samtools_backend,
                    primary_only=nonsplit_mode,
                    read_name_filter=set(read_to_barcode.keys()) if read_to_barcode else None,
                )
                mismatch_fwd = {
                    read_name: mismatch_base_identities[read_name]
                    for read_name in fwd_base_identities
                }
                mismatch_rev = {
                    read_name: mismatch_base_identities[read_name]
                    for read_name in rev_base_identities
                }
                quality_fwd = {
                    read_name: base_quality_scores[read_name] for read_name in fwd_base_identities
                }
                quality_rev = {
                    read_name: base_quality_scores[read_name] for read_name in rev_base_identities
                }
                read_span_fwd = {
                    read_name: read_span_masks[read_name] for read_name in fwd_base_identities
                }
                read_span_rev = {
                    read_name: read_span_masks[read_name] for read_name in rev_base_identities
                }
                if nonsplit_mode and nonsplit_chunk_count > 1:
                    # Write genuinely per-chunk-filtered files directly, instead
                    # of writing one full-corpus file set and aliasing it onto
                    # every pseudo-sample key: every chunk previously pointed at
                    # the SAME full-corpus files, so every batch's worker had to
                    # load and filter down from the entire dataset just to use
                    # its own ~1/nonsplit_chunk_count slice. This was the other
                    # (bigger) half of the per-batch memory blowup alongside the
                    # unsplit TSV (see _split_extract_tsv_by_read_filters).
                    for chunk_idx in range(nonsplit_chunk_count):
                        chunk_reads = read_name_filters.get(chunk_idx, set())
                        key = f"{chunk_idx}_{record}"
                        fwd_sequence_files = _write_sequence_batches(
                            _filter_reads_by_names(fwd_base_identities, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_fwd",
                            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
                            current_reference_length,
                            batch_size=100000,
                        )
                        rev_sequence_files = _write_sequence_batches(
                            _filter_reads_by_names(rev_base_identities, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_rev",
                            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
                            current_reference_length,
                            batch_size=100000,
                        )
                        sequence_batch_files[key] = fwd_sequence_files + rev_sequence_files
                        mismatch_fwd_files = _write_integer_batches(
                            _filter_reads_by_names(mismatch_fwd, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_mismatch_fwd",
                            batch_size=100000,
                        )
                        mismatch_rev_files = _write_integer_batches(
                            _filter_reads_by_names(mismatch_rev, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_mismatch_rev",
                            batch_size=100000,
                        )
                        mismatch_batch_files[key] = mismatch_fwd_files + mismatch_rev_files
                        quality_fwd_files = _write_integer_batches(
                            _filter_reads_by_names(quality_fwd, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_quality_fwd",
                            batch_size=100000,
                        )
                        quality_rev_files = _write_integer_batches(
                            _filter_reads_by_names(quality_rev, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_quality_rev",
                            batch_size=100000,
                        )
                        quality_batch_files[key] = quality_fwd_files + quality_rev_files
                        read_span_fwd_files = _write_integer_batches(
                            _filter_reads_by_names(read_span_fwd, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_read_span_fwd",
                            batch_size=100000,
                        )
                        read_span_rev_files = _write_integer_batches(
                            _filter_reads_by_names(read_span_rev, chunk_reads),
                            tmp_dir,
                            record,
                            f"{chunk_idx}_read_span_rev",
                            batch_size=100000,
                        )
                        read_span_batch_files[key] = read_span_fwd_files + read_span_rev_files
                else:
                    fwd_sequence_files = _write_sequence_batches(
                        fwd_base_identities,
                        tmp_dir,
                        record,
                        f"{bami}_fwd",
                        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
                        current_reference_length,
                        batch_size=100000,
                    )
                    rev_sequence_files = _write_sequence_batches(
                        rev_base_identities,
                        tmp_dir,
                        record,
                        f"{bami}_rev",
                        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
                        current_reference_length,
                        batch_size=100000,
                    )
                    sequence_batch_files[f"{bami}_{record}"] = (
                        fwd_sequence_files + rev_sequence_files
                    )
                    mismatch_fwd_files = _write_integer_batches(
                        mismatch_fwd,
                        tmp_dir,
                        record,
                        f"{bami}_mismatch_fwd",
                        batch_size=100000,
                    )
                    mismatch_rev_files = _write_integer_batches(
                        mismatch_rev,
                        tmp_dir,
                        record,
                        f"{bami}_mismatch_rev",
                        batch_size=100000,
                    )
                    mismatch_batch_files[f"{bami}_{record}"] = (
                        mismatch_fwd_files + mismatch_rev_files
                    )
                    quality_fwd_files = _write_integer_batches(
                        quality_fwd,
                        tmp_dir,
                        record,
                        f"{bami}_quality_fwd",
                        batch_size=100000,
                    )
                    quality_rev_files = _write_integer_batches(
                        quality_rev,
                        tmp_dir,
                        record,
                        f"{bami}_quality_rev",
                        batch_size=100000,
                    )
                    quality_batch_files[f"{bami}_{record}"] = (
                        quality_fwd_files + quality_rev_files
                    )
                    read_span_fwd_files = _write_integer_batches(
                        read_span_fwd,
                        tmp_dir,
                        record,
                        f"{bami}_read_span_fwd",
                        batch_size=100000,
                    )
                    read_span_rev_files = _write_integer_batches(
                        read_span_rev,
                        tmp_dir,
                        record,
                        f"{bami}_read_span_rev",
                        batch_size=100000,
                    )
                    read_span_batch_files[f"{bami}_{record}"] = (
                        read_span_fwd_files + read_span_rev_files
                    )
                del (
                    fwd_base_identities,
                    rev_base_identities,
                    mismatch_base_identities,
                    base_quality_scores,
                    read_span_masks,
                )
        ad.AnnData(
            X=np.random.rand(1, 1),
            uns={
                "sequence_batch_files": sequence_batch_files,
                "mismatch_batch_files": mismatch_batch_files,
                "quality_batch_files": quality_batch_files,
                "read_span_batch_files": read_span_batch_files,
            },
        ).write_h5ad(sequence_cache_path)

    # Chunk keys are now written directly per-chunk above (see the
    # nonsplit_mode branch in the cache-rebuild loop) rather than aliased from
    # a single full-corpus key afterward, so there is nothing to do here.
    ##########################################################################################

    ##########################################################################################
    # Iterate over records to analyze and return a dictionary keyed by the reference name that points to a tuple containing the top strand sequence and the complement
    record_seq_dict = {}
    for record in records_to_analyze:
        current_reference_length = reference_dict[record][0]
        delta_max_length = max_reference_length - current_reference_length
        sequence = reference_dict[record][1] + "N" * delta_max_length
        complement = (
            str(Seq(reference_dict[record][1]).complement()).upper() + "N" * delta_max_length
        )
        record_seq_dict[record] = (sequence, complement)
    ##########################################################################################

    ###################################################
    ###################################################
    # Begin iterating over batches -- either serially (max_workers is None, the
    # original behavior) or dispatched across a process pool (max_workers set).
    #
    # Precompute every batch's (tsv slice, sample-index slice) up front. This
    # sequential slicing must happen in this process (each batch consumes from a
    # shared, mutating tsv/bam path queue) -- but it is cheap (just list slicing),
    # unlike the actual per-batch work in _process_one_batch, which is what
    # benefits from running concurrently.
    batch_specs: list[tuple[int, list, object]] = []
    _tsv_remaining = list(tsv_path_list)
    for batch in range(batches):
        if batch == batches - 1:
            tsv_batch = _tsv_remaining
        else:
            tsv_batch = _tsv_remaining[:batch_size]
            _tsv_remaining = _tsv_remaining[batch_size:]
        if use_global_sample_indices:
            sample_indices_batch = list(
                range(batch * batch_size, batch * batch_size + len(tsv_batch))
            )
        else:
            sample_indices_batch = None
        logger.info("tsvs in batch {0} ".format(tsv_batch))
        batch_specs.append((batch, tsv_batch, sample_indices_batch))

    # `sample_record_batch_cache` (see _process_one_batch) holds up to one
    # (bam_index, record) key per TSV/sample in the batch, per record -- reset
    # only at batch boundaries -- so that many keys' on-disk files can be
    # resident in one worker at once.
    keys_per_batch = max(1, batch_size) * max(1, len(records_to_analyze))
    # Exact reads-per-batch is only cheaply known in non-split mode, where
    # `read_name_filters` gives the precise read set per chunk; leaving this at
    # 0 in split mode omits the modification-dict memory term rather than
    # guessing it from an indirect proxy (see _estimate_max_workers docstring).
    reads_per_batch = (
        max((len(chunk) for chunk in read_name_filters.values()), default=0)
        if read_name_filters
        else 0
    )
    n_mod_dict_types = _count_active_mod_dict_types(mods)

    resolved_max_workers = _resolve_max_workers(
        max_workers,
        len(batch_specs),
        threads,
        sequence_batch_files,
        mismatch_batch_files,
        quality_batch_files,
        read_span_batch_files,
        keys_per_batch=keys_per_batch,
        reads_per_batch=reads_per_batch,
        max_reference_length=max_reference_length,
        n_mod_dict_types=n_mod_dict_types,
    )

    # `max_workers is None` is a deliberate, explicit opt-out of multiprocessing
    # entirely (see _resolve_max_workers) -- respect that literally with a plain
    # in-process loop. Anything else (`"auto"` or an explicit int) means the
    # caller wanted parallelism and the memory estimate is what decided the
    # final count, possibly capping it down to 1 for safety -- that "1" must
    # still go through the pool below (even as a single-process pool) so it's
    # covered by the memory watchdog. A resolved-to-1 case is, by definition,
    # the memory-constrained case that most needs runtime protection if the
    # estimate itself turns out to be wrong.
    if max_workers is None:
        for batch, tsv_batch, sample_indices_batch in batch_specs:
            logger.info("Processing tsvs for batch {0} ".format(batch))
            _process_one_batch(
                batch,
                tsv_batch,
                sample_indices_batch,
                records_to_analyze,
                reference_dict,
                max_reference_length,
                mods,
                batch_size,
                threads,
                read_name_filters,
                use_global_sample_indices,
                sample_name_map,
                barcode_map,
                read_to_barcode,
                h5_dir,
                sequence_batch_files,
                mismatch_batch_files,
                quality_batch_files,
                read_span_batch_files,
            )
    else:
        import multiprocessing as mp

        from ..memory_guard import start_worker_watchdog

        logger.info(
            "Processing %d batches across up to %d worker processes",
            len(batch_specs),
            resolved_max_workers,
        )
        # Same inputs/estimate _resolve_max_workers just used to size the pool,
        # reused here as the per-worker kill threshold for platforms (macOS)
        # without a process-tree memory cap; see smftools.memory_guard. A no-op
        # on Linux, where enable_aggregate_memory_cap() (set up once at CLI
        # startup) already covers the whole process tree.
        per_worker_budget_bytes = _estimate_worker_peak_bytes(
            sequence_batch_files,
            mismatch_batch_files,
            quality_batch_files,
            read_span_batch_files,
            keys_per_batch=keys_per_batch,
            reads_per_batch=reads_per_batch,
            max_reference_length=max_reference_length,
            n_mod_dict_types=n_mod_dict_types,
        )
        with mp.Pool(processes=resolved_max_workers, maxtasksperchild=1) as pool:
            stop_watchdog = start_worker_watchdog(pool, per_worker_budget_bytes)
            try:
                async_results = [
                    pool.apply_async(
                        _process_one_batch,
                        (
                            batch,
                            tsv_batch,
                            sample_indices_batch,
                            records_to_analyze,
                            reference_dict,
                            max_reference_length,
                            mods,
                            batch_size,
                            threads,
                            read_name_filters,
                            use_global_sample_indices,
                            sample_name_map,
                            barcode_map,
                            read_to_barcode,
                            h5_dir,
                            sequence_batch_files,
                            mismatch_batch_files,
                            quality_batch_files,
                            read_span_batch_files,
                        ),
                    )
                    for batch, tsv_batch, sample_indices_batch in batch_specs
                ]
                pool.close()
                errors = []
                for (batch, _tsv_batch, _sample_indices_batch), result in zip(
                    batch_specs, async_results
                ):
                    try:
                        # A finite timeout matters here specifically because the
                        # watchdog above can kill a worker mid-task: a vanilla
                        # multiprocessing.Pool does not reliably raise for a
                        # result whose worker was killed out from under it, so
                        # an unbounded .get() could hang forever in that case.
                        # This is a hang backstop, not a performance SLA.
                        result.get(timeout=_BATCH_RESULT_TIMEOUT_SECONDS)
                    except Exception as exc:  # noqa: BLE001 - surface every worker failure
                        logger.error("Batch %d failed in worker process: %s", batch, exc)
                        errors.append((batch, exc))
                pool.join()
            finally:
                stop_watchdog()
            if errors:
                raise RuntimeError(
                    f"{len(errors)} of {len(batch_specs)} batches failed in worker "
                    f"processes: {[batch for batch, _ in errors]}"
                )

    # Iterate over all of the batched hdf5 files and concatenate them.
    files = h5_dir.iterdir()
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if "hdf5.h5ad" in hdf.name and hdf != final_hdf]
    combined_hdfs = [hdf for hdf in hdfs if "combined" in hdf.name]
    if len(combined_hdfs) > 0:
        hdfs = combined_hdfs
    else:
        pass
    # Sort file list by names and print the list of file names
    hdfs.sort()
    logger.info("{0} sample files found: {1}".format(len(hdfs), hdfs))
    hdf_paths = [hd5 for hd5 in hdfs]
    if not hdf_paths:
        raise RuntimeError(
            "modkit_extract_to_adata produced no batch hdf5 files to concatenate; "
            "no reads survived filtering or all samples/records were empty."
        )
    # Concatenate on disk (anndata.experimental.concat_on_disk) instead of eagerly
    # loading every batch hdf5 into memory before concatenating: the previous
    # `[ad.read_h5ad(f) for f in hdf_paths]` held all N batch files resident at
    # once (on top of whatever the worker pool above was still holding), which
    # was a second, independent memory spike that grew with batch count. This
    # streams each input's arrays through in chunks and never holds more than
    # one input's worth in memory. The result is still reloaded into memory
    # once afterward, since the annotation/consensus-sequence logic below
    # mutates `final_adata` in place and needs it as a normal AnnData.
    #
    # concat_on_disk's own path handling picks zarr vs h5py by literal
    # `Path.suffix == ".h5ad"`, which misreads our `*.h5ad.gz` batch files
    # (suffix is ".gz") as zarr stores and fails. Open real h5py.File handles
    # ourselves so it dispatches on the (correct) HDF5 group type instead.
    import h5py
    from anndata.experimental import concat_on_disk

    concat_tmp_path = h5_dir / f"_concat_tmp_{final_hdf}"
    if concat_tmp_path.exists():
        concat_tmp_path.unlink()
    logger.info(
        "Concatenating {} batch hdf5 files on disk (concat_on_disk)".format(len(hdf_paths))
    )
    with contextlib.ExitStack() as stack:
        input_groups = [stack.enter_context(h5py.File(p, mode="r")) for p in hdf_paths]
        output_group = stack.enter_context(h5py.File(concat_tmp_path, mode="w"))
        concat_on_disk(input_groups, output_group, join="outer", index_unique=None)
    final_adata = ad.read_h5ad(concat_tmp_path)
    concat_tmp_path.unlink()
    gc.collect()

    # Set obs columns to type 'category'
    for col in final_adata.obs.columns:
        final_adata.obs[col] = final_adata.obs[col].astype("category")

    final_adata.uns[f"{SEQUENCE_INTEGER_ENCODING}_map"] = dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT)
    final_adata.uns[f"{MISMATCH_INTEGER_ENCODING}_map"] = dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT)
    final_adata.uns[f"{SEQUENCE_INTEGER_DECODING}_map"] = {
        str(key): value for key, value in MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE.items()
    }

    consensus_bases = MODKIT_EXTRACT_SEQUENCE_BASES[:4]  # ignore N/PAD for consensus
    consensus_base_ints = [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[base] for base in consensus_bases]
    final_adata.uns["References"] = {}
    for record in records_to_analyze:
        # Add FASTA sequence to the object
        sequence = record_seq_dict[record][0]
        complement = record_seq_dict[record][1]
        final_adata.var[f"{record}_top_strand_FASTA_base"] = list(sequence)
        final_adata.var[f"{record}_bottom_strand_FASTA_base"] = list(complement)
        final_adata.uns[f"{record}_FASTA_sequence"] = sequence
        final_adata.uns["References"][f"{record}_FASTA_sequence"] = sequence
        # Add consensus sequence of samples mapped to the record to the object
        record_subset = final_adata[final_adata.obs[REFERENCE] == record]
        for strand in record_subset.obs[STRAND].cat.categories:
            strand_subset = record_subset[record_subset.obs[STRAND] == strand]
            for mapping_dir in strand_subset.obs[READ_MAPPING_DIRECTION].cat.categories:
                mapping_dir_subset = strand_subset[
                    strand_subset.obs[READ_MAPPING_DIRECTION] == mapping_dir
                ]
                encoded_sequences = mapping_dir_subset.layers[SEQUENCE_INTEGER_ENCODING]
                layer_counts = [
                    np.sum(encoded_sequences == base_int, axis=0)
                    for base_int in consensus_base_ints
                ]
                count_array = np.array(layer_counts)
                nucleotide_indexes = np.argmax(count_array, axis=0)
                consensus_sequence_list = [consensus_bases[i] for i in nucleotide_indexes]
                no_calls_mask = np.sum(count_array, axis=0) == 0
                if np.any(no_calls_mask):
                    consensus_sequence_list = np.array(consensus_sequence_list, dtype=object)
                    consensus_sequence_list[no_calls_mask] = "N"
                    consensus_sequence_list = consensus_sequence_list.tolist()
                final_adata.var[
                    f"{record}_{strand}_{mapping_dir}_consensus_sequence_from_all_samples"
                ] = consensus_sequence_list

    from .h5ad_functions import append_reference_strand_quality_stats

    append_reference_strand_quality_stats(final_adata)

    demux_type_mode = _resolve_demux_type_annotation_mode(
        input_already_demuxed, demux_backend, double_barcoded_path
    )
    if demux_type_mode == "already":
        final_adata.obs[DEMUX_TYPE] = ["already"] * final_adata.shape[0]
        final_adata.obs[DEMUX_TYPE] = final_adata.obs[DEMUX_TYPE].astype("category")
    elif demux_type_mode == "skip_smftools":
        # Skip demux_type annotation here - will be derived from BM tag after BAM tags are loaded
        logger.info("Skipping demux_type annotation (will be derived from BM tag)")
    elif demux_type_mode == "dorado_barcoding_summary":
        # Dorado backend - use barcoding_summary.txt
        from .h5ad_functions import add_demux_type_annotation

        double_barcoded_reads = double_barcoded_path / "barcoding_summary.txt"
        add_demux_type_annotation(final_adata, double_barcoded_reads)
    else:
        # Dorado backend but no double-barcoded demux summary is available (e.g.
        # skip_bam_split=True never produced per-end BM scoring). Nothing to derive
        # demux_type from, so skip annotation the same way the smftools-backend branch does.
        logger.info(
            "No double-barcoded demux summary available (double_barcoded_path is None); "
            "skipping demux_type annotation"
        )

    # Delete the individual h5ad files and only keep the final concatenated file
    if delete_batch_hdfs:
        delete_intermediate_h5ads_and_tmpdir(h5_dir, tmp_dir)

    return final_adata, final_adata_path
