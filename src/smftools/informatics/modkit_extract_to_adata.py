from __future__ import annotations

import concurrent.futures
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

    existing_h5s = h5_dir.iterdir()
    existing_h5s = [h5 for h5 in existing_h5s if ".h5ad.gz" in str(h5)]
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
        # Reuse the same TSV path for each read chunk; each pseudo-sample applies a read filter.
        tsvs = [base_tsvs[0]] * nonsplit_chunk_count
        read_name_filters = {idx: chunk for idx, chunk in enumerate(read_chunks)}
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
                sequence_batch_files[f"{bami}_{record}"] = fwd_sequence_files + rev_sequence_files
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
                mismatch_batch_files[f"{bami}_{record}"] = mismatch_fwd_files + mismatch_rev_files
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
                quality_batch_files[f"{bami}_{record}"] = quality_fwd_files + quality_rev_files
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

    # In non-split chunked mode, pseudo-sample indices share the same single-BAM sequence caches.
    if nonsplit_mode and nonsplit_chunk_count > 1:
        for chunk_idx in range(1, nonsplit_chunk_count):
            for record in records_to_analyze:
                src_key = f"0_{record}"
                dst_key = f"{chunk_idx}_{record}"
                if src_key in sequence_batch_files:
                    sequence_batch_files[dst_key] = sequence_batch_files[src_key]
                if src_key in mismatch_batch_files:
                    mismatch_batch_files[dst_key] = mismatch_batch_files[src_key]
                if src_key in quality_batch_files:
                    quality_batch_files[dst_key] = quality_batch_files[src_key]
                if src_key in read_span_batch_files:
                    read_span_batch_files[dst_key] = read_span_batch_files[src_key]
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
    # Begin iterating over batches
    for batch in range(batches):
        logger.info("Processing tsvs for batch {0} ".format(batch))
        # For the final batch, just take the remaining tsv and bam files
        if batch == batches - 1:
            tsv_batch = tsv_path_list
            bam_batch = bam_path_list
            if use_global_sample_indices:
                sample_indices_batch = list(
                    range(batch * batch_size, batch * batch_size + len(tsv_batch))
                )
            else:
                sample_indices_batch = None
        # For all other batches, take the next batch of tsvs and bams out of the file queue.
        else:
            tsv_batch = tsv_path_list[:batch_size]
            bam_batch = bam_path_list[:batch_size]
            tsv_path_list = tsv_path_list[batch_size:]
            bam_path_list = bam_path_list[batch_size:]
            if use_global_sample_indices:
                sample_indices_batch = list(
                    range(batch * batch_size, batch * batch_size + len(tsv_batch))
                )
            else:
                sample_indices_batch = None
        logger.info("tsvs in batch {0} ".format(tsv_batch))

        batch_already_processed = sum([1 for h5 in existing_h5s if f"_{batch}_" in h5.name])
        ###################################################
        if batch_already_processed:
            logger.debug(
                f"Batch {batch} has already been processed into h5ads. Skipping batch and using existing files"
            )
        else:
            ###################################################
            ### Add the tsvs as dataframes to a dictionary (dict_total) keyed by integer index. Also make modification specific dictionaries and strand specific dictionaries.
            # # Initialize dictionaries and place them in a list
            batch_dicts = ModkitBatchDictionaries()
            dict_list = batch_dicts.as_list()
            sample_types = batch_dicts.sample_types

            # # Step 1):Load the dict_total dictionary with all of the batch tsv files as dataframes.
            dict_total = parallel_load_tsvs(
                tsv_batch,
                records_to_analyze,
                reference_dict,
                batch,
                batch_size=len(tsv_batch),
                threads=threads,
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

                                # Vectorized methylation calculation with NaN for other codes
                                methylation_prob = np.full_like(
                                    probabilities, np.nan
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
                                        max_reference_length, np.nan
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
            for dict_index, dict_type in enumerate(dict_list):
                if dict_index not in dict_to_skip:
                    # Initialize an hdf5 file for the current modified strand
                    adata = None
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
                            X = temp_df.values
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
                                logger.info(f"Loading encoded sequences from {sequence_files}")
                                (
                                    encoded_reads,
                                    fwd_mapped_reads,
                                    rev_mapped_reads,
                                ) = _load_sequence_batches(sequence_files)
                                mismatch_reads: dict[str, np.ndarray] = {}
                                if mismatch_files:
                                    (
                                        mismatch_reads,
                                        _mismatch_fwd_reads,
                                        _mismatch_rev_reads,
                                    ) = _load_sequence_batches(mismatch_files)
                                quality_reads: dict[str, np.ndarray] = {}
                                if quality_files:
                                    quality_reads = _load_integer_batches(quality_files)
                                read_span_reads: dict[str, np.ndarray] = {}
                                if read_span_files:
                                    read_span_reads = _load_integer_batches(read_span_files)

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

                                # If final adata object already has a sample loaded, concatenate the current sample into the existing adata object
                                if adata:
                                    if temp_adata.shape[0] > 0:
                                        logger.info(
                                            "Concatenating {0} anndata object for sample {1}".format(
                                                sample_types[dict_index],
                                                final_sample_index,
                                            )
                                        )
                                        adata = ad.concat(
                                            [adata, temp_adata], join="outer", index_unique=None
                                        )
                                        del temp_adata
                                    else:
                                        logger.warning(
                                            f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata"
                                        )
                                else:
                                    if temp_adata.shape[0] > 0:
                                        logger.info(
                                            "Initializing {0} anndata object for sample {1}".format(
                                                sample_types[dict_index],
                                                final_sample_index,
                                            )
                                        )
                                        adata = temp_adata
                                    else:
                                        logger.warning(
                                            f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata"
                                        )

                                gc.collect()
                            else:
                                logger.warning(
                                    f"{sample} did not have any mapped reads on {record}_{dataset}_{strand}, omiting from final adata. Skipping sample."
                                )

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

            try:
                # Delete the batch dictionaries from memory
                del dict_list, adata
            except Exception:
                pass
            gc.collect()

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
    final_adata = None
    for hdf_index, hdf in enumerate(hdf_paths):
        logger.info("Reading in {} hdf5 file".format(hdfs[hdf_index]))
        temp_adata = ad.read_h5ad(hdf)
        if final_adata:
            logger.info(
                "Concatenating final adata object with {} hdf5 file".format(hdfs[hdf_index])
            )
            final_adata = ad.concat([final_adata, temp_adata], join="outer", index_unique=None)
        else:
            logger.info("Initializing final adata object with {} hdf5 file".format(hdfs[hdf_index]))
            final_adata = temp_adata
        del temp_adata

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

    if input_already_demuxed:
        final_adata.obs[DEMUX_TYPE] = ["already"] * final_adata.shape[0]
        final_adata.obs[DEMUX_TYPE] = final_adata.obs[DEMUX_TYPE].astype("category")
    elif demux_backend and demux_backend.lower() == "smftools":
        # Skip demux_type annotation here - will be derived from BM tag after BAM tags are loaded
        logger.info("Skipping demux_type annotation (will be derived from BM tag)")
    else:
        # Dorado backend - use barcoding_summary.txt
        from .h5ad_functions import add_demux_type_annotation

        double_barcoded_reads = double_barcoded_path / "barcoding_summary.txt"
        add_demux_type_annotation(final_adata, double_barcoded_reads)

    # Delete the individual h5ad files and only keep the final concatenated file
    if delete_batch_hdfs:
        delete_intermediate_h5ads_and_tmpdir(h5_dir, tmp_dir)

    return final_adata, final_adata_path
