from __future__ import annotations

import gc
import logging
import shutil
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Manager, Pool, current_process
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from smftools.constants import (
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    MODKIT_EXTRACT_SEQUENCE_BASES,
    MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE,
    MODKIT_EXTRACT_SEQUENCE_PADDING_BASE,
)
from smftools.logging_utils import get_logger, setup_logging
from smftools.optional_imports import require

from ..readwrite import make_dirs
from .bam_functions import count_aligned_reads, extract_base_identities
from .binarize_converted_base_identities import binarize_converted_base_identities
from .fasta_functions import find_conversion_sites

logger = get_logger(__name__)

if TYPE_CHECKING:
    import torch

torch = require("torch", extra="torch", purpose="converted BAM processing")


@dataclass(frozen=True)
class RecordFastaInfo:
    """Structured FASTA metadata for a single converted record.

    Attributes:
        sequence: Padded top-strand sequence for the record.
        complement: Padded bottom-strand sequence for the record.
        chromosome: Canonical chromosome name for the record.
        unconverted_name: FASTA record name for the unconverted reference.
        sequence_length: Length of the unpadded reference sequence.
        padding_length: Number of padded bases applied to reach max length.
        conversion: Conversion label (e.g., "unconverted", "5mC").
        strand: Strand label ("top" or "bottom").
        max_reference_length: Maximum reference length across all records.
    """

    sequence: str
    complement: str
    chromosome: str
    unconverted_name: str
    sequence_length: int
    padding_length: int
    conversion: str
    strand: str
    max_reference_length: int


@dataclass(frozen=True)
class SequenceEncodingConfig:
    """Configuration for integer sequence encoding.

    Attributes:
        base_to_int: Mapping of base characters to integer encodings.
        bases: Valid base characters used for encoding.
        padding_base: Base token used for padding.
        batch_size: Number of reads per temporary batch file.
    """

    base_to_int: Mapping[str, int]
    bases: tuple[str, ...]
    padding_base: str
    batch_size: int = 100000

    @property
    def padding_value(self) -> int:
        """Return the integer value used for padding positions."""
        return self.base_to_int[self.padding_base]

    @property
    def unknown_value(self) -> int:
        """Return the integer value used for unknown bases."""
        return self.base_to_int["N"]


SEQUENCE_ENCODING_CONFIG = SequenceEncodingConfig(
    base_to_int=MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    bases=MODKIT_EXTRACT_SEQUENCE_BASES,
    padding_base=MODKIT_EXTRACT_SEQUENCE_PADDING_BASE,
)


def converted_BAM_to_adata(
    converted_FASTA: str | Path,
    split_dir: Path,
    output_dir: Path,
    input_already_demuxed: bool,
    mapping_threshold: float,
    experiment_name: str,
    conversions: list[str],
    bam_suffix: str,
    device: str | torch.device = "cpu",
    num_threads: int = 8,
    deaminase_footprinting: bool = False,
    delete_intermediates: bool = True,
    double_barcoded_path: Path | None = None,
    samtools_backend: str | None = "auto",
) -> tuple[ad.AnnData | None, Path]:
    """Convert converted BAM files into an AnnData object with integer sequence encoding.

    Args:
        converted_FASTA: Path to the converted FASTA reference.
        split_dir: Directory containing converted BAM files.
        output_dir: Output directory for intermediate and final files.
        input_already_demuxed: Whether input reads were originally demultiplexed.
        mapping_threshold: Minimum fraction of aligned reads required for inclusion.
        experiment_name: Name for the output AnnData object.
        conversions: List of modification types (e.g., ``["unconverted", "5mC", "6mA"]``).
        bam_suffix: File suffix for BAM files.
        device: Torch device or device string.
        num_threads: Number of parallel processing threads.
        deaminase_footprinting: Whether the footprinting used direct deamination chemistry.
        delete_intermediates: Whether to remove intermediate files after processing.
        double_barcoded_path: Path to dorado demux summary file of double-ended barcodes.
        samtools_backend: Samtools backend choice for alignment parsing.

    Returns:
        tuple[anndata.AnnData | None, Path]: The AnnData object (if generated) and its path.

    Processing Steps:
        1. Resolve the best available torch device and create output directories.
        2. Load converted FASTA records and compute conversion sites.
        3. Filter BAMs based on mapping thresholds.
        4. Process each BAM in parallel, building per-sample H5AD files.
        5. Concatenate per-sample AnnData objects and attach reference metadata.
        6. Add demultiplexing annotations and clean intermediate artifacts.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.debug(f"Using device: {device}")

    ## Set Up Directories and File Paths
    h5_dir = output_dir / "h5ads"
    tmp_dir = output_dir / "tmp"
    final_adata = None
    final_adata_path = h5_dir / f"{experiment_name}.h5ad.gz"

    if final_adata_path.exists():
        logger.debug(f"{final_adata_path} already exists. Using existing AnnData object.")
        return final_adata, final_adata_path

    make_dirs([h5_dir, tmp_dir])

    bam_files = sorted(
        p
        for p in split_dir.iterdir()
        if p.is_file() and p.suffix == ".bam" and "unclassified" not in p.name
    )

    bam_path_list = bam_files

    bam_names = [bam.name for bam in bam_files]
    logger.info(f"Found {len(bam_files)} BAM files within {split_dir}: {bam_names}")

    ## Process Conversion Sites
    max_reference_length, record_FASTA_dict, chromosome_FASTA_dict = process_conversion_sites(
        converted_FASTA, conversions, deaminase_footprinting
    )

    ## Filter BAM Files by Mapping Threshold
    records_to_analyze = filter_bams_by_mapping_threshold(
        bam_path_list, bam_files, mapping_threshold, samtools_backend
    )

    ## Process BAMs in Parallel
    final_adata = process_bams_parallel(
        bam_path_list,
        records_to_analyze,
        record_FASTA_dict,
        chromosome_FASTA_dict,
        tmp_dir,
        h5_dir,
        num_threads,
        max_reference_length,
        device,
        deaminase_footprinting,
        samtools_backend,
    )

    final_adata.uns["sequence_integer_encoding_map"] = dict(
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
    )
    final_adata.uns["sequence_integer_decoding_map"] = {
        str(key): value for key, value in MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE.items()
    }

    final_adata.uns["References"] = {}
    for chromosome, [seq, comp] in chromosome_FASTA_dict.items():
        final_adata.var[f"{chromosome}_top_strand_FASTA_base"] = list(seq)
        final_adata.var[f"{chromosome}_bottom_strand_FASTA_base"] = list(comp)
        final_adata.uns[f"{chromosome}_FASTA_sequence"] = seq
        final_adata.uns["References"][f"{chromosome}_FASTA_sequence"] = seq

    final_adata.obs_names_make_unique()
    cols = final_adata.obs.columns

    # Make obs cols categorical
    for col in cols:
        final_adata.obs[col] = final_adata.obs[col].astype("category")

    consensus_bases = MODKIT_EXTRACT_SEQUENCE_BASES[:4]  # ignore N/PAD for consensus
    consensus_base_ints = [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[base] for base in consensus_bases]
    for ref_group in final_adata.obs["Reference_dataset_strand"].cat.categories:
        group_subset = final_adata[final_adata.obs["Reference_dataset_strand"] == ref_group]
        encoded_sequences = group_subset.layers["sequence_integer_encoding"]
        layer_counts = [
            np.sum(encoded_sequences == base_int, axis=0) for base_int in consensus_base_ints
        ]
        count_array = np.array(layer_counts)
        nucleotide_indexes = np.argmax(count_array, axis=0)
        consensus_sequence_list = [consensus_bases[i] for i in nucleotide_indexes]
        no_calls_mask = np.sum(count_array, axis=0) == 0
        if np.any(no_calls_mask):
            consensus_sequence_list = np.array(consensus_sequence_list, dtype=object)
            consensus_sequence_list[no_calls_mask] = "N"
            consensus_sequence_list = consensus_sequence_list.tolist()
        final_adata.var[f"{ref_group}_consensus_sequence_from_all_samples"] = (
            consensus_sequence_list
        )

    if input_already_demuxed:
        final_adata.obs["demux_type"] = ["already"] * final_adata.shape[0]
        final_adata.obs["demux_type"] = final_adata.obs["demux_type"].astype("category")
    else:
        from .h5ad_functions import add_demux_type_annotation

        double_barcoded_reads = double_barcoded_path / "barcoding_summary.txt"
        logger.info("Adding demux type to each read")
        add_demux_type_annotation(final_adata, double_barcoded_reads)

    ## Delete intermediate h5ad files and temp directories
    if delete_intermediates:
        logger.info("Deleting intermediate h5ad files")
        delete_intermediate_h5ads_and_tmpdir(h5_dir, tmp_dir)

    return final_adata, final_adata_path


def process_conversion_sites(
    converted_FASTA: str | Path,
    conversions: list[str] | None = None,
    deaminase_footprinting: bool = False,
) -> tuple[int, dict[str, RecordFastaInfo], dict[str, tuple[str, str]]]:
    """Extract conversion sites and FASTA metadata for converted references.

    Args:
        converted_FASTA: Path to the converted reference FASTA.
        conversions: List of modification types (e.g., ["unconverted", "5mC", "6mA"]).
        deaminase_footprinting: Whether the footprinting was done with direct deamination chemistry.

    Returns:
        tuple[int, dict[str, RecordFastaInfo], dict[str, tuple[str, str]]]:
            Maximum reference length, record metadata, and chromosome sequences.

    Processing Steps:
        1. Parse unconverted FASTA records to determine the max reference length.
        2. Build record metadata for unconverted and converted strands.
        3. Cache chromosome-level sequences for downstream annotation.
    """
    if conversions is None:
        conversions = ["unconverted", "5mC"]
    modification_dict: dict[str, dict] = {}
    record_FASTA_dict: dict[str, RecordFastaInfo] = {}
    chromosome_FASTA_dict: dict[str, tuple[str, str]] = {}
    max_reference_length = 0
    unconverted = conversions[0]
    conversion_types = conversions[1:]

    # Process the unconverted sequence once
    modification_dict[unconverted] = find_conversion_sites(
        converted_FASTA, unconverted, conversions, deaminase_footprinting
    )
    # Above points to record_dict[record.id] = [sequence_length, [], [], sequence, complement] with only unconverted record.id keys

    # Get **max sequence length** from unconverted records
    max_reference_length = max(values[0] for values in modification_dict[unconverted].values())

    # Add **unconverted records** to `record_FASTA_dict`
    for record, values in modification_dict[unconverted].items():
        sequence_length, top_coords, bottom_coords, sequence, complement = values

        if not deaminase_footprinting:
            chromosome = record.replace(f"_{unconverted}_top", "")
        else:
            chromosome = record

        # Store **original sequence**
        record_FASTA_dict[record] = RecordFastaInfo(
            sequence=sequence + "N" * (max_reference_length - sequence_length),
            complement=complement + "N" * (max_reference_length - sequence_length),
            chromosome=chromosome,
            unconverted_name=record,
            sequence_length=sequence_length,
            padding_length=max_reference_length - sequence_length,
            conversion=unconverted,
            strand="top",
            max_reference_length=max_reference_length,
        )

        if chromosome not in chromosome_FASTA_dict:
            chromosome_FASTA_dict[chromosome] = (
                sequence + "N" * (max_reference_length - sequence_length),
                complement + "N" * (max_reference_length - sequence_length),
            )

    # Process converted records
    for conversion in conversion_types:
        modification_dict[conversion] = find_conversion_sites(
            converted_FASTA, conversion, conversions, deaminase_footprinting
        )
        # Above points to record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence, complement] with only unconverted record.id keys

        for record, values in modification_dict[conversion].items():
            sequence_length, top_coords, bottom_coords, sequence, complement = values

            if not deaminase_footprinting:
                chromosome = record.split(f"_{unconverted}_")[0]  # Extract chromosome name
            else:
                chromosome = record

            # Add **both strands** for converted records
            for strand in ["top", "bottom"]:
                converted_name = f"{chromosome}_{conversion}_{strand}"
                unconverted_name = f"{chromosome}_{unconverted}_top"

                record_FASTA_dict[converted_name] = RecordFastaInfo(
                    sequence=sequence + "N" * (max_reference_length - sequence_length),
                    complement=complement + "N" * (max_reference_length - sequence_length),
                    chromosome=chromosome,
                    unconverted_name=unconverted_name,
                    sequence_length=sequence_length,
                    padding_length=max_reference_length - sequence_length,
                    conversion=conversion,
                    strand=strand,
                    max_reference_length=max_reference_length,
                )

    logger.debug("Updated record_FASTA_dict keys: %s", list(record_FASTA_dict.keys()))
    return max_reference_length, record_FASTA_dict, chromosome_FASTA_dict


def filter_bams_by_mapping_threshold(
    bam_path_list: list[Path],
    bam_files: list[Path],
    mapping_threshold: float,
    samtools_backend: str | None,
) -> set[str]:
    """Filter FASTA records based on per-BAM mapping thresholds.

    Args:
        bam_path_list: Ordered list of BAM paths to evaluate.
        bam_files: Matching list of BAM paths used for reporting.
        mapping_threshold: Minimum percentage of aligned reads to include a record.
        samtools_backend: Samtools backend choice for alignment parsing.

    Returns:
        set[str]: FASTA record IDs that pass the mapping threshold.

    Processing Steps:
        1. Count aligned/unaligned reads and per-record percentages.
        2. Collect record IDs that meet the mapping threshold.
    """
    records_to_analyze: set[str] = set()

    for i, bam in enumerate(bam_path_list):
        aligned_reads, unaligned_reads, record_counts = count_aligned_reads(bam, samtools_backend)
        aligned_percent = aligned_reads * 100 / (aligned_reads + unaligned_reads)
        logger.info(f"{aligned_percent:.2f}% of reads in {bam_files[i].name} aligned successfully.")

        for record, (count, percent) in record_counts.items():
            if percent >= mapping_threshold:
                records_to_analyze.add(record)

    logger.info(f"Analyzing the following FASTA records: {records_to_analyze}")
    return records_to_analyze


def _encode_sequence_array(
    read_sequence: np.ndarray,
    valid_length: int,
    config: SequenceEncodingConfig,
) -> np.ndarray:
    """Encode a base-identity array into integer values with padding.

    Args:
        read_sequence: Array of base calls (dtype "<U1").
        valid_length: Number of valid reference positions for this record.
        config: Integer encoding configuration.

    Returns:
        np.ndarray: Integer-encoded sequence with padding applied.

    Processing Steps:
        1. Initialize an array filled with the unknown base encoding.
        2. Map A/C/G/T/N bases into integer values.
        3. Mark positions beyond valid_length with the padding value.
    """
    read_sequence = np.asarray(read_sequence, dtype="<U1")
    encoded = np.full(read_sequence.shape, config.unknown_value, dtype=np.int16)
    for base in config.bases:
        encoded[read_sequence == base] = config.base_to_int[base]
    if valid_length < encoded.size:
        encoded[valid_length:] = config.padding_value
    return encoded


def _write_sequence_batches(
    base_identities: Mapping[str, np.ndarray],
    tmp_dir: Path,
    record: str,
    prefix: str,
    valid_length: int,
    config: SequenceEncodingConfig,
) -> list[str]:
    """Encode base identities into integer arrays and write batched H5AD files.

    Args:
        base_identities: Mapping of read name to base identity arrays.
        tmp_dir: Directory for temporary H5AD files.
        record: Reference record identifier.
        prefix: Prefix used to name batch files.
        valid_length: Valid reference length for padding determination.
        config: Integer encoding configuration.

    Returns:
        list[str]: Paths to written H5AD batch files.

    Processing Steps:
        1. Encode each read sequence into integers.
        2. Accumulate encoded reads into batches.
        3. Persist each batch to an H5AD file with `.uns` storage.
    """
    batch_files: list[str] = []
    batch: dict[str, np.ndarray] = {}
    batch_number = 0

    for read_name, sequence in base_identities.items():
        if sequence is None:
            continue
        batch[read_name] = _encode_sequence_array(sequence, valid_length, config)
        if len(batch) >= config.batch_size:
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
        batch_files: H5AD paths containing encoded sequences in `.uns`.

    Returns:
        tuple[dict[str, np.ndarray], set[str], set[str]]:
            Read-to-sequence mapping and sets of forward/reverse mapped reads.

    Processing Steps:
        1. Read each H5AD file.
        2. Merge `.uns` dictionaries into a single mapping.
        3. Track forward/reverse read IDs based on filename markers.
    """
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


def process_single_bam(
    bam_index: int,
    bam: Path,
    records_to_analyze: set[str],
    record_FASTA_dict: dict[str, RecordFastaInfo],
    chromosome_FASTA_dict: dict[str, tuple[str, str]],
    tmp_dir: Path,
    max_reference_length: int,
    device: torch.device,
    deaminase_footprinting: bool,
    samtools_backend: str | None,
) -> ad.AnnData | None:
    """Process a single BAM file into per-record AnnData objects.

    Args:
        bam_index: Index of the BAM within the processing batch.
        bam: Path to the BAM file.
        records_to_analyze: FASTA record IDs that passed the mapping threshold.
        record_FASTA_dict: FASTA metadata keyed by record ID.
        chromosome_FASTA_dict: Chromosome sequences for annotations.
        tmp_dir: Directory for temporary batch files.
        max_reference_length: Maximum reference length for padding.
        device: Torch device used for binarization.
        deaminase_footprinting: Whether direct deamination chemistry was used.
        samtools_backend: Samtools backend choice for alignment parsing.

    Returns:
        anndata.AnnData | None: Concatenated AnnData object or None if no data.

    Processing Steps:
        1. Extract base identities and mismatch profiles for each record.
        2. Binarize modified base identities into feature matrices.
        3. Encode read sequences into integer arrays and cache batches.
        4. Build AnnData layers/obs metadata for each record and concatenate.
    """
    adata_list: list[ad.AnnData] = []

    for record in records_to_analyze:
        sample = bam.stem
        record_info = record_FASTA_dict[record]
        chromosome = record_info.chromosome
        current_length = record_info.sequence_length
        mod_type, strand = record_info.conversion, record_info.strand
        sequence = chromosome_FASTA_dict[chromosome][0]

        # Extract Base Identities
        fwd_bases, rev_bases, mismatch_counts_per_read, mismatch_trend_per_read = (
            extract_base_identities(
                bam, record, range(current_length), max_reference_length, sequence, samtools_backend
            )
        )
        mismatch_trend_series = pd.Series(mismatch_trend_per_read)

        # Skip processing if both forward and reverse base identities are empty
        if not fwd_bases and not rev_bases:
            logger.debug(
                f"[Worker {current_process().pid}] Skipping {sample} - No valid base identities for {record}."
            )
            continue

        merged_bin = {}

        # Binarize the Base Identities if they exist
        if fwd_bases:
            fwd_bin = binarize_converted_base_identities(
                fwd_bases,
                strand,
                mod_type,
                bam,
                device,
                deaminase_footprinting,
                mismatch_trend_per_read,
            )
            merged_bin.update(fwd_bin)

        if rev_bases:
            rev_bin = binarize_converted_base_identities(
                rev_bases,
                strand,
                mod_type,
                bam,
                device,
                deaminase_footprinting,
                mismatch_trend_per_read,
            )
            merged_bin.update(rev_bin)

        # Skip if merged_bin is empty (no valid binarized data)
        if not merged_bin:
            logger.debug(
                f"[Worker {current_process().pid}] Skipping {sample} - No valid binarized data for {record}."
            )
            continue

        # Convert to DataFrame
        # for key in merged_bin:
        #     merged_bin[key] = merged_bin[key].cpu().numpy()  # Move to CPU & convert to NumPy
        bin_df = pd.DataFrame.from_dict(merged_bin, orient="index")
        sorted_index = sorted(bin_df.index)
        bin_df = bin_df.reindex(sorted_index)

        # Integer-encode reads if there is valid data
        batch_files: list[str] = []
        if fwd_bases:
            batch_files.extend(
                _write_sequence_batches(
                    fwd_bases,
                    tmp_dir,
                    record,
                    f"{bam_index}_fwd",
                    current_length,
                    SEQUENCE_ENCODING_CONFIG,
                )
            )

        if rev_bases:
            batch_files.extend(
                _write_sequence_batches(
                    rev_bases,
                    tmp_dir,
                    record,
                    f"{bam_index}_rev",
                    current_length,
                    SEQUENCE_ENCODING_CONFIG,
                )
            )

        if not batch_files:
            logger.debug(
                f"[Worker {current_process().pid}] Skipping {sample} - No valid encoded data for {record}."
            )
            continue

        gc.collect()

        encoded_reads, fwd_reads, rev_reads = _load_sequence_batches(batch_files)
        if not encoded_reads:
            logger.debug(
                f"[Worker {current_process().pid}] Skipping {sample} - No reads found in encoded data for {record}."
            )
            continue

        sequence_length = max_reference_length
        default_sequence = np.full(
            sequence_length, SEQUENCE_ENCODING_CONFIG.unknown_value, dtype=np.int16
        )
        if current_length < sequence_length:
            default_sequence[current_length:] = SEQUENCE_ENCODING_CONFIG.padding_value

        encoded_matrix = np.vstack(
            [encoded_reads.get(read_name, default_sequence) for read_name in sorted_index]
        )

        # Convert to AnnData
        X = bin_df.values.astype(np.float32)
        adata = ad.AnnData(X)
        adata.obs_names = bin_df.index.astype(str)
        adata.var_names = bin_df.columns.astype(str)
        adata.obs["Sample"] = [sample] * len(adata)
        try:
            barcode = sample.split("barcode")[1]
        except Exception:
            barcode = np.nan
        adata.obs["Barcode"] = [int(barcode)] * len(adata)
        adata.obs["Barcode"] = adata.obs["Barcode"].astype(str)
        adata.obs["Reference"] = [chromosome] * len(adata)
        adata.obs["Strand"] = [strand] * len(adata)
        adata.obs["Dataset"] = [mod_type] * len(adata)
        adata.obs["Reference_dataset_strand"] = [f"{chromosome}_{mod_type}_{strand}"] * len(adata)
        adata.obs["Reference_strand"] = [f"{chromosome}_{strand}"] * len(adata)
        adata.obs["Read_mismatch_trend"] = adata.obs_names.map(mismatch_trend_series)

        read_mapping_direction = []
        for read_id in adata.obs_names:
            if read_id in fwd_reads:
                read_mapping_direction.append("fwd")
            elif read_id in rev_reads:
                read_mapping_direction.append("rev")
            else:
                read_mapping_direction.append("unk")

        adata.obs["Read_mapping_direction"] = read_mapping_direction

        # Attach integer sequence encoding to layers
        adata.layers["sequence_integer_encoding"] = encoded_matrix

        adata_list.append(adata)

    return ad.concat(adata_list, join="outer") if adata_list else None


def timestamp():
    """Return a formatted timestamp for logging.

    Returns:
        str: Timestamp string in the format ``[YYYY-MM-DD HH:MM:SS]``.
    """
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def worker_function(
    bam_index: int,
    bam: Path,
    records_to_analyze: set[str],
    shared_record_FASTA_dict: dict[str, RecordFastaInfo],
    chromosome_FASTA_dict: dict[str, tuple[str, str]],
    tmp_dir: Path,
    h5_dir: Path,
    max_reference_length: int,
    device: torch.device,
    deaminase_footprinting: bool,
    samtools_backend: str | None,
    progress_queue,
    log_level: int,
    log_file: Path | None,
):
    """Process a single BAM and write the output to an H5AD file.

    Args:
        bam_index: Index of the BAM within the processing batch.
        bam: Path to the BAM file.
        records_to_analyze: FASTA record IDs that passed the mapping threshold.
        shared_record_FASTA_dict: Shared FASTA metadata keyed by record ID.
        chromosome_FASTA_dict: Chromosome sequences for annotations.
        tmp_dir: Directory for temporary batch files.
        h5_dir: Directory for per-BAM H5AD outputs.
        max_reference_length: Maximum reference length for padding.
        device: Torch device used for binarization.
        deaminase_footprinting: Whether direct deamination chemistry was used.
        samtools_backend: Samtools backend choice for alignment parsing.
        progress_queue: Queue used to signal completion.
        log_level: Logging level to configure in workers.
        log_file: Optional log file path.

    Processing Steps:
        1. Skip processing if an output H5AD already exists.
        2. Filter records to those present in the FASTA metadata.
        3. Run per-record processing and write AnnData output.
        4. Signal completion via the progress queue.
    """
    _ensure_worker_logging(log_level, log_file)
    worker_id = current_process().pid  # Get worker process ID
    sample = bam.stem

    try:
        logger.info(f"[Worker {worker_id}] Processing BAM: {sample}")

        h5ad_path = h5_dir / bam.with_suffix(".h5ad").name
        if h5ad_path.exists():
            logger.debug(f"[Worker {worker_id}] Skipping {sample}: Already processed.")
            progress_queue.put(sample)
            return

        # Filter records specific to this BAM
        bam_records_to_analyze = {
            record for record in records_to_analyze if record in shared_record_FASTA_dict
        }

        if not bam_records_to_analyze:
            logger.debug(
                f"[Worker {worker_id}] No valid records to analyze for {sample}. Skipping."
            )
            progress_queue.put(sample)
            return

        # Process BAM
        adata = process_single_bam(
            bam_index,
            bam,
            bam_records_to_analyze,
            shared_record_FASTA_dict,
            chromosome_FASTA_dict,
            tmp_dir,
            max_reference_length,
            device,
            deaminase_footprinting,
            samtools_backend,
        )

        if adata is not None:
            adata.write_h5ad(str(h5ad_path))
            logger.info(f"[Worker {worker_id}] Completed processing for BAM: {sample}")

            # Free memory
            del adata
            gc.collect()

        progress_queue.put(sample)

    except Exception:
        logger.warning(
            f"[Worker {worker_id}] ERROR while processing {sample}:\n{traceback.format_exc()}"
        )
        progress_queue.put(sample)  # Still signal completion to prevent deadlock


def process_bams_parallel(
    bam_path_list: list[Path],
    records_to_analyze: set[str],
    record_FASTA_dict: dict[str, RecordFastaInfo],
    chromosome_FASTA_dict: dict[str, tuple[str, str]],
    tmp_dir: Path,
    h5_dir: Path,
    num_threads: int,
    max_reference_length: int,
    device: torch.device,
    deaminase_footprinting: bool,
    samtools_backend: str | None,
) -> ad.AnnData | None:
    """Process BAM files in parallel and concatenate the resulting AnnData.

    Args:
        bam_path_list: List of BAM files to process.
        records_to_analyze: FASTA record IDs that passed the mapping threshold.
        record_FASTA_dict: FASTA metadata keyed by record ID.
        chromosome_FASTA_dict: Chromosome sequences for annotations.
        tmp_dir: Directory for temporary batch files.
        h5_dir: Directory for per-BAM H5AD outputs.
        num_threads: Number of worker processes.
        max_reference_length: Maximum reference length for padding.
        device: Torch device used for binarization.
        deaminase_footprinting: Whether direct deamination chemistry was used.
        samtools_backend: Samtools backend choice for alignment parsing.

    Returns:
        anndata.AnnData | None: Concatenated AnnData or None if no H5ADs produced.

    Processing Steps:
        1. Spawn worker processes to handle each BAM.
        2. Track completion via a multiprocessing queue.
        3. Concatenate per-BAM H5AD files into a final AnnData.
    """
    make_dirs(h5_dir)  # Ensure h5_dir exists

    logger.info(f"Starting parallel BAM processing with {num_threads} threads...")
    log_level, log_file = _get_logger_config()

    with Manager() as manager:
        progress_queue = manager.Queue()
        shared_record_FASTA_dict = manager.dict(record_FASTA_dict)

        with Pool(processes=num_threads) as pool:
            results = [
                pool.apply_async(
                    worker_function,
                    (
                        i,
                        bam,
                        records_to_analyze,
                        shared_record_FASTA_dict,
                        chromosome_FASTA_dict,
                        tmp_dir,
                        h5_dir,
                        max_reference_length,
                        device,
                        deaminase_footprinting,
                        samtools_backend,
                        progress_queue,
                        log_level,
                        log_file,
                    ),
                )
                for i, bam in enumerate(bam_path_list)
            ]

            logger.info(f"Submitting {len(results)} BAMs for processing.")

            # Track completed BAMs
            completed_bams = set()
            while len(completed_bams) < len(bam_path_list):
                try:
                    processed_bam = progress_queue.get(timeout=2400)  # Wait for a finished BAM
                    completed_bams.add(processed_bam)
                except Exception as e:
                    logger.error(f"Timeout waiting for worker process. Possible crash? {e}")
                    _log_async_result_errors(results, bam_path_list)

            pool.close()
            pool.join()  # Ensure all workers finish

    _log_async_result_errors(results, bam_path_list)

    # Final Concatenation Step
    h5ad_files = [f for f in h5_dir.iterdir() if f.suffix == ".h5ad"]

    if not h5ad_files:
        logger.warning(f"No valid H5AD files generated. Exiting.")
        return None

    logger.info(f"Concatenating {len(h5ad_files)} H5AD files into final output...")
    final_adata = ad.concat([ad.read_h5ad(f) for f in h5ad_files], join="outer")

    logger.info(f"Successfully generated final AnnData object.")
    return final_adata


def _log_async_result_errors(results, bam_path_list):
    """Log worker failures captured by multiprocessing AsyncResult objects.

    Args:
        results: Iterable of AsyncResult objects from multiprocessing.
        bam_path_list: List of BAM paths matching the async results.

    Processing Steps:
        1. Iterate over async results.
        2. Retrieve results to surface worker exceptions.
    """
    for bam, result in zip(bam_path_list, results):
        if not result.ready():
            continue
        try:
            result.get()
        except Exception as exc:
            logger.error("Worker process failed for %s: %s", bam, exc)


def _get_logger_config() -> tuple[int, Path | None]:
    """Return the active smftools logger level and optional file path.

    Returns:
        tuple[int, Path | None]: Log level and log file path (if configured).

    Processing Steps:
        1. Inspect the smftools logger for configured handlers.
        2. Extract log level and file handler path.
    """
    smftools_logger = logging.getLogger("smftools")
    level = smftools_logger.level
    if level == logging.NOTSET:
        level = logging.INFO
    log_file: Path | None = None
    for handler in smftools_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = Path(handler.baseFilename)
            break
    return level, log_file


def _ensure_worker_logging(log_level: int, log_file: Path | None) -> None:
    """Ensure worker processes have logging configured.

    Args:
        log_level: Logging level to configure.
        log_file: Optional log file path.

    Processing Steps:
        1. Check if handlers are already configured.
        2. Initialize logging with the provided level and file path.
    """
    smftools_logger = logging.getLogger("smftools")
    if not smftools_logger.handlers:
        setup_logging(level=log_level, log_file=log_file)


def delete_intermediate_h5ads_and_tmpdir(
    h5_dir: Union[str, Path, Iterable[str], None],
    tmp_dir: Optional[Union[str, Path]] = None,
    *,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Delete intermediate .h5ad files and a temporary directory.

    Args:
        h5_dir: Directory path or iterable of file paths to inspect for `.h5ad` files.
        tmp_dir: Optional directory to remove recursively.
        dry_run: If True, log what would be removed without deleting.
        verbose: If True, log progress and warnings.

    Processing Steps:
        1. Remove `.h5ad` files (excluding `.gz`) from the provided directory or list.
        2. Optionally remove the temporary directory tree.
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
                if name.endswith(".h5ad") and not name.endswith(".gz"):
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
