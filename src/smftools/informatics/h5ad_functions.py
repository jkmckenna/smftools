from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from smftools.constants import BASE_QUALITY_SCORES, READ_SPAN_MASK, REFERENCE_STRAND
from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)


def add_demux_type_annotation(
    adata,
    double_demux_source,
    sep: str = "\t",
    read_id_col: str = "read_id",
    barcode_col: str = "barcode",
):
    """
    Add adata.obs["demux_type"]:
        - "double" if read_id appears in the *double demux* TSV
        - "single" otherwise

    Rows where barcode == "unclassified" in the demux TSV are ignored.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose obs_names are read_ids.
    double_demux_source : str | Path | list[str]
        Either:
          - path to a TSV/TXT of dorado demux results
          - a list of read_ids
    """

    # -----------------------------
    # If it's a file → load TSV
    # -----------------------------
    if isinstance(double_demux_source, (str, Path)):
        file_path = Path(double_demux_source)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        df = pd.read_csv(file_path, sep=sep, dtype=str)

        # If the file has only one column → treat as a simple read list
        if df.shape[1] == 1:
            read_ids = df.iloc[:, 0].tolist()
        else:
            # Validate columns
            if read_id_col not in df.columns:
                raise ValueError(f"TSV must contain a '{read_id_col}' column.")
            if barcode_col not in df.columns:
                raise ValueError(f"TSV must contain a '{barcode_col}' column.")

            # Drop unclassified reads
            df = df[df[barcode_col].str.lower() != "unclassified"]

            # Extract read_ids
            read_ids = df[read_id_col].tolist()

    # -----------------------------
    # If user supplied list-of-ids
    # -----------------------------
    else:
        read_ids = list(double_demux_source)

    # Deduplicate for speed
    double_set = set(read_ids)

    # Boolean lookup in AnnData
    is_double = adata.obs_names.isin(double_set)

    adata.obs["demux_type"] = np.where(is_double, "double", "single")
    adata.obs["demux_type"] = adata.obs["demux_type"].astype("category")

    return adata


def add_demux_type_from_bm_tag(
    adata,
    bm_column: str = "BM",
):
    """
    Add adata.obs["demux_type"] based on the BM (barcode match type) tag from smftools demux.

    Mapping:
        - "both" → "double" (both ends matched same barcode)
        - "left_only", "right_only", "read_start_only", "read_end_only" → "single"
        - "mismatch", "unclassified" → "unclassified"

    Parameters
    ----------
    adata : AnnData
        AnnData object with BM column in obs.
    bm_column : str
        Name of the column containing BM tag values (default "BM").

    Returns
    -------
    AnnData
        The modified AnnData with demux_type column added.
    """
    if bm_column not in adata.obs.columns:
        logger.warning(
            f"Column '{bm_column}' not found in adata.obs. "
            "Cannot derive demux_type from BM tag. Setting all to 'unknown'."
        )
        adata.obs["demux_type"] = "unknown"
        adata.obs["demux_type"] = adata.obs["demux_type"].astype("category")
        return adata

    bm_values = adata.obs[bm_column].astype(str).str.lower()

    # Map BM values to demux_type
    demux_type = np.where(
        bm_values == "both",
        "double",
        np.where(
            bm_values.isin(["left_only", "right_only", "read_start_only", "read_end_only"]),
            "single",
            "unclassified",
        ),
    )

    adata.obs["demux_type"] = demux_type
    adata.obs["demux_type"] = adata.obs["demux_type"].astype("category")

    logger.info(
        "Derived demux_type from BM tag: double=%d, single=%d, unclassified=%d",
        (adata.obs["demux_type"] == "double").sum(),
        (adata.obs["demux_type"] == "single").sum(),
        (adata.obs["demux_type"] == "unclassified").sum(),
    )

    return adata


def append_reference_strand_quality_stats(
    adata,
    ref_column: str = REFERENCE_STRAND,
    quality_layer: str = BASE_QUALITY_SCORES,
    read_span_layer: str = READ_SPAN_MASK,
    uns_flag: str = "append_reference_strand_quality_stats_performed",
    force_redo: bool = False,
    bypass: bool = False,
) -> None:
    """Append per-position quality and error rate stats for each reference strand.

    Args:
        adata: AnnData object to annotate in-place.
        ref_column: Obs column defining reference strand groups.
        quality_layer: Layer containing base quality scores.
        read_span_layer: Optional layer marking covered positions (1=covered, 0=not covered).
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        bypass: Whether to skip this step.
    """
    if bypass:
        return

    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        return

    if ref_column not in adata.obs:
        logger.debug("Reference column '%s' not found; skipping quality stats.", ref_column)
        return

    if quality_layer not in adata.layers:
        logger.debug("Quality layer '%s' not found; skipping quality stats.", quality_layer)
        return

    ref_values = adata.obs[ref_column]
    references = (
        ref_values.cat.categories if hasattr(ref_values, "cat") else pd.Index(pd.unique(ref_values))
    )
    n_vars = adata.shape[1]
    has_span_mask = read_span_layer in adata.layers

    for ref in references:
        ref_mask = ref_values == ref
        ref_position_mask = adata.var.get(f"position_in_{ref}")
        if ref_position_mask is None:
            ref_position_mask = pd.Series(np.ones(n_vars, dtype=bool), index=adata.var.index)
        else:
            ref_position_mask = ref_position_mask.astype(bool)

        mean_quality = np.full(n_vars, np.nan, dtype=float)
        std_quality = np.full(n_vars, np.nan, dtype=float)
        mean_error = np.full(n_vars, np.nan, dtype=float)
        std_error = np.full(n_vars, np.nan, dtype=float)

        if ref_mask.sum() > 0:
            quality_matrix = np.asarray(adata.layers[quality_layer][ref_mask]).astype(float)
            quality_matrix[quality_matrix < 0] = np.nan
            if has_span_mask:
                coverage_mask = np.asarray(adata.layers[read_span_layer][ref_mask]) > 0
                quality_matrix = np.where(coverage_mask, quality_matrix, np.nan)

            mean_quality = np.nanmean(quality_matrix, axis=0)
            std_quality = np.nanstd(quality_matrix, axis=0)

            error_matrix = np.power(10.0, -quality_matrix / 10.0)
            mean_error = np.nanmean(error_matrix, axis=0)
            std_error = np.nanstd(error_matrix, axis=0)

        mean_quality = np.where(ref_position_mask.values, mean_quality, np.nan)
        std_quality = np.where(ref_position_mask.values, std_quality, np.nan)
        mean_error = np.where(ref_position_mask.values, mean_error, np.nan)
        std_error = np.where(ref_position_mask.values, std_error, np.nan)

        adata.var[f"{ref}_mean_base_quality"] = pd.Series(mean_quality, index=adata.var.index)
        adata.var[f"{ref}_std_base_quality"] = pd.Series(std_quality, index=adata.var.index)
        adata.var[f"{ref}_mean_error_rate"] = pd.Series(mean_error, index=adata.var.index)
        adata.var[f"{ref}_std_error_rate"] = pd.Series(std_error, index=adata.var.index)

    adata.uns[uns_flag] = True


def add_read_tag_annotations(
    adata,
    bam_files: Optional[List[str]] = None,
    read_tags: Optional[Dict[str, Dict[str, object]]] = None,
    tag_names: Optional[List[str]] = None,
    include_flags: bool = True,
    include_cigar: bool = True,
    extract_read_tags_from_bam_callable=None,
    samtools_backend: str | None = "auto",
):
    """Populate adata.obs with read tag metadata.

    Args:
        adata: AnnData to annotate (modified in-place).
        bam_files: Optional list of BAM files to extract tags from.
        read_tags: Optional mapping of read name to tag dict.
        tag_names: Optional list of BAM tag names to extract (e.g. ["NM", "MD", "MM", "ML"]).
        include_flags: Whether to add a FLAGS list column.
        include_cigar: Whether to add the CIGAR string column.
        extract_read_tags_from_bam_callable: Optional callable to extract tags from a BAM.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).

    Returns:
        None (mutates adata in-place).
    """
    if read_tags is None:
        read_tags = {}
        if bam_files:
            extractor = extract_read_tags_from_bam_callable or globals().get(
                "extract_read_tags_from_bam"
            )
            if extractor is None:
                raise ValueError(
                    "No `read_tags` provided and `extract_read_tags_from_bam` not found."
                )
            for bam in bam_files:
                bam_read_tags = extractor(
                    bam,
                    tag_names=tag_names,
                    include_flags=include_flags,
                    include_cigar=include_cigar,
                    samtools_backend=samtools_backend,
                )
                if not isinstance(bam_read_tags, dict):
                    raise ValueError(f"extract_read_tags_from_bam returned non-dict for {bam}")
                read_tags.update(bam_read_tags)

    if not read_tags:
        return

    df = pd.DataFrame.from_dict(read_tags, orient="index")
    df_reindexed = df.reindex(adata.obs_names)
    for column in df_reindexed.columns:
        adata.obs[column] = df_reindexed[column].values


def add_secondary_supplementary_alignment_flags(
    adata,
    bam_path: str | Path,
    *,
    uns_flag: str = "add_secondary_supplementary_flags_performed",
    bypass: bool = False,
    force_redo: bool = False,
    samtools_backend: str | None = "auto",
) -> None:
    """Annotate whether reads have secondary/supplementary alignments.

    Args:
        adata: AnnData to annotate (modified in-place).
        bam_path: Path to the aligned/sorted BAM to scan.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip annotation.
        force_redo: Whether to recompute even if ``uns_flag`` is set.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).
    """
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        return

    from .bam_functions import (
        extract_secondary_supplementary_alignment_spans,
        find_secondary_supplementary_read_names,
    )

    secondary_reads, supplementary_reads = find_secondary_supplementary_read_names(
        bam_path,
        adata.obs_names,
        samtools_backend=samtools_backend,
    )
    secondary_spans, supplementary_spans = extract_secondary_supplementary_alignment_spans(
        bam_path,
        adata.obs_names,
        samtools_backend=samtools_backend,
    )

    adata.obs["has_secondary_alignment"] = adata.obs_names.isin(secondary_reads)
    adata.obs["has_supplementary_alignment"] = adata.obs_names.isin(supplementary_reads)
    adata.obs["secondary_alignment_spans"] = [
        secondary_spans.get(read_name) for read_name in adata.obs_names
    ]
    adata.obs["supplementary_alignment_spans"] = [
        supplementary_spans.get(read_name) for read_name in adata.obs_names
    ]
    adata.uns[uns_flag] = True


def add_read_length_and_mapping_qc(
    adata,
    bam_files: Optional[List[str]] = None,
    read_metrics: Optional[Dict[str, Union[list, tuple]]] = None,
    uns_flag: str = "add_read_length_and_mapping_qc_performed",
    extract_read_features_from_bam_callable=None,
    bypass: bool = False,
    force_redo: bool = True,
    samtools_backend: str | None = "auto",
):
    """
    Populate adata.obs with read/mapping QC columns.

    Parameters
    ----------
    adata
        AnnData to annotate (modified in-place).
    bam_files
        Optional list of BAM files to extract metrics from. Ignored if read_metrics supplied.
    read_metrics
        Optional dict mapping obs_name -> [read_length, read_quality, reference_length, mapped_length,
        mapping_quality, reference_start, reference_end]
        If provided, this will be used directly and bam_files will be ignored.
    uns_flag
        key in final_adata.uns used to record that QC was performed (kept the name with original misspelling).
    extract_read_features_from_bam_callable
        Optional callable(bam_path) -> dict mapping read_name -> list/tuple of metrics.
        If not provided and bam_files is given, function will attempt to call `extract_read_features_from_bam`
        from the global namespace (your existing helper).

    Returns
    -------
    None (mutates final_adata in-place)
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

    # Build read_metrics dict either from provided arg or by extracting from bam files
    if read_metrics is None:
        read_metrics = {}
        if bam_files:
            extractor = extract_read_features_from_bam_callable or globals().get(
                "extract_read_features_from_bam"
            )
            if extractor is None:
                raise ValueError(
                    "No `read_metrics` provided and `extract_read_features_from_bam` not found."
                )
            for bam in bam_files:
                bam_read_metrics = extractor(bam, samtools_backend)
                if not isinstance(bam_read_metrics, dict):
                    raise ValueError(f"extract_read_features_from_bam returned non-dict for {bam}")
                read_metrics.update(bam_read_metrics)
        else:
            # nothing to do
            read_metrics = {}

    # Convert read_metrics dict -> DataFrame (rows = read id)
    # Values may be lists/tuples or scalars; prefer lists/tuples with 5 entries.
    if len(read_metrics) == 0:
        # fill with NaNs
        n = adata.n_obs
        adata.obs["read_length"] = np.full(n, np.nan)
        adata.obs["mapped_length"] = np.full(n, np.nan)
        adata.obs["reference_length"] = np.full(n, np.nan)
        adata.obs["read_quality"] = np.full(n, np.nan)
        adata.obs["mapping_quality"] = np.full(n, np.nan)
        adata.obs["reference_start"] = np.full(n, np.nan)
        adata.obs["reference_end"] = np.full(n, np.nan)
    else:
        # Build DF robustly
        # Convert values to lists where possible, else to [val, val, val...]
        max_cols = 7
        rows = {}
        for k, v in read_metrics.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals = list(v)
            else:
                # scalar -> replicate into 5 columns to preserve original behavior
                vals = [v] * max_cols
            # Ensure length >= 5
            if len(vals) < max_cols:
                vals = vals + [np.nan] * (max_cols - len(vals))
            rows[k] = vals[:max_cols]

        df = pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=[
                "read_length",
                "read_quality",
                "reference_length",
                "mapped_length",
                "mapping_quality",
                "reference_start",
                "reference_end",
            ],
        )

        # Reindex to final_adata.obs_names so order matches adata
        # If obs_names are not present as keys in df, the results will be NaN
        df_reindexed = df.reindex(adata.obs_names).astype(float)

        adata.obs["read_length"] = df_reindexed["read_length"].values
        adata.obs["mapped_length"] = df_reindexed["mapped_length"].values
        adata.obs["reference_length"] = df_reindexed["reference_length"].values
        adata.obs["read_quality"] = df_reindexed["read_quality"].values
        adata.obs["mapping_quality"] = df_reindexed["mapping_quality"].values
        adata.obs["reference_start"] = df_reindexed["reference_start"].values
        adata.obs["reference_end"] = df_reindexed["reference_end"].values

    # Compute ratio columns safely (avoid divide-by-zero and preserve NaN)
    # read_length_to_reference_length_ratio
    rl = pd.to_numeric(adata.obs["read_length"], errors="coerce").to_numpy(dtype=float)
    ref_len = pd.to_numeric(adata.obs["reference_length"], errors="coerce").to_numpy(dtype=float)
    mapped_len = pd.to_numeric(adata.obs["mapped_length"], errors="coerce").to_numpy(dtype=float)

    # safe divisions: use np.where to avoid warnings and replace inf with nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rl_to_ref = np.where((ref_len != 0) & np.isfinite(ref_len), rl / ref_len, np.nan)
        mapped_to_ref = np.where(
            (ref_len != 0) & np.isfinite(ref_len), mapped_len / ref_len, np.nan
        )
        mapped_to_read = np.where((rl != 0) & np.isfinite(rl), mapped_len / rl, np.nan)

    adata.obs["read_length_to_reference_length_ratio"] = rl_to_ref
    adata.obs["mapped_length_to_reference_length_ratio"] = mapped_to_ref
    adata.obs["mapped_length_to_read_length_ratio"] = mapped_to_read

    # Add read level raw modification signal: sum over X rows
    X = adata.X
    if sp.issparse(X):
        # sum returns (n_obs, 1) sparse matrix; convert to 1d array
        raw_sig = np.asarray(X.sum(axis=1)).ravel()
    else:
        raw_sig = np.asarray(X.sum(axis=1)).ravel()

    adata.obs["Raw_modification_signal"] = raw_sig

    # mark as done
    adata.uns[uns_flag] = True

    return None


def _collect_read_origins_from_pod5(pod5_path: str, target_ids: set[str]) -> dict[str, str]:
    """
    Worker function: scan one POD5 file and return a mapping
    {read_id: pod5_basename} only for read_ids in `target_ids`.
    """
    p5 = require("pod5", extra="ont", purpose="POD5 metadata")
    Reader = p5.Reader

    basename = os.path.basename(pod5_path)
    mapping: dict[str, str] = {}

    with Reader(pod5_path) as reader:
        for read in reader.reads():
            # Cast read id to string
            rid = str(read.read_id)
            if rid in target_ids:
                mapping[rid] = basename

    return mapping


def annotate_pod5_origin(
    adata,
    pod5_path_or_dir: str | Path,
    pattern: str = "*.pod5",
    n_jobs: int | None = None,
    fill_value: str | None = "unknown",
    verbose: bool = True,
    csv_path: str | None = None,
):
    """
    Add `pod5_origin` column to `adata.obs`, containing the POD5 basename
    each read came from.

    Parameters
    ----------
    adata
        AnnData with obs_names == read_ids (as strings).
    pod5_path_or_dir
        Directory containing POD5 files or path to a single POD5 file.
    pattern
        Glob pattern for POD5 files inside `pod5_dir`.
    n_jobs
        Number of worker processes. If None or <=1, runs serially.
    fill_value
        Value to use when a read_id is not found in any POD5 file.
        If None, leaves missing as NaN.
    verbose
        Print progress info.
    csv_path
        Path to a csv of the read to pod5 origin mapping

    Returns
    -------
    None (modifies `adata` in-place).
    """
    pod5_path_or_dir = Path(pod5_path_or_dir)

    # --- Resolve input into a list of pod5 files ---
    if pod5_path_or_dir.is_dir():
        pod5_files = sorted(str(p) for p in pod5_path_or_dir.glob(pattern))
        if not pod5_files:
            raise FileNotFoundError(
                f"No POD5 files matching {pattern!r} in {str(pod5_path_or_dir)!r}"
            )
    elif pod5_path_or_dir.is_file():
        if pod5_path_or_dir.suffix.lower() != ".pod5":
            raise ValueError(f"Expected a .pod5 file, got: {pod5_path_or_dir}")
        pod5_files = [str(pod5_path_or_dir)]
    else:
        raise FileNotFoundError(f"Path does not exist: {pod5_path_or_dir}")

    # Make sure obs_names are strings
    obs_names = adata.obs_names.astype(str)
    target_ids = set(obs_names)  # only these are interesting

    if verbose:
        logger.info(f"Found {len(pod5_files)} POD5 files.")
        logger.info(f"Tracking {len(target_ids)} read IDs from AnnData.")

    # --- Collect mappings (possibly multiprocessed) ---
    global_mapping: dict[str, str] = {}

    if n_jobs is None or n_jobs <= 1:
        # Serial version (less overhead, useful for debugging)
        if verbose:
            logger.debug("Running in SERIAL mode.")
        for f in pod5_files:
            if verbose:
                logger.debug(f"  Scanning {os.path.basename(f)} ...")
            part = _collect_read_origins_from_pod5(f, target_ids)
            global_mapping.update(part)
    else:
        if verbose:
            logger.debug(f"Running in PARALLEL mode with {n_jobs} workers.")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(_collect_read_origins_from_pod5, f, target_ids): f for f in pod5_files
            }
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    part = fut.result()
                except Exception as e:
                    logger.warning(f"Error while processing {f}: {e}")
                    continue
                global_mapping.update(part)
                if verbose:
                    logger.info(f"  Finished {os.path.basename(f)} ({len(part)} matching reads)")

    if verbose:
        logger.info(f"Total reads matched: {len(global_mapping)}")

    # --- Populate obs['pod5_origin'] in AnnData order, memory-efficiently ---
    origin = np.empty(adata.n_obs, dtype=object)
    default = None if fill_value is None else fill_value
    for i, rid in enumerate(obs_names):
        origin[i] = global_mapping.get(rid, default)

    adata.obs["pod5_origin"] = origin
    if verbose:
        logger.info("Assigned `pod5_origin` to adata.obs.")

    # --- Optionally write a CSV ---
    if csv_path is not None:
        if verbose:
            logger.info(f"Writing CSV mapping to: {csv_path}")

        # Create DataFrame in AnnData order for easier cross-referencing
        df = pd.DataFrame(
            {
                "read_id": obs_names,
                "pod5_origin": origin,
            }
        )
        df.to_csv(csv_path, index=False)

        if verbose:
            logger.info("CSV saved.")

    return global_mapping


def expand_bi_tag_columns(adata, bi_column="bi"):
    """Expand dorado bi array tag into individual score columns.

    The bi tag is a 7-element float array from dorado >= 1.3.1:
    - bi[0]: overall barcode score
    - bi[1]: top barcode start position
    - bi[2]: top barcode length
    - bi[3]: top (front) barcode score
    - bi[4]: bottom barcode end position
    - bi[5]: bottom barcode length
    - bi[6]: bottom (rear) barcode score

    This function expands the array into separate columns with descriptive names.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with bi tag in obs.
    bi_column : str, default "bi"
        Name of the column containing bi array.
    """
    import pandas as pd

    if bi_column not in adata.obs.columns:
        logger.debug(f"Column '{bi_column}' not found in adata.obs, skipping expansion")
        return

    logger.info(f"Expanding {bi_column} array into individual columns")

    bi_data = adata.obs[bi_column]

    # Initialize columns
    bi_overall = []
    bi_top_start = []
    bi_top_length = []
    bi_top_score = []
    bi_bottom_end = []
    bi_bottom_length = []
    bi_bottom_score = []

    for val in bi_data:
        if pd.isna(val) or val is None:
            bi_overall.append(np.nan)
            bi_top_start.append(np.nan)
            bi_top_length.append(np.nan)
            bi_top_score.append(np.nan)
            bi_bottom_end.append(np.nan)
            bi_bottom_length.append(np.nan)
            bi_bottom_score.append(np.nan)
        else:
            # val should be array-like
            bi_array = np.array(val) if not isinstance(val, np.ndarray) else val
            bi_overall.append(bi_array[0] if len(bi_array) > 0 else np.nan)
            bi_top_start.append(bi_array[1] if len(bi_array) > 1 else np.nan)
            bi_top_length.append(bi_array[2] if len(bi_array) > 2 else np.nan)
            bi_top_score.append(bi_array[3] if len(bi_array) > 3 else np.nan)
            bi_bottom_end.append(bi_array[4] if len(bi_array) > 4 else np.nan)
            bi_bottom_length.append(bi_array[5] if len(bi_array) > 5 else np.nan)
            bi_bottom_score.append(bi_array[6] if len(bi_array) > 6 else np.nan)

    adata.obs["bi_overall_score"] = bi_overall
    adata.obs["bi_top_start"] = bi_top_start
    adata.obs["bi_top_length"] = bi_top_length
    adata.obs["bi_top_score"] = bi_top_score
    adata.obs["bi_bottom_end"] = bi_bottom_end
    adata.obs["bi_bottom_length"] = bi_bottom_length
    adata.obs["bi_bottom_score"] = bi_bottom_score

    logger.info("Created columns: bi_overall_score, bi_top_score, bi_bottom_score, etc.")


def add_demux_type_from_bm_tag(adata, bm_column="BM"):
    """Add demux_type column to adata.obs from BM tag values.

    Maps BM tag values to demux_type categories:
    - "both" → "double_ended"
    - "left_only", "right_only", "read_start_only", "read_end_only" → "single_ended"
    - "unknown" or "unclassified" → "unclassified"

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    bm_column : str, default "BM"
        Name of column containing BM tag values.
    """
    import pandas as pd

    if bm_column not in adata.obs.columns:
        logger.warning(f"Column '{bm_column}' not found in adata.obs, cannot derive demux_type")
        return

    logger.info(f"Deriving demux_type from {bm_column} tag")

    def map_bm_to_demux_type(bm_value):
        if pd.isna(bm_value):
            return "unclassified"
        bm_str = str(bm_value).lower()
        if bm_str == "both":
            return "double"
        elif bm_str in ("left_only", "right_only", "read_start_only", "read_end_only"):
            return "single"
        else:  # unknown, unclassified
            return "unclassified"

    adata.obs["demux_type"] = adata.obs[bm_column].apply(map_bm_to_demux_type)

    # Log counts
    counts = adata.obs["demux_type"].value_counts()
    logger.info(f"demux_type counts: {counts.to_dict()}")
