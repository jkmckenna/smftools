from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

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
