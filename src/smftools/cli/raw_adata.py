"""CLI pipeline for tool-heavy BAM preparation and ragged raw extraction."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import (
    BAM_OUTPUTS_DIR,
    BED_OUTPUTS_DIR,
    FASTA_OUTPUTS_DIR,
    MODKIT_EXTRACT_CALL_CODE_CANONICAL,
    MODKIT_EXTRACT_CALL_CODE_MODIFIED,
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE,
    MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE,
    MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB,
    MODKIT_EXTRACT_TSV_COLUMN_CHROM,
    MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE,
    MODKIT_EXTRACT_TSV_COLUMN_READ_ID,
    MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION,
    MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND,
    MODKIT_OUTPUTS_DIR,
    RAW_DIR,
    SPLIT_DIR,
)
from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def _conversion_signal(record: dict[str, object], *, deaminase: bool) -> list[float]:
    """Return query-coordinate conversion/deamination signal for one read."""
    bases = [MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE[int(value)] for value in record["sequence"]]
    result = np.full(len(bases), np.nan, dtype=np.float32)
    dataset = str(record.get("dataset", "unconverted"))
    strand = str(record.get("strand", "top"))
    trend = str(record.get("Read_mismatch_trend", "none"))
    if deaminase:
        mapping = {"C": 0.0, "T": 1.0} if trend == "C->T" else None
        if trend == "G->A":
            mapping = {"G": 0.0, "A": 1.0}
    elif dataset == "unconverted":
        mapping = None
    else:
        mappings = {
            ("top", "5mC"): {"C": 1.0, "T": 0.0},
            ("bottom", "5mC"): {"G": 1.0, "A": 0.0},
            ("top", "6mA"): {"A": 1.0, "G": 0.0},
            ("bottom", "6mA"): {"T": 1.0, "C": 0.0},
        }
        mapping = mappings.get((strand, dataset))
    if mapping:
        for index, base in enumerate(bases):
            if base in mapping:
                result[index] = mapping[base]
    return result.tolist()


def _load_read_sidecar(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_parquet(path)
    read_column = "read_name" if "read_name" in frame else "read_id"
    if read_column not in frame:
        raise ValueError(f"sidecar {path} lacks a read_name/read_id column")
    return frame.drop_duplicates(read_column).set_index(read_column)


def _attach_obs_metadata(
    frame: pd.DataFrame,
    *,
    cfg,
    bam_path: Path,
    barcode_sidecar: str | Path | None,
    umi_sidecar: str | Path | None,
) -> pd.DataFrame:
    """Attach scalar barcode, UMI, and read-QC metadata to ragged records."""
    from ..informatics.bam_functions import extract_read_features_from_bam

    frame = frame.set_index("read_id", drop=False)
    for sidecar in (
        _load_read_sidecar(barcode_sidecar),
        _load_read_sidecar(umi_sidecar),
    ):
        if sidecar is not None:
            for column in sidecar.columns:
                frame[column] = sidecar[column].reindex(frame.index)

    if "BC" in frame:
        frame["barcode"] = frame["BC"].fillna("unclassified").astype(str)
    elif "barcode" not in frame:
        frame["barcode"] = "unknown"
    frame["sample"] = frame["barcode"]
    frame["Experiment_name"] = str(cfg.experiment_name)
    frame["Experiment_name_and_barcode"] = (
        frame["Experiment_name"] + "_" + frame["barcode"].astype(str)
    )

    metrics = extract_read_features_from_bam(
        bam_path, samtools_backend=cfg.samtools_backend, primary_only=True
    )
    metric_columns = (
        "read_length",
        "read_quality",
        "reference_length",
        "mapped_length",
        "mapping_quality",
        "reference_start_metric",
        "reference_end",
    )
    metric_frame = pd.DataFrame.from_dict(metrics, orient="index", columns=metric_columns)
    frame = frame.join(metric_frame)
    read_length = pd.to_numeric(frame["read_length"], errors="coerce")
    reference_length = pd.to_numeric(frame["reference_length"], errors="coerce")
    mapped_length = pd.to_numeric(frame["mapped_length"], errors="coerce")
    frame["read_length_to_reference_length_ratio"] = read_length / reference_length
    frame["mapped_length_to_reference_length_ratio"] = mapped_length / reference_length
    frame["mapped_length_to_read_length_ratio"] = mapped_length / read_length
    if getattr(cfg, "skip_unclassified", False):
        frame = frame.loc[frame["barcode"] != "unclassified"]
    return frame.reset_index(drop=True)


def _direct_probability(call_code: object, probability: object) -> float:
    value = float(probability)
    if call_code in MODKIT_EXTRACT_CALL_CODE_MODIFIED:
        return value
    if call_code in MODKIT_EXTRACT_CALL_CODE_CANONICAL:
        return 1.0 - value
    return float("nan")


def _attach_direct_signals(
    frame: pd.DataFrame, mod_tsv_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Attach combined and per-base/strand query-coordinate modkit signals."""
    from ..informatics.ragged_store import cigar_query_length, iter_cigar_aligned_pairs

    frame = frame.set_index("read_id", drop=False)
    frame["modification_signal"] = [
        [float("nan")] * cigar_query_length(cigar) for cigar in frame["cigar"]
    ]
    signal_columns: set[str] = set()
    tsv_paths = sorted(mod_tsv_dir.glob("*.tsv"))
    if not tsv_paths:
        raise FileNotFoundError(f"no modkit extract TSVs found under {mod_tsv_dir}")
    calls = pd.concat((pd.read_csv(path, sep="\t") for path in tsv_paths), ignore_index=True)
    calls[MODKIT_EXTRACT_TSV_COLUMN_READ_ID] = calls[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].astype(str)
    calls = calls.loc[calls[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].isin(frame.index)]

    for read_id, group in calls.groupby(MODKIT_EXTRACT_TSV_COLUMN_READ_ID, sort=False):
        record = frame.loc[read_id]
        reference_to_query = {
            reference: query
            for query, reference in iter_cigar_aligned_pairs(
                str(record["cigar"]), int(record["reference_start"])
            )
        }
        combined = list(record["modification_signal"])
        channel_arrays: dict[str, list[float]] = {}
        for _, call in group.iterrows():
            if str(call[MODKIT_EXTRACT_TSV_COLUMN_CHROM]) != str(record["reference"]):
                continue
            reference_position = int(call[MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION])
            query_position = reference_to_query.get(reference_position)
            if query_position is None:
                continue
            probability = _direct_probability(
                call[MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE],
                call[MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB],
            )
            base = str(call[MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE])
            strand = "plus" if str(call[MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND]) == "+" else "minus"
            safe_base = re.sub(r"[^A-Za-z0-9]+", "_", base)
            column = f"modification_signal_{safe_base}_{strand}"
            signal_columns.add(column)
            channel = channel_arrays.setdefault(column, [float("nan")] * len(combined))
            channel[query_position] = probability
            if np.isnan(combined[query_position]):
                combined[query_position] = probability
            else:
                combined[query_position] += probability
        frame.at[read_id, "modification_signal"] = combined
        for column, values in channel_arrays.items():
            if column not in frame:
                frame[column] = pd.Series(index=frame.index, dtype=object)
            frame.at[read_id, column] = values
    return frame.reset_index(drop=True), sorted(signal_columns)


def build_ragged_records(
    cfg,
    *,
    fasta: Path,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None = None,
    umi_sidecar: str | Path | None = None,
    mod_tsv_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, object]]:
    """Extract the aligned BAM into the canonical one-row-per-read raw schema."""
    from ..informatics.bam_functions import extract_read_relative_base_identities
    from ..informatics.fasta_functions import get_native_references

    reference_map = get_native_references(fasta)
    rows: list[dict[str, object]] = []
    reference_lengths: dict[str, int] = {}
    modality = str(cfg.smf_modality)
    deaminase = modality == "deaminase"

    if modality in {"conversion", "deaminase"}:
        from ..informatics.converted_BAM_to_adata import process_conversion_sites

        _, record_info, chromosome_sequences = process_conversion_sites(
            fasta, cfg.conversion_types, deaminase
        )
    else:
        record_info = {}
        chromosome_sequences = {
            reference: (sequence, sequence)
            for reference, (_length, sequence) in reference_map.items()
        }

    for record, (record_length, sequence) in reference_map.items():
        extracted = extract_read_relative_base_identities(
            aligned_bam,
            record,
            sequence,
            samtools_backend=cfg.samtools_backend,
            primary_only=True,
        )
        for row in extracted:
            if modality in {"conversion", "deaminase"}:
                info = record_info[record]
                strand = info.strand
                if deaminase and row["Read_mismatch_trend"] == "G->A":
                    strand = "bottom"
                row["reference"] = info.chromosome
                row["strand"] = strand
                row["dataset"] = info.conversion
                row["Reference_strand"] = f"{info.chromosome}_{strand}"
                row["modification_signal"] = _conversion_signal(row, deaminase=deaminase)
                reference_lengths[row["Reference_strand"]] = info.sequence_length
            else:
                reference_lengths[row["Reference_strand"]] = record_length
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no primary mapped reads were extracted from {aligned_bam}")

    frame = _attach_obs_metadata(
        pd.DataFrame(rows),
        cfg=cfg,
        bam_path=aligned_bam,
        barcode_sidecar=barcode_sidecar,
        umi_sidecar=umi_sidecar,
    )
    signal_columns: list[str] = []
    if modality == "direct":
        if mod_tsv_dir is None:
            raise ValueError("direct raw extraction requires mod_tsv_dir")
        frame, signal_columns = _attach_direct_signals(frame, mod_tsv_dir)

    references = {
        f"{reference}_FASTA_sequence": sequence
        for reference, (sequence, _complement) in chromosome_sequences.items()
    }
    return (
        frame,
        reference_lengths,
        {
            "References": references,
            "signal_columns": signal_columns,
            "modality": modality,
            "sequence_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
            "mismatch_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
            "sequence_integer_decoding_map": {
                str(key): value for key, value in MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE.items()
            },
        },
    )


def raw_adata(config_path: str):
    """Run BAM preparation through section 6 and emit ragged raw artifacts."""
    from ..readwrite import safe_read_h5ad
    from .helpers import get_adata_paths, load_experiment_config
    from .load_adata import load_adata_core

    cfg = load_experiment_config(config_path)
    raw_root = Path(cfg.output_directory) / RAW_DIR
    cfg.informatics_outputs_path = raw_root
    cfg.bam_outputs_path = raw_root / BAM_OUTPUTS_DIR
    cfg.fasta_outputs_path = raw_root / FASTA_OUTPUTS_DIR
    cfg.bed_outputs_path = raw_root / BED_OUTPUTS_DIR
    cfg.modkit_outputs_path = raw_root / MODKIT_OUTPUTS_DIR
    cfg.split_path = cfg.bam_outputs_path / SPLIT_DIR
    paths = get_adata_paths(cfg)
    if paths.raw_spine and paths.raw_spine.exists() and not cfg.force_redo_load_adata:
        spine, _ = safe_read_h5ad(paths.raw_spine)
        if "ragged_store" in spine.uns:
            logger.info("Raw ragged store already exists: %s", paths.raw_spine)
            return spine, paths.raw_spine, cfg
    return load_adata_core(cfg, paths, config_path=config_path, raw_only=True)
