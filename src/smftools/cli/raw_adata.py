"""CLI pipeline for tool-heavy BAM preparation and ragged raw extraction."""

from __future__ import annotations

import re
import time
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
    PARTITIONED_STAGE_NONEMPTY_DIRECTORIES,
    PARTITIONED_STAGE_REQUIRED_ARTIFACTS,
    RAW_DIR,
    REFERENCE_INTERVAL_MAP_FILENAME,
    REGION_CATALOG_DIRNAME,
    REGION_CATALOG_FILENAMES,
    SPLIT_DIR,
)
from smftools.logging_utils import get_logger, mark_stage_outcome, stage_logging_lifecycle

from ..informatics.molecule_identity import (
    BASECALL_PARENT_READ_ID_COLUMN,
    BASECALL_READ_ID_COLUMN,
    POD5_IDENTITY_EVIDENCE_COLUMN,
    POD5_IDENTITY_STATUS_COLUMN,
    POD5_READ_ID_COLUMN,
    alignment_segment_id,
    namespaced_source_id,
)

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
    metrics: dict | None = None,
) -> pd.DataFrame:
    """Attach scalar barcode, UMI, and read-QC metadata to ragged records.

    ``metrics``, when given, is used as-is instead of calling
    ``extract_read_features_from_bam`` internally -- that call scans the whole
    BAM regardless of which reads ``frame`` actually contains, so a caller
    processing one reference's frame at a time (see ``build_ragged_records_
    streaming``) should compute it once, upfront, and pass the same dict to
    every call rather than re-scanning the whole BAM once per reference.
    """
    from ..informatics.barcode_sidecar import read_barcode_identity_sidecar
    from ..informatics.ragged_store import cigar_max_indel_runs

    frame = frame.set_index("read_id", drop=False)
    if barcode_sidecar is not None and Path(barcode_sidecar).is_file():
        barcode_frame = read_barcode_identity_sidecar(barcode_sidecar).set_index(
            "read_name", drop=False
        )
        for column in barcode_frame.columns:
            if column != "read_name":
                frame[column] = barcode_frame[column].reindex(frame.index)
    for sidecar in (_load_read_sidecar(umi_sidecar),):
        if sidecar is not None:
            for column in sidecar.columns:
                frame[column] = sidecar[column].reindex(frame.index)

    if "barcode" not in frame:
        frame["barcode"] = "unknown"
    else:
        frame["barcode"] = frame["barcode"].fillna("unknown").astype(str)
    if "sample" not in frame:
        frame["sample"] = frame["barcode"]
    else:
        frame["sample"] = frame["sample"].fillna(frame["barcode"]).astype(str)
    frame["Experiment_name"] = str(cfg.experiment_name)
    identity_namespace = pd.Series(str(cfg.experiment_name), index=frame.index)
    if "namespace" in frame:
        declared_namespace = frame["namespace"].fillna("").astype(str)
        identity_namespace = declared_namespace.where(declared_namespace != "", identity_namespace)
    frame["Experiment_name_and_barcode"] = identity_namespace + "_" + frame["barcode"].astype(str)

    if metrics is None:
        from ..informatics.bam_functions import extract_read_features_from_bam

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

    # Longest internal insertion/deletion run per read, from the alignment CIGAR.
    # Carried onto the molecule spine so preprocessing can filter reads with large
    # internal indels (e.g. spurious gaps) without re-reading the BAM.
    if "cigar" in frame:
        indel_runs = [cigar_max_indel_runs(str(cigar)) for cigar in frame["cigar"]]
        frame["max_insertion_length"] = [runs[0] for runs in indel_runs]
        frame["max_deletion_length"] = [runs[1] for runs in indel_runs]
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
    frame: pd.DataFrame,
    mod_tsv_dir: Path | None = None,
    *,
    tsv_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach combined and per-base/strand query-coordinate modkit signals.

    Reads either every TSV under ``mod_tsv_dir`` (a whole directory) or,
    when the caller already narrowed things down to a small pre-split chunk
    (see ``_split_modkit_tsv_by_bucket``), exactly the file(s) in
    ``tsv_paths`` -- the same join logic serves both without duplicating it.
    Production code always uses ``tsv_paths`` now (``_extract_direct_
    reference_modkit``, one bucket's own small chunk); ``mod_tsv_dir`` is
    kept for joining an arbitrary whole directory of TSVs directly.
    """
    from ..informatics.ragged_store import cigar_query_length, iter_cigar_aligned_pairs

    frame = frame.set_index("read_id", drop=False)
    frame["modification_signal"] = [
        [float("nan")] * cigar_query_length(cigar) for cigar in frame["cigar"]
    ]
    signal_columns: set[str] = set()
    if tsv_paths is None:
        if mod_tsv_dir is None:
            raise ValueError("either mod_tsv_dir or tsv_paths must be given")
        tsv_paths = sorted(mod_tsv_dir.glob("*.tsv")) + sorted(mod_tsv_dir.glob("*.tsv.gz"))
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


def _resolve_direct_call(code_bytes: dict[str, int]) -> tuple[str, float]:
    """Pick the winning call at one query position from its ML byte(s).

    Mirrors modkit's own per-position resolution: canonical's probability is
    ``1 - sum(modified probabilities)`` (the SAM MM/ML spec's implicit
    "everything else" state), and whichever of {canonical, each listed
    modified code} has the highest probability wins. Verified empirically
    against a real modkit-extract TSV on real direct-modality data: this
    reproduces modkit's ``call_code``/``call_prob`` columns exactly (0
    mismatches across every explicitly-called position in a sampled read) --
    see dev/pipeline_scaling_audit.md's Track B notes.
    """
    canonical_prob = 1.0 - sum(value / 255.0 for value in code_bytes.values())
    best_code, best_prob = "-", canonical_prob
    for code, ml_byte in code_bytes.items():
        modified_prob = ml_byte / 255.0
        if modified_prob > best_prob:
            best_code, best_prob = code, modified_prob
    return best_code, best_prob


def _attach_direct_signals_from_bam(
    frame: pd.DataFrame,
    aligned_bam: Path,
    *,
    window_start: int | None = None,
    window_end: int | None = None,
    impute_uncalled_canonical: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach combined and per-base/strand query-coordinate modification signals.

    ``window_start``/``window_end``, when given, scope the BAM scan to
    ``[window_start, window_end)`` instead of the whole chromosome -- purely
    an optimization for callers already restricted to a sub-reference window
    (see ``_extract_direct_reference``'s parallel windowed path): reads
    outside ``frame``'s own read-id set are skipped regardless (``wanted``
    below), so correctness never depends on this scoping, only how much of
    the BAM gets re-scanned per call.

    Decodes the aligned BAM's own MM/ML tags via ``pysam.AlignedSegment.
    modified_bases`` instead of joining a modkit-extract TSV (``_attach_direct_
    signals``'s approach) -- avoids the external ``modkit extract`` subprocess
    and its whole-file TSV entirely, and needs only the same aligned BAM
    already open for read extraction, so it's streaming-compatible (see
    dev/pipeline_scaling_audit.md's Track B notes). Selected via
    ``cfg.direct_signal_backend == "pysam"`` -- not the default for now (see
    that field's docstring in ``experiment_config.py``): its decode produces
    a lower downstream QC-pass rate than modkit's own output on real data,
    root cause not yet understood.

    ``modified_bases`` (not ``modified_bases_forward``) is used deliberately:
    its query positions are relative to the BAM-stored, CIGAR-relative query
    sequence -- the same coordinate space ``cigar``/``modification_signal``
    already use throughout this codebase. ``modified_bases_forward`` re-orients
    positions to original (pre-alignment) sequencing direction instead, a
    different, incompatible coordinate space for reverse-strand reads.

    One deliberate behavior difference from ``_attach_direct_signals``:
    positions with no explicit MM/ML tag entry are left ``NaN`` (no signal),
    rather than modkit's own convention of filling them with a synthetic
    "canonical, probability 1.0" row. Verified against a real modkit TSV: its
    ``inferred=True`` rows exist exactly where no ML entry does (0 mismatches
    either direction, sampled over a real read) -- leaving them ``NaN`` is more
    correct (no information genuinely means no information), not a divergence
    to reconcile.

    ``impute_uncalled_canonical`` (default ``False``) opts back into modkit's
    convention for A/B comparison: for every canonical-base position in the
    read's own query sequence with no explicit MM/ML entry, fill probability
    ``0.0`` (canonical) instead of leaving ``NaN``. Only affects positions of
    a canonical base that has at least one explicit call elsewhere in the
    read (``calls_by_base``); a read with zero calls for a base is left
    exactly as before, matching modkit's own per-read gating.
    """
    import pysam

    from ..informatics.ragged_store import cigar_query_length

    frame = frame.set_index("read_id", drop=False)
    frame["modification_signal"] = [
        [float("nan")] * cigar_query_length(cigar) for cigar in frame["cigar"]
    ]
    signal_columns: set[str] = set()

    windowed = window_start is not None and window_end is not None

    bam = pysam.AlignmentFile(str(aligned_bam), "rb")
    try:
        for chrom, group in frame.groupby("reference", sort=False):
            wanted = set(group.index)
            fetch_iter = (
                bam.fetch(reference=str(chrom), start=window_start, stop=window_end)
                if windowed
                else bam.fetch(reference=str(chrom))
            )
            for read in fetch_iter:
                if read.is_secondary or read.is_supplementary or read.is_unmapped:
                    continue
                read_id = alignment_segment_id(read)
                if read_id not in wanted:
                    continue
                record = frame.loc[read_id]
                combined = list(record["modification_signal"])
                channel_arrays: dict[str, list[float]] = {}
                strand = "plus" if not read.is_reverse else "minus"

                calls_by_base: dict[str, dict[int, dict[str, int]]] = {}
                for (canonical_base, _strand_bit, code), calls in read.modified_bases.items():
                    per_position = calls_by_base.setdefault(canonical_base, {})
                    for query_position, ml_byte in calls:
                        per_position.setdefault(query_position, {})[code] = ml_byte

                query_sequence = read.query_sequence if impute_uncalled_canonical else None

                for canonical_base, per_position in calls_by_base.items():
                    safe_base = re.sub(r"[^A-Za-z0-9]+", "_", canonical_base)
                    column = f"modification_signal_{safe_base}_{strand}"
                    signal_columns.add(column)
                    channel = channel_arrays.setdefault(column, [float("nan")] * len(combined))
                    for query_position, code_bytes in per_position.items():
                        if query_position >= len(combined):
                            continue
                        call_code, winning_probability = _resolve_direct_call(code_bytes)
                        # _resolve_direct_call returns the winning STATE's own
                        # confidence -- e.g. 0.96 when canonical wins with 96%
                        # confidence, not "4% modified". _direct_probability
                        # (already used by the modkit-TSV join path,
                        # _attach_direct_signals) converts that into a
                        # consistent P(modified) by inverting canonical-wins
                        # calls; skipping this step here previously stored the
                        # raw canonical confidence directly, so a read that
                        # was e.g. 96% confidently UNMODIFIED at a position
                        # showed up as a 0.96 (strong-looking) modification
                        # signal -- systematically inflating and decorrelating
                        # Raw_modification_signal from true methylation level.
                        probability = _direct_probability(call_code, winning_probability)
                        channel[query_position] = probability
                        if np.isnan(combined[query_position]):
                            combined[query_position] = probability
                        else:
                            combined[query_position] += probability

                    if query_sequence:
                        base_upper = canonical_base.upper()
                        for query_position, base in enumerate(query_sequence):
                            if query_position >= len(combined) or query_position in per_position:
                                continue
                            if base.upper() != base_upper:
                                continue
                            channel[query_position] = 0.0
                            if np.isnan(combined[query_position]):
                                combined[query_position] = 0.0

                frame.at[read_id, "modification_signal"] = combined
                for column, values in channel_arrays.items():
                    if column not in frame:
                        frame[column] = pd.Series(index=frame.index, dtype=object)
                    frame.at[read_id, column] = values
    finally:
        bam.close()
    return frame.reset_index(drop=True), sorted(signal_columns)


def _attach_pod5_metadata(frame: pd.DataFrame, *, cfg) -> pd.DataFrame:
    """Validate each basecall's source POD5 identity and attach POD5 metadata.

    Dorado split children use the BAM ``pi`` tag as their candidate POD5 UUID;
    unsplit calls use their original BAM query name. A candidate is promoted to
    ``pod5_read_id`` only when it exists in the configured POD5 dataset. This
    identity validation always runs for available POD5 inputs, even when the
    optional sequencing-metadata enrichment is disabled. Scalar ``pod5_*``
    columns are carried onto the molecule spine by ``raw_store``; the optional
    full current trace (``pod5_current_pa``) stays in the parquet shard.
    """
    if str(getattr(cfg, "input_type", "")).lower() != "pod5":
        return frame

    frame = frame.copy()
    if BASECALL_READ_ID_COLUMN not in frame:
        frame[BASECALL_READ_ID_COLUMN] = frame["read_id"].astype(str)
    basecall_ids = frame[BASECALL_READ_ID_COLUMN].fillna("").astype(str)
    basecall_ids = basecall_ids.where(basecall_ids.str.len() > 0, frame["read_id"].astype(str))
    parent_ids = frame.get(
        BASECALL_PARENT_READ_ID_COLUMN,
        pd.Series(None, index=frame.index, dtype=object),
    )
    parent_ids = parent_ids.fillna("").astype(str).str.strip()
    candidates = parent_ids.where(parent_ids.str.len() > 0, basecall_ids)
    candidate_sources = parent_ids.str.len() > 0

    pod5_path = getattr(cfg, "input_data_path", None)
    if pod5_path is None or not Path(pod5_path).exists():
        logger.warning("input_type=pod5 but input_data_path is missing; skipping POD5 metadata")
        frame[POD5_READ_ID_COLUMN] = None
        frame[POD5_IDENTITY_STATUS_COLUMN] = "source_unavailable"
        frame[POD5_IDENTITY_EVIDENCE_COLUMN] = "pod5_source_unavailable"
        return frame

    from ..informatics.pod5_functions import extract_pod5_read_metadata

    include_metadata = bool(getattr(cfg, "extract_pod5_metadata", True))
    metadata = extract_pod5_read_metadata(
        pod5_path,
        target_ids=candidates,
        n_jobs=getattr(cfg, "threads", 1),
        include_current=(include_metadata and bool(getattr(cfg, "raw_store_pod5_current", False))),
        verbose=False,
    )
    matched = (
        candidates.isin(metadata.index.astype(str))
        if not metadata.empty
        else candidates.ne(candidates)
    )
    frame[POD5_READ_ID_COLUMN] = [
        candidate if is_matched else None
        for candidate, is_matched in zip(candidates, matched, strict=True)
    ]
    frame[POD5_IDENTITY_STATUS_COLUMN] = np.where(matched, "resolved", "unresolved")
    frame[POD5_IDENTITY_EVIDENCE_COLUMN] = np.select(
        [matched & candidate_sources, matched & ~candidate_sources, candidate_sources],
        ["bam_pi+pod5_index", "bam_qname+pod5_index", "bam_pi_not_found"],
        default="bam_qname_not_found",
    )

    if not metadata.empty:
        metadata.index = metadata.index.astype(str)
        columns = metadata.columns if include_metadata else ["pod5_origin"]
        for column in columns:
            if column in metadata:
                frame[column] = candidates.map(metadata[column]).to_numpy()
    if not matched.all():
        logger.warning(
            "Could not validate POD5 origin identity for %d/%d extracted read(s)",
            int((~matched).sum()),
            len(frame),
        )
    logger.info(
        "Validated POD5 origin identity for %d/%d extracted read(s)",
        int(matched.sum()),
        len(frame),
    )
    return frame


def _read_move_tables(
    bam_path: Path, target_ids: set[str], *, primary_only: bool = True
) -> dict[str, tuple[list, int]]:
    """Return ``{read_id: (mv, ts)}`` for reads carrying the dorado move table."""
    from ..informatics.bam_functions import _require_pysam

    pysam = _require_pysam()
    tables: dict[str, tuple[list, int]] = {}
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if primary_only and (read.is_secondary or read.is_supplementary):
                continue
            read_id = alignment_segment_id(read)
            if read_id not in target_ids or read_id in tables or not read.has_tag("mv"):
                continue
            ts = int(read.get_tag("ts")) if read.has_tag("ts") else 0
            tables[read_id] = (list(read.get_tag("mv")), ts)
    return tables


def _attach_signal_features(frame: pd.DataFrame, *, cfg, aligned_bam: Path) -> pd.DataFrame:
    """Attach per-base current mean/std/dwell/start from the move table + POD5 signal.

    Composes the dorado move table (``mv``/``ts`` BAM tags, from ``--emit-moves``)
    with the raw POD5 current to produce read-relative signal-feature arrays; these
    densify to reference-grid layers via ``materialize_ragged``. Requires POD5 input
    with move tables preserved on the aligned BAM; skips gracefully otherwise.
    """
    if str(getattr(cfg, "input_type", "")).lower() != "pod5":
        return frame
    if not getattr(cfg, "extract_signal_features", True):
        return frame
    pod5_path = getattr(cfg, "input_data_path", None)
    if pod5_path is None or not Path(pod5_path).exists():
        return frame

    target_ids = set(frame["read_id"].astype(str))
    move_tables = _read_move_tables(Path(aligned_bam), target_ids)
    if not move_tables:
        logger.warning(
            "No move tables (mv tag) found on %s; skipping current signal features. "
            "Re-run with emit_moves=True and an aligner that preserves tags.",
            aligned_bam,
        )
        return frame

    from ..informatics.pod5_functions import iter_pod5_signals
    from ..informatics.signal_features import SIGNAL_FEATURE_COLUMNS, read_signal_features

    frame = frame.set_index("read_id", drop=False)
    for column in SIGNAL_FEATURE_COLUMNS:
        frame[column] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    reverse_by_read = (frame["mapping_direction"].astype(str) == "rev").to_dict()
    seq_len_by_read = {
        read_id: (len(sequence) if sequence is not None else 0)
        for read_id, sequence in frame["sequence"].items()
    }

    pod5_by_read = frame.get(POD5_READ_ID_COLUMN, frame["read_id"]).to_dict()
    reads_by_pod5: dict[str, list[str]] = {}
    for read_id, pod5_read_id in pod5_by_read.items():
        if pd.isna(pod5_read_id) or read_id not in move_tables:
            continue
        reads_by_pod5.setdefault(str(pod5_read_id), []).append(str(read_id))

    attached = 0
    for pod5_read_id, signal in iter_pod5_signals(pod5_path, read_ids=list(reads_by_pod5)):
        for read_id in reads_by_pod5.get(pod5_read_id, []):
            mv, ts = move_tables[read_id]
            features = read_signal_features(
                mv,
                ts,
                bool(reverse_by_read.get(read_id, False)),
                signal,
                expected_bases=seq_len_by_read.get(read_id),
            )
            if features is None:
                continue
            for column in SIGNAL_FEATURE_COLUMNS:
                frame.at[read_id, column] = features[column].tolist()
            attached += 1

    logger.info("Attached current signal features for %d/%d read(s)", attached, len(frame))
    return frame.reset_index(drop=True)


def _split_by_reference_strand(frame: pd.DataFrame):
    """Split a frame into one sub-frame per distinct ``Reference_strand``.

    A deaminase read's ``Reference_strand`` is decided per-read (a chromosome's
    canonical strand can be overridden to "_bottom" by that read's own
    mismatch trend), so a single chromosome's extracted frame can contain a
    mix of "_top" and "_bottom" rows even though it came from one FASTA
    record. Streaming shard writers (``raw_store._write_raw_shards_streaming``)
    require each yielded group to be single-``Reference_strand`` -- they label
    a whole group from its first row -- so callers must split here before
    handing a frame off.
    """
    for _, strand_frame in frame.groupby("Reference_strand", sort=True, observed=True):
        yield strand_frame


def _yield_flush_result(
    flush_result: tuple[list[pd.DataFrame], bool] | None,
    chromosome: str,
    strands_seen: dict[str, set[str]],
):
    """Split one ``_ChromosomeGroupAccumulator`` flush into per-strand yields.

    Yields ``(reference_strand, frame_or_None, is_final)``. ``is_final`` here
    is per-``Reference_strand``, not per-chromosome: on a chromosome's actual
    final flush, every ``Reference_strand`` ever seen for it -- not just the
    ones present in this last batch -- must be marked final, since no more
    data for any of them is ever coming (a chromosome can yield "_top" and
    "_bottom" unevenly across flushes; see ``_ChromosomeGroupAccumulator``).
    Strands with no new rows in a chromosome's final flush still get a
    ``frame=None`` marker so callers can finalize their own bookkeeping
    (e.g. ``plan_references``) for them.
    """
    if flush_result is None:
        return
    frames, is_final = flush_result
    seen = strands_seen.setdefault(chromosome, set())
    yielded_now: set[str] = set()
    if frames:
        combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        for sub_frame in _split_by_reference_strand(combined):
            strand = str(sub_frame["Reference_strand"].iloc[0])
            seen.add(strand)
            yielded_now.add(strand)
            yield strand, sub_frame, is_final
    if is_final:
        for strand in seen - yielded_now:
            yield strand, None, True
        strands_seen.pop(chromosome, None)


class _ChromosomeGroupAccumulator:
    """Accumulate per-record completions into per-chromosome combined frames.

    A single physical chromosome+strand can be split across multiple
    ``reference_map`` records -- conversion modality aligns each chromosome
    against several conversion-state variants (e.g. ``conversion_types=
    ["5mC"]`` produces ``"{chrom}_unconverted_top"``, ``"{chrom}_5mC_top"``,
    and ``"{chrom}_5mC_bottom"`` as three separate alignment targets/records
    for one chromosome), all of which normalize to the same final
    ``Reference_strand`` once extracted. Deaminase modality, by contrast, has
    exactly one record per chromosome (its top/bottom split happens per-read
    from mismatch trend, not via separate alignment targets), so every
    chromosome here has exactly one contributing record.

    A chromosome's data must not be handed to the streaming shard writer
    piecemeal per-record -- if one record's buckets finished and were written
    while a chromosome sibling record was still outstanding, the writer would
    see multiple separate groups for the same ``Reference_strand``, and it
    always started a fresh shard-index count per group, so a later record's
    group silently overwrote an earlier one's shard file on disk even though
    ``obs.parquet`` ended up with pointers for both (real data loss,
    confirmed on a real conversion-modality dataset before this fix: 3 of 4
    references affected, up to 23% of reads for one).

    That correctness requirement is about *records*, not about how much of a
    chromosome's data can be flushed at once -- holding one entire
    chromosome's ragged data in memory before writing anything doesn't scale
    with experiment size (a single deaminase chromosome with 700k+ reads put
    ~87GB in the parent process; see dev/pipeline_scaling_audit.md). So this
    accumulator supports two independent operations: ``add_partial`` feeds
    newly-available frames for a record (call it any number of times, e.g.
    once per completed bucket) and returns a bounded-size flush the moment
    the chromosome's accumulated row count crosses ``flush_threshold`` --
    safe to do mid-chromosome, since ``write_raw_store_streaming``'s shard
    writer now tracks a persistent shard-index per (reference, start_bin)
    across repeated groups instead of restarting at 0. ``complete`` marks a
    record as having no more data coming, and only that call can resolve a
    chromosome's sibling-completion tracking (its own docstring concern is
    unchanged -- a chromosome isn't *finished* until every record sharing it
    has called ``complete``, this just no longer requires buffering
    everything in between).
    """

    def __init__(self, record_chromosome: dict[str, str], *, flush_threshold: int | None = None):
        self._record_chromosome = dict(record_chromosome)
        self._remaining: dict[str, set[str]] = {}
        for record, chromosome in self._record_chromosome.items():
            self._remaining.setdefault(chromosome, set()).add(record)
        self._pending: dict[str, list[pd.DataFrame]] = {}
        self._pending_rows: dict[str, int] = {}
        self._flush_threshold = flush_threshold

    def add_partial(
        self, record: str, frames: list[pd.DataFrame]
    ) -> tuple[list[pd.DataFrame], bool] | None:
        """Feed newly-available frames for ``record`` (e.g. one bucket's rows).

        Call any number of times per record, including zero times for a
        record that dispatched no work. Never resolves sibling-completion
        tracking -- call ``complete`` separately once ``record`` has no more
        data coming.

        Returns ``(frames_to_write, False)`` if the chromosome's accumulated
        row count just crossed ``flush_threshold`` (a bounded-size, non-final
        flush), else ``None``.
        """
        chromosome = self._record_chromosome[record]
        if frames:
            self._pending.setdefault(chromosome, []).extend(frames)
            self._pending_rows[chromosome] = self._pending_rows.get(chromosome, 0) + sum(
                len(f) for f in frames
            )
        if (
            self._flush_threshold is not None
            and self._pending_rows.get(chromosome, 0) >= self._flush_threshold
        ):
            flushed = self._pending.pop(chromosome, [])
            self._pending_rows[chromosome] = 0
            return (flushed, False)
        return None

    def complete(self, record: str) -> tuple[list[pd.DataFrame], bool] | None:
        """Mark ``record`` as having no more data coming.

        Returns ``(frames_to_write, True)`` -- whatever is left pending for
        the chromosome, possibly empty -- once every record sharing
        ``record``'s chromosome has completed, else ``None``.
        """
        chromosome = self._record_chromosome[record]
        remaining = self._remaining[chromosome]
        remaining.discard(record)
        if remaining:
            return None
        flushed = self._pending.pop(chromosome, [])
        self._pending_rows.pop(chromosome, None)
        return (flushed, True)


def _map_references_parallel(
    items, worker, *, max_workers: int, worker_kwargs: dict, pool_label: str | None = None
):
    """Run ``worker(*args, **worker_kwargs)`` once per item in ``items``.

    Sequential when ``max_workers <= 1``. Otherwise runs in a process pool --
    each reference's extraction (``extract_read_relative_base_identities``,
    ``alignment_to_ragged_record``'s per-base Python list construction) is
    CPU-bound pure Python, so a thread pool would still serialize on the GIL;
    only separate processes give real concurrency here. Each worker needs
    only its own BAM file handle, opened independently (pysam handles can't
    be shared across processes anyway).

    Yields ``(args, result)`` pairs as each future completes, not in
    submission order -- callers need ``args`` back (not just ``result``) to
    know which reference/window a given result belongs to, since completion
    order is scheduler-driven, not submission order. Downstream
    (``write_raw_store_streaming``) already documents that spine.obs row
    order need not match the original per-reference order, so this
    reordering is not a behavior change, only something already accounted
    for.

    At most ``max_workers`` tasks are ever in flight at once -- the next
    item is only submitted once a completed result has actually been
    retrieved (via ``yield``) by the caller. Submitting every item upfront
    (the previous behavior) decouples submission from consumption: with
    ``max_workers`` processes computing large per-bucket ragged DataFrames
    continuously and a single-threaded caller draining them (bookkeeping +
    parquet writes), completed-but-unretrieved results piled up in the
    executor's internal queue without bound -- this is what actually caused
    a ~90GB parent-process blowup on real data even after bounding the
    downstream chromosome accumulator (see dev/pipeline_scaling_audit.md);
    that fix only bounded what happens *after* retrieval, not how far ahead
    of the consumer the pool was allowed to compute. This backpressure
    caps peak parent-side memory at O(max_workers * per_bucket_result_size)
    regardless of total experiment size.
    """
    from ..memory_guard import (
        require_memory_headroom,
        require_task_admission,
        resolve_memory_budget_bytes,
        resolve_pool_budget,
        start_worker_watchdog,
    )
    from ..perf_log import get_perf_logger

    cfg = worker_kwargs.get("cfg")
    n_items = len(items) if hasattr(items, "__len__") else max_workers
    if cfg is not None:
        pool_budget = resolve_pool_budget(
            cfg,
            n_items,
            estimator="raw_extraction_bucket_peak",
        )
        require_task_admission(pool_budget, pool_label=pool_label)
        max_workers = min(max_workers, pool_budget.max_workers)
    else:
        pool_budget = None
    perf = get_perf_logger()
    pool_id = perf.next_pool_id() if perf is not None else None
    if perf is not None:
        perf.pool_start(
            pool_id,
            n_tasks=n_items,
            max_workers=max_workers,
            stage_component="raw_extraction",
            pool_budget=pool_budget.as_dict() if pool_budget is not None else None,
        )
    if max_workers <= 1:
        if pool_label:
            logger.info("[%s] running sequentially", pool_label)
        try:
            for index, args in enumerate(items):
                if cfg is not None:
                    require_memory_headroom(
                        cfg,
                        operation_label=pool_label,
                        estimator="raw_extraction_bucket_peak",
                    )
                started = time.monotonic()
                result = worker(*args, **worker_kwargs)
                if perf is not None:
                    rows, bases = _raw_result_work_counts(result)
                    perf.task_complete(
                        pool_id,
                        task_index=index,
                        completed=index + 1,
                        total=n_items,
                        duration_seconds=time.monotonic() - started,
                        rows=rows,
                        bases=bases,
                    )
                yield args, result
        finally:
            if perf is not None:
                perf.pool_end(pool_id, final_max_workers=1, n_retries=0)
        return

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from ..parallel_utils import configure_worker_threads

    per_worker_budget_bytes = (
        resolve_memory_budget_bytes(cfg) // max_workers if cfg is not None else 0
    )
    poll_interval = float(getattr(cfg, "perf_log_sample_interval_seconds", 2.0) or 2.0)
    items_iter = iter(enumerate(items))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=configure_worker_threads,
        initargs=(1,),
    ) as pool:
        stop_watchdog = start_worker_watchdog(
            pool,
            per_worker_budget_bytes,
            poll_interval,
            perf_logger=perf,
            pool_id=pool_id,
            pool_label=pool_label,
        )
        try:
            future_to_args = {}
            exhausted = object()
            next_item = next(items_iter, exhausted)
            submitted_at = {}
            completed = 0
            while next_item is not exhausted and len(future_to_args) < max_workers:
                index, args = next_item
                future = pool.submit(worker, *args, **worker_kwargs)
                future_to_args[future] = (index, args)
                submitted_at[future] = time.monotonic()
                next_item = next(items_iter, exhausted)
            while future_to_args or next_item is not exhausted:
                if not future_to_args:
                    if cfg is None:
                        index, args = next_item
                        future = pool.submit(worker, *args, **worker_kwargs)
                        future_to_args[future] = (index, args)
                        submitted_at[future] = time.monotonic()
                        next_item = next(items_iter, exhausted)
                        continue
                    blocked_budget = resolve_pool_budget(
                        cfg,
                        1,
                        estimator="raw_extraction_bucket_peak",
                    )
                    require_task_admission(blocked_budget, pool_label=pool_label)
                done, _pending = wait(future_to_args, return_when=FIRST_COMPLETED)
                for future in done:
                    index, args = future_to_args.pop(future)
                    result = future.result()
                    completed += 1
                    if perf is not None:
                        rows, bases = _raw_result_work_counts(result)
                        perf.task_complete(
                            pool_id,
                            task_index=index,
                            completed=completed,
                            total=n_items,
                            duration_seconds=time.monotonic() - submitted_at.pop(future),
                            rows=rows,
                            bases=bases,
                        )
                    yield args, result
                if cfg is None:
                    target_in_flight = max_workers
                else:
                    refill_budget = resolve_pool_budget(
                        cfg,
                        max_workers,
                        estimator="raw_extraction_bucket_peak",
                    )
                    target_in_flight = min(max_workers, refill_budget.max_in_flight)
                while next_item is not exhausted and len(future_to_args) < target_in_flight:
                    index, args = next_item
                    future = pool.submit(worker, *args, **worker_kwargs)
                    future_to_args[future] = (index, args)
                    submitted_at[future] = time.monotonic()
                    next_item = next(items_iter, exhausted)
        finally:
            stop_watchdog()
            if perf is not None:
                perf.pool_end(pool_id, final_max_workers=max_workers, n_retries=0)


def _raw_result_work_counts(result) -> tuple[int, int]:
    """Return rows and sequence bases from a raw extraction worker result."""
    frame = result[0] if isinstance(result, tuple) and result else result
    if not isinstance(frame, pd.DataFrame):
        return 0, 0
    rows = len(frame)
    if "sequence" not in frame:
        return rows, 0
    bases = sum(len(value) for value in frame["sequence"] if hasattr(value, "__len__"))
    return rows, bases


def _read_ids_for_reference(aligned_bam: Path, record: str) -> list[str]:
    """Primary-mapped read_ids for one reference, in BAM traversal order.

    A cheap pre-scan (name only, no CIGAR/sequence/tag decode -- the
    expensive part of extraction) used to build balanced read-id buckets
    before dispatching the real per-bucket extraction work; see
    ``_bucket_read_ids``.
    """
    import pysam

    read_ids: list[str] = []
    with pysam.AlignmentFile(str(aligned_bam), "rb") as bam:
        for read in bam.fetch(record):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            read_ids.append(alignment_segment_id(read))
    return read_ids


def _bucket_read_ids(read_ids: list[str], n_buckets: int) -> list[set[str]]:
    """Split ``read_ids`` into ``n_buckets`` buckets by round-robin assignment.

    Genomic-position windowing was tried first (split ``[0, record_length)``
    into sub-ranges, fetch each independently) and found badly imbalanced on
    real amplicon data: many reads share an *exact* ``reference_start`` (PCR/
    library duplication at a fixed primer site), so no position-based
    boundary can split them apart -- one window still absorbed the majority
    of a reference's reads regardless of how the boundaries were chosen
    (equal-width, even read-count quantiles -- both tried, both still
    imbalanced by that clustering). Round-robin over read *identity* instead
    of position sidesteps the problem entirely: buckets differ in size by at
    most one read, regardless of how reads cluster genomically. Each worker
    still fetches the *whole* reference (cheap iteration) but only extracts
    reads in its own bucket, via ``extract_read_relative_base_identities``'s
    existing ``read_name_filter`` parameter -- trading N-way redundant (but
    cheap) iteration for exact balance, rather than N-way redundant (and
    expensive) per-base extraction.
    """
    if n_buckets <= 1:
        return [set(read_ids)] if read_ids else []
    buckets: list[set[str]] = [set() for _ in range(n_buckets)]
    for index, read_id in enumerate(read_ids):
        buckets[index % n_buckets].add(read_id)
    return [bucket for bucket in buckets if bucket]


def _n_buckets_for_reference(
    n_reads: int,
    max_workers: int,
    *,
    min_reads_per_bucket: int = 500,
    max_reads_per_bucket: int = 4000,
) -> int:
    """How many buckets to split one reference's reads into for parallel
    extraction.

    Two, separate constraints combine here, and conflating them previously
    caused large experiments to blow their per-worker memory budget:

    - Parallelism: parallelizing per-chromosome alone caps concurrency at the
      reference count and load-balances poorly when read depth is uneven
      across references (an amplicon panel with one 40x-oversequenced short
      locus and several lightly-sequenced ones, say) -- splitting each
      reference's reads into several buckets instead lets the pool's
      work-stealing scheduler balance uneven per-bucket cost dynamically.
      Bucket count is floored so buckets don't get so small that per-bucket
      overhead (a fetch call, a worker round-trip) would dominate the actual
      extraction work -- this alone would cap bucket count at ``max_workers``.
    - Memory: each bucket's ragged per-read data is held in memory by
      whichever worker processes it (see ``_map_references_parallel``, whose
      own concurrency is bounded separately by
      ``memory_guard.resolve_max_workers``). Capping bucket *count* at
      ``max_workers`` regardless of ``n_reads`` left bucket *size* --
      and so per-worker memory -- scaling linearly with reference read
      count: fine for a small experiment, but a large one could put tens of
      thousands of reads in a single bucket and exceed the per-worker
      memory budget entirely (see dev/pipeline_scaling_audit.md). Bucket
      count must grow with ``n_reads`` past ``max_workers`` once
      ``n_reads / max_workers`` would exceed ``max_reads_per_bucket``, so
      bucket size -- and therefore memory -- stays experiment-size-independent.
    """
    if n_reads <= 0:
        return 1
    if max_workers <= 1:
        by_memory = -(-n_reads // max_reads_per_bucket)  # ceil division
        return max(1, by_memory)
    by_parallelism = min(max_workers, max(1, n_reads // min_reads_per_bucket))
    by_memory = -(-n_reads // max_reads_per_bucket)  # ceil division
    return max(1, by_parallelism, by_memory)


def _split_modkit_tsv_by_bucket(
    tsv_paths: list[Path],
    read_id_to_bucket_id: dict[str, int],
    output_dir: Path,
    *,
    chunksize: int = 2_000_000,
) -> dict[int, Path]:
    """Stream-split a (possibly huge) modkit-extract TSV into one small file
    per read-id bucket, so parallel workers each hold only their own bucket's
    rows instead of one process joining the whole file serially.

    Reads ``tsv_paths`` via ``pandas.read_csv``'s ``chunksize`` (bounded
    memory regardless of total TSV size -- the same 75M-row TSV that needs
    ~40GB loaded whole streams through in fixed-size pieces here), routing
    each chunk's rows to their bucket's output file by ``read_id`` using the
    same read-id -> bucket assignment already computed for the pysam
    backend's per-reference parallel dispatch (``_bucket_read_ids``), so
    both backends parallelize identically from the caller's point of view.
    Rows whose read_id has no bucket assignment (not a wanted primary read)
    are dropped -- the same effective filter as ``_attach_direct_signals``'s
    ``calls.loc[...isin(frame.index)]``, just applied before the split
    instead of after a whole-file load.

    Bucket ids (not reference names) name the output files -- reference
    names can contain characters unsafe for filenames (this codebase's own
    FASTA records use bare colons, e.g. ``"chr1:1000-3000"``) and the caller
    already has a bucket id for every item it dispatches, so there is no
    reason to round-trip through a sanitized name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    output_paths: dict[int, Path] = {}
    try:
        for tsv_path in tsv_paths:
            for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=chunksize):
                bucket_ids = chunk[MODKIT_EXTRACT_TSV_COLUMN_READ_ID].map(read_id_to_bucket_id)
                wanted = chunk.loc[bucket_ids.notna()].copy()
                if wanted.empty:
                    continue
                wanted["_bucket_id"] = bucket_ids.loc[wanted.index].astype(int)
                for bucket_id, sub in wanted.groupby("_bucket_id", sort=False):
                    sub = sub.drop(columns="_bucket_id")
                    handle = handles.get(bucket_id)
                    if handle is None:
                        path = output_dir / f"bucket_{bucket_id:06d}.tsv"
                        output_paths[bucket_id] = path
                        handle = open(path, "w")
                        handles[bucket_id] = handle
                    sub.to_csv(handle, sep="\t", index=False, header=handle.tell() == 0)
    finally:
        for handle in handles.values():
            handle.close()
    return output_paths


def _extract_convertible_reference(
    record: str,
    sequence: str,
    read_name_filter: set[str] | None,
    metrics: dict,
    info,
    deaminase: bool,
    *,
    cfg,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None,
    umi_sidecar: str | Path | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Extract+attach one reference bucket's frame for conversion/deaminase.

    Module-level (not a closure) so it can run in a worker process via
    ``_map_references_parallel`` -- see ``_build_ragged_records_streaming_
    convertible``. ``read_name_filter`` may be a read-id bucket (parallelizing
    a single large/deep reference across several workers, see
    ``_bucket_read_ids``) or ``None`` (the whole reference, one bucket).
    ``metrics`` must already be sliced down to just this bucket's read_ids by
    the caller -- passing the whole-experiment metrics dict (tens of MB) to
    every one of dozens of worker tasks was itself the dominant cost of
    parallelizing (measured: 23.8MB x 26 tasks ~= 620MB of redundant pickled
    IPC transfer on a real 220K-read run, dwarfing the actual per-bucket
    extraction work each task does).
    Returns ``(frame_or_None, signal_columns)`` for a uniform contract with
    ``_extract_direct_reference``; conversion/deaminase never produces
    per-base/strand channel columns, so ``signal_columns`` is always empty
    here. The caller is responsible for combining a reference's bucket
    results and splitting by ``Reference_strand`` before writing -- not done
    here, since a bucket is only part of a reference's data.
    """
    from ..informatics.bam_functions import extract_read_relative_base_identities

    extracted = extract_read_relative_base_identities(
        aligned_bam,
        record,
        sequence,
        samtools_backend=cfg.samtools_backend,
        primary_only=True,
        read_name_filter=read_name_filter,
    )
    if not extracted:
        return None, []
    rows: list[dict[str, object]] = []
    for row in extracted:
        strand = info.strand
        if deaminase and row["Read_mismatch_trend"] == "G->A":
            strand = "bottom"
        row["reference"] = info.chromosome
        row["strand"] = strand
        row["dataset"] = info.conversion
        row["Reference_strand"] = f"{info.chromosome}_{strand}"
        row["modification_signal"] = _conversion_signal(row, deaminase=deaminase)
        rows.append(row)
    frame = _attach_obs_metadata(
        pd.DataFrame(rows),
        cfg=cfg,
        bam_path=aligned_bam,
        barcode_sidecar=barcode_sidecar,
        umi_sidecar=umi_sidecar,
        metrics=metrics,
    )
    frame = _attach_pod5_metadata(frame, cfg=cfg)
    frame = _attach_signal_features(frame, cfg=cfg, aligned_bam=aligned_bam)
    return frame, []


def _extract_direct_reference(
    record: str,
    sequence: str,
    read_name_filter: set[str] | None,
    metrics: dict,
    *,
    cfg,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None,
    umi_sidecar: str | Path | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Extract+attach one reference bucket's frame for direct modality (pysam
    backend). Module-level (not a closure) so it can run in a worker process
    via ``_map_references_parallel`` -- see ``_build_ragged_records_streaming_
    direct``. See ``_extract_convertible_reference`` for the bucket/combine
    contract this mirrors, including why ``metrics`` must already be sliced
    to this bucket's read_ids by the caller.
    """
    from ..informatics.bam_functions import extract_read_relative_base_identities

    extracted = extract_read_relative_base_identities(
        aligned_bam,
        record,
        sequence,
        samtools_backend=cfg.samtools_backend,
        primary_only=True,
        read_name_filter=read_name_filter,
    )
    if not extracted:
        return None, []
    frame = _attach_obs_metadata(
        pd.DataFrame(extracted),
        cfg=cfg,
        bam_path=aligned_bam,
        barcode_sidecar=barcode_sidecar,
        umi_sidecar=umi_sidecar,
        metrics=metrics,
    )
    frame = _attach_pod5_metadata(frame, cfg=cfg)
    frame = _attach_signal_features(frame, cfg=cfg, aligned_bam=aligned_bam)
    # frame's read_id set already equals read_name_filter (extraction above
    # applied it), so _attach_direct_signals_from_bam's own wanted-read-id
    # filtering is exact regardless of position -- no window scoping needed.
    frame, found_columns = _attach_direct_signals_from_bam(
        frame,
        aligned_bam,
        impute_uncalled_canonical=bool(
            getattr(cfg, "direct_signal_impute_uncalled_canonical", False)
        ),
    )
    return frame, found_columns


def _extract_direct_reference_modkit(
    record: str,
    sequence: str,
    read_name_filter: set[str] | None,
    metrics: dict,
    split_tsv_path: Path | None,
    *,
    cfg,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None,
    umi_sidecar: str | Path | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Extract+attach one reference bucket's frame for direct modality (modkit
    backend). Mirrors ``_extract_direct_reference`` exactly except for the
    modification-signal step: joins against this bucket's own small
    pre-split modkit-extract TSV chunk (``_split_modkit_tsv_by_bucket``)
    instead of decoding MM/ML tags from the BAM directly, so the join logic
    itself (``_attach_direct_signals``) is unchanged -- only how much of the
    whole-experiment TSV any one worker has to hold in memory.
    """
    from ..informatics.bam_functions import extract_read_relative_base_identities

    extracted = extract_read_relative_base_identities(
        aligned_bam,
        record,
        sequence,
        samtools_backend=cfg.samtools_backend,
        primary_only=True,
        read_name_filter=read_name_filter,
    )
    if not extracted:
        return None, []
    frame = _attach_obs_metadata(
        pd.DataFrame(extracted),
        cfg=cfg,
        bam_path=aligned_bam,
        barcode_sidecar=barcode_sidecar,
        umi_sidecar=umi_sidecar,
        metrics=metrics,
    )
    frame = _attach_pod5_metadata(frame, cfg=cfg)
    frame = _attach_signal_features(frame, cfg=cfg, aligned_bam=aligned_bam)
    if split_tsv_path is None or not split_tsv_path.exists():
        return frame, []
    frame, found_columns = _attach_direct_signals(frame, tsv_paths=[split_tsv_path])
    return frame, found_columns


def _build_ragged_records_streaming_convertible(
    cfg,
    *,
    fasta: Path,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None = None,
    umi_sidecar: str | Path | None = None,
) -> tuple[object, dict[str, int], dict[str, object]]:
    """Streaming variant of ``build_ragged_records`` for ``conversion``/``deaminase``.

    Yields one reference's fully extracted+attached frame at a time (via the
    returned generator) instead of accumulating every reference's rows into
    one experiment-wide frame before returning -- pairs with
    ``informatics.raw_store.write_raw_store_streaming``, which never holds
    more than one reference's ragged array data in memory either.

    ``reference_lengths``/the ``extra_uns`` metadata dict are fully computed
    upfront from the FASTA alone (``process_conversion_sites`` needs no BAM
    data) rather than incrementally as rows are seen -- this resolves what
    would otherwise be a circular dependency: ``write_raw_store_streaming``
    needs ``reference_lengths`` before it can process even the first
    reference, but ``build_ragged_records``'s original incremental
    population only completes after every read across the whole experiment
    has been seen.
    """
    from ..informatics.bam_functions import extract_read_features_from_bam
    from ..informatics.converted_BAM_to_adata import process_conversion_sites
    from ..informatics.fasta_functions import get_native_references
    from ..informatics.reference_identity import reference_uid as _reference_uid

    modality = str(cfg.smf_modality)
    deaminase = modality == "deaminase"

    reference_map = get_native_references(fasta)
    _, record_info, chromosome_sequences = process_conversion_sites(
        fasta, cfg.conversion_types, deaminase
    )

    reference_lengths: dict[str, int] = {}
    for record in reference_map:
        info = record_info[record]
        if deaminase:
            # A deaminase read's *own* mismatch trend (not the reference's
            # canonical strand) decides whether it's named "_top" or
            # "_bottom" (see the per-row override below) -- so both namings
            # need a length entry for every chromosome, not just info.strand's
            # single canonical value.
            reference_lengths[f"{info.chromosome}_top"] = info.sequence_length
            reference_lengths[f"{info.chromosome}_bottom"] = info.sequence_length
        else:
            reference_lengths[f"{info.chromosome}_{info.strand}"] = info.sequence_length

    references = {
        f"{reference}_FASTA_sequence": sequence
        for reference, (sequence, _complement) in chromosome_sequences.items()
    }
    reference_uids: dict[str, str] = {}
    for reference_strand, length in reference_lengths.items():
        chromosome = str(reference_strand).rsplit("_", 1)[0]
        seq_pair = chromosome_sequences.get(chromosome)
        if seq_pair is not None:
            reference_uids[str(reference_strand)] = _reference_uid(seq_pair[0], length)

    extra_uns = {
        "References": references,
        "reference_uids": reference_uids,
        "signal_columns": [],
        "modality": modality,
        "sequence_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
        "mismatch_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
        "sequence_integer_decoding_map": {
            str(key): value for key, value in MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE.items()
        },
    }

    max_workers = max(1, int(getattr(cfg, "threads", 1) or 1))

    def _reference_frames():
        # Computed once, up front -- extract_read_features_from_bam scans the
        # whole BAM regardless of which reads it's asked about, so calling it
        # again inside the per-reference loop below would re-scan the whole
        # BAM once per reference instead of once total.
        metrics = extract_read_features_from_bam(
            aligned_bam, samtools_backend=cfg.samtools_backend, primary_only=True
        )
        # Split per reference into several read-count-balanced buckets, not
        # just one item per reference -- parallelizing per-reference alone
        # caps concurrency at the reference count and load-balances poorly
        # when read depth is uneven across references (see
        # _n_buckets_for_reference/_bucket_read_ids).
        #
        # IMPORTANT: reference_map has one entry per alignment target, not one
        # per chromosome -- conversion modality aligns against multiple
        # conversion-state variants of each chromosome+strand (e.g.
        # "6B6_unconverted_top" and "6B6_5mC_top" both belong to chromosome
        # "6B6"). _ChromosomeGroupAccumulator waits for every record sharing a
        # chromosome to complete before combining+splitting by
        # Reference_strand -- see its docstring for why yielding per-record
        # instead silently loses data.
        buckets_remaining: dict[str, int] = {}
        record_chromosome: dict[str, str] = {}
        items = []
        for record, (_length, sequence) in reference_map.items():
            info = record_info[record]
            record_chromosome[record] = info.chromosome
            read_ids = _read_ids_for_reference(aligned_bam, record)
            n_buckets = _n_buckets_for_reference(
                len(read_ids),
                max_workers,
                max_reads_per_bucket=int(getattr(cfg, "raw_bucket_max_reads", 4000)),
            )
            buckets = _bucket_read_ids(read_ids, n_buckets)
            buckets_remaining[record] = len(buckets)
            for bucket in buckets:
                # Sliced to this bucket's own read_ids -- passing the whole
                # experiment's metrics dict to every bucket task is itself the
                # dominant IPC cost at scale (see _extract_convertible_reference).
                metrics_slice = {rid: metrics[rid] for rid in bucket if rid in metrics}
                items.append((record, sequence, bucket, metrics_slice, info, deaminase))
        worker_kwargs = dict(
            cfg=cfg,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
        )
        flush_threshold = int(getattr(cfg, "raw_shard_flush_max_reads", 20_000))
        accumulator = _ChromosomeGroupAccumulator(
            record_chromosome, flush_threshold=flush_threshold
        )
        strands_seen: dict[str, set[str]] = {}
        any_rows = False
        from ..memory_guard import resolve_max_workers

        # Records with zero buckets never dispatch a task, so they'd never
        # reach the completion loop below -- mark them done up front so their
        # chromosome siblings aren't blocked waiting on them forever.
        for record, remaining in buckets_remaining.items():
            if remaining == 0:
                yield from _yield_flush_result(
                    accumulator.complete(record), record_chromosome[record], strands_seen
                )

        for args, (bucket_frame, _found_columns) in _map_references_parallel(
            items,
            _extract_convertible_reference,
            max_workers=resolve_max_workers(cfg, len(items)),
            worker_kwargs=worker_kwargs,
            pool_label=f"raw extraction (convertible, {len(items)} buckets)",
        ):
            record = args[0]
            chromosome = record_chromosome[record]
            frames_for_bucket = (
                [bucket_frame] if bucket_frame is not None and not bucket_frame.empty else []
            )
            for strand, frame, is_final in _yield_flush_result(
                accumulator.add_partial(record, frames_for_bucket), chromosome, strands_seen
            ):
                any_rows = any_rows or frame is not None
                yield strand, frame, is_final
            buckets_remaining[record] -= 1
            if buckets_remaining[record] != 0:
                continue
            for strand, frame, is_final in _yield_flush_result(
                accumulator.complete(record), chromosome, strands_seen
            ):
                any_rows = any_rows or frame is not None
                yield strand, frame, is_final
        if not any_rows:
            raise RuntimeError(f"no primary mapped reads were extracted from {aligned_bam}")

    return _reference_frames(), reference_lengths, extra_uns


def _build_ragged_records_streaming_direct(
    cfg,
    *,
    fasta: Path,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None = None,
    umi_sidecar: str | Path | None = None,
    mod_tsv_paths: list[Path] | None = None,
) -> tuple[object, dict[str, int], dict[str, object]]:
    """Streaming variant of ``build_ragged_records`` for ``direct`` modality.

    Supports both ``direct_signal_backend`` values. ``pysam`` (the default)
    decodes each read's own MM/ML tags directly (``_attach_direct_signals_
    from_bam``), needing nothing beyond the same aligned BAM already open for
    extraction, so it streams per reference exactly like conversion/deaminase.

    ``modkit`` needs ``mod_tsv_paths`` -- the modkit-extract TSV(s), already
    produced by the caller before this function runs. One flat TSV covers the
    whole experiment (modkit has no per-reference/per-chunk output mode), so
    it can't be streamed per reference the way the BAM itself can; instead
    it's streamed once via ``pandas.read_csv(chunksize=...)`` and split into
    the same per-reference read-id buckets used for pysam-backend
    parallelism (``_split_modkit_tsv_by_bucket``), bounding both per-worker
    memory and giving the modkit backend the same real multi-core
    parallelism the pysam backend already has, instead of one process
    joining the whole file serially (see dev/pipeline_scaling_audit.md's
    Track B notes for why the old whole-frame-only path existed).
    """
    from ..informatics.bam_functions import extract_read_features_from_bam
    from ..informatics.fasta_functions import get_native_references
    from ..informatics.reference_identity import reference_uid as _reference_uid

    backend = str(getattr(cfg, "direct_signal_backend", "modkit"))
    if backend == "modkit" and not mod_tsv_paths:
        raise ValueError("direct_signal_backend='modkit' requires mod_tsv_paths")

    reference_map = get_native_references(fasta)
    chromosome_sequences = {
        reference: (sequence, sequence) for reference, (_length, sequence) in reference_map.items()
    }

    reference_lengths: dict[str, int] = {}
    for record, (record_length, _sequence) in reference_map.items():
        # direct modality's strand ("top"/"bottom") is decided per-read from
        # alignment orientation, not a per-chromosome constant, so both
        # namings need a length entry for every chromosome (mirrors deaminase
        # above, for the same reason).
        reference_lengths[f"{record}_top"] = record_length
        reference_lengths[f"{record}_bottom"] = record_length

    references = {
        f"{reference}_FASTA_sequence": sequence
        for reference, (sequence, _complement) in chromosome_sequences.items()
    }
    reference_uids: dict[str, str] = {}
    for reference_strand, length in reference_lengths.items():
        chromosome = str(reference_strand).rsplit("_", 1)[0]
        seq_pair = chromosome_sequences.get(chromosome)
        if seq_pair is not None:
            reference_uids[str(reference_strand)] = _reference_uid(seq_pair[0], length)

    # Mutated in place by _reference_frames() as new channels are discovered;
    # write_raw_store_streaming only reads extra_uns after fully consuming the
    # generator, so the final set is complete by the time it's written to
    # spine.uns (dict(extra_uns) copies the outer dict, not this inner list).
    signal_columns: list[str] = []
    extra_uns = {
        "References": references,
        "reference_uids": reference_uids,
        "signal_columns": signal_columns,
        "modality": "direct",
        "sequence_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
        "mismatch_integer_encoding_map": dict(MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT),
        "sequence_integer_decoding_map": {
            str(key): value for key, value in MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE.items()
        },
    }

    max_workers = max(1, int(getattr(cfg, "threads", 1) or 1))

    def _reference_frames():
        metrics = extract_read_features_from_bam(
            aligned_bam, samtools_backend=cfg.samtools_backend, primary_only=True
        )
        buckets_remaining: dict[str, int] = {}
        items = []
        read_id_to_bucket_id: dict[str, int] = {}
        bucket_id_counter = 0
        for record, (_record_length, sequence) in reference_map.items():
            read_ids = _read_ids_for_reference(aligned_bam, record)
            n_buckets = _n_buckets_for_reference(
                len(read_ids),
                max_workers,
                max_reads_per_bucket=int(getattr(cfg, "raw_bucket_max_reads", 4000)),
            )
            buckets = _bucket_read_ids(read_ids, n_buckets)
            buckets_remaining[record] = len(buckets)
            for bucket in buckets:
                metrics_slice = {rid: metrics[rid] for rid in bucket if rid in metrics}
                if backend == "modkit":
                    for read_id in bucket:
                        read_id_to_bucket_id[read_id] = bucket_id_counter
                    items.append((record, sequence, bucket, metrics_slice, bucket_id_counter))
                    bucket_id_counter += 1
                else:
                    items.append((record, sequence, bucket, metrics_slice))

        split_dir: Path | None = None
        if backend == "modkit":
            split_dir = Path(cfg.modkit_outputs_path) / "tsv_split_buckets"
            split_paths = _split_modkit_tsv_by_bucket(
                list(mod_tsv_paths), read_id_to_bucket_id, split_dir
            )
            items = [
                (record, sequence, bucket, metrics_slice, split_paths.get(bucket_id))
                for record, sequence, bucket, metrics_slice, bucket_id in items
            ]

        worker = (
            _extract_direct_reference_modkit if backend == "modkit" else _extract_direct_reference
        )
        worker_kwargs = dict(
            cfg=cfg,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
        )
        pending: dict[str, list[pd.DataFrame]] = {}
        pending_rows: dict[str, int] = {}
        strands_seen: dict[str, set[str]] = {}
        any_rows = False
        flush_threshold = int(getattr(cfg, "raw_shard_flush_max_reads", 20_000))
        from ..memory_guard import resolve_max_workers

        try:
            for args, (bucket_frame, found_columns) in _map_references_parallel(
                items,
                worker,
                max_workers=resolve_max_workers(cfg, len(items)),
                worker_kwargs=worker_kwargs,
                pool_label=f"raw extraction (direct/{backend}, {len(items)} buckets)",
            ):
                record = args[0]
                for column in found_columns:
                    if column not in signal_columns:
                        signal_columns.append(column)
                if bucket_frame is not None and not bucket_frame.empty:
                    pending.setdefault(record, []).append(bucket_frame)
                    pending_rows[record] = pending_rows.get(record, 0) + len(bucket_frame)
                buckets_remaining[record] -= 1
                is_final = buckets_remaining[record] == 0
                if is_final or pending_rows.get(record, 0) >= flush_threshold:
                    frames = pending.pop(record, [])
                    pending_rows[record] = 0
                    for strand, frame, strand_is_final in _yield_flush_result(
                        (frames, is_final), record, strands_seen
                    ):
                        any_rows = any_rows or frame is not None
                        yield strand, frame, strand_is_final
        finally:
            if split_dir is not None:
                import shutil

                shutil.rmtree(split_dir, ignore_errors=True)
        if not any_rows:
            raise RuntimeError(f"no primary mapped reads were extracted from {aligned_bam}")

    return _reference_frames(), reference_lengths, extra_uns


def build_ragged_records_streaming(
    cfg,
    *,
    fasta: Path,
    aligned_bam: Path,
    barcode_sidecar: str | Path | None = None,
    umi_sidecar: str | Path | None = None,
    mod_tsv_paths: list[Path] | None = None,
) -> tuple[object, dict[str, int], dict[str, object]]:
    """Streaming variant of ``build_ragged_records``.

    Dispatches by modality: ``conversion``/``deaminase`` always stream (their
    modification-signal source is derivable from the FASTA alone); ``direct``
    streams for either ``direct_signal_backend`` value -- ``modkit`` needs
    ``mod_tsv_paths`` (the caller must already have run ``modkit extract``),
    which ``_build_ragged_records_streaming_direct`` streams and splits into
    per-bucket chunks itself rather than joining the whole file in one
    process (see that function's docstring).

    The returned generator yields ``(reference_strand, frame_or_None,
    is_final)`` -- a low-read-depth reference may still appear as a single
    item, but a high-read-depth one is flushed across several bounded-size
    items (each capped around ``cfg.raw_shard_flush_max_reads``) instead of
    accumulating that reference's whole ragged frame in memory first, which
    otherwise scales with experiment size rather than staying bounded (see
    dev/pipeline_scaling_audit.md). ``write_raw_store_streaming`` is the
    intended consumer of this exact shape.
    """
    modality = str(cfg.smf_modality)
    if modality == "direct":
        backend = str(getattr(cfg, "direct_signal_backend", "modkit"))
        if backend == "modkit" and not mod_tsv_paths:
            raise ValueError(
                "build_ragged_records_streaming with direct_signal_backend='modkit' "
                "requires mod_tsv_paths (run modkit extract first)"
            )
        return _build_ragged_records_streaming_direct(
            cfg,
            fasta=fasta,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
            mod_tsv_paths=mod_tsv_paths,
        )
    if modality in {"conversion", "deaminase"}:
        return _build_ragged_records_streaming_convertible(
            cfg,
            fasta=fasta,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
        )
    raise ValueError(f"build_ragged_records_streaming does not support modality {modality!r}")


def build_partitioned_ragged_records_streaming(
    cfg,
    *,
    fasta: Path,
    partitions: list[tuple[Path, str, str | Path | None]],
    umi_sidecar: str | Path | None = None,
) -> tuple[object, dict[str, int], dict[str, object]]:
    """Stream canonical alignment partitions without concatenating their BAMs.

    Args:
        cfg: Resolved experiment configuration.
        fasta: Exact prepared alignment FASTA.
        partitions: Canonically ordered ``(BAM, namespace, barcode-sidecar)`` tuples.
        umi_sidecar: Optional read-level UMI sidecar shared by all partitions.

    Returns:
        The same generator, reference-length, and metadata contract as
        :func:`build_ragged_records_streaming`.
    """
    if not partitions:
        raise ValueError("partitioned raw extraction requires at least one alignment source")
    prepared = []
    expected_lengths: dict[str, int] | None = None
    common_uns: dict[str, object] | None = None
    for aligned_bam, namespace, barcode_sidecar in partitions:
        frames, lengths, extra_uns = build_ragged_records_streaming(
            cfg,
            fasta=fasta,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
        )
        if expected_lengths is None:
            expected_lengths = lengths
            common_uns = extra_uns
        elif lengths != expected_lengths:
            raise ValueError("alignment partitions resolved incompatible reference lengths")
        prepared.append((frames, str(namespace or "")))

    def _frames():
        for frames, namespace in prepared:
            for reference, frame, is_final in frames:
                if frame is not None and namespace:
                    frame = frame.copy()
                    frame["source_read_id"] = frame["read_id"].astype(str)
                    frame["read_id"] = frame["read_id"].map(
                        lambda value: namespaced_source_id(namespace, value)
                    )
                    if "template_id" in frame:
                        frame["template_id"] = frame["template_id"].map(
                            lambda value: namespaced_source_id(namespace, value)
                        )
                    frame["namespace"] = namespace
                yield reference, frame, is_final

    assert expected_lengths is not None and common_uns is not None
    return _frames(), expected_lengths, common_uns


@stage_logging_lifecycle
def raw_adata(config_path: str):
    """Run BAM preparation through section 6 and emit ragged raw artifacts."""
    from ..logging_utils import setup_stage_logging
    from ..perf_log import perf_substep
    from ..readwrite import safe_read_h5ad
    from .helpers import (
        get_adata_paths,
        load_experiment_config,
        partitioned_stage_is_complete,
        publish_stage_outputs,
        raw_input_artifact_ids,
        stage_config_hash,
        stage_lifecycle,
    )
    from .load_adata import load_adata_core

    cfg = load_experiment_config(config_path)
    raw_root = Path(cfg.output_directory) / RAW_DIR
    setup_stage_logging(cfg, raw_root)
    cfg.informatics_outputs_path = raw_root
    cfg.bam_outputs_path = raw_root / BAM_OUTPUTS_DIR
    cfg.fasta_outputs_path = raw_root / FASTA_OUTPUTS_DIR
    cfg.bed_outputs_path = raw_root / BED_OUTPUTS_DIR
    cfg.modkit_outputs_path = raw_root / MODKIT_OUTPUTS_DIR
    cfg.split_path = cfg.bam_outputs_path / SPLIT_DIR
    # Raw is the one stage authorized to replace an invalid selector. Every
    # downstream caller uses the strict default and refuses canonical fallback
    # when a current generation is corrupt.
    paths = get_adata_paths(cfg, allow_invalid_raw=True)
    required = PARTITIONED_STAGE_REQUIRED_ARTIFACTS["raw"]
    from ..informatics.experiment_manifest import restore_previous_complete_state
    from ..informatics.input_manifest import (
        input_manifest_artifact_paths,
        resolve_input_manifest_readonly,
    )
    from ..informatics.raw_append import (
        RawAppendAssembly,
        assemble_raw_append,
        discard_raw_append_assembly,
        plan_raw_append,
    )
    from ..informatics.raw_generation import (
        RawGenerationError,
        publish_raw_generation,
        raw_generation_dependencies,
        resolve_current_raw_generation,
    )
    from ..informatics.raw_store import (
        BARCODE_INDEX_FILENAME,
        INTERVAL_CATALOG_FILENAME,
        MOLECULE_INDEX_DIRNAME,
        MOLECULES_FILENAME,
        SEGMENT_INDEX_DIRNAME,
        SEGMENTS_FILENAME,
    )
    from ..informatics.sidecar_manifest import sidecar_manifest_path

    append_assembly: RawAppendAssembly | None = None
    append_plan = None
    raw_config_hash = stage_config_hash(cfg, "raw")
    requested_raw_input_ids = raw_input_artifact_ids(cfg)

    def publication_inputs():
        canonical_raw_root = Path(cfg.output_directory) / RAW_DIR
        sources = {
            "spine": canonical_raw_root / "spine.h5ad",
            "ragged_store": canonical_raw_root / "raw",
            "interval_catalog": canonical_raw_root / INTERVAL_CATALOG_FILENAME,
            "obs": canonical_raw_root / "obs.parquet",
            "molecules": Path(cfg.output_directory) / MOLECULES_FILENAME,
            "molecule_index": Path(cfg.output_directory) / MOLECULE_INDEX_DIRNAME,
            "segments": Path(cfg.output_directory) / SEGMENTS_FILENAME,
            "segment_index": Path(cfg.output_directory) / SEGMENT_INDEX_DIRNAME,
            "reference_interval_map": (
                Path(cfg.output_directory) / REFERENCE_INTERVAL_MAP_FILENAME
            ),
            "sidecar_manifest": sidecar_manifest_path(canonical_raw_root),
            **input_manifest_artifact_paths(cfg.output_directory),
        }
        if append_assembly is not None:
            sources.update(append_assembly.sources)
        barcode_index = canonical_raw_root / BARCODE_INDEX_FILENAME
        if "barcode_index" not in sources and barcode_index.exists():
            sources["barcode_index"] = barcode_index
        regions = {
            scope: Path(cfg.output_directory) / REGION_CATALOG_DIRNAME / filename
            for scope, filename in REGION_CATALOG_FILENAMES.items()
            if (Path(cfg.output_directory) / REGION_CATALOG_DIRNAME / filename).exists()
        }
        dependencies = raw_generation_dependencies(
            sources["spine"],
            sources["sidecar_manifest"],
            run_root=cfg.output_directory,
            owned_artifacts={**sources, **regions},
        )
        return sources, dependencies, regions

    def publish_generation():
        sources, dependencies, regions = publication_inputs()
        return publish_raw_generation(
            cfg.output_directory,
            sources,
            config_hash=raw_config_hash,
            input_artifact_ids=requested_raw_input_ids,
            dependencies=dependencies,
            region_artifacts=regions,
            reuse_generation=(
                current_generation[0] if append_plan is not None and append_plan.eligible else None
            ),
            source_transition=(
                append_plan.transition.to_dict()
                if append_plan is not None and append_plan.eligible
                else None
            ),
        )

    def lifecycle_outputs(generation):
        outputs = {
            key: Path(path)
            for key, path in generation.items()
            if key not in {"generation_id", "sidecar_manifest"} and isinstance(path, Path)
        }
        outputs["manifest"] = Path(generation["sidecar_manifest"])
        outputs["generation_spine"] = Path(generation["spine"])
        for scope in REGION_CATALOG_FILENAMES:
            generation_key = f"region:{scope}"
            if generation_key in generation:
                outputs[f"{scope}_regions"] = Path(generation[generation_key])
        return outputs

    try:
        current_generation = resolve_current_raw_generation(raw_root)
    except RawGenerationError as exc:
        current_generation = None
        logger.warning("Raw current generation is invalid; rebuilding: %s", exc)
    if current_generation is not None and not cfg.force_redo_load_adata:
        generation_id = str(current_generation[1]["generation_id"])
        current_spine = current_generation[0] / "spine.h5ad"
        if partitioned_stage_is_complete(
            cfg,
            "raw",
            required=required,
            extra_matches={"generation_id": generation_id},
            allow_previous_complete=True,
        ):
            spine, _ = safe_read_h5ad(current_spine)
            logger.info("Raw generation is already complete: %s", current_generation[0])
            # The skip may have been authorized by a retained complete record
            # after an attempt that was killed before it could write a terminal
            # state. The artifacts it describes were just validated, so make it
            # the live record rather than leaving the stage reported as running.
            if restore_previous_complete_state(cfg.output_directory, "raw"):
                logger.info("Restored the retained complete raw record after an abandoned attempt")
            mark_stage_outcome("skipped", reason="compatible raw generation is already complete")
            return spine, current_spine, cfg
        requested_manifest = resolve_input_manifest_readonly(
            input_manifest_path=cfg.input_manifest_path,
            input_paths=None if cfg.input_manifest_path else cfg.input_files,
            alignment_mode=cfg.alignment_mode,
            modality=cfg.smf_modality,
            barcode_map=cfg.fastq_barcode_map,
            auto_pair=cfg.fastq_auto_pairing,
        )
        append_plan = plan_raw_append(
            current_generation[0],
            requested_manifest,
            run_root=cfg.output_directory,
            config_hash=raw_config_hash,
            input_artifact_ids=requested_raw_input_ids,
        )
        if append_plan.eligible:
            cfg._raw_append_source_ids = append_plan.transition.added_source_ids
            logger.info(
                "Raw source transition is append-only: reusing %d and processing %d source(s)",
                len(append_plan.transition.reused_source_ids),
                len(append_plan.transition.added_source_ids),
            )
        else:
            logger.warning(
                "Raw generation requires a full recompute (%s): %s",
                append_plan.reason,
                current_generation[0],
            )

    legacy_required = tuple(
        key
        for key in required
        if key not in {"generation_spine", "generation_manifest", "generation", "current"}
    )
    canonical_spine = raw_root / "spine.h5ad"
    migrate_legacy = (
        current_generation is None
        and canonical_spine.exists()
        and not cfg.force_redo_load_adata
        and partitioned_stage_is_complete(cfg, "raw", required=legacy_required)
    )

    with stage_lifecycle(
        cfg,
        "raw",
        input_artifact_ids=requested_raw_input_ids,
    ) as lifecycle:
        if migrate_legacy:
            logger.info("Migrating compatible legacy raw artifacts into an immutable generation")
        else:
            with perf_substep("raw_pipeline"):
                result = load_adata_core(cfg, paths, config_path=config_path, raw_only=True)
        if append_plan is not None and append_plan.eligible:
            append_assembly = assemble_raw_append(
                cfg.output_directory,
                current_generation[0],
                transition=append_plan.transition.to_dict(),
                analysis_mode=getattr(cfg, "analysis_mode", "auto"),
                load_cache_mode=getattr(cfg, "load_cache_mode", "auto"),
                max_full_matrix_gb=float(getattr(cfg, "max_full_matrix_gb", 8.0)),
                genome_tile_size=int(getattr(cfg, "genome_tile_size", 10_000)),
                genome_tile_halo=int(getattr(cfg, "genome_tile_halo", 1_000)),
            )
        try:
            generation = publish_generation()
        finally:
            discard_raw_append_assembly(append_assembly)
        outputs = lifecycle_outputs(generation)
        generation_spine = Path(generation["spine"])
        spine, _ = safe_read_h5ad(generation_spine)
        publish_stage_outputs(
            lifecycle,
            outputs,
            required=required,
            task_catalog_key=None,
            checksum_keys=(
                "interval_catalog",
                "reference_interval_map",
                "alignment_regions",
                "analysis_regions",
                "plot_regions",
                "manifest",
                "input_manifest_csv",
                "input_manifest_json",
                "input_resolution_report",
                "generation_manifest",
                "current",
            ),
            schema_versions={
                "raw": 3,
                "raw_generation": 1,
                "identity": 1,
                "region_catalog": 1,
                "reference_interval_map": 1,
                "input_manifest": 1,
            },
            extra={
                "n_molecules": int(spine.n_obs),
                "generation_id": str(generation["generation_id"]),
                "source_transition": (
                    append_plan.transition.to_dict()
                    if append_plan is not None and append_plan.eligible
                    else None
                ),
                "append_reused_molecules": (
                    append_assembly.reused_molecules if append_assembly is not None else 0
                ),
                "append_added_molecules": (
                    append_assembly.added_molecules if append_assembly is not None else 0
                ),
            },
            nonempty_directory_keys=PARTITIONED_STAGE_NONEMPTY_DIRECTORIES["raw"],
        )
        from ..informatics.experiment_spine import write_experiment_spine

        write_experiment_spine(cfg.output_directory)
    return spine, generation_spine, cfg
