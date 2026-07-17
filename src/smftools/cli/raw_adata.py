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
    from ..informatics.ragged_store import cigar_max_indel_runs

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
    ``cfg.direct_signal_backend == "pysam"`` (the default).

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
                read_id = read.query_name
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
                        _call_code, probability = _resolve_direct_call(code_bytes)
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
    """Link each read to its origin POD5 and attach scalar sequencing/signal metadata.

    Only runs for POD5 inputs. Scalar ``pod5_*`` columns are carried onto the
    molecule spine by ``raw_store``; the optional full current trace
    (``pod5_current_pa``) stays in the parquet shard.
    """
    if str(getattr(cfg, "input_type", "")).lower() != "pod5":
        return frame
    if not getattr(cfg, "extract_pod5_metadata", True):
        return frame
    pod5_path = getattr(cfg, "input_data_path", None)
    if pod5_path is None or not Path(pod5_path).exists():
        logger.warning("input_type=pod5 but input_data_path is missing; skipping POD5 metadata")
        return frame

    from ..informatics.pod5_functions import extract_pod5_read_metadata

    metadata = extract_pod5_read_metadata(
        pod5_path,
        target_ids=frame["read_id"].astype(str),
        n_jobs=getattr(cfg, "threads", 1),
        include_current=bool(getattr(cfg, "raw_store_pod5_current", False)),
        verbose=False,
    )
    if metadata.empty:
        logger.warning("No POD5 metadata matched the extracted reads")
        return frame

    frame = frame.set_index("read_id", drop=False)
    for column in metadata.columns:
        frame[column] = metadata[column].reindex(frame.index)
    logger.info("Linked %d read(s) to origin POD5 with sequencing/signal metadata", len(metadata))
    return frame.reset_index(drop=True)


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
            read_id = str(read.query_name)
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

    attached = 0
    for read_id, signal in iter_pod5_signals(pod5_path, read_ids=list(move_tables)):
        if read_id not in move_tables or read_id not in frame.index:
            continue
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

    A group must not be finalized (yielded to the streaming shard writer)
    until every record sharing its chromosome has completed -- finalizing as
    soon as one record's own buckets finish would hand the writer multiple
    separate "complete" groups for the same ``Reference_strand``, and it
    always starts a fresh group at ``shard_index=0``, so a later record's
    group silently overwrites an earlier one's shard file on disk even
    though ``obs.parquet`` ends up with pointers for both (real data loss,
    confirmed on a real conversion-modality dataset before this fix: 3 of 4
    references affected, up to 23% of reads for one).
    """

    def __init__(self, record_chromosome: dict[str, str]):
        self._record_chromosome = dict(record_chromosome)
        self._remaining: dict[str, set[str]] = {}
        for record, chromosome in self._record_chromosome.items():
            self._remaining.setdefault(chromosome, set()).add(record)
        self._pending: dict[str, list[pd.DataFrame]] = {}

    def complete(self, record: str, frames: list[pd.DataFrame]) -> list[pd.DataFrame] | None:
        """Mark one record's fully-accumulated frames as done.

        ``frames`` may be empty (a record that dispatched no work, e.g. zero
        read_ids, or whose buckets all returned no rows) -- it must still be
        marked complete so its chromosome siblings aren't blocked forever.

        Returns the chromosome's combined frame list once every record
        sharing its chromosome has completed, else ``None``.
        """
        chromosome = self._record_chromosome[record]
        if frames:
            self._pending.setdefault(chromosome, []).extend(frames)
        remaining = self._remaining[chromosome]
        remaining.discard(record)
        if remaining:
            return None
        return self._pending.pop(chromosome, [])


def _map_references_parallel(items, worker, *, max_workers: int, worker_kwargs: dict):
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
    """
    if max_workers <= 1:
        for args in items:
            yield args, worker(*args, **worker_kwargs)
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..memory_guard import (
        _limit_blas_threads_in_worker,
        resolve_memory_budget_bytes,
        start_worker_watchdog,
    )

    cfg = worker_kwargs.get("cfg")
    per_worker_budget_bytes = (
        resolve_memory_budget_bytes(cfg) // max_workers if cfg is not None else 0
    )
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_limit_blas_threads_in_worker
    ) as pool:
        stop_watchdog = start_worker_watchdog(pool, per_worker_budget_bytes)
        try:
            future_to_args = {
                pool.submit(worker, *args, **worker_kwargs): args for args in items
            }
            for future in as_completed(future_to_args):
                yield future_to_args[future], future.result()
        finally:
            stop_watchdog()


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
            read_ids.append(read.query_name)
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


def _n_buckets_for_reference(n_reads: int, max_workers: int, *, min_reads_per_bucket: int = 500) -> int:
    """How many buckets to split one reference's reads into for parallel
    extraction.

    Parallelizing per-chromosome alone caps concurrency at the reference
    count and load-balances poorly when read depth is uneven across
    references (an amplicon panel with one 40x-oversequenced short locus and
    several lightly-sequenced ones, say) -- splitting each reference's reads
    into several buckets instead lets the pool's work-stealing scheduler
    balance uneven per-bucket cost dynamically, rather than being stuck
    waiting on one large indivisible per-reference unit. Capped so buckets
    don't get so small that per-bucket overhead (a fetch call, a worker
    round-trip) would dominate the actual extraction work.
    """
    if max_workers <= 1:
        return 1
    return max(1, min(max_workers, n_reads // min_reads_per_bucket))


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
            n_buckets = _n_buckets_for_reference(len(read_ids), max_workers)
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
        accumulator = _ChromosomeGroupAccumulator(record_chromosome)
        record_pending: dict[str, list[pd.DataFrame]] = {}
        any_rows = False
        from ..memory_guard import resolve_max_workers

        # Records with zero buckets never dispatch a task, so they'd never
        # reach the completion loop below -- mark them done up front so their
        # chromosome siblings aren't blocked waiting on them forever.
        for record, remaining in buckets_remaining.items():
            if remaining == 0:
                accumulator.complete(record, [])

        for args, (bucket_frame, _found_columns) in _map_references_parallel(
            items,
            _extract_convertible_reference,
            max_workers=resolve_max_workers(cfg, len(items)),
            worker_kwargs=worker_kwargs,
        ):
            record = args[0]
            if bucket_frame is not None and not bucket_frame.empty:
                record_pending.setdefault(record, []).append(bucket_frame)
            buckets_remaining[record] -= 1
            if buckets_remaining[record] != 0:
                continue
            frames = accumulator.complete(record, record_pending.pop(record, []))
            if frames is None:
                continue
            if frames:
                any_rows = True
                combined = (
                    frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
                )
                yield from _split_by_reference_strand(combined)
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

    backend = str(getattr(cfg, "direct_signal_backend", "pysam"))
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
            n_buckets = _n_buckets_for_reference(len(read_ids), max_workers)
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

        worker = _extract_direct_reference_modkit if backend == "modkit" else _extract_direct_reference
        worker_kwargs = dict(
            cfg=cfg,
            aligned_bam=aligned_bam,
            barcode_sidecar=barcode_sidecar,
            umi_sidecar=umi_sidecar,
        )
        pending: dict[str, list[pd.DataFrame]] = {}
        any_rows = False
        from ..memory_guard import resolve_max_workers

        try:
            for args, (bucket_frame, found_columns) in _map_references_parallel(
                items,
                worker,
                max_workers=resolve_max_workers(cfg, len(items)),
                worker_kwargs=worker_kwargs,
            ):
                record = args[0]
                for column in found_columns:
                    if column not in signal_columns:
                        signal_columns.append(column)
                if bucket_frame is not None and not bucket_frame.empty:
                    pending.setdefault(record, []).append(bucket_frame)
                buckets_remaining[record] -= 1
                if buckets_remaining[record] == 0:
                    frames = pending.pop(record, [])
                    if frames:
                        any_rows = True
                        combined = (
                            frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
                        )
                        yield from _split_by_reference_strand(combined)
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
    """
    modality = str(cfg.smf_modality)
    if modality == "direct":
        backend = str(getattr(cfg, "direct_signal_backend", "pysam"))
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
    raise ValueError(
        f"build_ragged_records_streaming does not support modality {modality!r}"
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
