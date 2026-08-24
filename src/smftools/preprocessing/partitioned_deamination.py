"""Per-molecule deamination evidence over raw ragged molecule shards.

Mirrors `partitioned_variant`: sparse per-position evidence, located segments,
and a per-molecule summary, written as a preprocess sidecar. See
`deamination_evidence` for the classification and segmentation rules.

Computed in preprocess rather than raw, deliberately. `ragged_store._read_record`
already walks the CIGAR against the reference and derives deamination votes, but
keeps only scalars. Capturing the per-position arrays there would bake a
threshold-free artifact into an immutable tier and force re-ingestion of every
existing raw generation -- for information that is fully reproducible here from
`SEQUENCE` + `CIGAR` + `REFERENCE_START`, exactly as
`partitioned_variant._observed_bases` reproduces observed bases. Keeping it in
preprocess also keeps the support threshold a config knob rather than a
property of stored data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from ..informatics.molecule_identity import EXPERIMENT_UID_COLUMN, MOLECULE_UID_COLUMN
from ..informatics.partition_read import load_spine
from ..informatics.physical_layout import portable_parquet_row_group_rows
from ..informatics.ragged_store import (
    CIGAR,
    READ_ID,
    REFERENCE_START,
    SEQUENCE,
    iter_cigar_aligned_pairs,
)
from ..logging_utils import get_logger
from .deamination_evidence import (
    DeaminationSubstitution,
    deamination_substitutions,
    observe_read_deamination,
    segment_deamination,
)

logger = get_logger(__name__)

DEAMINATION_SUBDIR = "deamination"
DEAMINATION_TASK_STORE = "task_store"
DEAMINATION_OBS_SIDECAR = "deamination_obs"
CHIMERA_COLUMN = "deaminase_segment_chimera"

_BASE_DECODER = {
    int(value): str(base).upper() for base, value in MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT.items()
}


def deamination_reporting_enabled(cfg) -> bool:
    """Whether this experiment has deamination chemistry at all.

    Gated on the substitution set rather than on a modality string, so `direct`
    -- which has no such chemistry and, in general, no chimeras -- pays nothing,
    and a modality that later gains a conversion map is picked up automatically.
    """
    if bool(getattr(cfg, "bypass_deamination_segmentation", False)):
        logger.info(
            "Deamination segmentation bypassed by config; the scalar "
            "deaminase_PCR_chimera column is unaffected"
        )
        return False
    return bool(
        deamination_substitutions(
            getattr(cfg, "smf_modality", None),
            list(getattr(cfg, "conversion_types", []) or []),
        )
    )


def _execute_deamination_batch(
    rows: list[tuple[str, str, str, int, list]],
    reference_strand: str,
    reference_sequence: str,
    experiment_uid: str,
    excluded_positions: tuple[int, ...],
    substitutions: tuple[DeaminationSubstitution, ...],
    penalty_scale: float,
    min_segment: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Segment one batch of reads. Module-level and picklable for the pool.

    Batches are cut *within* shards, not per shard. The `251105` raw store has
    8 ragged shards and one holds 20,258 of 28,302 reads, so shard-level
    dispatch would leave one worker doing 72% of the work and cap the speedup
    near 1.4x regardless of core count.
    """
    events: list[dict] = []
    segments: list[dict] = []
    summaries: list[dict] = []
    excluded = set(excluded_positions)
    for read_id, molecule_uid, cigar, reference_start, sequence in rows:
        observed = _observed_bases_from_parts(sequence, cigar, reference_start)
        observations = observe_read_deamination(
            reference_sequence, observed, substitutions, excluded_positions=excluded
        )
        read_segments, summary = segment_deamination(
            observations, penalty_scale=penalty_scale, min_segment_size=min_segment
        )
        common = {
            EXPERIMENT_UID_COLUMN: experiment_uid,
            "read_id": read_id,
            MOLECULE_UID_COLUMN: molecule_uid,
            "reference": reference_strand,
        }
        for observation in observations:
            events.append(
                {
                    **common,
                    "position": observation.position,
                    "strand": observation.strand,
                    "converted": observation.converted,
                }
            )
        for segment in read_segments:
            segments.append(
                {
                    **common,
                    "start": segment.start,
                    "end": segment.end,
                    "strand": segment.strand,
                    "n_observations": segment.n_observations,
                    "n_converted": segment.n_converted,
                }
            )
        summaries.append(
            {
                **common,
                "n_observations": summary.n_observations,
                "efficiency": summary.efficiency,
                "error_rate": summary.error_rate,
                "segment_count": summary.segment_count,
                "strands_present": ",".join(summary.strands_present),
                "dominant_strand": summary.dominant_strand or "",
                "switch_positions": ",".join(str(p) for p in summary.switch_positions),
                CHIMERA_COLUMN: summary.is_chimeric,
            }
        )
    return events, segments, summaries


def _observed_bases_from_parts(sequence, cigar: str, reference_start: int) -> dict[int, str]:
    """Reference-position -> observed base, from the ragged row's parts alone.

    Takes the three fields rather than a Series so a batch ships only what the
    walk needs; the full row carries quality, mismatch and signal columns that
    would otherwise be pickled to every worker for nothing.
    """
    values = list(sequence)
    observed: dict[int, str] = {}
    for query_position, reference_position in iter_cigar_aligned_pairs(
        str(cigar), int(reference_start)
    ):
        if query_position >= len(values):
            raise ValueError("ragged sequence is shorter than its CIGAR query span")
        base = _BASE_DECODER.get(int(values[query_position]), "N")
        if base != "N":
            observed[reference_position] = base
    return observed


def _observed_bases(row: pd.Series) -> dict[int, str]:
    """Reference-position -> observed base for one ragged row."""
    sequence = list(row[SEQUENCE])
    observed: dict[int, str] = {}
    for query_position, reference_position in iter_cigar_aligned_pairs(
        str(row[CIGAR]), int(row[REFERENCE_START])
    ):
        if query_position >= len(sequence):
            raise ValueError("ragged sequence is shorter than its CIGAR query span")
        base = _BASE_DECODER.get(int(sequence[query_position]), "N")
        if base != "N":
            observed[reference_position] = base
    return observed


class _StreamingParquetSink:
    """Append row dicts to a parquet file without holding them all.

    The batch results they come from are the other half of `F44`: streaming the
    arguments into the pool bounded its input, but every completed result was
    still retained until the pool finished and then copied into a list and again
    into a DataFrame -- three simultaneous copies of the whole output. On the
    `260820` run that grew linearly at ~5.2 GiB/min with no plateau (`F45`).

    The schema is taken from the first flush and every later batch is built
    against it, so a batch whose column happens to be all-null cannot silently
    change a column's type. A batch that genuinely does not fit the established
    schema raises rather than writing something subtly wrong.
    """

    def __init__(self, path: Path, *, flush_rows: int = 50_000) -> None:
        self._path = path
        self._flush_rows = max(1, int(flush_rows))
        self._buffer: list[dict] = []
        self._writer = None
        self._schema = None
        self.n_rows = 0

    def extend(self, rows) -> None:
        rows = list(rows)
        if not rows:
            return
        self._buffer.extend(rows)
        self.n_rows += len(rows)
        if len(self._buffer) >= self._flush_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._schema is None:
            table = pa.Table.from_pylist(self._buffer)
            self._schema = table.schema
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._path, self._schema)
        else:
            table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> Path:
        self._flush()
        if self._writer is not None:
            self._writer.close()
        else:
            # No rows at all: still publish the artifact the caller expects.
            _write_parquet(pd.DataFrame(), self._path)
        return self._path


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, row_group_size=portable_parquet_row_group_rows(frame))
    return path


def append_deamination_annotations(obs_path, deamination_obs_path) -> "Path":
    """Merge per-molecule deamination summaries into the preprocess obs sidecar.

    Mirrors `variant_reporting.append_variant_reporting_annotations`. Carries the
    chimera call plus the evidence behind it -- efficiency, error rate, segment
    count, switch positions -- so a surprising classification can be checked
    without recomputing the lane.
    """
    obs_path = Path(obs_path)
    summary = pd.read_parquet(deamination_obs_path)
    if summary.empty:
        return obs_path
    obs = pd.read_parquet(obs_path)
    carried = [
        "read_id",
        "n_observations",
        "efficiency",
        "error_rate",
        "segment_count",
        "strands_present",
        "dominant_strand",
        "switch_positions",
        CHIMERA_COLUMN,
    ]
    summary = summary[[column for column in carried if column in summary.columns]]
    summary = summary.rename(
        columns={
            "n_observations": "deamination_observation_count",
            "efficiency": "deamination_efficiency",
            "error_rate": "deamination_error_rate",
            "segment_count": "deamination_segment_count",
            "strands_present": "deamination_strands_present",
            "dominant_strand": "deamination_dominant_strand",
            "switch_positions": "deamination_switch_positions",
        }
    )
    summary["read_id"] = summary["read_id"].astype(str)
    obs["read_id"] = obs["read_id"].astype(str)
    merged = obs.merge(summary, on="read_id", how="left")
    # A read the lane could not summarize is not "not chimeric" -- it is
    # unmeasured. False is correct for the flag only because the composite
    # column treats an absent *method* separately from an unmeasured read.
    merged[CHIMERA_COLUMN] = merged[CHIMERA_COLUMN].fillna(False).astype(bool)
    merged.to_parquet(obs_path, index=False)
    logger.info(
        "Merged deamination annotations for %d of %d read(s)",
        int(summary["read_id"].isin(obs["read_id"]).sum()),
        len(obs),
    )
    return obs_path


def execute_partitioned_deamination(
    spine_path: str | Path,
    output_dir: str | Path,
    *,
    cfg,
    excluded_positions_by_reference: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, Path]:
    """Compute deamination evidence, segments, and per-molecule summaries.

    ``excluded_positions_by_reference`` drops known variant informative sites
    from the evidence, so a reference difference is never counted as a
    deamination event (`EGL-20a`). The catalog those come from is derived from
    the references alone, which is what keeps the two lanes acyclic.
    """
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    substitutions: Sequence[DeaminationSubstitution] = deamination_substitutions(
        getattr(cfg, "smf_modality", None),
        list(getattr(cfg, "conversion_types", []) or []),
    )
    if not substitutions:
        raise ValueError("deamination evidence requires a conversion or deaminase modality")

    penalty_scale = float(getattr(cfg, "deaminase_segment_penalty_scale", 3.0) or 3.0)
    min_segment = max(1, int(getattr(cfg, "deaminase_segment_min_observations", 3) or 1))
    excluded = {
        str(reference): set(int(position) for position in positions)
        for reference, positions in (excluded_positions_by_reference or {}).items()
    }

    spine = load_spine(spine_path, verbose=False)
    references = {
        str(key).removesuffix("_FASTA_sequence"): str(value).upper()
        for key, value in dict(spine.uns.get("References", {}) or {}).items()
        if str(key).endswith("_FASTA_sequence")
    }
    if not references:
        raise ValueError("raw spine carries no reference sequences")

    obs = spine.obs
    required = {"ragged_shard", "Reference_strand", EXPERIMENT_UID_COLUMN, MOLECULE_UID_COLUMN}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"raw spine lacks deamination identity columns: {sorted(missing)}")

    batch_size = max(1, int(getattr(cfg, "deaminase_segment_batch_reads", 250) or 250))

    def _iter_batches(*, with_payload: bool = True):
        """Yield one segmentation batch at a time, reading one shard per group.

        A generator rather than a list because each batch carries its reads'
        full sequences. Building them all first held ~65 GiB before a single
        task dispatched, against a 76.8 GiB ceiling -- so segmentation was
        refused admission for a 125 MiB task on a run that had already eaten
        the budget (`F44`). Streaming bounds it to the in-flight batches.

        ``with_payload=False`` runs the identical loop while projecting the
        sequence column away, so the batch *count* the pool needs up front costs
        a scalar-only read (~0.5s across 124 shards) instead of a second pass
        over 65 GiB. Both passes share this one definition deliberately: a
        separate counting routine that drifted from the yielding one would
        mis-size the pool in a way nothing would catch.
        """
        for group_path, shard_obs in obs.groupby("ragged_shard", sort=True, observed=True):
            reference_strands = shard_obs["Reference_strand"].astype(str).unique()
            if len(reference_strands) != 1:
                raise ValueError(f"raw shard {group_path!r} spans multiple references")
            reference_strand = str(reference_strands[0])
            reference = reference_strand.rsplit("_", 1)[0]
            sequence = references.get(reference)
            if sequence is None:
                if with_payload:
                    # The counting pass walks the same loop, so warn once.
                    logger.warning(
                        "No reference sequence for %s; skipping deamination evidence",
                        reference_strand,
                    )
                continue
            shard_excluded = tuple(
                sorted(excluded.get(reference_strand, excluded.get(reference, set())))
            )
            molecule_by_read = dict(
                zip(
                    shard_obs.get(
                        "read_id", pd.Series(shard_obs.index, index=shard_obs.index)
                    ).astype(str),
                    shard_obs[MOLECULE_UID_COLUMN].astype(str),
                    strict=True,
                )
            )
            experiment_uid = str(shard_obs[EXPERIMENT_UID_COLUMN].astype(str).iloc[0])

            columns = None if with_payload else [READ_ID]
            frame = pd.read_parquet(spine_path.parent / str(group_path), columns=columns)
            frame[READ_ID] = frame[READ_ID].astype(str)
            rows: list[tuple] = []
            for row in frame.sort_values(READ_ID, kind="stable").to_dict("records"):
                read_id = str(row[READ_ID])
                molecule_uid = molecule_by_read.get(read_id)
                if molecule_uid is None:
                    continue
                rows.append(
                    (
                        read_id,
                        molecule_uid,
                        str(row[CIGAR]),
                        int(row[REFERENCE_START]),
                        row[SEQUENCE],
                    )
                    if with_payload
                    else None
                )
            for offset in range(0, len(rows), batch_size):
                yield (
                    rows[offset : offset + batch_size] if with_payload else (),
                    reference_strand,
                    sequence,
                    experiment_uid,
                    shard_excluded,
                    tuple(substitutions),
                    penalty_scale,
                    min_segment,
                )

    root = output_dir / DEAMINATION_SUBDIR
    events_sink = _StreamingParquetSink(root / DEAMINATION_TASK_STORE / "events.parquet")
    segments_sink = _StreamingParquetSink(root / DEAMINATION_TASK_STORE / "segments.parquet")
    summaries_sink = _StreamingParquetSink(
        root / DEAMINATION_OBS_SIDECAR / "deamination_obs.parquet"
    )
    chimeric_count = [0]

    def _consume(_index, result) -> None:
        """Write one batch's rows straight through; retain nothing but counts."""
        events, segments, summaries = result
        events_sink.extend(events)
        segments_sink.extend(segments)
        summaries_sink.extend(summaries)
        chimeric_count[0] += sum(1 for row in summaries if row.get(CHIMERA_COLUMN))

    n_batches = sum(1 for _ in _iter_batches(with_payload=False))
    if n_batches == 1 or getattr(cfg, "threads", 1) in (None, 1):
        # Sequential streams both ways too: one batch in, one batch straight out.
        for index, batch in enumerate(_iter_batches()):
            _consume(index, _execute_deamination_batch(*batch))
    elif n_batches:
        from ..memory_guard import run_tasks_parallel

        run_tasks_parallel(
            _execute_deamination_batch,
            _iter_batches,
            cfg=cfg,
            n_items=n_batches,
            on_result=_consume,
            pool_label=f"deamination segmentation ({n_batches} batches)",
            per_item_memory_mb=max(1.0, batch_size * 0.5),
            estimator="deamination_batch_peak",
        )

    outputs = {
        "events": events_sink.close(),
        "segments": segments_sink.close(),
        "obs": summaries_sink.close(),
    }
    logger.info(
        "Deamination evidence: %d molecule(s), %d observation(s), %d segment(s), %d chimeric "
        "(penalty_scale=%.1f)",
        summaries_sink.n_rows,
        events_sink.n_rows,
        segments_sink.n_rows,
        chimeric_count[0],
        penalty_scale,
    )
    return outputs
