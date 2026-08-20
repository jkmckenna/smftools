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
    return bool(
        deamination_substitutions(
            getattr(cfg, "smf_modality", None),
            list(getattr(cfg, "conversion_types", []) or []),
        )
    )


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


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, row_group_size=portable_parquet_row_group_rows(frame))
    return path


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

    event_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for group_path, shard_obs in obs.groupby("ragged_shard", sort=True, observed=True):
        reference_strands = shard_obs["Reference_strand"].astype(str).unique()
        if len(reference_strands) != 1:
            raise ValueError(f"raw shard {group_path!r} spans multiple references")
        reference_strand = str(reference_strands[0])
        # `Reference_strand` is "<reference>_<top|bottom>"; the stored sequence
        # is keyed by the bare reference name.
        reference = reference_strand.rsplit("_", 1)[0]
        sequence = references.get(reference)
        if sequence is None:
            logger.warning(
                "No reference sequence for %s; skipping deamination evidence", reference_strand
            )
            continue
        shard_excluded = excluded.get(reference_strand, excluded.get(reference, set()))
        molecule_by_read = dict(
            zip(
                shard_obs.get("read_id", pd.Series(shard_obs.index, index=shard_obs.index)).astype(
                    str
                ),
                shard_obs[MOLECULE_UID_COLUMN].astype(str),
                strict=True,
            )
        )
        experiment_uid = str(shard_obs[EXPERIMENT_UID_COLUMN].astype(str).iloc[0])

        frame = pd.read_parquet(spine_path.parent / str(group_path))
        frame[READ_ID] = frame[READ_ID].astype(str)
        for row in frame.sort_values(READ_ID, kind="stable").to_dict("records"):
            read_id = str(row[READ_ID])
            if read_id not in molecule_by_read:
                continue
            observations = observe_read_deamination(
                sequence,
                _observed_bases(pd.Series(row)),
                substitutions,
                excluded_positions=shard_excluded,
            )
            segments, summary = segment_deamination(
                observations,
                penalty_scale=penalty_scale,
                min_segment_size=min_segment,
            )
            common = {
                EXPERIMENT_UID_COLUMN: experiment_uid,
                "read_id": read_id,
                MOLECULE_UID_COLUMN: molecule_by_read[read_id],
                "reference": reference_strand,
            }
            for observation in observations:
                event_rows.append(
                    {
                        **common,
                        "position": observation.position,
                        "strand": observation.strand,
                        "converted": observation.converted,
                    }
                )
            for segment in segments:
                segment_rows.append(
                    {
                        **common,
                        "start": segment.start,
                        "end": segment.end,
                        "strand": segment.strand,
                        "n_observations": segment.n_observations,
                        "n_converted": segment.n_converted,
                    }
                )
            summary_rows.append(
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

    root = output_dir / DEAMINATION_SUBDIR
    outputs = {
        "events": _write_parquet(
            pd.DataFrame(event_rows), root / DEAMINATION_TASK_STORE / "events.parquet"
        ),
        "segments": _write_parquet(
            pd.DataFrame(segment_rows), root / DEAMINATION_TASK_STORE / "segments.parquet"
        ),
        "obs": _write_parquet(
            pd.DataFrame(summary_rows), root / DEAMINATION_OBS_SIDECAR / "deamination_obs.parquet"
        ),
    }
    chimeric = int(pd.DataFrame(summary_rows)[CHIMERA_COLUMN].sum()) if summary_rows else 0
    logger.info(
        "Deamination evidence: %d molecule(s), %d observation(s), %d segment(s), %d chimeric "
        "(penalty_scale=%.1f)",
        len(summary_rows),
        len(event_rows),
        len(segment_rows),
        chimeric,
        penalty_scale,
    )
    return outputs
