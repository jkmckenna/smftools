"""Per-reference storage planning for locus and genome analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import pandas as pd

from smftools.constants import REFERENCE_STRAND

BYTES_PER_DENSE_POSITION = 8
VALID_ANALYSIS_MODES = frozenset({"auto", "locus", "genome"})
VALID_CACHE_MODES = frozenset({"auto", "full", "tiled", "none"})


@dataclass(frozen=True)
class ReferencePlan:
    """Materialization policy for one stranded reference."""

    reference: str
    analysis_mode: str
    cache_mode: str
    reference_length: int
    n_reads: int
    estimated_dense_bytes: int
    tile_size: int
    tile_halo: int

    def to_dict(self) -> dict[str, object]:
        """Return an HDF5/JSON-friendly representation."""
        return asdict(self)


def _validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def plan_references(
    frame: pd.DataFrame,
    reference_lengths: Mapping[str, int],
    *,
    analysis_mode: str = "auto",
    load_cache_mode: str = "auto",
    max_full_matrix_gb: float = 8.0,
    tile_size: int = 10_000,
    tile_halo: int = 1_000,
) -> list[ReferencePlan]:
    """Choose locus/genome and full/tiled policies independently per reference."""
    analysis_mode = str(analysis_mode).lower()
    load_cache_mode = str(load_cache_mode).lower()
    if analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of {sorted(VALID_ANALYSIS_MODES)}")
    if load_cache_mode not in VALID_CACHE_MODES:
        raise ValueError(f"load_cache_mode must be one of {sorted(VALID_CACHE_MODES)}")
    _validate_positive("max_full_matrix_gb", max_full_matrix_gb)
    _validate_positive("tile_size", tile_size)
    if tile_halo < 0:
        raise ValueError("tile_halo must be non-negative")
    if REFERENCE_STRAND not in frame:
        raise ValueError(f"frame must contain {REFERENCE_STRAND!r}")

    counts = frame[REFERENCE_STRAND].astype(str).value_counts()
    missing = sorted(set(counts.index).difference(map(str, reference_lengths)))
    if missing:
        raise KeyError(f"missing reference lengths for: {missing}")

    threshold = int(max_full_matrix_gb * 1024**3)
    plans: list[ReferencePlan] = []
    for reference in sorted(counts.index):
        length = int(reference_lengths[reference])
        _validate_positive(f"reference length for {reference}", length)
        n_reads = int(counts[reference])
        estimated = n_reads * length * BYTES_PER_DENSE_POSITION
        resolved_analysis = (
            ("locus" if estimated <= threshold else "genome")
            if analysis_mode == "auto"
            else analysis_mode
        )
        resolved_cache = (
            ("full" if resolved_analysis == "locus" else "tiled")
            if load_cache_mode == "auto"
            else load_cache_mode
        )
        plans.append(
            ReferencePlan(
                reference=reference,
                analysis_mode=resolved_analysis,
                cache_mode=resolved_cache,
                reference_length=length,
                n_reads=n_reads,
                estimated_dense_bytes=estimated,
                tile_size=int(tile_size),
                tile_halo=int(tile_halo),
            )
        )
    return plans


def iter_reference_tiles(plan: ReferencePlan) -> Iterable[tuple[int, int, int, int]]:
    """Yield ``core_start, core_end, load_start, load_end`` for a reference."""
    for core_start in range(0, plan.reference_length, plan.tile_size):
        core_end = min(core_start + plan.tile_size, plan.reference_length)
        yield (
            core_start,
            core_end,
            max(0, core_start - plan.tile_halo),
            min(plan.reference_length, core_end + plan.tile_halo),
        )
