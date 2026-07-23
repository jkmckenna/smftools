"""Small deterministic partitioned-store fixture for pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from smftools.informatics.raw_store import write_raw_store

FixtureModality = Literal["conversion", "deaminase", "direct"]

REFERENCE_LENGTHS = {
    "fixture_ref_a_top": 16,
    "fixture_ref_b_top": 20,
}

ACCEPTANCE_REFERENCE_LENGTHS = {
    "fixture_ref_a_top": 16,
    "fixture_ref_a_bottom": 16,
    "fixture_ref_b_top": 20,
    "fixture_ref_b_bottom": 20,
}


@dataclass(frozen=True)
class PartitionedPipelineFixture:
    """Paths and source records for one compact partitioned experiment store."""

    root: Path
    run_dir: Path
    raw_store_dir: Path
    basecalled_bam: Path
    modality: FixtureModality
    frame: pd.DataFrame
    paths: dict[str, object]

    @property
    def read_ids(self) -> tuple[str, ...]:
        """Return canonical fixture read IDs in input order."""
        return tuple(self.frame["read_id"].astype(str))


def make_partitioned_ragged_frame(
    modality: FixtureModality = "conversion",
    *,
    reads_per_stratum: int = 2,
    include_strand_derivatives: bool = False,
) -> pd.DataFrame:
    """Return deterministic reads spanning reference, strand, and barcode strata."""
    if modality not in {"conversion", "deaminase", "direct"}:
        raise ValueError(f"unsupported fixture modality: {modality!r}")
    if reads_per_stratum < 1:
        raise ValueError("reads_per_stratum must be positive")

    signal_templates = {
        "conversion": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "deaminase": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "direct": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    }
    rows: list[dict[str, object]] = []
    reference_lengths = (
        ACCEPTANCE_REFERENCE_LENGTHS if include_strand_derivatives else REFERENCE_LENGTHS
    )
    for reference_index, reference in enumerate(reference_lengths):
        strand = "bottom" if reference.endswith("_bottom") else "top"
        reference_name = reference.removesuffix(f"_{strand}")
        for barcode_index, barcode in enumerate(("barcode01", "barcode02")):
            for replicate in range(reads_per_stratum):
                offset = (
                    reference_index * 2 * reads_per_stratum
                    + barcode_index * reads_per_stratum
                    + replicate
                )
                signal = signal_templates[modality]
                rows.append(
                    {
                        "read_id": f"{modality}_read_{offset:02d}",
                        "reference": reference_name,
                        "Reference_strand": reference,
                        "sample": barcode,
                        "barcode": barcode,
                        "strand": strand,
                        "mapping_direction": "rev" if strand == "bottom" else "fwd",
                        "Dataset": modality,
                        "Experiment_name": "partitioned_pipeline_fixture",
                        "reference_start": replicate * 4,
                        "cigar": "8M",
                        "aligned_length": 8,
                        "sequence": [int((offset + position) % 4) for position in range(8)],
                        "quality": [30 + (offset % 5)] * 8,
                        "mismatch": [4] * 8,
                        "modification_signal": signal[offset % 2 :] + signal[: offset % 2],
                    }
                )
    return pd.DataFrame(rows)


def build_partitioned_pipeline_fixture(
    root: str | Path,
    *,
    modality: FixtureModality = "conversion",
    analysis_mode: str = "locus",
    reads_per_stratum: int = 2,
    include_strand_derivatives: bool = False,
) -> PartitionedPipelineFixture:
    """Write and return a portable raw-store fixture rooted below ``root``."""
    root = Path(root)
    run_dir = root / f"{modality}_experiment"
    raw_store_dir = run_dir / "raw_outputs"
    basecalled_bam = run_dir / "inputs" / "basecalled.bam"
    basecalled_bam.parent.mkdir(parents=True, exist_ok=True)
    basecalled_bam.touch()

    frame = make_partitioned_ragged_frame(
        modality,
        reads_per_stratum=reads_per_stratum,
        include_strand_derivatives=include_strand_derivatives,
    )
    reference_lengths = (
        ACCEPTANCE_REFERENCE_LENGTHS if include_strand_derivatives else REFERENCE_LENGTHS
    )
    paths = write_raw_store(
        frame,
        raw_store_dir,
        reference_lengths=reference_lengths,
        shard_size=2,
        start_bin_size=8,
        analysis_mode=analysis_mode,
        bam_path=basecalled_bam,
        extra_uns={"fixture_modality": modality, "analysis_mode": analysis_mode},
    )
    return PartitionedPipelineFixture(
        root=root,
        run_dir=run_dir,
        raw_store_dir=raw_store_dir,
        basecalled_bam=basecalled_bam,
        modality=modality,
        frame=frame,
        paths=paths,
    )
