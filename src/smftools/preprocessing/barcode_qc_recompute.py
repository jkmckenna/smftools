"""Barcode QC for an existing run, without regenerating it (`BQC-03`).

Preprocess writes the spike-in contamination QC (`EGL-31`) and the ligation /
tagging model (`BQC-02`) as it publishes a generation. Runs published before
those existed, or before the per-end evidence they read (`BQC-01`), have
neither. This recomputes both from the current raw generation's molecule
table and writes them to ``raw_outputs/barcode_qc/<raw generation id>/``,
outside the generation, which stays immutable.

The tagging model is fitted on every primary read, with per-end evidence
rebuilt from artifacts the run already holds -- the molecule table drops
unclassified reads when ``skip_unclassified`` is set, and those include the
mismatch reads the model needs:

- clip lengths at each read end (read orientation), reference span and the
  supplementary flag, from the aligned BAM named by the alignment manifest;
- per-end barcode names and edit distances, from the native extractor's
  positional sidecar (`B1` / `B2` / `B5` / `B6`).

A run demultiplexed by dorado or MinKNOW has no positional sidecar, so no
per-end names: the tagging model then reports why it was skipped, and only
the contamination counts are written.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

from smftools.logging_utils import get_logger

from ..informatics.sidecar_manifest import resolve_sidecar
from .barcode_contamination_qc import write_barcode_contamination_qc
from .barcode_tagging_model import kit_barcode_count, write_barcode_tagging_qc

logger = get_logger(__name__)

RAW_STAGE_DIR = "raw_outputs"
QC_DIRNAME = "barcode_qc"
#: Per-end barcode calls written by ``--extract`` beside the QC outputs.
EXTRACTED_POSITIONAL = "positional.parquet"


class BarcodeQCRecomputeError(RuntimeError):
    """The run lacks what barcode QC needs."""


def current_raw_generation(output_root: str | Path) -> Path:
    stage = Path(output_root) / RAW_STAGE_DIR
    pointer = stage / "current.json"
    if not pointer.is_file():
        raise BarcodeQCRecomputeError(f"no current raw generation under {stage}")
    current = json.loads(pointer.read_text(encoding="utf-8"))
    generation = stage / current["generation_path"]
    if not generation.is_dir():
        raise BarcodeQCRecomputeError(f"current raw generation missing: {generation}")
    return generation


def _sidecar(output_root: Path, generation: Path, key: str) -> Path | None:
    for directory in (generation, output_root / RAW_STAGE_DIR):
        found = resolve_sidecar(directory / "sidecar_manifest.json", key)
        if found is not None:
            return found
    return None


def _aligned_bam(output_root: Path, generation: Path) -> Path | None:
    manifest = _sidecar(output_root, generation, "alignment_manifest")
    if manifest is None:
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    bam = manifest.parent / payload.get("artifacts", {}).get("bam", {}).get("path", "")
    return bam if bam.is_file() else None


def _primary_alignments(bam: Path) -> pd.DataFrame:
    """Every primary alignment: reference span, orientation, end clips (read
    orientation) and the SA flag."""
    from ..informatics.ragged_store import cigar_end_clips, cigar_reference_length

    rows: list[tuple] = []
    lengths: dict[str, int] = {}
    if shutil.which("samtools"):
        header = subprocess.run(
            ["samtools", "view", "-H", str(bam)], capture_output=True, text=True, check=True
        ).stdout
        for line in header.splitlines():
            if line.startswith("@SQ"):
                fields = dict(item.split(":", 1) for item in line.split("\t")[1:])
                lengths[fields["SN"]] = int(fields["LN"])
        process = subprocess.Popen(
            ["samtools", "view", "-F", "0x904", str(bam)], stdout=subprocess.PIPE, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            fields = line.split("\t", 12)
            reverse = bool(int(fields[1]) & 16)
            start = int(fields[3]) - 1
            front, rear = cigar_end_clips(fields[5], reverse)
            rows.append(
                (
                    fields[0],
                    fields[2],
                    start,
                    start + cigar_reference_length(fields[5]),
                    reverse,
                    front,
                    rear,
                    "\tSA:Z:" in line,
                )
            )
        if process.wait() != 0:
            raise BarcodeQCRecomputeError(f"samtools view failed on {bam}")
    else:
        from ..informatics.bam_functions import _require_pysam

        pysam = _require_pysam()
        with pysam.AlignmentFile(str(bam), "rb") as handle:
            lengths = dict(zip(handle.references, handle.lengths))
            for read in handle.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                front, rear = cigar_end_clips(read.cigarstring, read.is_reverse)
                rows.append(
                    (
                        read.query_name,
                        read.reference_name,
                        read.reference_start,
                        read.reference_end,
                        read.is_reverse,
                        front,
                        rear,
                        read.has_tag("SA"),
                    )
                )
    frame = pd.DataFrame(
        rows,
        columns=[
            "read_name",
            "reference",
            "reference_start",
            "reference_end",
            "reverse",
            "front_clip_length",
            "rear_clip_length",
            "has_supplementary_alignment",
        ],
    )
    frame["reference_length"] = frame["reference"].map(lengths)
    frame["Reference_strand"] = frame["reference"].astype(str)
    frame["Read_mapping_direction"] = frame["reverse"].map({True: "rev", False: "fwd"})
    return frame.drop_duplicates("read_name")


def barcode_kit_settings(cfg) -> dict:
    """The barcode extractor's settings for a config, resolved as the raw stage
    resolves them (config over kit YAML over defaults)."""
    from smftools.constants import BARCODE_KIT_ALIASES

    from ..informatics.bam_functions import (
        BarcodeKitConfig,
        _build_flanking_from_adapters,
        load_barcode_references_from_yaml,
        resolve_barcode_config,
    )

    kit = getattr(cfg, "barcode_kit", None)
    if kit == "custom":
        path = getattr(cfg, "custom_barcode_yaml", None)
    else:
        path = BARCODE_KIT_ALIASES.get(str(kit or ""))
    if not path:
        raise BarcodeQCRecomputeError(f"no barcode kit YAML for barcode_kit={kit!r}")
    loaded = load_barcode_references_from_yaml(path)
    if isinstance(loaded, BarcodeKitConfig):
        kit_config = loaded
    else:
        references, length = loaded
        adapters = getattr(cfg, "barcode_adapters", [None, None]) or [None, None]
        kit_config = BarcodeKitConfig(
            barcodes=references,
            barcode_length=length,
            flanking=_build_flanking_from_adapters(adapters)
            if any(adapter is not None for adapter in adapters)
            else None,
        )
    resolved = resolve_barcode_config(kit_config, cfg)
    return {
        "barcode_adapters": getattr(cfg, "barcode_adapters", [None, None]),
        "barcode_references": kit_config.barcodes,
        "barcode_length": kit_config.barcode_length,
        "barcode_search_window": getattr(cfg, "barcode_search_window", 200),
        "barcode_max_edit_distance": resolved["barcode_max_edit_distance"],
        "barcode_adapter_matcher": getattr(cfg, "barcode_adapter_matcher", "edlib"),
        "barcode_composite_max_edits": resolved["barcode_composite_max_edits"],
        "barcode_min_separation": resolved.get("barcode_min_separation"),
        "require_both_ends": bool(getattr(cfg, "barcode_both_ends", False)),
        "min_barcode_score": getattr(cfg, "barcode_min_score", None),
        "barcode_kit_config": kit_config,
        "barcode_ends": resolved["barcode_ends"],
        "barcode_amplicon_gap_tolerance": resolved["barcode_amplicon_gap_tolerance"],
    }


def extract_positional(
    cfg, output_root: str | Path, generation: Path, out: Path, *, threads: int | None = None
) -> Path:
    """Run the native barcode extractor over the aligned BAM's primary reads.

    Reverse alignments are flipped back to read orientation by the extractor,
    so the aligned BAM serves as well as the unaligned basecalls for reads that
    still carry their barcodes (not trimmed at basecalling). Writes
    ``positional.parquet`` into ``out``; an existing one is reused.
    """
    from ..informatics.bam_functions import extract_and_assign_barcodes_in_bam

    target = out / EXTRACTED_POSITIONAL
    if target.is_file():
        return target
    bam = _aligned_bam(Path(output_root), generation)
    if bam is None:
        raise BarcodeQCRecomputeError("no aligned BAM to extract barcodes from")
    out.mkdir(parents=True, exist_ok=True)
    staging = out / f".{EXTRACTED_POSITIONAL}.partial"
    extract_and_assign_barcodes_in_bam(
        bam,
        **barcode_kit_settings(cfg),
        samtools_backend=getattr(cfg, "samtools_backend", "auto"),
        threads=threads or getattr(cfg, "threads", None),
        output_path=staging,
        primary_only=True,
    )
    staging.replace(target)
    return target


def read_level_evidence(
    output_root: str | Path, generation: Path, *, positional: Path | None = None
) -> tuple[pd.DataFrame | None, list[str]]:
    """Per-end evidence for **every** primary read, from the aligned BAM and
    the native extractor's positional sidecar.

    The molecule table is not used for this: with `skip_unclassified` it drops
    reads whose two ends name different barcodes -- the reads the tagging
    model is fitted from (on one run, 237 of 246 cleanly double-barcoded
    mismatch reads, against 0.3 % of double-same reads).
    """
    output_root = Path(output_root)
    bam = _aligned_bam(output_root, generation)
    if positional is None:
        positional = _sidecar(output_root, generation, "barcode_positional")
    if bam is None:
        return None, ["no aligned BAM: per-end evidence unavailable"]
    if positional is None:
        return None, [
            "no positional barcode sidecar (not demultiplexed by the native extractor): "
            "per-end names and edit distances unavailable"
        ]
    reads = _primary_alignments(bam)
    calls = pd.read_parquet(positional, columns=["read_name", "B1", "B2", "B5", "B6"])
    calls = calls.drop_duplicates("read_name").set_index("read_name")
    calls = calls.reindex(reads["read_name"])
    reads["barcode_front"] = calls["B5"].fillna("").astype(str).to_numpy()
    reads["barcode_rear"] = calls["B6"].fillna("").astype(str).to_numpy()
    reads["barcode_front_edit_distance"] = pd.to_numeric(calls["B1"], errors="coerce").to_numpy()
    reads["barcode_rear_edit_distance"] = pd.to_numeric(calls["B2"], errors="coerce").to_numpy()
    return reads, [f"per-end evidence from {bam.name} and {positional.name} ({len(reads)} reads)"]


def recompute_barcode_qc(
    output_root: str | Path,
    *,
    barcode_kit: str | None,
    spike_in_references: Iterable[str] = (),
    pool: Iterable[str] | None = None,
    read_level: bool = True,
    unclassified_dropped: bool = False,
    extract_with=None,
    threads: int | None = None,
) -> dict[str, object]:
    """Write `EGL-31` and `BQC-02` QC for the run's current raw generation.

    The spike-in counts come from the molecule table. The tagging model reads
    every primary read (:func:`read_level_evidence`) when the run holds an
    aligned BAM and a positional barcode sidecar; otherwise, or with
    ``read_level=False``, the molecule table (which needs `BQC-01` columns,
    and lacks mismatch reads when ``unclassified_dropped``).

    ``extract_with`` (the run's experiment config) runs the native barcode
    extractor over the aligned BAM when the run has no positional sidecar --
    runs demultiplexed by MinKNOW / dorado whose reads kept their barcodes.
    """
    output_root = Path(output_root)
    generation = current_raw_generation(output_root)
    molecules = generation / "molecules.parquet"
    if not molecules.is_file():
        raise BarcodeQCRecomputeError(f"no molecule table in {generation}")
    obs = pd.read_parquet(molecules)
    out = output_root / RAW_STAGE_DIR / QC_DIRNAME / generation.name
    out.mkdir(parents=True, exist_ok=True)
    references = list(spike_in_references)
    notes: list[str] = []
    paths: dict[str, object] = {"output_dir": out, "notes": notes}
    paths.update(
        {
            f"contamination_{key}": value
            for key, value in write_barcode_contamination_qc(
                obs, out, spike_in_references=references
            ).items()
        }
    )
    positional = None
    if read_level and extract_with is not None:
        if _sidecar(output_root, generation, "barcode_positional") is None:
            positional = extract_positional(
                extract_with, output_root, generation, out, threads=threads
            )
            notes.append(f"per-end barcodes extracted from the aligned BAM into {positional.name}")
    evidence, source_notes = (
        (None, [])
        if not read_level
        else read_level_evidence(output_root, generation, positional=positional)
    )
    notes.extend(source_notes)
    source = "reads"
    if evidence is None:
        evidence, source = obs, "molecule table"
        if unclassified_dropped:
            notes.append("molecule table without unclassified reads: mismatch reads are missing")
    try:
        paths.update(
            {
                f"tagging_{key}": value
                for key, value in write_barcode_tagging_qc(
                    evidence,
                    out,
                    barcode_kit=barcode_kit,
                    spike_in_references=references,
                    pool=pool,
                    kit_size=kit_barcode_count(barcode_kit),
                    unclassified_dropped=source == "molecule table" and unclassified_dropped,
                ).items()
            }
        )
    except KeyError as error:
        notes.append(f"tagging model skipped: {error}")
    (out / "recompute.json").write_text(
        json.dumps(
            {
                "raw_generation": generation.name,
                "barcode_kit": barcode_kit,
                "spike_in_references": references,
                "pool": sorted(pool) if pool else None,
                "tagging_evidence": source,
                "notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for note in notes:
        logger.info("Barcode QC: %s", note)
    return paths
