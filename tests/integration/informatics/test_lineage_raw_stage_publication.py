"""The descendant raw stage, run for real, publishes beside the parent.

The unit coverage for `SRB-05b` drives the raw stage through an injected runner.
This exercises the actual `raw_adata` publication path, because the hazard this
program keeps rediscovering is that a seam looks right until something really
runs through it.
"""

from __future__ import annotations

import csv
import json
from array import array
from pathlib import Path

import pytest

from smftools.cli.raw_adata import raw_adata
from smftools.informatics.raw_generation import (
    resolve_current_raw_generation,
    validate_raw_generation,
)

pysam = pytest.importorskip("pysam")
pytestmark = pytest.mark.integration

_LINEAGE_PROVENANCE = {
    "lineage_id": "a" * 64,
    "origin_experiment_uid": "uid-a",
    "parent_raw_generation_id": "parent-a",
    "parent_preprocess_generation_id": None,
    "selection_id": "b" * 64,
    "source_resolution_digest": None,
    "basecall_id": "c" * 64,
    "generation_kind": "selected_cohort",
    "identity_map": None,
}


def _write_aligned_bam(path: Path) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "ref", "LN": 12}],
        "PG": [{"ID": "external", "PN": "external-aligner", "VN": "1"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "read-1"
        read.query_sequence = "ACGT"
        read.query_qualities = pysam.qualitystring_to_array("IIII")
        read.reference_id = 0
        read.reference_start = 0
        read.cigarstring = "4M"
        read.mapping_quality = 60
        read.set_tag("BC", "barcode01")
        read.set_tag("MM", "C+m,0;")
        read.set_tag("ML", array("B", [200]))
        bam.write(read)
    pysam.index(str(path))


def _write_config(path: Path, *, bam_path: Path, fasta: Path, run_root: Path) -> Path:
    rows = [
        ("variable", "value", "help", "options", "type"),
        ("alignment_mode", "existing", "", "", "str"),
        ("input_data_path", str(bam_path), "", "", "str"),
        ("output_directory", str(run_root), "", "", "str"),
        ("fasta", str(fasta), "", "", "str"),
        ("experiment_id", "existing-direct", "", "", "str"),
        ("experiment_name", "existing-direct", "", "", "str"),
        ("smf_modality", "direct", "", "", "str"),
        ("direct_signal_backend", "pysam", "", "", "str"),
        ("input_already_demuxed", "True", "", "", "bool"),
        ("skip_bam_split", "True", "", "", "bool"),
        ("skip_bam_qc", "True", "", "", "bool"),
        ("threads", "1", "", "", "int"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def test_a_descendant_publishes_beside_the_parent_without_taking_current(tmp_path, monkeypatch):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">ref\nACGTACGTACGT\n", encoding="utf-8")
    parent_bam = tmp_path / "parent.bam"
    _write_aligned_bam(parent_bam)
    run_root = tmp_path / "run"
    parent_config = _write_config(
        tmp_path / "parent_config.csv",
        bam_path=parent_bam,
        fasta=fasta,
        run_root=run_root,
    )
    monkeypatch.setattr(
        "smftools.informatics.bam_functions.align_and_sort_BAM",
        lambda *_args, **_kwargs: pytest.fail("existing mode invoked an aligner"),
    )

    raw_adata(str(parent_config))
    parent_dir, parent_manifest = resolve_current_raw_generation(run_root / "raw_outputs")

    descendant_bam = tmp_path / "calls.bam"
    _write_aligned_bam(descendant_bam)
    descendant_config = _write_config(
        tmp_path / "descendant_config.csv",
        bam_path=descendant_bam,
        fasta=fasta,
        run_root=run_root,
    )

    _, descendant_spine, _ = raw_adata(
        str(descendant_config),
        lineage_provenance=dict(_LINEAGE_PROVENANCE),
    )

    descendant_dir = Path(descendant_spine).parent
    still_current, still_manifest = resolve_current_raw_generation(run_root / "raw_outputs")
    descendant = validate_raw_generation(descendant_dir, run_root=run_root)

    # Both generations exist; the parent is still what ordinary readers resolve.
    assert descendant_dir != parent_dir
    assert still_current == parent_dir
    assert still_manifest["generation_id"] == parent_manifest["generation_id"]
    assert descendant["lineage"] == _LINEAGE_PROVENANCE
    assert descendant["schema_version"] == 3
    pointer = json.loads((run_root / "raw_outputs" / "current.json").read_text(encoding="utf-8"))
    assert pointer["generation_id"] == parent_manifest["generation_id"]


def test_the_parent_generation_records_no_lineage(tmp_path, monkeypatch):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">ref\nACGTACGTACGT\n", encoding="utf-8")
    bam_path = tmp_path / "parent.bam"
    _write_aligned_bam(bam_path)
    run_root = tmp_path / "run"
    config = _write_config(
        tmp_path / "config.csv",
        bam_path=bam_path,
        fasta=fasta,
        run_root=run_root,
    )
    monkeypatch.setattr(
        "smftools.informatics.bam_functions.align_and_sort_BAM",
        lambda *_args, **_kwargs: pytest.fail("existing mode invoked an aligner"),
    )

    raw_adata(str(config))

    generation_dir, _ = resolve_current_raw_generation(run_root / "raw_outputs")
    assert validate_raw_generation(generation_dir, run_root=run_root)["lineage"] is None
