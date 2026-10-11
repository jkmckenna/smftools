"""Barcode QC recomputed for an existing run (`BQC-03`)."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from smftools.preprocessing.barcode_qc_recompute import (
    BarcodeQCRecomputeError,
    read_level_evidence,
    recompute_barcode_qc,
)

pysam = pytest.importorskip("pysam")
pytestmark = pytest.mark.unit

REFERENCE_LENGTH = 40
GENERATION = "a" * 32


def _reads():
    """(name, reverse, cigar, front, rear, front distance, rear distance)."""
    reads = []
    for index in range(30):
        reads.append((f"same{index}", index % 2 == 1, "60S40M50S", "NB01", "NB01", 1, 2))
    reads.append(("mismatch", False, "60S40M50S", "NB01", "NB02", 1, 1))
    reads.append(("no_rear", True, "60S40M", "NB02", None, 0, None))
    return reads


def _run(tmp_path, *, positional=True, molecule_rows=None):
    root = tmp_path / "run"
    raw = root / "raw_outputs"
    generation = raw / "generations" / GENERATION
    generation.mkdir(parents=True)
    (raw / "current.json").write_text(
        json.dumps({"generation_id": GENERATION, "generation_path": f"generations/{GENERATION}"})
    )
    reads = _reads()
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "amp", "LN": REFERENCE_LENGTH}]}
    bam = raw / "aligned_sorted.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for name, reverse, cigar, *_rest in reads:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.reference_id = 0
            read.reference_start = 0
            read.cigarstring = cigar
            length = sum(int(n) for n in re.findall(r"(\d+)[MIS]", cigar))
            read.query_sequence = "A" * length
            read.query_qualities = pysam.qualitystring_to_array("I" * length)
            read.is_reverse = reverse
            read.mapping_quality = 60
            out.write(read)
    (raw / "alignment_manifest.json").write_text(
        json.dumps({"artifacts": {"bam": {"path": "aligned_sorted.bam"}}})
    )
    sidecars = {
        "alignment_manifest": {
            "path": "alignment_manifest.json",
            "path_kind": "relative",
            "anchor": "manifest_parent",
        }
    }
    if positional:
        pd.DataFrame(
            [
                {"read_name": name, "B1": d1, "B2": d2, "B5": front, "B6": rear}
                for name, _reverse, _cigar, front, rear, d1, d2 in reads
            ]
        ).to_parquet(raw / "positional.parquet", index=False)
        sidecars["barcode_positional"] = {
            "path": "positional.parquet",
            "path_kind": "relative",
            "anchor": "manifest_parent",
        }
    (raw / "sidecar_manifest.json").write_text(json.dumps({"sidecars": sidecars}))
    names = molecule_rows or [name for name, *_rest in reads if name != "mismatch"]
    pd.DataFrame(
        {
            "read_id": names,
            "basecall_read_id": names,
            "Reference_strand": "amp_top",
            "barcode_assigned": "NB01",
            "barcode_front": "NB01",
            "barcode_rear": "NB01",
            "demux_type": "double",
        }
    ).to_parquet(generation / "molecules.parquet", index=False)
    return root


def test_read_level_evidence_keeps_reads_the_molecule_table_dropped(tmp_path):
    root = _run(tmp_path)
    evidence, notes = read_level_evidence(root, root / "raw_outputs" / "generations" / GENERATION)
    rows = evidence.set_index("read_name")
    assert "mismatch" in rows.index  # absent from the molecule table
    assert rows.loc["mismatch", "barcode_rear"] == "NB02"
    # Reverse records swap the CIGAR's clips into read orientation.
    assert tuple(rows.loc["same1", ["front_clip_length", "rear_clip_length"]]) == (50, 60)
    assert tuple(rows.loc["no_rear", ["front_clip_length", "rear_clip_length"]]) == (0, 60)
    assert rows.loc["same0", "reference_length"] == REFERENCE_LENGTH
    assert "aligned_sorted.bam" in notes[0]


def test_recompute_writes_both_reports_outside_the_generation(tmp_path):
    root = _run(tmp_path)
    result = recompute_barcode_qc(root, barcode_kit="SQK-NBD114-24", pool=["NB01", "NB02"])
    out = root / "raw_outputs" / "barcode_qc" / GENERATION
    assert result["output_dir"] == out
    summary = json.loads((out / "barcode_contamination_qc" / "tagging_summary.json").read_text())
    assert summary["mismatch"] == 1 and summary["double_same"] == 30
    assert json.loads((out / "recompute.json").read_text())["tagging_evidence"] == "reads"
    assert (out / "barcode_contamination_qc" / "contamination_summary.json").is_file()
    generation = root / "raw_outputs" / "generations" / GENERATION
    assert sorted(path.name for path in generation.iterdir()) == ["molecules.parquet"]


def test_without_positional_sidecar_the_model_records_why(tmp_path):
    root = _run(tmp_path, positional=False)
    result = recompute_barcode_qc(
        root, barcode_kit="SQK-NBD114-24", pool=["NB01"], unclassified_dropped=True
    )
    notes = " ".join(result["notes"])
    assert "native extractor" in notes
    assert "tagging model skipped" in notes  # molecule table has no BQC-01 columns


def test_missing_raw_generation_is_an_error(tmp_path):
    with pytest.raises(BarcodeQCRecomputeError, match="no current raw generation"):
        recompute_barcode_qc(tmp_path, barcode_kit="SQK-NBD114-24")


ADAPTER = "TTGACCAGTAGGCAGTCAATCGGATCCTAG"  # + 8 bp barcode = a "barcode" end clip
KIT = {"NB01": "AAGGTTAA", "NB02": "CCTTGGCC"}


def _revcomp(sequence):
    return sequence[::-1].translate(str.maketrans("ACGT", "TGCA"))


def test_extract_builds_per_end_calls_for_runs_demultiplexed_upstream(tmp_path):
    """A MinKNOW-demultiplexed run has no positional sidecar; `--extract` makes one
    from the aligned BAM (reverse records flipped back to read orientation)."""
    from types import SimpleNamespace

    root = _run(tmp_path, positional=False)
    raw = root / "raw_outputs"
    insert = ("GATTACA" * 6)[:REFERENCE_LENGTH]
    reads = {
        "same": (False, ADAPTER + KIT["NB01"] + insert + _revcomp(ADAPTER + KIT["NB01"])),
        "mismatch": (True, ADAPTER + KIT["NB01"] + insert + _revcomp(ADAPTER + KIT["NB02"])),
    }
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "amp", "LN": REFERENCE_LENGTH}]}
    with pysam.AlignmentFile(str(raw / "aligned_sorted.bam"), "wb", header=header) as out:
        for name, (reverse, sequence) in reads.items():
            record = sequence if not reverse else _revcomp(sequence)
            clip = len(ADAPTER) + 8
            read = pysam.AlignedSegment()
            read.query_name = name
            read.reference_id = 0
            read.reference_start = 0
            read.cigarstring = f"{clip}S{REFERENCE_LENGTH}M{clip}S"
            read.query_sequence = record
            read.query_qualities = pysam.qualitystring_to_array("I" * len(record))
            read.is_reverse = reverse
            out.write(read)
    kit = tmp_path / "kit.yaml"
    kit.write_text(
        "name: TEST\nflanking:\n  adapter_side: "
        + ADAPTER
        + "\nbarcode_ends: both\nbarcode_max_edit_distance: 1\n"
        "barcode_composite_max_edits: 0\nbarcodes:\n"
        + "".join(f"  {name}: {sequence}\n" for name, sequence in KIT.items())
    )
    cfg = SimpleNamespace(
        barcode_kit="custom",
        custom_barcode_yaml=str(kit),
        barcode_adapter_matcher="exact",
        barcode_search_window=60,
        samtools_backend="python",
        threads=1,
    )

    result = recompute_barcode_qc(
        root, barcode_kit="SQK-NBD114-24", pool=["NB01", "NB02"], extract_with=cfg
    )

    out = root / "raw_outputs" / "barcode_qc" / GENERATION
    calls = pd.read_parquet(out / "positional.parquet").set_index("read_name")
    assert tuple(calls.loc["same", ["B5", "B6"]]) == ("NB01", "NB01")
    assert tuple(calls.loc["mismatch", ["B5", "B6", "BM"]]) == ("NB01", "NB02", "mismatch")
    assert any("extracted from the aligned BAM" in note for note in result["notes"])
    summary = json.loads((out / "barcode_contamination_qc" / "tagging_summary.json").read_text())
    assert summary["mismatch"] == 1 and summary["double_same"] == 1
