"""Progress reporting while reconciling read identity (`F23`).

The pass is single-threaded over every read and took 27.7 minutes on a 1.75M
read run while logging nothing, which is indistinguishable from a hang. These
pin that it reports, and — more importantly — that reporting did not change
what it returns.
"""

from __future__ import annotations

import pysam
import pytest

from smftools.informatics import barcode_sidecar
from smftools.informatics.barcode_sidecar import _bam_records

pytestmark = pytest.mark.unit


def _write_bam(path, n_reads, *, tags=True):
    header = {
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "ref", "LN": 1000}],
        "RG": [{"ID": "rg1", "SM": "sample1"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for index in range(n_reads):
            read = pysam.AlignedSegment()
            read.query_name = f"read{index:07d}"
            read.query_sequence = "ACGT"
            read.flag = 4
            read.query_qualities = pysam.qualitystring_to_array("IIII")
            if tags:
                read.set_tag("RG", "rg1")
                read.set_tag("BC", "barcode01")
                read.set_tag("BM", "both")
            out.write(read)
    return path


def test_records_are_returned_for_every_read(tmp_path):
    records = _bam_records(_write_bam(tmp_path / "in.bam", 20))
    assert len(records) == 20


def test_tag_evidence_is_collected(tmp_path):
    records = _bam_records(_write_bam(tmp_path / "in.bam", 3))
    first = records["read0000000"]
    assert first["bam_barcode"] == "barcode01"
    assert first["bam_bm"] == "both"
    assert first["read_group"] == "rg1"
    assert first["bam_sample"] == "sample1"


def test_a_completion_line_is_always_emitted(tmp_path, caplog):
    """Even a short run must say it finished, so the phase is visible at all."""
    with caplog.at_level("INFO", logger="smftools.informatics.barcode_sidecar"):
        _bam_records(_write_bam(tmp_path / "in.bam", 5))

    assert any("Identity reconciliation complete" in record.message for record in caplog.records)
    assert any("Reconciling read identity" in record.message for record in caplog.records)


def test_periodic_progress_is_emitted_for_long_scans(tmp_path, caplog, monkeypatch):
    """The interval is what makes a 28-minute pass legible."""
    monkeypatch.setattr(barcode_sidecar, "_IDENTITY_PROGRESS_INTERVAL", 5)

    with caplog.at_level("INFO", logger="smftools.informatics.barcode_sidecar"):
        _bam_records(_write_bam(tmp_path / "in.bam", 17))

    progress = [
        r for r in caplog.records if "reads scanned," in r.message and "complete" not in r.message
    ]
    assert len(progress) == 3, "one line per 5 reads at 17 reads"


def test_no_progress_lines_below_the_interval(tmp_path, caplog):
    """A short pass must not become chatty."""
    with caplog.at_level("INFO", logger="smftools.informatics.barcode_sidecar"):
        _bam_records(_write_bam(tmp_path / "in.bam", 10))

    progress = [
        r for r in caplog.records if "reads scanned," in r.message and "complete" not in r.message
    ]
    assert progress == []


def test_secondary_and_supplementary_are_still_skipped(tmp_path):
    """Counting scanned reads must not change which records are retained."""
    path = tmp_path / "in.bam"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "ref", "LN": 1000}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for index, flag in enumerate((4, 256, 2048, 4)):
            read = pysam.AlignedSegment()
            read.query_name = f"read{index}"
            read.query_sequence = "ACGT"
            read.flag = flag
            read.query_qualities = pysam.qualitystring_to_array("IIII")
            out.write(read)

    records = _bam_records(path)

    assert set(records) == {"read0", "read3"}
