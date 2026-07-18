from __future__ import annotations

import gzip

import pandas as pd
import pytest

from smftools.informatics.fastq_export import (
    _sanitize_filename,
    write_fastq_manifest,
    write_fastq_per_barcode,
)
from smftools.informatics.partition_read import load_spine
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.sequence_encoding import phred_to_fastq_quality_string


def test_phred_to_fastq_quality_string_encodes_and_clamps():
    assert phred_to_fastq_quality_string([0, 10, 30, 40]) == "!+?I"
    # ragged store's -1 "missing quality" sentinel clamps to Phred 0
    assert phred_to_fastq_quality_string([-1, -1]) == "!!"
    # values above the Phred+33 printable range clamp to 93
    assert phred_to_fastq_quality_string([200]) == "~"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("bc01", "bc01"),
        ("weird/barcode name", "weird_barcode_name"),
        ("", "unknown"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert _sanitize_filename(raw) == expected


def _read(read_id, barcode, seq_ints, qual_ints, reference_start=0):
    cigar = f"{len(seq_ints)}M"
    return dict(
        read_id=read_id,
        reference="ref",
        Reference_strand="ref_top",
        barcode=barcode,
        sample=barcode,
        reference_start=reference_start,
        cigar=cigar,
        aligned_length=len(seq_ints),
        sequence=seq_ints,
        quality=qual_ints,
        mismatch=[4] * len(seq_ints),
    )


def _write_store(tmp_path, rows):
    frame = pd.DataFrame(rows)
    out = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 3_000_000},
        analysis_mode="locus",
    )
    return load_spine(out["spine"]), out["spine"].parent


def test_write_fastq_per_barcode_decodes_records_across_shards(tmp_path):
    # read3 is in a different start_bin, forcing a second parquet shard, to
    # exercise the "read each shard once" grouping path.
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30, 30, 30, 30]),  # ACGT
        _read("read2", "bc02", [3, 2, 1, 0], [20, 20, 20, 20]),  # TGCA
        _read("read3", "bc01", [1, 1, 1, 1], [10, 10, 10, 10], reference_start=2_000_000),  # CCCC
    ]
    spine, base_dir = _write_store(tmp_path, rows)
    assert spine.obs["ragged_shard"].nunique() == 2

    manifest = write_fastq_per_barcode(
        spine.obs, base_dir, tmp_path / "fastq_out", group_labels="Barcode"
    )

    assert {barcode: info["n_reads"] for barcode, info in manifest.items()} == {
        "bc01": 2,
        "bc02": 1,
    }
    with gzip.open(manifest["bc01"]["path"], "rt") as handle:
        content = handle.read()
    assert content == "@read1\nACGT\n+\n????\n@read3\nCCCC\n+\n++++\n"
    with gzip.open(manifest["bc02"]["path"], "rt") as handle:
        assert handle.read() == "@read2\nTGCA\n+\n5555\n"


def test_write_fastq_per_barcode_filters_by_read_ids(tmp_path):
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30] * 4),
        _read("read2", "bc01", [3, 2, 1, 0], [20] * 4),
    ]
    spine, base_dir = _write_store(tmp_path, rows)

    manifest = write_fastq_per_barcode(
        spine.obs, base_dir, tmp_path / "fastq_out", read_ids={"read1"}, group_labels="Barcode"
    )
    assert set(manifest) == {"bc01"}
    assert manifest["bc01"]["n_reads"] == 1


def test_write_fastq_per_barcode_external_group_labels_and_missing_fallback(tmp_path):
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], [30] * 4),
        _read("read2", "bc01", [3, 2, 1, 0], [20] * 4),
    ]
    spine, base_dir = _write_store(tmp_path, rows)
    # read2 intentionally missing from the external label series
    labels = pd.Series({"read1": "custom_name"})

    manifest = write_fastq_per_barcode(
        spine.obs, base_dir, tmp_path / "fastq_out", group_labels=labels
    )
    assert manifest["custom_name"]["n_reads"] == 1
    assert manifest["unknown"]["n_reads"] == 1


def test_write_fastq_per_barcode_no_matching_reads_returns_empty(tmp_path):
    rows = [_read("read1", "bc01", [0, 1, 2, 3], [30] * 4)]
    spine, base_dir = _write_store(tmp_path, rows)

    manifest = write_fastq_per_barcode(
        spine.obs, base_dir, tmp_path / "fastq_out", read_ids={"nonexistent"}
    )
    assert manifest == {}


def test_write_fastq_per_barcode_requires_ragged_shard_column(tmp_path):
    obs = pd.DataFrame({"Barcode": ["bc01"]}, index=["read1"])
    with pytest.raises(KeyError, match="ragged_shard"):
        write_fastq_per_barcode(obs, tmp_path, tmp_path / "out")


def test_write_fastq_per_barcode_plain_text_output(tmp_path):
    rows = [_read("read1", "bc01", [0, 1, 2, 3], [30] * 4)]
    spine, base_dir = _write_store(tmp_path, rows)

    manifest = write_fastq_per_barcode(
        spine.obs, base_dir, tmp_path / "fastq_out", gzip_output=False, group_labels="Barcode"
    )
    path = manifest["bc01"]["path"]
    assert path.suffix == ".fastq"
    assert path.read_text() == "@read1\nACGT\n+\n????\n"


def test_write_fastq_manifest_writes_csv(tmp_path):
    manifest = {
        "bc02": {"path": tmp_path / "bc02.fastq.gz", "n_reads": 3},
        "bc01": {"path": tmp_path / "bc01.fastq.gz", "n_reads": 5},
    }
    path = write_fastq_manifest(tmp_path, manifest)
    frame = pd.read_csv(path)
    assert list(frame["barcode"]) == ["bc01", "bc02"]
    assert list(frame["n_reads"]) == [5, 3]
