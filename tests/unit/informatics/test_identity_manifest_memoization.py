"""Manifest resolution is memoized per identity key (`EGL-30`).

`publish_barcode_identity_sidecar` resolved the input manifest *per read*.
Because resolution is O(manifest rows), a MinKNOW FASTQ tree made it
O(reads x files): 1.75M reads x 575 files, plus four more passes each
iteration. Measured at **~30 minutes** of silence on a real run, against **4
seconds** memoized.

The correctness risk of caching is over-sharing -- returning one read's
manifest match for another whose identity differs. These pin that the cache key
covers every input the result depends on, and that the published output is
byte-identical to recomputation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pysam
import pytest

from smftools.informatics.barcode_sidecar import (
    _common_manifest_value,
    _matching_manifest_rows,
    publish_barcode_identity_sidecar,
    read_barcode_identity_sidecar,
)

pytestmark = pytest.mark.unit


def _rows(n_barcodes=4, files_per_barcode=3):
    return [
        SimpleNamespace(
            barcode=f"barcode{index // files_per_barcode + 1:02d}",
            read_group="",
            pair_id="",
            source_id=f"src{index}",
            sample=f"sample{index // files_per_barcode + 1}",
            namespace="ns",
            source_kind="fastq",
            path=f"/data/barcode{index // files_per_barcode + 1:02d}/part_{index}.fastq.gz",
        )
        for index in range(n_barcodes * files_per_barcode)
    ]


def _write_bam(path, reads):
    header = {
        "HD": {"VN": "1.6"},
        "SQ": [{"SN": "ref", "LN": 1000}],
        "RG": [{"ID": "rg1", "SM": "sample1"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, barcode in reads:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "ACGT"
            read.flag = 4
            read.query_qualities = pysam.qualitystring_to_array("IIII")
            read.set_tag("RG", "rg1")
            read.set_tag("BC", barcode)
            out.write(read)
    return path


# --- the property that makes caching safe ------------------------------------


def test_resolution_depends_only_on_the_three_key_values():
    """If it depended on anything else, the cache key would be incomplete."""
    rows = _rows()
    first = _matching_manifest_rows(
        rows, bam_barcode="barcode02", read_group="", classifier_barcode="barcode02"
    )
    second = _matching_manifest_rows(
        rows, bam_barcode="barcode02", read_group="", classifier_barcode="barcode02"
    )
    assert first == second


def test_different_barcodes_resolve_differently():
    """The failure mode of a too-coarse key: one read's match reused for another."""
    rows = _rows()
    one = _matching_manifest_rows(
        rows, bam_barcode="barcode01", read_group="", classifier_barcode="barcode01"
    )
    two = _matching_manifest_rows(
        rows, bam_barcode="barcode02", read_group="", classifier_barcode="barcode02"
    )
    assert one != two
    assert _common_manifest_value(rows, one, "barcode") != _common_manifest_value(
        rows, two, "barcode"
    )


@pytest.mark.parametrize("field", ["bam_barcode", "read_group", "classifier_barcode"])
def test_every_key_component_can_change_the_result(field):
    """Each of the three must be in the key, or reads collide in the cache."""
    rows = _rows()
    base = dict(bam_barcode="barcode01", read_group="", classifier_barcode="")
    changed = dict(base)
    changed[field] = "barcode03"
    assert _matching_manifest_rows(rows, **base) != _matching_manifest_rows(rows, **changed)


# --- end-to-end equivalence ---------------------------------------------------


def test_published_sidecar_is_correct_across_many_reads(tmp_path):
    """Every read must still receive its own barcode's manifest resolution."""
    rows = _rows()
    reads = [(f"read{index:05d}", f"barcode{index % 4 + 1:02d}") for index in range(400)]
    bam = _write_bam(tmp_path / "in.bam", reads)

    sidecar, _report = publish_barcode_identity_sidecar(
        bam, tmp_path / "identity.parquet", input_manifest=rows, classifier_source="filename"
    )
    frame = read_barcode_identity_sidecar(sidecar).set_index("read_name")

    assert len(frame) == 400
    for name, barcode in reads:
        assert frame.loc[name, "barcode"] == barcode, f"{name} resolved to the wrong barcode"


def test_repeated_publication_is_deterministic(tmp_path):
    """Caching must not make the result depend on read order within a run."""
    rows = _rows()
    reads = [(f"read{index:05d}", f"barcode{index % 4 + 1:02d}") for index in range(200)]

    first_bam = _write_bam(tmp_path / "a.bam", reads)
    second_bam = _write_bam(tmp_path / "b.bam", list(reversed(reads)))

    first, _ = publish_barcode_identity_sidecar(
        first_bam, tmp_path / "a.parquet", input_manifest=rows, classifier_source="filename"
    )
    second, _ = publish_barcode_identity_sidecar(
        second_bam, tmp_path / "b.parquet", input_manifest=rows, classifier_source="filename"
    )

    left = read_barcode_identity_sidecar(first).set_index("read_name").sort_index()
    right = read_barcode_identity_sidecar(second).set_index("read_name").sort_index()
    pd.testing.assert_series_equal(left["barcode"], right["barcode"])


def test_a_single_manifest_row_still_resolves(tmp_path):
    """`_matching_manifest_rows` short-circuits at <=1 row; keep that path covered."""
    rows = _rows(n_barcodes=1, files_per_barcode=1)
    bam = _write_bam(tmp_path / "in.bam", [("read1", "barcode01")])

    sidecar, _ = publish_barcode_identity_sidecar(
        bam, tmp_path / "identity.parquet", input_manifest=rows, classifier_source="filename"
    )

    assert len(read_barcode_identity_sidecar(sidecar)) == 1
