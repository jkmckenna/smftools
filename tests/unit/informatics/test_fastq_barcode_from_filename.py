"""Barcode inference when converting a MinKNOW FASTQ tree to BAM (`F21`).

MinKNOW names files `<flowcell>_pass_<barcode>_<runid>_<chunk>.fastq.gz`. The
converter took the *last underscore token* as the barcode, which is the chunk
index -- so every barcode's chunk 0 was merged into a "barcode" called `0`,
chunk 3 into `3`, and so on. Reads from genuinely different barcodes ended up
sharing one label, and nothing downstream could tell.

These call `concatenate_fastqs_to_bam` for real rather than re-implementing the
nested helper, because the helper is what shipped wrong and a copy of it in a
test would have agreed with the bug.
"""

from __future__ import annotations

import gzip

import pytest

from smftools.informatics.bam_functions import concatenate_fastqs_to_bam

pytestmark = pytest.mark.unit


def _write_fastq(path, n_reads=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for index in range(n_reads):
            handle.write(f"@read_{path.stem}_{index}\nACGTACGTAC\n+\nIIIIIIIIII\n")
    return path


def _minknow_tree(root, barcodes, chunks):
    paths = []
    for barcode in barcodes:
        for chunk in range(chunks):
            paths.append(
                _write_fastq(
                    root
                    / "fastq_pass"
                    / barcode
                    / f"FBF00818_pass_{barcode}_f58ece73_8080b7fd_{chunk}.fastq.gz"
                )
            )
    return paths


def test_barcode_comes_from_the_barcode_token_not_the_chunk_index(tmp_path):
    paths = _minknow_tree(tmp_path, ["barcode01", "barcode11"], chunks=3)

    summary = concatenate_fastqs_to_bam(
        paths, tmp_path / "out.bam", auto_pair=False, progress=False
    )

    assert set(summary["barcodes"]) == {"barcode01", "barcode11"}


def test_chunks_of_one_barcode_do_not_become_separate_barcodes(tmp_path):
    """The old behaviour produced one "barcode" per chunk index."""
    paths = _minknow_tree(tmp_path, ["barcode01"], chunks=5)

    summary = concatenate_fastqs_to_bam(
        paths, tmp_path / "out.bam", auto_pair=False, progress=False
    )

    assert set(summary["barcodes"]) == {"barcode01"}


def test_two_barcodes_never_collapse_into_one(tmp_path):
    """The damaging half: chunk 0 of every barcode shared the label "0"."""
    paths = _minknow_tree(tmp_path, ["barcode01", "barcode02", "barcode32"], chunks=2)

    summary = concatenate_fastqs_to_bam(
        paths, tmp_path / "out.bam", auto_pair=False, progress=False
    )

    assert len(set(summary["barcodes"])) == 3


def test_a_two_digit_barcode_is_read_whole(tmp_path):
    paths = _minknow_tree(tmp_path, ["barcode32"], chunks=1)
    summary = concatenate_fastqs_to_bam(
        paths, tmp_path / "out.bam", auto_pair=False, progress=False
    )
    assert set(summary["barcodes"]) == {"barcode32"}


def test_the_parent_directory_is_used_when_the_name_carries_nothing(tmp_path):
    """MinKNOW files reads under `fastq_pass/<barcode>/` regardless of naming."""
    path = _write_fastq(tmp_path / "fastq_pass" / "barcode07" / "reads_part_4.fastq.gz")

    summary = concatenate_fastqs_to_bam(
        [path], tmp_path / "out.bam", auto_pair=False, progress=False
    )

    assert set(summary["barcodes"]) == {"barcode07"}


def test_an_explicit_barcode_map_still_wins(tmp_path):
    """Callers that know better must keep overriding inference."""
    paths = _minknow_tree(tmp_path, ["barcode01"], chunks=1)

    summary = concatenate_fastqs_to_bam(
        paths,
        tmp_path / "out.bam",
        auto_pair=False,
        progress=False,
        barcode_map={str(paths[0]): "my_sample"},
    )

    assert set(summary["barcodes"]) == {"my_sample"}


def test_non_minknow_names_keep_the_previous_behaviour(tmp_path):
    """The fallback must not regress for trees this heuristic was written for."""
    path = _write_fastq(tmp_path / "reads_sampleA.fastq.gz")

    summary = concatenate_fastqs_to_bam(
        [path], tmp_path / "out.bam", auto_pair=False, progress=False
    )

    assert set(summary["barcodes"]) == {"sampleA"}
