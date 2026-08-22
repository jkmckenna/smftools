"""MinKNOW chunk indices must not be read as paired-end mates (`F20`).

MinKNOW writes one FASTQ per barcode *per chunk*, numbered `_0`.. `_N`. The
generic mate pattern matches a bare trailing `_1`/`_2`, so without a
discriminator chunk 1 is read as mate R1 and chunk 2 as mate R2.

The loud failure is the lucky case -- a two-chunk barcode raises "must contain
exactly one R1 and one R2". The dangerous case is three or more chunks, where
`_1` and `_2` pair *with each other*: two chunks of the same barcode, from
single-end nanopore data, silently ingested as a mate pair while the rest go in
unpaired. Nothing in the output shows it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smftools.informatics.input_manifest import (
    _chunked_fastq_prefixes,
    _infer_fastq_metadata,
)

pytestmark = pytest.mark.unit


def _minknow(barcode: str, indices) -> list[Path]:
    return [Path(f"FBF00818_pass_{barcode}_f58ece73_8080b7fd_{i}.fastq.gz") for i in indices]


# --- the discriminator -------------------------------------------------------


def test_a_zero_index_proves_chunk_numbering():
    """An R1/R2 pair never has an `_0` sibling."""
    assert _chunked_fastq_prefixes(_minknow("barcode46", (0, 1)))


def test_an_index_above_two_proves_chunk_numbering():
    assert _chunked_fastq_prefixes(_minknow("barcode01", range(17)))


def test_a_bare_one_two_family_stays_ambiguous():
    """Real Illumina pairs are named exactly this way.

    Only positive evidence overrides the existing behaviour; {1, 2} alone is
    genuinely ambiguous and must keep pairing.
    """
    assert (
        _chunked_fastq_prefixes([Path("sample_1.fastq.gz"), Path("sample_2.fastq.gz")])
        == frozenset()
    )


def test_a_lone_mate_one_stays_ambiguous():
    assert _chunked_fastq_prefixes([Path("sample_1.fastq.gz")]) == frozenset()


def test_families_are_tracked_independently():
    """One chunked barcode must not disable pairing for an unrelated sample."""
    paths = [
        *_minknow("barcode01", range(4)),
        Path("illumina_1.fastq.gz"),
        Path("illumina_2.fastq.gz"),
    ]
    families = _chunked_fastq_prefixes(paths)
    assert any("barcode01" in prefix for prefix in families)
    assert not any("illumina" in prefix for prefix in families)


def test_files_without_a_numeric_suffix_are_ignored():
    assert _chunked_fastq_prefixes([Path("reads.fastq.gz")]) == frozenset()


# --- what inference then produces --------------------------------------------


def test_a_chunk_is_inferred_as_unpaired():
    paths = _minknow("barcode01", range(17))
    families = _chunked_fastq_prefixes(paths)

    metadata, names = _infer_fastq_metadata(paths[1], families)

    assert metadata["mate"] == "unpaired"
    assert "pair_id" not in names


def test_the_two_chunk_case_that_used_to_fail_is_unpaired():
    """`barcode46` had exactly `_0` and `_1` and raised on ingestion."""
    paths = _minknow("barcode46", (0, 1))
    families = _chunked_fastq_prefixes(paths)

    for path in paths:
        assert _infer_fastq_metadata(path, families)[0]["mate"] == "unpaired"


def test_the_silent_case_no_longer_pairs():
    """`_1` and `_2` of one barcode must not become each other's mates."""
    paths = _minknow("barcode01", range(17))
    families = _chunked_fastq_prefixes(paths)

    mates = {
        path.name: _infer_fastq_metadata(path, families)[0]["mate"]
        for path in paths
        if path.name.endswith(("_1.fastq.gz", "_2.fastq.gz"))
    }
    assert set(mates.values()) == {"unpaired"}


def test_genuine_illumina_pairs_still_pair():
    """The fix must not cost paired-end support."""
    paths = [Path("sample_1.fastq.gz"), Path("sample_2.fastq.gz")]
    families = _chunked_fastq_prefixes(paths)

    first, _ = _infer_fastq_metadata(paths[0], families)
    second, _ = _infer_fastq_metadata(paths[1], families)

    assert (first["mate"], second["mate"]) == ("R1", "R2")
    assert first["pair_id"] == second["pair_id"] == "sample"


def test_explicit_r1_r2_naming_is_unaffected():
    paths = [Path("sample_R1.fastq.gz"), Path("sample_R2.fastq.gz")]
    families = _chunked_fastq_prefixes(paths)
    assert _infer_fastq_metadata(paths[0], families)[0]["mate"] == "R1"


def test_inference_without_family_context_is_unchanged():
    """Callers that pass no context keep the previous behaviour exactly."""
    assert _infer_fastq_metadata(Path("sample_1.fastq.gz"))[0]["mate"] == "R1"
