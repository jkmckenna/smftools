"""Choosing which representation of a run's reads to ingest (`BCS-01`-`BCS-04`).

A run directory holds POD5 signal, a `fastq_pass` tree, and sometimes BAMs from
whichever model was run. It used to be refused outright as `input_data_path`
("mixed recognized input types"), so the practice was to point at one
subdirectory by hand. The config now expresses a *model* and selection picks the
representation satisfying it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from smftools.informatics.basecall_provenance import (
    BasecallProvenance,
    is_bare_selector,
    model_family,
    model_version,
    read_fastq_provenance,
)
from smftools.informatics.basecall_source_selection import (
    capability_suffices,
    describe_rejection,
    is_excluded,
    model_matches,
    select_read_source,
)

pytestmark = pytest.mark.unit

HAC_5 = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
HAC_4 = "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"
SUP_5 = "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"


def _fastq(path: Path, model: str | None) -> Path:
    """One gzipped FASTQ record with a MinKNOW-shaped header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f" basecall_model_version_id={model}" if model else ""
    with gzip.open(path, "wt") as handle:
        handle.write(f"@read0 runid=abc ch=1{tag}\nACGT\n+\nIIII\n")
    return path


def _discovery(**kinds) -> dict:
    found = {
        f"{kind}_paths": []
        for kind in ("fastq", "bam", "cram", "sam", "pod5", "fast5", "h5ad", "other")
    }
    for kind, paths in kinds.items():
        found[f"{kind}_paths"] = list(paths)
    found["all_files_searched"] = sum(len(v) for v in found.values() if isinstance(v, list))
    return found


# --- provenance -------------------------------------------------------------


def test_fastq_provenance_is_read_from_the_header(tmp_path):
    path = _fastq(tmp_path / "reads.fastq.gz", HAC_5)
    provenance = read_fastq_provenance(path)
    assert provenance.model == HAC_5
    assert provenance.family == "hac"
    assert provenance.version == (5, 0, 0)
    # FASTQ is sequence-only however it was produced.
    assert provenance.carries_modifications is False


def test_fastq_without_a_model_is_not_an_error(tmp_path):
    """Plenty of FASTQ carries no provenance; it simply cannot satisfy a model."""
    assert read_fastq_provenance(_fastq(tmp_path / "bare.fastq.gz", None)) is None


def test_unreadable_source_is_not_an_error(tmp_path):
    assert read_fastq_provenance(tmp_path / "absent.fastq.gz") is None


@pytest.mark.parametrize(
    ("name", "family", "version"),
    [(HAC_5, "hac", (5, 0, 0)), (SUP_5, "sup", (5, 0, 0)), ("hac", "hac", ())],
)
def test_model_name_parsing(name, family, version):
    assert model_family(name) == family
    assert model_version(name) == version


# --- the match policy -------------------------------------------------------


def test_bare_selector_accepts_any_version_of_its_family():
    assert is_bare_selector("hac")
    assert model_matches("hac", HAC_5)
    assert model_matches("hac", HAC_4)
    assert not model_matches("hac", SUP_5)


def test_qualified_selector_demands_an_exact_match():
    assert not is_bare_selector(HAC_5)
    assert model_matches(HAC_5, HAC_5)
    assert not model_matches(HAC_5, HAC_4)


def test_a_source_recording_no_model_never_matches():
    assert not model_matches("hac", None)


# --- the capability rule ----------------------------------------------------


@pytest.mark.parametrize("modality", ["deaminase", "conversion"])
def test_sequence_only_modalities_accept_fastq(modality):
    assert capability_suffices(BasecallProvenance(model=HAC_5), "fastq", modality)


def test_direct_modality_refuses_a_canonical_fastq():
    """Model identity alone is not enough: direct reads MM/ML, FASTQ has none."""
    assert not capability_suffices(BasecallProvenance(model=HAC_5), "fastq", "direct")


def test_direct_modality_accepts_a_modification_tagged_bam():
    assert capability_suffices(
        BasecallProvenance(model=HAC_5, carries_modifications=True), "bam", "direct"
    )
    assert not capability_suffices(
        BasecallProvenance(model=HAC_5, carries_modifications=False), "bam", "direct"
    )


# --- selection --------------------------------------------------------------


def test_matching_fastq_is_selected(tmp_path):
    reads = _fastq(tmp_path / "fastq_pass" / "reads.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads]), model_selector="hac", modality="deaminase"
    )
    assert selection.resolved
    assert selection.kind == "fastq"
    assert selection.paths == (reads,)


def test_fastq_fail_is_never_selected(tmp_path):
    """The exclusion the manual workaround existed to achieve."""
    good = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    bad = _fastq(tmp_path / "fastq_fail" / "b.fastq.gz", HAC_5)
    assert is_excluded(bad) and not is_excluded(good)
    selection = select_read_source(
        _discovery(fastq=[good, bad]), model_selector="hac", modality="deaminase"
    )
    assert selection.paths == (good,)


def test_unmatched_model_with_signal_present_means_basecall(tmp_path):
    reads = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads], pod5=[tmp_path / "signal.pod5"]),
        model_selector="sup",
        modality="deaminase",
    )
    assert not selection.resolved
    assert selection.must_basecall


def test_unmatched_model_without_signal_is_explained(tmp_path):
    reads = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads]), model_selector="sup", modality="deaminase"
    )
    assert not selection.resolved and not selection.must_basecall
    message = describe_rejection(selection, model_selector="sup")
    # The refusal has to say what was found and why each candidate lost.
    assert "a.fastq.gz" in message
    assert HAC_5 in message


def test_newest_version_wins_within_a_family(tmp_path):
    older = _fastq(tmp_path / "fastq_pass" / "old.fastq.gz", HAC_4)
    newer = _fastq(tmp_path / "fastq_pass" / "new.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[older, newer]), model_selector="hac", modality="deaminase"
    )
    assert selection.paths == (newer,)


def test_detached_sources_satisfy_nothing(tmp_path):
    reads = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads], pod5=[tmp_path / "signal.pod5"]),
        model_selector="hac",
        modality="deaminase",
        reachable=False,
    )
    assert not selection.resolved
    assert all("detached" in c.reason for c in selection.candidates)


def test_direct_modality_falls_through_to_basecalling(tmp_path):
    """A matching FASTQ cannot serve direct, so signal must be basecalled."""
    reads = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads], pod5=[tmp_path / "signal.pod5"]),
        model_selector="hac",
        modality="direct",
    )
    assert selection.must_basecall


def _bam(path: Path, model: str, *, with_mods: bool = False) -> Path:
    """A minimal unaligned BAM carrying Dorado-shaped read-group provenance."""
    pysam = pytest.importorskip("pysam")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "HD": {"VN": "1.6", "SO": "unknown"},
        "RG": [{"ID": "rg0", "PL": "ONT", "DS": f"basecall_model={model} runid=abc"}],
        "PG": [{"ID": "basecaller", "PN": "dorado", "VN": "0.9.0"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        record = pysam.AlignedSegment(out.header)
        record.query_name = "read0"
        record.query_sequence = "ACGT"
        record.flag = 4
        record.query_qualities = pysam.qualitystring_to_array("IIII")
        record.set_tag("RG", "rg0")
        if with_mods:
            record.set_tag("MM", "C+m?,0;")
            record.set_tag("ML", [200])
        out.write(record)
    return path


def test_bam_is_preferred_over_fastq_when_both_qualify(tmp_path):
    """BAM preserves tags, read groups and any existing alignment; FASTQ does not."""
    reads = _fastq(tmp_path / "fastq_pass" / "a.fastq.gz", HAC_5)
    alignment = _bam(tmp_path / "basecalls" / "a.bam", HAC_5)
    selection = select_read_source(
        _discovery(fastq=[reads], bam=[alignment]),
        model_selector="hac",
        modality="deaminase",
    )
    assert selection.kind == "bam"
    assert selection.paths == (alignment,)


def test_bam_provenance_records_the_dorado_version_without_gating(tmp_path):
    """Recorded and reported, never a reason to reject a source."""
    from smftools.informatics.basecall_provenance import read_bam_provenance

    provenance = read_bam_provenance(_bam(tmp_path / "a.bam", HAC_5))
    assert provenance.model == HAC_5
    assert provenance.dorado_version == "0.9.0"
    assert model_matches("hac", provenance.model)


def test_direct_modality_selects_a_modification_tagged_bam(tmp_path):
    tagged = _bam(tmp_path / "mods" / "a.bam", HAC_5, with_mods=True)
    selection = select_read_source(
        _discovery(bam=[tagged]), model_selector="hac", modality="direct"
    )
    assert selection.kind == "bam"


def test_direct_modality_rejects_an_untagged_bam(tmp_path):
    plain = _bam(tmp_path / "plain" / "a.bam", HAC_5)
    selection = select_read_source(
        _discovery(bam=[plain], pod5=[tmp_path / "s.pod5"]),
        model_selector="hac",
        modality="direct",
    )
    assert selection.must_basecall
