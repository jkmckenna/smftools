import json
import shutil
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from smftools.cli.load_adata import (
    _prepare_existing_alignment,
    _validate_alignment_executables,
)
from smftools.informatics.alignment_manifest import (
    AlignmentManifestError,
    read_alignment_manifest,
)
from smftools.informatics.alignment_validation import (
    AlignmentValidationError,
    normalize_existing_alignment,
    prepare_alignment_reference_bundle,
    validate_existing_alignment,
)
from smftools.informatics.sidecar_manifest import resolve_sidecar

pysam = pytest.importorskip("pysam")


def _fasta(path, *, name="ref", sequence="ACGTACGTACGT"):
    path.write_text(f">{name}\n{sequence}\n", encoding="utf-8")
    return path


def _bam(
    path,
    *,
    reference_name="ref",
    reference_length=12,
    starts=(1,),
    sort_order="coordinate",
    direct=False,
    program=True,
    paired_flags=None,
    sequence=True,
    qualities=True,
    cigar=True,
):
    header = {
        "HD": {"VN": "1.6", "SO": sort_order},
        "SQ": [{"SN": reference_name, "LN": reference_length}],
    }
    if program:
        header["PG"] = [{"ID": "minimap2", "PN": "minimap2", "VN": "2.28"}]
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index, start in enumerate(starts):
            read = pysam.AlignedSegment()
            read.query_name = f"read-{index}"
            if sequence:
                read.query_sequence = "ACGT"
            if qualities:
                read.query_qualities = pysam.qualitystring_to_array("IIII")
            read.reference_id = 0
            read.reference_start = start
            if cigar:
                read.cigarstring = "4M"
            read.mapping_quality = 60
            if paired_flags is not None:
                read.flag = paired_flags[index]
            if direct:
                read.set_tag("MM", "C+m,0;")
                read.set_tag("ML", array("B", [200]))
            bam.write(read)
    return path


def _signatures(path):
    with pysam.AlignmentFile(str(path), "rb") as bam:
        return sorted(read.to_string() for read in bam.fetch(until_eof=True))


def test_valid_existing_alignment_is_owned_indexed_and_manifested(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(tmp_path / "source.bam", starts=(1, 5))
    source_bytes = source.read_bytes()
    manifest = SimpleNamespace(digest="input-digest")
    sidecars = tmp_path / "raw_outputs" / "sidecar_manifest.json"

    first = _prepare_existing_alignment(
        output_directory=tmp_path,
        source_bam=source,
        reference_fasta=fasta,
        reference_bundle={"schema_version": 1, "digest": "reference-digest"},
        resolved_input_manifest=manifest,
        sidecar_manifest=sidecars,
        modality="conversion",
        threads=1,
        force_redo=False,
    )
    second = _prepare_existing_alignment(
        output_directory=tmp_path,
        source_bam=source,
        reference_fasta=fasta,
        reference_bundle={"schema_version": 1, "digest": "reference-digest"},
        resolved_input_manifest=manifest,
        sidecar_manifest=sidecars,
        modality="conversion",
        threads=1,
        force_redo=False,
    )

    assert second == first
    assert first[0] != source
    assert first[0].is_file() and first[1].is_file()
    assert source.read_bytes() == source_bytes
    assert not source.with_suffix(".bam.bai").exists()
    payload = read_alignment_manifest(first[2])
    assert payload["validation"]["source"]["external_aligner"] == "minimap2"
    assert payload["validation"]["normalization_applied"] is False
    assert resolve_sidecar(sidecars, "alignment_manifest") == first[2]


def test_existing_mode_does_not_probe_alignment_executables(monkeypatch):
    def unexpected_probe(_command):
        raise AssertionError("existing mode must not probe an aligner")

    monkeypatch.setattr("smftools.cli.load_adata.check_executable_exists", unexpected_probe)
    cfg = SimpleNamespace(
        alignment_mode="existing",
        input_type="bam",
        input_already_demuxed=True,
        aligner="dorado",
    )

    _validate_alignment_executables(cfg)


def test_unsorted_alignment_is_sorted_without_changing_records(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(tmp_path / "source.bam", starts=(5, 1), sort_order="unsorted")
    before = _signatures(source)

    output, bai, source_summary, normalized_summary = normalize_existing_alignment(
        source,
        tmp_path / "owned" / "aligned.bam",
        fasta,
        modality="conversion",
        threads=1,
    )

    assert source_summary.coordinate_sorted is False
    assert normalized_summary.coordinate_sorted is True
    assert normalized_summary.source_index_valid is True
    assert bai.is_file()
    assert _signatures(output) == before


@pytest.mark.parametrize(
    ("reference_name", "reference_length"),
    [("other", 12), ("ref", 13)],
)
def test_reference_name_or_length_mismatch_fails(tmp_path, reference_name, reference_length):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(
        tmp_path / "source.bam",
        reference_name=reference_name,
        reference_length=reference_length,
    )

    with pytest.raises(AlignmentValidationError, match="@SQ names, lengths, or order"):
        validate_existing_alignment(source, fasta, modality="conversion")


def test_malformed_bam_fails_cleanly(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    malformed = tmp_path / "malformed.bam"
    malformed.write_bytes(b"not-a-bam")

    with pytest.raises(AlignmentValidationError, match="Could not read existing BAM"):
        validate_existing_alignment(malformed, fasta, modality="conversion")


def test_direct_existing_alignment_requires_mm_ml(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    missing = _bam(tmp_path / "missing.bam")
    valid = _bam(tmp_path / "valid.bam", direct=True)

    with pytest.raises(AlignmentValidationError, match="requires valid MM/ML"):
        validate_existing_alignment(missing, fasta, modality="direct")
    assert validate_existing_alignment(valid, fasta, modality="direct").mm_ml_primary_records == 1


def test_missing_external_aligner_provenance_is_recorded_as_unknown(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(tmp_path / "source.bam", program=False)

    summary = validate_existing_alignment(source, fasta, modality="conversion")

    assert summary.external_aligner == "unknown"
    assert summary.program_records == ()


@pytest.mark.parametrize(
    ("field", "kwargs", "message"),
    [
        ("sequence", {"sequence": False, "qualities": False}, "no query sequence"),
        ("qualities", {"qualities": False}, "no base qualities"),
        ("CIGAR", {"cigar": False}, "no CIGAR"),
    ],
)
def test_required_alignment_fields_fail(tmp_path, field, kwargs, message):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(tmp_path / f"missing-{field}.bam", **kwargs)

    with pytest.raises(AlignmentValidationError, match=message):
        validate_existing_alignment(source, fasta, modality="conversion")


def test_invalid_and_currently_unsupported_paired_flags_fail_early(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    invalid = _bam(tmp_path / "invalid.bam", paired_flags=(0x1,))
    valid_pair_flag = _bam(tmp_path / "paired.bam", paired_flags=(0x1 | 0x40,))

    with pytest.raises(AlignmentValidationError, match="exactly one of read1/read2"):
        validate_existing_alignment(invalid, fasta, modality="conversion")
    with pytest.raises(AlignmentValidationError, match="Paired existing alignments"):
        validate_existing_alignment(valid_pair_flag, fasta, modality="conversion")


def test_manifest_reader_rejects_owned_artifact_corruption(tmp_path):
    fasta = _fasta(tmp_path / "reference.fa")
    source = _bam(tmp_path / "source.bam")
    prepared = _prepare_existing_alignment(
        output_directory=tmp_path,
        source_bam=source,
        reference_fasta=fasta,
        reference_bundle={"schema_version": 1, "digest": "reference-digest"},
        resolved_input_manifest=SimpleNamespace(digest="input-digest"),
        sidecar_manifest=tmp_path / "raw_outputs" / "sidecar_manifest.json",
        modality="conversion",
        threads=1,
        force_redo=False,
    )
    payload = json.loads(prepared[2].read_text())
    assert payload["state"] == "complete"
    prepared[0].write_bytes(b"corrupt")

    with pytest.raises(AlignmentManifestError, match="checksum mismatch"):
        read_alignment_manifest(prepared[2])


def test_relocated_existing_alignment_manifest_resolves_owned_artifacts(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    fasta = _fasta(source_root / "reference.fa")
    source = _bam(source_root / "source.bam")
    run_root = tmp_path / "run"
    prepared = _prepare_existing_alignment(
        output_directory=run_root,
        source_bam=source,
        reference_fasta=fasta,
        reference_bundle={"schema_version": 1, "digest": "reference-digest"},
        resolved_input_manifest=SimpleNamespace(digest="input-digest"),
        sidecar_manifest=run_root / "raw_outputs" / "sidecar_manifest.json",
        modality="conversion",
        threads=1,
        force_redo=False,
    )
    relative_manifest = prepared[2].relative_to(run_root)
    relocated = tmp_path / "relocated"
    shutil.copytree(run_root, relocated)
    shutil.rmtree(source_root)

    payload = read_alignment_manifest(relocated / relative_manifest)

    assert payload["state"] == "complete"
    assert (
        resolve_sidecar(relocated / "raw_outputs" / "sidecar_manifest.json", "alignment_manifest")
        == relocated / relative_manifest
    )


def test_prepared_reference_bundle_is_content_identified_and_reusable_for_validation(tmp_path):
    source_fasta = _fasta(tmp_path / "reference.fa")

    prepared_fasta, manifest_path = prepare_alignment_reference_bundle(
        source_fasta,
        tmp_path / "bundle",
        modality="direct",
    )
    bam = _bam(tmp_path / "aligned.bam", direct=True)

    assert validate_existing_alignment(bam, prepared_fasta, modality="direct").primary_records == 1
    payload = json.loads(manifest_path.read_text())
    assert payload["state"] == "complete"
    assert payload["prepared_fasta"]["path"] == prepared_fasta.name
    assert len(payload["prepared_fasta"]["sha256"]) == 64


def test_prepared_reference_reduction_does_not_index_source_fasta(tmp_path):
    source_fasta = _fasta(tmp_path / "reference.fa")
    bed = tmp_path / "regions.bed"
    bed.write_text("ref\t2\t6\n", encoding="utf-8")

    prepared_fasta, _ = prepare_alignment_reference_bundle(
        source_fasta,
        tmp_path / "bundle",
        modality="direct",
        alignment_regions_bed=bed,
    )

    assert prepared_fasta.read_text() == ">ref:2-6\nGTAC\n"
    assert not Path(f"{source_fasta}.fai").exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("missing\t0\t4\n", "unknown FASTA record"),
        ("ref\t-1\t4\n", "invalid coordinates"),
        ("ref\t4\t13\n", "invalid coordinates"),
        ("ref\t4\n", "fewer than three fields"),
    ],
)
def test_prepared_reference_rejects_invalid_bed(tmp_path, contents, message):
    source_fasta = _fasta(tmp_path / "reference.fa")
    bed = tmp_path / "regions.bed"
    bed.write_text(contents, encoding="utf-8")

    with pytest.raises(AlignmentValidationError, match=message):
        prepare_alignment_reference_bundle(
            source_fasta,
            tmp_path / "bundle",
            modality="direct",
            alignment_regions_bed=bed,
        )

    assert not Path(f"{source_fasta}.fai").exists()
