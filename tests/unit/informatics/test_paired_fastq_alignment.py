from pathlib import Path

import pytest

from smftools.informatics.alignment_adapters import AlignmentEnvironment, AlignmentRequest
from smftools.informatics.alignment_adapters.builtin import Minimap2Adapter
from smftools.informatics.bam_functions import concatenate_fastqs_to_bam
from smftools.informatics.molecule_identity import alignment_segment_id

pysam = pytest.importorskip("pysam")


def _fastq(path: Path, records: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(f"@{name}\n{sequence}\n+\n{'I' * len(sequence)}\n" for name, sequence in records),
        encoding="utf-8",
    )
    return path


def _environment() -> AlignmentEnvironment:
    return AlignmentEnvironment("minimap2 2.28", (2, 28, 0), "python", "pysam 0.23")


def _request(tmp_path: Path, input_bam: Path) -> AlignmentRequest:
    return AlignmentRequest(
        reference_fasta=tmp_path / "reference.fa",
        input_bam=input_bam,
        aligned_bam=tmp_path / "work" / "aligned.bam",
        source_layout="paired_bam",
        modality="conversion",
        aligner_args=("-a", "-x", "sr"),
        threads=2,
    )


@pytest.mark.parametrize(
    ("r1_name", "r2_name"),
    [
        ("template/1", "template/2"),
        ("template_R1", "template_R2"),
        ("template 1:N:0:ATCACG", "template 2:N:0:ATCACG"),
    ],
)
def test_paired_fastq_normalization_validates_names_and_preserves_mates(tmp_path, r1_name, r2_name):
    r1 = _fastq(tmp_path / "sample_R1.fastq", [(r1_name, "ACGT")])
    r2 = _fastq(tmp_path / "sample_R2.fastq", [(r2_name, "TGCA")])
    output = tmp_path / "reads.bam"

    summary = concatenate_fastqs_to_bam(
        [(r1, r2)], output, progress=False, samtools_backend="python"
    )

    with pysam.AlignmentFile(str(output), "rb", check_sq=False) as bam:
        reads = list(bam.fetch(until_eof=True))
    assert summary["paired_pairs_written"] == 1
    assert summary["singletons_written"] == 0
    assert [read.query_name for read in reads] == ["template", "template"]
    assert [alignment_segment_id(read) for read in reads] == ["template/1", "template/2"]
    assert [read.get_tag("BC") for read in reads] == ["sample", "sample"]


@pytest.mark.parametrize(
    ("r1_records", "r2_records", "message"),
    [
        ([("one/1", "AC")], [("two/2", "TG")], "out of sync"),
        (
            [("one/1", "AC"), ("two/1", "GT")],
            [("one/2", "TG")],
            "unequal record counts",
        ),
        ([("one/2", "AC")], [("one/2", "TG")], "assigned as R1"),
    ],
)
def test_invalid_paired_fastqs_fail_without_partial_bam(tmp_path, r1_records, r2_records, message):
    r1 = _fastq(tmp_path / "sample_R1.fastq", r1_records)
    r2 = _fastq(tmp_path / "sample_R2.fastq", r2_records)
    output = tmp_path / "reads.bam"

    with pytest.raises(ValueError, match=message):
        concatenate_fastqs_to_bam([(r1, r2)], output, progress=False, samtools_backend="python")

    assert not output.exists()


def test_minimap2_paired_adapter_stages_two_synchronized_streams(tmp_path):
    r1 = _fastq(tmp_path / "tumor_S1_L001_R1_001.fastq", [("one/1", "ACGT")])
    r2 = _fastq(tmp_path / "tumor_S1_L001_R2_001.fastq", [("one/2", "TGCA")])
    bam = tmp_path / "reads.bam"
    concatenate_fastqs_to_bam(
        [(r1, r2)],
        bam,
        barcode_map={r1: "barcode01", r2: "barcode01"},
        read_group_map={r1: "lane-1", r2: "lane-1"},
        progress=False,
        samtools_backend="python",
    )
    adapter = Minimap2Adapter()
    request = _request(tmp_path, bam)

    inputs = adapter.prepare_input(request, _environment())

    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].read_text(encoding="utf-8").startswith("@one/1 BC:Z:barcode01\tRG:Z:lane-1\n")
    assert inputs[1].read_text(encoding="utf-8").startswith("@one/2 BC:Z:barcode01\tRG:Z:lane-1\n")
    assert adapter.build_argv(request, inputs)[-3:] == [
        str(request.reference_fasta),
        str(inputs[0]),
        str(inputs[1]),
    ]
    assert "-y" in adapter.build_argv(request, inputs)
    assert adapter.normalized_argv(request)[-3:] == [
        "$REFERENCE",
        "$INPUT_R1_FASTQ",
        "$INPUT_R2_FASTQ",
    ]


def test_paired_fastq_metadata_must_be_consistent(tmp_path):
    r1 = _fastq(tmp_path / "sample_R1.fastq", [("one/1", "ACGT")])
    r2 = _fastq(tmp_path / "sample_R2.fastq", [("one/2", "TGCA")])

    with pytest.raises(ValueError, match="conflicting barcode metadata"):
        concatenate_fastqs_to_bam(
            [(r1, r2)],
            tmp_path / "reads.bam",
            barcode_map={r1: "barcode01", r2: "barcode02"},
            progress=False,
            samtools_backend="python",
        )
