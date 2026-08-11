import random
import shutil

import pytest

from smftools.informatics.alignment_adapters import AlignmentRequest, get_alignment_adapter
from smftools.informatics.alignment_manifest import (
    read_alignment_manifest,
    write_alignment_manifest,
)
from smftools.informatics.alignment_validation import validate_existing_alignment
from smftools.informatics.bam_functions import (
    concatenate_fastqs_to_bam,
    extract_read_features_from_bam,
    extract_read_relative_base_identities,
    extract_read_tags_from_bam,
)
from smftools.informatics.raw_intermediate_manifest import artifact_checksum

pysam = pytest.importorskip("pysam")
pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("minimap2") is None, reason="minimap2 is not installed")
def test_minimap2_adapter_executes_and_publishes_validated_manifest(tmp_path):
    rng = random.Random(7)
    reference_sequence = "".join(rng.choices("ACGT", k=1400))
    query_sequence = reference_sequence[200:1200]
    reference = tmp_path / "reference.fa"
    reference.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")

    source_bam = tmp_path / "source.bam"
    with pysam.AlignmentFile(source_bam, "wb", header={"HD": {"VN": "1.6"}}) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "read-1"
        read.query_sequence = query_sequence
        read.query_qualities = pysam.qualitystring_to_array("I" * len(query_sequence))
        read.flag = 4
        bam.write(read)

    workspace = tmp_path / "alignment"
    adapter = get_alignment_adapter("minimap2")
    environment = adapter.validate_environment("python")
    result = adapter.execute(
        AlignmentRequest(
            reference_fasta=reference,
            input_bam=source_bam,
            aligned_bam=workspace / "aligned.bam",
            source_layout="single_bam",
            modality="deaminase",
            aligner_args=("-a", "-x", "map-ont", "--MD", "-Y"),
            threads=1,
            align_from_bam=False,
        ),
        environment,
        artifact_checksum(reference),
    )
    validation = validate_existing_alignment(
        result.aligned_sorted_bam,
        reference,
        modality="deaminase",
    )
    manifest = write_alignment_manifest(
        workspace / "alignment_manifest.json",
        input_manifest_digest="input-digest",
        reference_bundle={"schema_version": 1, "digest": "reference-digest"},
        prepared_reference_sha256=artifact_checksum(reference),
        source_bam=source_bam,
        source_sha256=artifact_checksum(source_bam),
        normalized_bam=result.aligned_sorted_bam,
        normalized_bai=result.aligned_sorted_bai,
        validation={"normalized": validation.to_dict()},
        alignment_mode="align",
        adapter=result.provenance,
    )

    payload = read_alignment_manifest(manifest)
    assert validation.mapped_primary_records == 1
    assert payload["adapter"]["name"] == "minimap2"
    assert payload["adapter"]["normalized_argv"][-2:] == ["$REFERENCE", "$INPUT_FASTQ"]
    assert not (workspace / "aligned.bam").exists()
    assert not (workspace / "alignment_input.fastq").exists()


@pytest.mark.skipif(shutil.which("minimap2") is None, reason="minimap2 is not installed")
def test_minimap2_adapter_preserves_paired_segment_and_mate_identity(tmp_path):
    rng = random.Random(19)
    reference_sequence = "".join(rng.choices("ACGT", k=1800))
    complement = str.maketrans("ACGT", "TGCA")
    r1_sequence = reference_sequence[300:500]
    r2_sequence = reference_sequence[700:900].translate(complement)[::-1]
    reference = tmp_path / "reference.fa"
    reference.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    r1 = tmp_path / "sample_R1.fastq"
    r2 = tmp_path / "sample_R2.fastq"
    r1.write_text(f"@template/1\n{r1_sequence}\n+\n{'I' * 200}\n", encoding="utf-8")
    r2.write_text(f"@template/2\n{r2_sequence}\n+\n{'I' * 200}\n", encoding="utf-8")
    source_bam = tmp_path / "source.bam"
    concatenate_fastqs_to_bam(
        [(r1, r2)],
        source_bam,
        barcode_map={r1: "barcode01", r2: "barcode01"},
        read_group_map={r1: "sample-rg", r2: "sample-rg"},
        progress=False,
        samtools_backend="python",
    )

    workspace = tmp_path / "alignment"
    adapter = get_alignment_adapter("minimap2")
    result = adapter.execute(
        AlignmentRequest(
            reference_fasta=reference,
            input_bam=source_bam,
            aligned_bam=workspace / "aligned.bam",
            source_layout="paired_bam",
            modality="conversion",
            aligner_args=("-a", "-x", "sr", "--MD", "-Y"),
            threads=1,
        ),
        adapter.validate_environment("python"),
        artifact_checksum(reference),
    )

    validation = validate_existing_alignment(
        result.aligned_sorted_bam, reference, modality="conversion"
    )
    with pysam.AlignmentFile(result.aligned_sorted_bam, "rb") as bam:
        reads = list(bam.fetch(until_eof=True))
    raw_records = extract_read_relative_base_identities(
        result.aligned_sorted_bam,
        "ref",
        reference_sequence,
        samtools_backend="python",
    )
    metrics = extract_read_features_from_bam(
        result.aligned_sorted_bam, samtools_backend="python", primary_only=True
    )
    tags = extract_read_tags_from_bam(
        result.aligned_sorted_bam, samtools_backend="python", primary_only=True
    )

    assert validation.paired_primary_records == 2
    assert validation.proper_pair_primary_records == 2
    assert {read.query_name for read in reads} == {"template"}
    assert {read.get_tag("BC") for read in reads} == {"barcode01"}
    assert {read.get_tag("RG") for read in reads} == {"sample-rg"}
    assert {read.is_read1 for read in reads} == {True, False}
    assert all(read.next_reference_id == read.reference_id for read in reads)
    assert sorted(read.template_length for read in reads) == [-600, 600]
    assert {record["read_id"] for record in raw_records} == {"template/1", "template/2"}
    assert {record["template_id"] for record in raw_records} == {"template"}
    assert {record["mate"] for record in raw_records} == {"R1", "R2"}
    assert sorted(record["template_length"] for record in raw_records) == [-600, 600]
    assert set(metrics) == {"template/1", "template/2"}
    assert set(tags) == {"template/1", "template/2"}
    assert result.provenance["normalized_argv"][-3:] == [
        "$REFERENCE",
        "$INPUT_R1_FASTQ",
        "$INPUT_R2_FASTQ",
    ]
    assert not (workspace / "alignment_input_R1.fastq").exists()
    assert not (workspace / "alignment_input_R2.fastq").exists()


@pytest.mark.parametrize(
    ("adapter_name", "required_tools"),
    [
        ("bwa-mem2", ("bwa-mem2",)),
        ("bowtie2", ("bowtie2", "bowtie2-build")),
    ],
)
@pytest.mark.parametrize("paired", [False, True], ids=["single", "paired"])
def test_short_read_adapter_executes_with_native_index(
    tmp_path, adapter_name, required_tools, paired
):
    missing = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"required tools are not installed: {', '.join(missing)}")

    rng = random.Random(31)
    reference_sequence = "".join(rng.choices("ACGT", k=2400))
    complement = str.maketrans("ACGT", "TGCA")
    r1_sequence = reference_sequence[300:500]
    r2_sequence = reference_sequence[900:1100].translate(complement)[::-1]
    reference = tmp_path / "reference with spaces.fa"
    reference.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
    r1 = tmp_path / "reads R1.fastq"
    r1.write_text(f"@template/1\n{r1_sequence}\n+\n{'I' * 200}\n", encoding="utf-8")
    inputs = [r1]
    if paired:
        r2 = tmp_path / "reads R2.fastq"
        r2.write_text(f"@template/2\n{r2_sequence}\n+\n{'I' * 200}\n", encoding="utf-8")
        inputs = [(r1, r2)]
    source_bam = tmp_path / "source reads.bam"
    concatenate_fastqs_to_bam(
        inputs,
        source_bam,
        progress=False,
        samtools_backend="python",
    )

    workspace = tmp_path / "alignment workspace"
    adapter = get_alignment_adapter(adapter_name)
    result = adapter.execute(
        AlignmentRequest(
            reference_fasta=reference,
            input_bam=source_bam,
            aligned_bam=workspace / "aligned.bam",
            source_layout="paired_bam" if paired else "single_bam",
            modality="conversion",
            threads=1,
            align_from_bam=False,
        ),
        adapter.validate_environment("python"),
        artifact_checksum(reference),
    )

    validation = validate_existing_alignment(
        result.aligned_sorted_bam, reference, modality="conversion"
    )
    assert validation.mapped_primary_records == (2 if paired else 1)
    assert validation.paired_primary_records == (2 if paired else 0)
    assert result.provenance["reference_index"]["strategy"] == ("content_addressed_native_index")
    assert result.provenance["reference_index"]["index_files"]
    assert not (workspace / "alignment_input.fastq").exists()
    assert not (workspace / "alignment_input_R1.fastq").exists()
    assert not (workspace / "alignment_input_R2.fastq").exists()
