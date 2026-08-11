import random
import shutil

import pytest

from smftools.informatics.alignment_adapters import AlignmentRequest, get_alignment_adapter
from smftools.informatics.alignment_manifest import (
    read_alignment_manifest,
    write_alignment_manifest,
)
from smftools.informatics.alignment_validation import validate_existing_alignment
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
