"""Built-in Dorado and minimap2 alignment adapters."""

from __future__ import annotations

from pathlib import Path

from .base import (
    AlignmentAdapter,
    AlignmentCapabilities,
    AlignmentEnvironment,
    AlignmentRequest,
)


class Minimap2Adapter(AlignmentAdapter):
    """Adapter preserving the legacy minimap2 argument contract."""

    name = "minimap2"
    executable = "minimap2"
    minimum_version = (2, 24, 0)
    capabilities = AlignmentCapabilities(
        source_layouts=("single_bam",),
        supports_paired_end=False,
        supports_bam_input=True,
        supports_fastq_input=True,
        preserves_mm_ml_from_bam=True,
        preserves_mm_ml_from_fastq=False,
    )

    def prepare_input(self, request: AlignmentRequest, environment: AlignmentEnvironment) -> Path:
        if request.align_from_bam:
            return request.input_bam
        from ..bam_functions import _bam_to_fastq_with_pysam, _bam_to_fastq_with_samtools

        fastq = request.aligned_bam.with_name("alignment_input.fastq")
        try:
            if environment.samtools_backend == "python":
                _bam_to_fastq_with_pysam(request.input_bam, fastq)
            else:
                _bam_to_fastq_with_samtools(request.input_bam, fastq)
        except Exception:
            fastq.unlink(missing_ok=True)
            raise
        return fastq

    def build_argv(self, request: AlignmentRequest, execution_input: Path) -> list[str]:
        argv = [self.executable, *request.aligner_args]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([str(request.reference_fasta), str(execution_input)])
        return argv

    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        argv = [self.executable, *request.aligner_args]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend(["$REFERENCE", "$INPUT_BAM" if request.align_from_bam else "$INPUT_FASTQ"])
        return argv


class DoradoAdapter(AlignmentAdapter):
    """Adapter preserving the legacy Dorado aligner argument contract."""

    name = "dorado"
    executable = "dorado"
    minimum_version = (0, 7, 0)
    capabilities = AlignmentCapabilities(
        source_layouts=("single_bam",),
        supports_paired_end=False,
        supports_bam_input=True,
        supports_fastq_input=False,
        preserves_mm_ml_from_bam=True,
        preserves_mm_ml_from_fastq=False,
    )

    def prepare_input(self, request: AlignmentRequest, environment: AlignmentEnvironment) -> Path:
        return request.input_bam

    def preserves_mm_ml(self, request: AlignmentRequest) -> bool:
        """Dorado always consumes the BAM directly; ``align_from_bam`` is minimap2-only."""
        return self.capabilities.preserves_mm_ml_from_bam

    def build_argv(self, request: AlignmentRequest, execution_input: Path) -> list[str]:
        argv = [self.executable, "aligner"]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([*request.aligner_args, str(request.reference_fasta), str(execution_input)])
        return argv

    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        argv = [self.executable, "aligner"]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([*request.aligner_args, "$REFERENCE", "$INPUT_BAM"])
        return argv
