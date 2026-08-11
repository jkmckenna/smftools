"""Built-in Dorado and minimap2 alignment adapters."""

from __future__ import annotations

from pathlib import Path

from .base import (
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentCapabilities,
    AlignmentEnvironment,
    AlignmentInputs,
    AlignmentRequest,
    prepare_sequence_fastqs,
)


class Minimap2Adapter(AlignmentAdapter):
    """Adapter preserving the legacy minimap2 argument contract."""

    name = "minimap2"
    executable = "minimap2"
    minimum_version = (2, 24, 0)
    capabilities = AlignmentCapabilities(
        source_layouts=("single_bam", "paired_bam"),
        supports_paired_end=True,
        supports_bam_input=True,
        supports_fastq_input=True,
        preserves_mm_ml_from_bam=True,
        preserves_mm_ml_from_fastq=False,
    )

    def validate_request(self, request: AlignmentRequest) -> None:
        """Reject direct BAM passthrough for paired sequence alignment."""
        super().validate_request(request)
        if request.source_layout == "paired_bam" and request.align_from_bam:
            raise AlignmentAdapterError(
                "Paired minimap2 alignment requires the canonical two-FASTQ route; "
                "set align_from_bam=false."
            )

    def prepare_input(
        self, request: AlignmentRequest, environment: AlignmentEnvironment
    ) -> AlignmentInputs:
        if request.align_from_bam:
            return request.input_bam
        return prepare_sequence_fastqs(request, environment)

    def _aligner_args(self, request: AlignmentRequest) -> tuple[str, ...]:
        args = request.aligner_args
        if request.source_layout == "paired_bam" and "-y" not in args:
            return (*args, "-y")
        return args

    def build_argv(self, request: AlignmentRequest, execution_input: AlignmentInputs) -> list[str]:
        inputs = execution_input if isinstance(execution_input, tuple) else (execution_input,)
        argv = [self.executable, *self._aligner_args(request)]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([str(request.reference_fasta), *(str(path) for path in inputs)])
        return argv

    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        argv = [self.executable, *self._aligner_args(request)]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        if request.source_layout == "paired_bam":
            inputs = ["$INPUT_R1_FASTQ", "$INPUT_R2_FASTQ"]
        else:
            inputs = ["$INPUT_BAM" if request.align_from_bam else "$INPUT_FASTQ"]
        argv.extend(["$REFERENCE", *inputs])
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

    def build_argv(self, request: AlignmentRequest, execution_input: AlignmentInputs) -> list[str]:
        if isinstance(execution_input, tuple):
            raise AlignmentAdapterError("Dorado accepts exactly one BAM input.")
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
