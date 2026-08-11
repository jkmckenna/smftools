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
        if request.source_layout == "paired_bam":
            return self._prepare_paired_fastqs(request)
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

    def _prepare_paired_fastqs(self, request: AlignmentRequest) -> tuple[Path, Path]:
        """Split a canonical paired unaligned BAM into synchronized mate streams."""
        from ..bam_functions import _require_pysam

        r1_fastq = request.aligned_bam.with_name("alignment_input_R1.fastq")
        r2_fastq = request.aligned_bam.with_name("alignment_input_R2.fastq")
        r1_fastq.parent.mkdir(parents=True, exist_ok=True)
        pysam = _require_pysam()

        def _write_record(handle, read, mate: int) -> None:
            if read.query_sequence is None or read.query_qualities is None:
                raise AlignmentAdapterError(
                    f"Paired input record {read.query_name!r} lacks sequence or qualities."
                )
            tags = [f"{tag}:Z:{read.get_tag(tag)}" for tag in ("BC", "RG") if read.has_tag(tag)]
            comment = " " + "\t".join(tags) if tags else ""
            quality = pysam.array_to_qualitystring(read.query_qualities)
            handle.write(
                f"@{read.query_name}/{mate}{comment}\n{read.query_sequence}\n+\n{quality}\n"
            )

        try:
            with (
                pysam.AlignmentFile(str(request.input_bam), "rb", check_sq=False) as bam,
                r1_fastq.open("w", encoding="utf-8") as r1_handle,
                r2_fastq.open("w", encoding="utf-8") as r2_handle,
            ):
                iterator = iter(bam.fetch(until_eof=True))
                pair_number = 0
                while True:
                    try:
                        first = next(iterator)
                    except StopIteration:
                        break
                    try:
                        second = next(iterator)
                    except StopIteration as exc:
                        raise AlignmentAdapterError(
                            "Canonical paired BAM ended with an unmatched mate."
                        ) from exc
                    pair_number += 1
                    reads = {
                        1: first if first.is_read1 else second,
                        2: first if first.is_read2 else second,
                    }
                    if (
                        not first.is_paired
                        or not second.is_paired
                        or first.query_name != second.query_name
                        or set(reads) != {1, 2}
                        or not reads[1].is_read1
                        or not reads[2].is_read2
                    ):
                        raise AlignmentAdapterError(
                            "Canonical paired BAM is not synchronized at pair "
                            f"{pair_number}: expected adjacent R1/R2 records with one query name."
                        )
                    _write_record(r1_handle, reads[1], 1)
                    _write_record(r2_handle, reads[2], 2)
        except Exception:
            r1_fastq.unlink(missing_ok=True)
            r2_fastq.unlink(missing_ok=True)
            raise
        return r1_fastq, r2_fastq

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
