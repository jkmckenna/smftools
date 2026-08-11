"""Core contracts and shared execution for alignment adapters."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

ALIGNMENT_ADAPTER_SCHEMA_VERSION = 1
_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?")


class AlignmentAdapterError(RuntimeError):
    """Raised when adapter selection, validation, or execution fails."""


AlignmentInputs: TypeAlias = Path | tuple[Path, ...]


@dataclass(frozen=True)
class AlignmentCapabilities:
    """Static input and tag-preservation capabilities for one adapter."""

    source_layouts: tuple[str, ...]
    supports_paired_end: bool
    supports_bam_input: bool
    supports_fastq_input: bool
    preserves_mm_ml_from_bam: bool
    preserves_mm_ml_from_fastq: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        payload = asdict(self)
        payload["source_layouts"] = list(self.source_layouts)
        return payload


@dataclass(frozen=True)
class AlignmentEnvironment:
    """Validated executable versions and resolved sort/index backend."""

    adapter_version: str
    adapter_version_tuple: tuple[int, int, int]
    samtools_backend: str
    sort_index_version: str
    index_builder_version: str | None = None

    @property
    def tool_versions(self) -> dict[str, str]:
        """Return tool versions for intermediate compatibility identity."""
        return {
            "adapter": self.adapter_version,
            "sort_index": self.sort_index_version,
            **(
                {"index_builder": self.index_builder_version}
                if self.index_builder_version is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class AlignmentRequest:
    """Validated inputs needed to produce one sorted, indexed alignment."""

    reference_fasta: Path
    input_bam: Path
    aligned_bam: Path
    source_layout: str
    modality: str
    aligner_args: tuple[str, ...] = ()
    threads: int | None = None
    align_from_bam: bool = False


@dataclass(frozen=True)
class AlignmentExecutionResult:
    """Owned adapter outputs plus deterministic provenance."""

    aligned_sorted_bam: Path
    aligned_sorted_bai: Path
    provenance: Mapping[str, Any]


def _parse_version(output: str, executable: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.search(output)
    if match is None:
        raise AlignmentAdapterError(
            f"Could not parse {executable} version from output: {output!r}."
        )
    return tuple(int(value or 0) for value in match.groups())  # type: ignore[return-value]


def probe_executable_version(
    executable: str,
    minimum: tuple[int, int, int],
    *,
    version_args: tuple[str, ...] = ("--version",),
) -> tuple[str, tuple[int, int, int]]:
    """Require an executable with a parseable version at or above ``minimum``."""
    if shutil.which(executable) is None:
        raise AlignmentAdapterError(
            f"Alignment adapter requires executable {executable!r} in PATH."
        )
    try:
        completed = subprocess.run(
            [executable, *version_args],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AlignmentAdapterError(f"Could not probe {executable} version: {exc}") from exc
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip()
    )
    if completed.returncode != 0 or not output:
        raise AlignmentAdapterError(
            f"Could not probe {executable} version (exit {completed.returncode})."
        )
    version = _parse_version(output, executable)
    if version < minimum:
        required = ".".join(map(str, minimum))
        observed = ".".join(map(str, version))
        raise AlignmentAdapterError(
            f"{executable} {observed} is unsupported; install {executable} >= {required}."
        )
    version_line = next(
        line.strip() for line in output.splitlines() if _VERSION_PATTERN.search(line)
    )
    return version_line, version


class AlignmentAdapter(ABC):
    """Base class for shell-free aligner execution and owned normalization."""

    name: str
    executable: str
    minimum_version: tuple[int, int, int]
    capabilities: AlignmentCapabilities
    reference_index_parameters: Mapping[str, Any] = {}
    version_args: tuple[str, ...] = ("--version",)
    tag_preservation_limits: tuple[str, ...] = ()

    def validate_environment(self, samtools_backend: str) -> AlignmentEnvironment:
        """Probe required versions before an alignment workspace is staged."""
        from ..bam_functions import _require_pysam, _resolve_samtools_backend

        adapter_version, parsed = probe_executable_version(
            self.executable,
            self.minimum_version,
            version_args=self.version_args,
        )
        resolved_backend = _resolve_samtools_backend(samtools_backend)
        if resolved_backend == "python":
            pysam = _require_pysam()
            sort_version = f"pysam {pysam.__version__}"
        else:
            sort_version, _ = probe_executable_version("samtools", (1, 10, 0))
        return AlignmentEnvironment(
            adapter_version=adapter_version,
            adapter_version_tuple=parsed,
            samtools_backend=resolved_backend,
            sort_index_version=sort_version,
        )

    def validate_request(self, request: AlignmentRequest) -> None:
        """Reject unsupported source layouts and lossy direct-signal routes."""
        if request.source_layout not in self.capabilities.source_layouts:
            supported = ", ".join(self.capabilities.source_layouts)
            remedy = (
                "Paired alignment is introduced by IAR-08; provide a single-end/long-read "
                "source for now."
                if request.source_layout.startswith("paired")
                else f"Choose a compatible adapter layout ({supported})."
            )
            raise AlignmentAdapterError(
                f"Adapter {self.name!r} does not support source layout "
                f"{request.source_layout!r}. {remedy}"
            )
        if request.align_from_bam and not self.capabilities.supports_bam_input:
            raise AlignmentAdapterError(
                f"Adapter {self.name!r} does not accept BAM alignment input; "
                "set align_from_bam=false to use canonical FASTQ staging."
            )
        preserves_mm_ml = self.preserves_mm_ml(request)
        if request.modality.strip().lower() == "direct" and not preserves_mm_ml:
            raise AlignmentAdapterError(
                f"Adapter {self.name!r} would convert the BAM through sequence-only FASTQ and "
                "discard MM/ML tags. Use dorado or a validated tag-preserving BAM route."
            )

    def preserves_mm_ml(self, request: AlignmentRequest) -> bool:
        """Return whether this request's concrete input route preserves MM/ML."""
        return (
            self.capabilities.preserves_mm_ml_from_bam
            if request.align_from_bam
            else self.capabilities.preserves_mm_ml_from_fastq
        )

    def reference_plan(
        self, reference_sha256: str, environment: AlignmentEnvironment
    ) -> dict[str, Any]:
        """Return semantic reference-index identity for this adapter/version."""
        payload = {
            "schema_version": ALIGNMENT_ADAPTER_SCHEMA_VERSION,
            "adapter": self.name,
            "adapter_version": list(environment.adapter_version_tuple),
            "reference_sha256": str(reference_sha256),
            "index_parameters": dict(self.reference_index_parameters),
            "strategy": "adapter_in_memory",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "identity": hashlib.sha256(encoded).hexdigest()}

    def prepare_reference(
        self,
        request: AlignmentRequest,
        environment: AlignmentEnvironment,
        reference_sha256: str,
    ) -> tuple[Path, dict[str, Any]]:
        """Return the reference consumed by the adapter and its semantic plan."""
        return request.reference_fasta, self.reference_plan(reference_sha256, environment)

    @abstractmethod
    def prepare_input(
        self, request: AlignmentRequest, environment: AlignmentEnvironment
    ) -> AlignmentInputs:
        """Prepare and return one or more execution inputs."""

    @abstractmethod
    def build_argv(self, request: AlignmentRequest, execution_input: AlignmentInputs) -> list[str]:
        """Build the exact argument vector for this request."""

    @abstractmethod
    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        """Build relocation-independent argv provenance."""

    def _stream_stderr(self, stderr) -> None:
        for line in stderr:
            logger.info("[%s] %s", self.name, line.rstrip())

    def _run_aligner(self, argv: list[str], output_bam: Path) -> None:
        output_bam.parent.mkdir(parents=True, exist_ok=True)
        with output_bam.open("wb") as output_handle:
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=output_handle,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except OSError as exc:
                raise AlignmentAdapterError(f"Could not start {self.name}: {exc}") from exc
            assert process.stderr is not None
            self._stream_stderr(process.stderr)
            return_code = process.wait()
        if return_code != 0:
            raise AlignmentAdapterError(f"{self.name} failed with exit code {return_code}.")

    def execute(
        self,
        request: AlignmentRequest,
        environment: AlignmentEnvironment,
        reference_sha256: str,
    ) -> AlignmentExecutionResult:
        """Execute alignment, coordinate sort, and index as one adapter operation."""
        from ..bam_functions import (
            _index_bam_with_pysam,
            _index_bam_with_samtools,
            _sort_bam_with_pysam,
            _sort_bam_with_samtools,
        )

        self.validate_request(request)
        request.aligned_bam.parent.mkdir(parents=True, exist_ok=True)
        reference, reference_plan = self.prepare_reference(request, environment, reference_sha256)
        execution_input = self.prepare_input(request, environment)
        execution_request = replace(request, reference_fasta=reference)
        argv = self.build_argv(execution_request, execution_input)
        sorted_bam = request.aligned_bam.with_name(
            f"{request.aligned_bam.stem}_sorted{request.aligned_bam.suffix}"
        )
        bai = Path(f"{sorted_bam}.bai")
        execution_inputs = (
            execution_input if isinstance(execution_input, tuple) else (execution_input,)
        )
        temporary_inputs = tuple(path for path in execution_inputs if path != request.input_bam)
        try:
            self._run_aligner(argv, request.aligned_bam)
            threads = str(request.threads) if request.threads else None
            if environment.samtools_backend == "python":
                _sort_bam_with_pysam(request.aligned_bam, sorted_bam, threads=threads)
                _index_bam_with_pysam(sorted_bam, threads=threads)
            else:
                _sort_bam_with_samtools(request.aligned_bam, sorted_bam, threads=threads)
                _index_bam_with_samtools(sorted_bam, threads=threads)
        except Exception:
            request.aligned_bam.unlink(missing_ok=True)
            sorted_bam.unlink(missing_ok=True)
            bai.unlink(missing_ok=True)
            raise
        finally:
            for temporary_input in temporary_inputs:
                temporary_input.unlink(missing_ok=True)
        request.aligned_bam.unlink(missing_ok=True)
        provenance = {
            "schema_version": ALIGNMENT_ADAPTER_SCHEMA_VERSION,
            "name": self.name,
            "version": environment.adapter_version,
            "normalized_argv": self.normalized_argv(request),
            "source_layout": request.source_layout,
            "capabilities": self.capabilities.to_dict(),
            "tag_preservation_limits": list(self.tag_preservation_limits),
            "reference_index": reference_plan,
            "sort_index": {
                "backend": environment.samtools_backend,
                "version": environment.sort_index_version,
            },
        }
        return AlignmentExecutionResult(sorted_bam, bai, provenance)


def prepare_sequence_fastqs(
    request: AlignmentRequest, environment: AlignmentEnvironment
) -> AlignmentInputs:
    """Stage canonical single-end or synchronized paired FASTQ input."""
    if request.source_layout == "paired_bam":
        return _prepare_paired_fastqs(request)
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


def _prepare_paired_fastqs(request: AlignmentRequest) -> tuple[Path, Path]:
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
        handle.write(f"@{read.query_name}/{mate}{comment}\n{read.query_sequence}\n+\n{quality}\n")

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
