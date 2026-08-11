"""Indexed short-read alignment adapters for BWA-MEM2 and Bowtie2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ...readwrite import atomic_write_json
from ..raw_intermediate_manifest import artifact_checksum
from .base import (
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentCapabilities,
    AlignmentEnvironment,
    AlignmentInputs,
    AlignmentRequest,
    prepare_sequence_fastqs,
    probe_executable_version,
)


class _IndexedFastqAdapter(AlignmentAdapter):
    """Shared content-addressed reference indexing for short-read adapters."""

    index_executable: str
    index_version_args: tuple[str, ...] = ("--version",)
    index_minimum_version: tuple[int, int, int]
    reference_index_parameters: Mapping[str, Any] = {"format": "native"}
    capabilities = AlignmentCapabilities(
        source_layouts=("single_bam", "paired_bam"),
        supports_paired_end=True,
        supports_bam_input=False,
        supports_fastq_input=True,
        preserves_mm_ml_from_bam=False,
        preserves_mm_ml_from_fastq=False,
    )
    tag_preservation_limits = (
        "Sequence-only FASTQ staging does not preserve arbitrary BAM auxiliary tags.",
        "MM/ML modification tags are not preserved; direct-modification requests are rejected.",
        "Barcode and read-group authority remains in canonical ingestion sidecars/manifests.",
    )
    managed_aligner_options: tuple[str, ...] = ()

    def validate_request(self, request: AlignmentRequest) -> None:
        """Reject lossy routes and arguments owned by the adapter contract."""
        super().validate_request(request)
        conflicts = [
            argument
            for argument in request.aligner_args
            if any(
                argument == option
                or argument.startswith(f"{option}=")
                or (
                    option.startswith("-")
                    and not option.startswith("--")
                    and argument.startswith(option)
                )
                for option in self.managed_aligner_options
            )
        ]
        if conflicts:
            raise AlignmentAdapterError(
                f"Adapter {self.name!r} manages these options internally: " + ", ".join(conflicts)
            )

    def validate_environment(self, samtools_backend: str) -> AlignmentEnvironment:
        """Validate the aligner, index builder, and shared sort backend."""
        environment = super().validate_environment(samtools_backend)
        if self.index_executable == self.executable:
            builder_version = environment.adapter_version
        else:
            builder_version, _parsed = probe_executable_version(
                self.index_executable,
                self.index_minimum_version,
                version_args=self.index_version_args,
            )
        return replace(environment, index_builder_version=builder_version)

    def reference_plan(
        self, reference_sha256: str, environment: AlignmentEnvironment
    ) -> dict[str, Any]:
        """Return content-addressed native-index identity."""
        plan = super().reference_plan(reference_sha256, environment)
        plan["strategy"] = "content_addressed_native_index"
        plan["index_builder"] = {
            "executable": self.index_executable,
            "version": environment.index_builder_version,
        }
        identity_payload = {key: value for key, value in plan.items() if key != "identity"}
        plan["identity"] = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return plan

    def prepare_input(
        self, request: AlignmentRequest, environment: AlignmentEnvironment
    ) -> AlignmentInputs:
        """Stage sequence-only FASTQ input from the canonical unaligned BAM."""
        return prepare_sequence_fastqs(request, environment)

    def prepare_reference(
        self,
        request: AlignmentRequest,
        environment: AlignmentEnvironment,
        reference_sha256: str,
    ) -> tuple[Path, dict[str, Any]]:
        """Build or reuse an atomic content-addressed native reference index."""
        plan = self.reference_plan(reference_sha256, environment)
        cache_parent = request.aligned_bam.parent / "reference_indexes"
        index_root = cache_parent / f"{self.name}-{plan['identity'][:20]}"
        manifest_path = index_root / "index_manifest.json"
        prefix = index_root / "reference"
        reusable_files = self._validated_index_files(manifest_path, plan)
        if reusable_files is not None:
            return prefix, {**plan, "index_files": reusable_files, "reused": True}

        cache_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.name}-index-", dir=cache_parent))
        staging_prefix = staging / "reference"
        try:
            argv = self.build_index_argv(request.reference_fasta, staging_prefix)
            completed = subprocess.run(
                argv,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise AlignmentAdapterError(
                    f"{self.index_executable} index build failed with exit code "
                    f"{completed.returncode}: {detail}"
                )
            staged_files = self.index_files(staging_prefix)
            if not staged_files:
                raise AlignmentAdapterError(
                    f"{self.index_executable} completed without producing a valid index."
                )
            index_root.mkdir(parents=True, exist_ok=True)
            final_files = []
            for staged_file in staged_files:
                destination = index_root / staged_file.name
                os.replace(staged_file, destination)
                final_files.append(destination)
            records = [
                {
                    "path": path.name,
                    "sha256": artifact_checksum(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(final_files)
            ]
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "state": "complete",
                    "plan": plan,
                    "normalized_argv": self.normalized_index_argv(),
                    "files": records,
                },
            )
            return prefix, {**plan, "index_files": records, "reused": False}
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _validated_index_files(
        self, manifest_path: Path, plan: Mapping[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Return validated index records or ``None`` for a cache miss."""
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != 1
                or payload.get("state") != "complete"
                or payload.get("plan") != dict(plan)
            ):
                return None
            records = payload.get("files")
            if not isinstance(records, list) or not records:
                return None
            for record in records:
                path = manifest_path.parent / str(record["path"])
                if (
                    path.parent != manifest_path.parent
                    or artifact_checksum(path) != record["sha256"]
                ):
                    return None
            return records
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def build_index_argv(self, reference: Path, prefix: Path) -> list[str]:
        """Return the exact native index-builder argv."""
        raise NotImplementedError

    def normalized_index_argv(self) -> list[str]:
        """Return relocation-independent index-builder provenance."""
        raise NotImplementedError

    def index_files(self, prefix: Path) -> list[Path]:
        """Return a complete staged native index or an empty list."""
        raise NotImplementedError


class BwaMem2Adapter(_IndexedFastqAdapter):
    """BWA-MEM2 adapter for single-end and paired-end sequence alignment."""

    name = "bwa-mem2"
    executable = "bwa-mem2"
    minimum_version = (2, 2, 1)
    version_args = ("version",)
    index_executable = "bwa-mem2"
    index_minimum_version = minimum_version
    index_suffixes = (".0123", ".amb", ".ann", ".bwt.2bit.64", ".pac")
    managed_aligner_options = ("-t", "-o")

    def build_index_argv(self, reference: Path, prefix: Path) -> list[str]:
        return [self.index_executable, "index", "-p", str(prefix), str(reference)]

    def normalized_index_argv(self) -> list[str]:
        return [self.index_executable, "index", "-p", "$REFERENCE_INDEX", "$REFERENCE"]

    def index_files(self, prefix: Path) -> list[Path]:
        files = [Path(f"{prefix}{suffix}") for suffix in self.index_suffixes]
        return files if all(path.is_file() for path in files) else []

    def build_argv(self, request: AlignmentRequest, execution_input: AlignmentInputs) -> list[str]:
        inputs = execution_input if isinstance(execution_input, tuple) else (execution_input,)
        argv = [self.executable, "mem"]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([*request.aligner_args, str(request.reference_fasta)])
        argv.extend(str(path) for path in inputs)
        return argv

    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        argv = [self.executable, "mem"]
        if request.threads:
            argv.extend(["-t", str(request.threads)])
        argv.extend([*request.aligner_args, "$REFERENCE_INDEX"])
        argv.extend(
            ["$INPUT_R1_FASTQ", "$INPUT_R2_FASTQ"]
            if request.source_layout == "paired_bam"
            else ["$INPUT_FASTQ"]
        )
        return argv


class Bowtie2Adapter(_IndexedFastqAdapter):
    """Bowtie2 adapter for single-end and paired-end sequence alignment."""

    name = "bowtie2"
    executable = "bowtie2"
    minimum_version = (2, 4, 0)
    index_executable = "bowtie2-build"
    index_minimum_version = minimum_version
    managed_aligner_options = (
        "-p",
        "--threads",
        "-x",
        "--index",
        "-U",
        "--unpaired",
        "-1",
        "--mates1",
        "-2",
        "--mates2",
        "-S",
    )

    def build_index_argv(self, reference: Path, prefix: Path) -> list[str]:
        return [self.index_executable, str(reference), str(prefix)]

    def normalized_index_argv(self) -> list[str]:
        return [self.index_executable, "$REFERENCE", "$REFERENCE_INDEX"]

    def index_files(self, prefix: Path) -> list[Path]:
        small = [Path(f"{prefix}.{number}.bt2") for number in (1, 2, 3, 4)] + [
            Path(f"{prefix}.rev.{number}.bt2") for number in (1, 2)
        ]
        large = [Path(f"{prefix}.{number}.bt2l") for number in (1, 2, 3, 4)] + [
            Path(f"{prefix}.rev.{number}.bt2l") for number in (1, 2)
        ]
        return (
            small
            if all(path.is_file() for path in small)
            else large
            if all(path.is_file() for path in large)
            else []
        )

    def build_argv(self, request: AlignmentRequest, execution_input: AlignmentInputs) -> list[str]:
        argv = [self.executable, *request.aligner_args]
        if request.threads:
            argv.extend(["-p", str(request.threads)])
        argv.extend(["-x", str(request.reference_fasta)])
        if isinstance(execution_input, tuple):
            argv.extend(["-1", str(execution_input[0]), "-2", str(execution_input[1])])
        else:
            argv.extend(["-U", str(execution_input)])
        return argv

    def normalized_argv(self, request: AlignmentRequest) -> list[str]:
        argv = [self.executable, *request.aligner_args]
        if request.threads:
            argv.extend(["-p", str(request.threads)])
        argv.extend(["-x", "$REFERENCE_INDEX"])
        if request.source_layout == "paired_bam":
            argv.extend(["-1", "$INPUT_R1_FASTQ", "-2", "$INPUT_R2_FASTQ"])
        else:
            argv.extend(["-U", "$INPUT_FASTQ"])
        return argv
