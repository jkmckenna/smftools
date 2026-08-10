"""Semantic identity and commit records for raw-ingestion intermediates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from smftools.constants import RAW_DIR

from ..readwrite import atomic_write_json

RAW_INTERMEDIATE_MANIFEST_SCHEMA_VERSION = 1
RAW_INTERMEDIATE_ROOT = "intermediates"
RAW_INTERMEDIATE_COMMIT_FILENAME = "commit.json"
FILE_HASH_CHUNK_SIZE = 1024 * 1024
_OPERATION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class RawIntermediateError(RuntimeError):
    """Raised when an intermediate cannot be validated or committed safely."""


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def sha256_file(path: str | Path) -> str:
    """Stream one file through SHA-256 and reject concurrent mutation."""
    path = Path(path)
    try:
        before = path.stat()
        signature = _stat_signature(before)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            if _stat_signature(os.fstat(handle.fileno())) != signature:
                raise RawIntermediateError(f"Artifact changed before hashing began: {path}")
            while block := handle.read(FILE_HASH_CHUNK_SIZE):
                digest.update(block)
            after_handle = os.fstat(handle.fileno())
        after_path = path.stat()
    except RawIntermediateError:
        raise
    except OSError as exc:
        raise RawIntermediateError(f"Could not hash artifact {path}: {exc}") from exc
    if _stat_signature(after_handle) != signature or _stat_signature(after_path) != signature:
        raise RawIntermediateError(f"Artifact changed while hashing: {path}")
    return digest.hexdigest()


def _directory_digest(path: Path) -> str:
    records = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size_bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
        )
    return _json_digest(records)


def artifact_checksum(path: str | Path) -> str:
    """Return a content checksum for one file or recursively for one directory."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return _directory_digest(path)
    raise RawIntermediateError(f"Intermediate artifact is missing: {path}")


def executable_version(command: str) -> str | None:
    """Return a concise executable version string without failing planning."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def alignment_reference_bundle(cfg: Any) -> dict[str, Any]:
    """Return the content-derived alignment-reference bundle identity for planning."""
    fasta_value = getattr(cfg, "fasta", None)
    if not fasta_value:
        raise RawIntermediateError("Alignment reference FASTA is not configured.")
    fasta = Path(fasta_value).expanduser().resolve(strict=False)
    artifacts = [{"role": "fasta", "sha256": sha256_file(fasta)}]
    alignment_bed_value = getattr(cfg, "alignment_regions_bed", None)
    if alignment_bed_value:
        alignment_bed = Path(alignment_bed_value).expanduser().resolve(strict=False)
        artifacts.append({"role": "alignment_regions_bed", "sha256": sha256_file(alignment_bed)})
    semantics = {
        "modality": str(getattr(cfg, "smf_modality", "") or ""),
        "conversions": list(getattr(cfg, "conversion_types", ()) or ()),
        "strands": list(getattr(cfg, "strands", ()) or ()),
    }
    payload = {"schema_version": 1, "artifacts": artifacts, "semantics": semantics}
    return {**payload, "digest": _json_digest(payload)}


@dataclass(frozen=True)
class IntermediateSpec:
    """Expected semantic inputs and compatibility policy for one operation."""

    operation: str
    input_artifacts: tuple[tuple[str, str], ...]
    operation_config: Mapping[str, Any] = field(default_factory=dict)
    tool_versions: Mapping[str, str | None] = field(default_factory=dict)
    tool_version_policy: str = "strict"

    def __post_init__(self) -> None:
        operation = str(self.operation).strip().lower()
        if not _OPERATION_PATTERN.fullmatch(operation):
            raise ValueError(f"Invalid raw intermediate operation name: {self.operation!r}")
        if self.tool_version_policy not in {"strict", "provenance_only"}:
            raise ValueError("tool_version_policy must be 'strict' or 'provenance_only'.")
        artifacts = tuple((str(name), str(checksum)) for name, checksum in self.input_artifacts)
        if any(not name or not checksum for name, checksum in artifacts):
            raise ValueError("Intermediate input artifact identities must be nonempty.")
        if len({name for name, _ in artifacts}) != len(artifacts):
            raise ValueError("Intermediate input artifact names must be unique.")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "input_artifacts", artifacts)
        object.__setattr__(
            self,
            "operation_config",
            MappingProxyType(dict(sorted(self.operation_config.items()))),
        )
        object.__setattr__(
            self,
            "tool_versions",
            MappingProxyType(
                {
                    str(key): None if value is None else str(value)
                    for key, value in sorted(self.tool_versions.items())
                }
            ),
        )

    def compatibility_payload(self) -> dict[str, Any]:
        """Return fields whose changes invalidate reuse under this policy."""
        payload = {
            "schema_version": RAW_INTERMEDIATE_MANIFEST_SCHEMA_VERSION,
            "operation": self.operation,
            "input_artifacts": [
                {"artifact_id": name, "checksum": checksum}
                for name, checksum in self.input_artifacts
            ],
            "operation_config": dict(self.operation_config),
            "tool_version_policy": self.tool_version_policy,
        }
        if self.tool_version_policy == "strict":
            payload["tool_versions"] = dict(self.tool_versions)
        return payload

    @property
    def compatibility_key(self) -> str:
        return _json_digest(self.compatibility_payload())


@dataclass(frozen=True)
class IntermediateWorkspace:
    """One immutable execution revision for a semantic operation key."""

    root: Path
    manifest_path: Path
    spec: IntermediateSpec
    reusable: bool
    revision_id: str


def _revision_manifests(output_directory: str | Path, spec: IntermediateSpec) -> list[Path]:
    root = (
        Path(output_directory)
        / RAW_DIR
        / RAW_INTERMEDIATE_ROOT
        / spec.operation
        / spec.compatibility_key
        / "revisions"
    )
    return sorted(root.glob(f"*/{RAW_INTERMEDIATE_COMMIT_FILENAME}"), reverse=True)


def validate_intermediate_commit(
    manifest_path: str | Path, spec: IntermediateSpec
) -> dict[str, Any] | None:
    """Return a complete compatible commit record, or ``None`` for any mismatch."""
    manifest_path = Path(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RAW_INTERMEDIATE_MANIFEST_SCHEMA_VERSION:
            return None
        if payload.get("state") != "complete":
            return None
        if payload.get("compatibility_key") != spec.compatibility_key:
            return None
        if payload.get("compatibility") != spec.compatibility_payload():
            return None
        outputs = payload.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return None
        root = manifest_path.parent
        for record in outputs:
            relative_path = Path(str(record["path"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            path = root / relative_path
            if artifact_checksum(path) != record.get("sha256"):
                return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RawIntermediateError):
        return None
    return payload


def prepare_intermediate(
    output_directory: str | Path,
    spec: IntermediateSpec,
    *,
    force_redo: bool = False,
) -> IntermediateWorkspace:
    """Reuse a valid commit or allocate a new immutable execution revision."""
    if not force_redo:
        for manifest_path in _revision_manifests(output_directory, spec):
            payload = validate_intermediate_commit(manifest_path, spec)
            if payload is not None:
                return IntermediateWorkspace(
                    root=manifest_path.parent,
                    manifest_path=manifest_path,
                    spec=spec,
                    reusable=True,
                    revision_id=str(payload["revision_id"]),
                )
    revision_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:12]}"
    )
    root = (
        Path(output_directory)
        / RAW_DIR
        / RAW_INTERMEDIATE_ROOT
        / spec.operation
        / spec.compatibility_key
        / "revisions"
        / revision_id
    )
    root.mkdir(parents=True, exist_ok=False)
    return IntermediateWorkspace(
        root=root,
        manifest_path=root / RAW_INTERMEDIATE_COMMIT_FILENAME,
        spec=spec,
        reusable=False,
        revision_id=revision_id,
    )


def commit_intermediate(
    workspace: IntermediateWorkspace,
    outputs: Mapping[str, str | Path],
) -> Path:
    """Validate outputs and atomically publish a complete immutable commit record."""
    if workspace.reusable:
        raise RawIntermediateError("A reused intermediate revision cannot be recommitted.")
    records = []
    for artifact_id, raw_path in sorted(outputs.items()):
        path = Path(raw_path)
        try:
            relative = path.resolve().relative_to(workspace.root.resolve())
        except ValueError as exc:
            raise RawIntermediateError(
                f"Committed intermediate output must be owned by its revision: {path}"
            ) from exc
        records.append(
            {
                "artifact_id": str(artifact_id),
                "path": relative.as_posix(),
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": artifact_checksum(path),
            }
        )
    if not records:
        raise RawIntermediateError("Cannot commit an intermediate without outputs.")
    payload = {
        "schema_version": RAW_INTERMEDIATE_MANIFEST_SCHEMA_VERSION,
        "state": "complete",
        "operation": workspace.spec.operation,
        "revision_id": workspace.revision_id,
        "compatibility_key": workspace.spec.compatibility_key,
        "compatibility": workspace.spec.compatibility_payload(),
        "tool_versions": dict(workspace.spec.tool_versions),
        "outputs": records,
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(workspace.manifest_path, payload)
    if validate_intermediate_commit(workspace.manifest_path, workspace.spec) is None:
        raise RawIntermediateError("Published intermediate commit failed validation.")
    return workspace.manifest_path


def committed_output(workspace: IntermediateWorkspace, artifact_id: str) -> Path | None:
    """Resolve one output from a reusable committed revision."""
    payload = validate_intermediate_commit(workspace.manifest_path, workspace.spec)
    if payload is None:
        return None
    for record in payload["outputs"]:
        if record.get("artifact_id") == artifact_id:
            return workspace.root / str(record["path"])
    return None


def input_artifact_pairs(
    values: Sequence[tuple[str, str]] | Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Normalize explicitly ordered intermediate input identities."""
    return tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
