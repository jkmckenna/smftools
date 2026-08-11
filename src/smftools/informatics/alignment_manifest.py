"""Versioned manifests for owned alignment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..readwrite import atomic_write_json
from .raw_intermediate_manifest import artifact_checksum

ALIGNMENT_MANIFEST_SCHEMA_VERSION = 1


class AlignmentManifestError(ValueError):
    """Raised when an alignment manifest is missing or invalid."""


def write_alignment_manifest(
    path: str | Path,
    *,
    input_manifest_digest: str,
    reference_bundle: Mapping[str, Any],
    prepared_reference_sha256: str,
    source_bam: str | Path,
    source_sha256: str,
    normalized_bam: str | Path,
    normalized_bai: str | Path,
    validation: Mapping[str, Any],
    alignment_mode: str = "existing",
    adapter: Mapping[str, Any] | None = None,
) -> Path:
    """Publish a complete alignment manifest with relative owned artifacts."""
    path = Path(path)
    root = path.parent.resolve()

    def owned_record(artifact_path: str | Path) -> dict[str, Any]:
        artifact = Path(artifact_path)
        try:
            relative = artifact.resolve().relative_to(root)
        except ValueError as exc:
            raise AlignmentManifestError(
                f"Alignment artifact is not owned by the manifest directory: {artifact}"
            ) from exc
        return {
            "path": relative.as_posix(),
            "size_bytes": artifact.stat().st_size,
            "sha256": artifact_checksum(artifact),
        }

    payload = {
        "schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        "state": "complete",
        "alignment_mode": str(alignment_mode),
        "input_manifest_digest": str(input_manifest_digest),
        "reference_bundle": {
            **dict(reference_bundle),
            "prepared_fasta_sha256": str(prepared_reference_sha256),
            "prepared_reference_records": validation.get("normalized", {}).get(
                "reference_records", []
            ),
        },
        "source": {
            "path_hint": Path(source_bam).name,
            "sha256": str(source_sha256),
        },
        "artifacts": {
            "bam": owned_record(normalized_bam),
            "bai": owned_record(normalized_bai),
        },
        "validation": dict(validation),
    }
    if adapter is not None:
        payload["adapter"] = dict(adapter)
    atomic_write_json(path, payload)
    read_alignment_manifest(path)
    return path


def read_alignment_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a complete alignment manifest and its owned artifacts."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ALIGNMENT_MANIFEST_SCHEMA_VERSION:
            raise AlignmentManifestError("Unsupported alignment manifest schema version.")
        mode = payload.get("alignment_mode")
        if payload.get("state") != "complete" or mode not in {"align", "existing"}:
            raise AlignmentManifestError("Alignment manifest is not a complete alignment.")
        if mode == "align" and not isinstance(payload.get("adapter"), dict):
            raise AlignmentManifestError("Generated alignment manifest lacks adapter provenance.")
        if not payload.get("input_manifest_digest"):
            raise AlignmentManifestError("Alignment manifest lacks input-manifest identity.")
        reference = payload.get("reference_bundle")
        if (
            not isinstance(reference, dict)
            or not reference.get("digest")
            or not reference.get("prepared_fasta_sha256")
        ):
            raise AlignmentManifestError("Alignment manifest lacks prepared-reference identity.")
        if not isinstance(payload.get("validation"), dict):
            raise AlignmentManifestError("Alignment manifest lacks validation output.")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"bam", "bai"}:
            raise AlignmentManifestError("Alignment manifest must own BAM and BAI artifacts.")
        for record in artifacts.values():
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise AlignmentManifestError("Alignment manifest contains an unsafe artifact path.")
            artifact = path.parent / relative
            if artifact_checksum(artifact) != record.get("sha256"):
                raise AlignmentManifestError(f"Alignment artifact checksum mismatch: {artifact}")
    except AlignmentManifestError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AlignmentManifestError(f"Invalid alignment manifest {path}: {exc}") from exc
    return payload
