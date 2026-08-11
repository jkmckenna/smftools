"""Portable, checksummed manifests for sequence-only and lossless exports."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..readwrite import atomic_write_json
from .raw_intermediate_manifest import artifact_checksum

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
BUNDLE_INPUT_MANIFEST_FILENAME = "inputs.csv"
BUNDLE_IDENTITY_MAP_FILENAME = "identity_map.csv"
BUNDLE_KINDS = frozenset({"sequence_only", "lossless_bam"})


class ExportBundleError(ValueError):
    """Raised when an export bundle is incomplete, unsafe, or corrupt."""


def _safe_relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ExportBundleError(f"Bundle artifact is outside the bundle root: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise ExportBundleError(f"Unsafe bundle artifact path: {path}")
    return relative.as_posix()


def artifact_record(root: str | Path, path: str | Path, **metadata: Any) -> dict[str, Any]:
    """Return a portable checksummed record for one bundle file."""
    root = Path(root)
    path = Path(path)
    if not path.is_file():
        raise ExportBundleError(f"Bundle artifact is missing or not a file: {path}")
    return {
        "path": _safe_relative(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": artifact_checksum(path),
        **metadata,
    }


def write_bundle_manifest(
    root: str | Path,
    *,
    bundle_kind: str,
    scope: str,
    modality: str,
    selection: Mapping[str, Any],
    source_generations: Iterable[Mapping[str, Any]],
    artifacts: Iterable[Mapping[str, Any]],
    lost_capabilities: Iterable[str],
) -> Path:
    """Write and validate the authoritative bundle manifest as the final artifact."""
    root = Path(root)
    if bundle_kind not in BUNDLE_KINDS:
        raise ExportBundleError(f"Unsupported bundle kind: {bundle_kind!r}")
    input_manifest = root / BUNDLE_INPUT_MANIFEST_FILENAME
    identity_map = root / BUNDLE_IDENTITY_MAP_FILENAME
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "state": "complete",
        "bundle_kind": bundle_kind,
        "scope": str(scope),
        "modality": str(modality or "unknown"),
        "selection": dict(selection),
        "source_generations": [dict(item) for item in source_generations],
        "lost_capabilities": sorted({str(item) for item in lost_capabilities}),
        "canonical_input_manifest": artifact_record(root, input_manifest),
        "identity_map": artifact_record(root, identity_map),
        "artifacts": [dict(item) for item in artifacts],
    }
    path = root / BUNDLE_MANIFEST_FILENAME
    atomic_write_json(path, payload)
    read_bundle_manifest(path)
    return path


def _resolve_record(root: Path, record: Mapping[str, Any]) -> Path:
    relative = Path(str(record.get("path", "")))
    resolved = (root / relative).resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not resolved.is_relative_to(root.resolve())
    ):
        raise ExportBundleError("Bundle manifest contains an unsafe artifact path")
    if not resolved.is_file():
        raise ExportBundleError(f"Bundle artifact is missing: {resolved}")
    if resolved.stat().st_size != int(record.get("size_bytes", -1)):
        raise ExportBundleError(f"Bundle artifact size mismatch: {resolved}")
    if artifact_checksum(resolved) != str(record.get("sha256", "")):
        raise ExportBundleError(f"Bundle artifact checksum mismatch: {resolved}")
    return resolved


def read_bundle_manifest(path: str | Path) -> dict[str, Any]:
    """Read a complete bundle and validate every advertised artifact checksum."""
    path = Path(path)
    if path.is_dir():
        path = path / BUNDLE_MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ExportBundleError("Unsupported export-bundle schema version")
        if payload.get("state") != "complete" or payload.get("bundle_kind") not in BUNDLE_KINDS:
            raise ExportBundleError("Export bundle is not complete")
        root = path.parent
        for key in ("canonical_input_manifest", "identity_map"):
            record = payload.get(key)
            if not isinstance(record, dict):
                raise ExportBundleError(f"Export bundle lacks {key}")
            _resolve_record(root, record)
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ExportBundleError("Export bundle contains no input artifacts")
        for record in artifacts:
            if not isinstance(record, dict):
                raise ExportBundleError("Export bundle artifact record is invalid")
            _resolve_record(root, record)
    except ExportBundleError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExportBundleError(f"Invalid export bundle {path}: {exc}") from exc
    return payload


def resolve_bundle_input_manifest(path: str | Path) -> Path:
    """Validate a bundle JSON and return its canonical relative input manifest."""
    path = Path(path)
    payload = read_bundle_manifest(path)
    return _resolve_record(path.parent, payload["canonical_input_manifest"])


def _atomic_write_rows(
    path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_identity_map(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write the reversible bundle-to-source identity mapping."""
    path = Path(path)
    records = [dict(row) for row in rows]
    fields = [
        "bundle_read_id",
        "bundle_template_id",
        "experiment_id",
        "experiment_uid",
        "namespace",
        "source_read_id",
        "template_id",
        "molecule_uid",
        "segment_id",
        "segment_uid",
        "mate",
        "sample",
        "barcode",
        "artifact_path",
    ]
    _atomic_write_rows(path, fields, ({key: row.get(key, "") for key in fields} for row in records))
    return path


def write_canonical_input_manifest(
    path: str | Path,
    declarations: Iterable[Mapping[str, Any]],
    *,
    alignment_mode: str,
    modality: str,
) -> Path:
    """Write full schema-1 rows with relative paths and verified source identities."""
    from .input_manifest import _CSV_COLUMNS, resolve_input_manifest_readonly

    path = Path(path)
    initial = [dict(row) for row in declarations]
    _atomic_write_rows(path, list(_CSV_COLUMNS), initial)
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=path,
        alignment_mode=alignment_mode,
        modality=modality,
    )
    relative_by_resolved = {
        str((path.parent / str(row["path"])).resolve()): str(row["path"]) for row in initial
    }
    canonical = []
    for row in resolved.rows:
        record = row.csv_record()
        record["path"] = relative_by_resolved[row.path]
        canonical.append(record)
    _atomic_write_rows(path, list(_CSV_COLUMNS), canonical)
    return path
