"""`data verify`: re-checksum a replica's declared raw sources (`PSR-11`).

A catalog entry only proves that a run's input manifest, at scan time,
self-consistently named a digest -- `read_resolved_input_manifest` already
checks that. It says nothing about whether the raw bytes the manifest
describes still match today. This recomputes each declared source's SHA-256
directly from the file, deliberately bypassing
`smftools.informatics.input_manifest`'s stat-signature cache: that cache
exists to make repeat ingestion cheap, which is exactly the shortcut a
verification step must not take, since a file corrupted without its mtime
changing is precisely the failure mode checksums exist to catch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..constants import RAW_DIR
from ..informatics.input_manifest import INPUT_MANIFEST_DIRNAME, RESOLVED_INPUT_MANIFEST_JSON
from ..informatics.input_manifest import (
    read_resolved_input_manifest as _read_resolved_input_manifest,
)
from .replica_catalog import ResolvedReplica

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNREACHABLE = "unreachable"

_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class VerifyRow:
    """The verification outcome for one declared source file."""

    path: str
    status: str
    expected_sha256: str
    actual_sha256: Optional[str] = None


@dataclass(frozen=True)
class VerifyResult:
    """Every declared source's verification outcome for one replica."""

    replica: ResolvedReplica
    manifest_digest: str
    rows: tuple[VerifyRow, ...]

    @property
    def ok(self) -> bool:
        """Whether every reachable source matched its recorded checksum."""
        return all(row.status != STATUS_MISMATCH for row in self.rows)

    @property
    def unreachable_count(self) -> int:
        return sum(1 for row in self.rows if row.status == STATUS_UNREACHABLE)

    @property
    def mismatch_count(self) -> int:
        return sum(1 for row in self.rows if row.status == STATUS_MISMATCH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(replica_root: str | Path) -> Path:
    """Where a replica's resolved input manifest lives, given its run root."""
    return Path(replica_root) / RAW_DIR / INPUT_MANIFEST_DIRNAME / RESOLVED_INPUT_MANIFEST_JSON


def verify_replica(resolved: ResolvedReplica) -> VerifyResult:
    """Re-checksum every reachable source declared by `resolved`'s manifest.

    A declared source that does not currently exist at its recorded path is
    reported as `unreachable`, not `mismatch` -- raw input being archived
    elsewhere is the expected, common case (`PSR-01`), not a verification
    failure.

    Raises:
        InputManifestError: The replica's manifest is missing or unreadable.
    """
    manifest_path = manifest_path_for(resolved.resolved_path)
    manifest = _read_resolved_input_manifest(manifest_path)

    rows: list[VerifyRow] = []
    for row in manifest.rows:
        source_path = Path(row.path)
        if not source_path.is_file():
            rows.append(
                VerifyRow(path=row.path, status=STATUS_UNREACHABLE, expected_sha256=row.sha256)
            )
            continue
        actual = _sha256_file(source_path)
        status = STATUS_OK if actual == row.sha256 else STATUS_MISMATCH
        rows.append(
            VerifyRow(
                path=row.path, status=status, expected_sha256=row.sha256, actual_sha256=actual
            )
        )
    return VerifyResult(replica=resolved, manifest_digest=manifest.digest, rows=tuple(rows))
