from __future__ import annotations

from pathlib import Path

import pytest

from smftools.data.replica_catalog import Replica, ResolvedReplica
from smftools.data.volume_verify import (
    STATUS_MISMATCH,
    STATUS_OK,
    STATUS_UNREACHABLE,
    verify_replica,
)
from smftools.informatics.input_manifest import InputManifestError, resolve_input_manifest

pytestmark = pytest.mark.unit


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _published_replica(tmp_path: Path, run_name: str = "exp1") -> tuple[ResolvedReplica, Path]:
    source = _write(tmp_path / "sources" / "sample.fastq", b"@one\nAC\n+\n!!\n")
    run_root = tmp_path / run_name
    resolve_input_manifest(output_directory=run_root, input_paths=[source])
    replica = Replica(volume_id="vol-1", path=".", digest="ignored", verified_at="ignored")
    resolved = ResolvedReplica(replica=replica, mount_path=run_root)
    return resolved, source


def test_verify_replica_all_sources_ok(tmp_path: Path) -> None:
    resolved, _ = _published_replica(tmp_path)

    outcome = verify_replica(resolved)

    assert outcome.ok is True
    assert outcome.mismatch_count == 0
    assert outcome.unreachable_count == 0
    assert all(row.status == STATUS_OK for row in outcome.rows)


def test_verify_replica_detects_a_mismatch(tmp_path: Path) -> None:
    resolved, source = _published_replica(tmp_path)
    source.write_bytes(b"corrupted bytes, different content entirely")

    outcome = verify_replica(resolved)

    assert outcome.ok is False
    assert outcome.mismatch_count == 1
    assert outcome.rows[0].status == STATUS_MISMATCH
    assert outcome.rows[0].actual_sha256 != outcome.rows[0].expected_sha256


def test_verify_replica_reports_an_unreachable_source_without_failing(tmp_path: Path) -> None:
    resolved, source = _published_replica(tmp_path)
    source.unlink()

    outcome = verify_replica(resolved)

    assert outcome.ok is True  # unreachable is not a mismatch
    assert outcome.unreachable_count == 1
    assert outcome.rows[0].status == STATUS_UNREACHABLE
    assert outcome.rows[0].actual_sha256 is None


def test_verify_replica_raises_when_the_manifest_is_missing(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    replica = Replica(volume_id="vol-1", path=".", digest="x", verified_at="x")
    resolved = ResolvedReplica(replica=replica, mount_path=empty_root)

    with pytest.raises(InputManifestError):
        verify_replica(resolved)
