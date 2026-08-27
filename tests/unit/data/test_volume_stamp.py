from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from smftools.data.volume_stamp import STAMP_FILENAME, init_volume, read_volume_stamp

pytestmark = pytest.mark.unit


def test_init_volume_writes_a_stamp_with_a_fresh_id(tmp_path: Path) -> None:
    stamp, created = init_volume(tmp_path, label="archive-01", kind="archive")

    assert created is True
    assert stamp.label == "archive-01"
    assert stamp.kind == "archive"
    assert stamp.volume_id
    on_disk = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert on_disk == stamp.to_dict()


def test_read_volume_stamp_returns_none_when_unstamped(tmp_path: Path) -> None:
    assert read_volume_stamp(tmp_path) is None


def test_reinit_is_idempotent_and_keeps_the_original_identity(tmp_path: Path) -> None:
    first, _ = init_volume(tmp_path, label="archive-01", kind="archive")

    second, created = init_volume(tmp_path, label="renamed-drive", kind="backup")

    assert created is False
    assert second == first
    assert second.label == "archive-01"
    assert second.kind == "archive"


def test_stamp_is_written_once_never_rewritten_on_disk(tmp_path: Path) -> None:
    init_volume(tmp_path, label="archive-01", kind="archive")
    stamp_path = tmp_path / STAMP_FILENAME
    original_bytes = stamp_path.read_bytes()

    init_volume(tmp_path, label="a-different-name", kind="working")

    assert stamp_path.read_bytes() == original_bytes


def test_reattached_volume_keeps_its_identity_across_relabeling(tmp_path: Path) -> None:
    original_mount = tmp_path / "orig-mount"
    original_mount.mkdir()
    stamp, _ = init_volume(original_mount, label="workbench-ssd", kind="working")

    # Simulate the drive being unplugged, renamed at the OS level, and
    # reattached at a different mount path -- the stamp file travels with it
    # unchanged, since nothing about volume_id is derived from mount point or
    # OS-reported label.
    relocated_mount = tmp_path / "renamed-mount"
    shutil.copytree(original_mount, relocated_mount)

    reread = read_volume_stamp(relocated_mount)

    assert reread is not None
    assert reread.volume_id == stamp.volume_id
    assert reread.created == stamp.created


def test_init_volume_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown volume kind"):
        init_volume(tmp_path, label="archive-01", kind="not-a-real-kind")


def test_init_volume_requires_an_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "not-there"

    with pytest.raises(FileNotFoundError):
        init_volume(missing, label="archive-01", kind="archive")


def test_read_volume_stamp_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / STAMP_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_volume_stamp(tmp_path)


def test_read_volume_stamp_rejects_missing_fields(tmp_path: Path) -> None:
    (tmp_path / STAMP_FILENAME).write_text(json.dumps({"volume_id": "abc"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing field"):
        read_volume_stamp(tmp_path)
