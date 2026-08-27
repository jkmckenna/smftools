from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.data import volume_discovery
from smftools.data.volume_discovery import discover_volumes, platform_mount_root_candidates
from smftools.data.volume_stamp import init_volume

pytestmark = pytest.mark.unit


def _stamped(base: Path, name: str, *, label: str | None = None, kind: str = "archive") -> Path:
    volume_dir = base / name
    volume_dir.mkdir(parents=True)
    init_volume(volume_dir, label=label or name, kind=kind)
    return volume_dir


def test_finds_stamped_volumes_under_a_mount_root(tmp_path):
    volumes_root = tmp_path / "Volumes"
    volumes_root.mkdir()
    a = _stamped(volumes_root, "archive-01", kind="archive")
    _unstamped = volumes_root / "not-a-smftools-volume"
    _unstamped.mkdir()

    found = discover_volumes(mount_roots=[volumes_root], extra_paths=[])

    assert len(found) == 1
    assert found[0].mount_path == a
    assert found[0].stamp.label == "archive-01"


def test_extra_paths_are_checked_directly_not_enumerated(tmp_path):
    network_mount = tmp_path / "net" / "lab-storage"
    network_mount.mkdir(parents=True)
    init_volume(network_mount, label="lab-storage", kind="working")

    found = discover_volumes(mount_roots=[], extra_paths=[network_mount])

    assert len(found) == 1
    assert found[0].mount_path == network_mount


def test_no_mount_roots_or_extra_paths_finds_nothing(tmp_path):
    assert discover_volumes(mount_roots=[], extra_paths=[]) == []


def test_unreadable_stamp_is_skipped_with_a_warning_not_raised(tmp_path, caplog):
    volumes_root = tmp_path / "Volumes"
    volumes_root.mkdir()
    good = _stamped(volumes_root, "archive-01")
    corrupt = volumes_root / "corrupt-drive"
    corrupt.mkdir()
    (corrupt / ".smftools-volume.json").write_text("{not json", encoding="utf-8")

    found = discover_volumes(mount_roots=[volumes_root], extra_paths=[])

    assert [item.mount_path for item in found] == [good]


def test_same_volume_id_attached_twice_keeps_the_first(tmp_path):
    volumes_root = tmp_path / "Volumes"
    volumes_root.mkdir()
    first = _stamped(volumes_root, "copy-a")
    stamp_payload = json.loads((first / ".smftools-volume.json").read_text())

    second = volumes_root / "copy-b"
    second.mkdir()
    (second / ".smftools-volume.json").write_text(json.dumps(stamp_payload), encoding="utf-8")

    found = discover_volumes(mount_roots=[volumes_root], extra_paths=[])

    assert len(found) == 1
    assert found[0].mount_path == first


def test_config_dir_reaches_configured_extra_search_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("SMFTOOLS_VOLUME_SEARCH_PATHS", raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))

    network_mount = tmp_path / "net" / "lab-storage"
    network_mount.mkdir(parents=True)
    init_volume(network_mount, label="lab-storage", kind="working")

    config_dir = tmp_path / "lab"
    config_dir.mkdir()
    (config_dir / "roots.toml").write_text(
        f'[volumes]\nextra_search_paths = ["{network_mount}"]\n', encoding="utf-8"
    )

    found = discover_volumes(mount_roots=[], config_dir=config_dir)

    assert len(found) == 1
    assert found[0].mount_path == network_mount


def test_platform_mount_root_candidates_expands_depth_two_roots(tmp_path, monkeypatch):
    """A depth-2 convention (e.g. /media/<user>) expands to its existing users."""
    media_root = tmp_path / "media"
    (media_root / "alice").mkdir(parents=True)
    (media_root / "bob").mkdir(parents=True)
    volumes_root = tmp_path / "Volumes"
    volumes_root.mkdir()

    monkeypatch.setattr(
        volume_discovery,
        "MOUNT_ROOTS",
        (
            ((str(volumes_root),), 1),
            (tuple(media_root.parts), 2),
        ),
    )

    candidates = platform_mount_root_candidates()

    assert volumes_root in candidates
    assert media_root / "alice" in candidates
    assert media_root / "bob" in candidates
    assert media_root not in candidates
