from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.data.replica_catalog import replicas_for
from smftools.data.volume_scan import scan_and_catalog, scan_volume
from smftools.data.volume_stamp import init_volume
from smftools.informatics.input_manifest import resolve_input_manifest

pytestmark = pytest.mark.unit


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _publish_run(run_root: Path, *, source_name: str = "sample.fastq") -> str:
    """Publish a real, valid resolved input manifest under `run_root`; return its digest."""
    source = _write(
        run_root.parent / f"_sources_{run_root.name}" / source_name, b"@one\nAC\n+\n!!\n"
    )
    result = resolve_input_manifest(output_directory=run_root, input_paths=[source])
    return result.digest


def test_scan_volume_finds_a_published_run(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    run_root = mount / "analyses" / "exp1"
    digest = _publish_run(run_root)

    found = scan_volume(mount)

    assert len(found) == 1
    assert found[0].relative_path == "analyses/exp1"
    assert found[0].digest == digest
    assert found[0].warning is None


def test_scan_volume_finds_nothing_in_an_unpublished_tree(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    (mount / "some" / "empty" / "tree").mkdir(parents=True)

    assert scan_volume(mount) == []


def test_scan_volume_prunes_generations_directories(tmp_path: Path) -> None:
    """A manifest nested under a `generations/` dir must never be reached."""
    mount = tmp_path / "mount"
    buried = mount / "generations" / "fake-id" / "raw_outputs" / "input_manifest"
    buried.mkdir(parents=True)
    # Deliberately invalid -- if this were ever read, it would surface as a
    # warning rather than silently vanish, so this proves pruning, not luck.
    (buried / "resolved_input_manifest.json").write_text("not json", encoding="utf-8")

    assert scan_volume(mount) == []


def test_scan_volume_reports_unreadable_manifest_as_a_warning(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    manifest_dir = mount / "exp1" / "raw_outputs" / "input_manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "resolved_input_manifest.json").write_text("not json", encoding="utf-8")

    found = scan_volume(mount)

    assert len(found) == 1
    assert found[0].digest == ""
    assert found[0].warning is not None


def test_scan_volume_finds_multiple_runs_sorted_by_path(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    _publish_run(mount / "b_exp", source_name="b.fastq")
    _publish_run(mount / "a_exp", source_name="a.fastq")

    found = scan_volume(mount)

    assert [run.relative_path for run in found] == ["a_exp", "b_exp"]


def test_scan_and_catalog_requires_a_stamped_mount(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()

    with pytest.raises(ValueError, match="has not been stamped"):
        scan_and_catalog(mount, {})


def test_scan_and_catalog_registers_a_replica(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    stamp, _ = init_volume(mount, label="archive-01", kind="archive")
    digest = _publish_run(mount / "exp1")

    catalog, runs = scan_and_catalog(mount, {})

    assert len(runs) == 1
    replicas = replicas_for(catalog, digest)
    assert len(replicas) == 1
    assert replicas[0].volume_id == stamp.volume_id
    assert replicas[0].path == "exp1"


def test_scan_and_catalog_skips_unreadable_manifests(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    manifest_dir = mount / "exp1" / "raw_outputs" / "input_manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "resolved_input_manifest.json").write_text("not json", encoding="utf-8")

    catalog, runs = scan_and_catalog(mount, {})

    assert len(runs) == 1
    assert catalog == {}


def test_scan_and_catalog_does_not_mutate_the_input_catalog(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    _publish_run(mount / "exp1")
    original: dict = {}

    scan_and_catalog(mount, original)

    assert original == {}
