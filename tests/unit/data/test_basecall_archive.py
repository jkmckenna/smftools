"""Write a basecall generation back to its POD5 archive (`BCS-08`, `BCS-09`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.data.basecall_archive import (
    ARCHIVE_MANIFEST_FILENAME,
    BASECALLS_ARCHIVE_DIRNAME,
    BasecallArchiveError,
    archive_basecall_generation,
    enclosing_volume_id,
)
from smftools.data.volume_stamp import init_volume
from smftools.informatics.basecall_generation import publish_basecall_generation

pytestmark = pytest.mark.unit


def _seed_generation(run_root: Path, *, model: str = "hac", dorado_version: str = "1.0.0") -> str:
    bam_source = run_root / "source.bam"
    bam_source.write_bytes(b"fake-bam-bytes")
    outputs = publish_basecall_generation(
        run_root,
        bam_path=bam_source,
        model=model,
        modality="canonical",
        config_hash="hash-1",
        input_artifact_ids=["input-manifest:digest", "source:a:sha"],
        dorado_version=dorado_version,
    )
    return str(outputs["generation_id"])


def test_enclosing_volume_id_finds_the_nearest_stamped_ancestor(tmp_path: Path) -> None:
    stamp, _ = init_volume(tmp_path, label="archive-01", kind="archive")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert enclosing_volume_id(nested) == stamp.volume_id


def test_enclosing_volume_id_returns_none_when_unstamped(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert enclosing_volume_id(nested) is None


def test_archive_basecall_generation_refuses_without_a_current_generation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(BasecallArchiveError, match="no current basecall generation"):
        archive_basecall_generation(run_root, archive_root=tmp_path / "archive")


def test_archive_basecall_generation_copies_bam_and_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    generation_id = _seed_generation(run_root, model="hac", dorado_version="1.0.0")
    archive_root = tmp_path / "archive"

    result = archive_basecall_generation(run_root, archive_root=archive_root)

    assert result["status"] == "archived"
    assert result["generation_id"] == generation_id
    dest_dir = archive_root / BASECALLS_ARCHIVE_DIRNAME / "hac@1.0.0"
    assert result["path"] == dest_dir
    manifest = json.loads((dest_dir / ARCHIVE_MANIFEST_FILENAME).read_text())
    assert manifest["generation_id"] == generation_id
    assert manifest["model"] == "hac"
    bam_files = list(dest_dir.glob("*.bam"))
    assert len(bam_files) == 1
    assert bam_files[0].read_bytes() == b"fake-bam-bytes"
    assert not list(dest_dir.glob("*.partial"))


def test_archive_basecall_generation_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _seed_generation(run_root)
    archive_root = tmp_path / "archive"

    first = archive_basecall_generation(run_root, archive_root=archive_root)
    second = archive_basecall_generation(run_root, archive_root=archive_root)

    assert first["status"] == "archived"
    assert second["status"] == "already_archived"
    assert second["generation_id"] == first["generation_id"]
    bam_files = list(Path(second["path"]).glob("*.bam"))
    assert len(bam_files) == 1  # no duplicate written


def test_archive_basecall_generation_recopies_a_corrupted_destination(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _seed_generation(run_root)
    archive_root = tmp_path / "archive"
    first = archive_basecall_generation(run_root, archive_root=archive_root)

    bam_files = list(Path(first["path"]).glob("*.bam"))
    bam_files[0].write_bytes(b"corrupted")

    result = archive_basecall_generation(run_root, archive_root=archive_root)

    assert result["status"] == "archived"
    bam_files = list(Path(result["path"]).glob("*.bam"))
    assert bam_files[0].read_bytes() == b"fake-bam-bytes"


def test_archive_basecall_generation_reports_same_volume(tmp_path: Path) -> None:
    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    init_volume(volume_root, label="shared", kind="archive")
    run_root = volume_root / "run"
    run_root.mkdir()
    _seed_generation(run_root)
    archive_root = volume_root / "archive"

    result = archive_basecall_generation(run_root, archive_root=archive_root)

    assert result["same_volume"] is True


def test_archive_basecall_generation_reports_different_volume(tmp_path: Path) -> None:
    source_volume = tmp_path / "source_volume"
    source_volume.mkdir()
    init_volume(source_volume, label="source", kind="working")
    dest_volume = tmp_path / "dest_volume"
    dest_volume.mkdir()
    init_volume(dest_volume, label="dest", kind="archive")
    run_root = source_volume / "run"
    run_root.mkdir()
    _seed_generation(run_root)
    archive_root = dest_volume / "archive"

    result = archive_basecall_generation(run_root, archive_root=archive_root)

    assert result["same_volume"] is False


def test_archive_basecall_generation_reports_none_when_unstamped(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _seed_generation(run_root)

    result = archive_basecall_generation(run_root, archive_root=tmp_path / "archive")

    assert result["same_volume"] is None
