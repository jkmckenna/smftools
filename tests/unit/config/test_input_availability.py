"""Raw input may be archived to a detached volume once a run's outputs exist.

`PSR-01`: config loading used to discover input files eagerly, so an absent path
raised and *every* stage command failed the moment an archive drive was
unplugged -- including stages that read no raw input at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from smftools.config.discover_input_files import discover_input_files, input_kind_for_path
from smftools.config.input_availability import (
    INPUT_MISSING,
    INPUT_OFFLINE,
    INPUT_PRESENT,
    InputAvailability,
    detached_volume_for,
    require_input_available,
    resolve_input_availability,
    restore_offline_input_identity,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_volume_env(tmp_path, monkeypatch):
    """Never let a `PSR-12` exact-availability test see this machine's real volumes/config."""
    from smftools.data import volume_discovery

    monkeypatch.setattr(volume_discovery, "platform_mount_root_candidates", lambda: [])
    monkeypatch.delenv("SMFTOOLS_VOLUME_SEARCH_PATHS", raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))


def test_existing_path_is_present(tmp_path):
    assert resolve_input_availability(tmp_path).state == INPUT_PRESENT


def test_absent_path_on_an_attached_filesystem_is_missing(tmp_path):
    availability = resolve_input_availability(tmp_path / "not-here")
    assert availability.state == INPUT_MISSING
    assert availability.volume is None


@pytest.mark.parametrize(
    ("path", "volume"),
    [
        ("/Volumes/ArchiveDrive/run/pod5", "/Volumes/ArchiveDrive"),
        ("/mnt/archive01/run/pod5", "/mnt/archive01"),
        ("/media/someone/Archive/run/pod5", "/media/someone/Archive"),
        ("/run/media/someone/Archive/run/pod5", "/run/media/someone/Archive"),
    ],
)
def test_path_on_a_detached_volume_is_offline(path, volume):
    """Every platform's removable-mount convention must be recognised."""
    availability = resolve_input_availability(path)
    assert availability.state == INPUT_OFFLINE
    assert availability.volume == Path(volume)


def test_mount_root_itself_is_not_treated_as_a_volume():
    """`/Volumes` exists on macOS; a bare mount root names no volume to attach."""
    assert detached_volume_for(Path("/Volumes")) is None


def test_offline_input_names_the_volume_to_attach():
    availability = resolve_input_availability("/Volumes/ArchiveDrive/run/pod5")
    with pytest.raises(FileNotFoundError, match="ArchiveDrive"):
        require_input_available(availability, stage="raw")


def test_missing_input_is_not_reported_as_archived(tmp_path):
    """The two states must read differently, or a typo looks like an unplugged drive."""
    availability = resolve_input_availability(tmp_path / "typo")
    with pytest.raises(FileNotFoundError, match="mistyped") as excinfo:
        require_input_available(availability, stage="raw")
    assert "not attached" not in str(excinfo.value)


def test_present_input_does_not_raise(tmp_path):
    require_input_available(InputAvailability(state=INPUT_PRESENT, path=tmp_path), stage="raw")


def _write_resolved_manifest(run_root: Path, source_paths: list[str]) -> None:
    manifest_dir = run_root / "raw_outputs" / "input_manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "resolved_input_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_directory": "/Volumes/ArchiveDrive/run",
                "source_count": len(source_paths),
                "sources": [{"path": p} for p in source_paths],
            }
        ),
        encoding="utf-8",
    )


def test_offline_identity_is_restored_from_the_resolved_manifest(tmp_path):
    """`input_type`/`input_files` feed the stage config hash.

    Left unset while the volume is detached, a finished raw generation would look
    incompatible and every downstream stage would be sent back to ingestion for
    data it cannot reach.
    """
    _write_resolved_manifest(
        tmp_path,
        [
            "/Volumes/ArchiveDrive/run/fastq_pass/bc01/reads_0.fastq.gz",
            "/Volumes/ArchiveDrive/run/fastq_pass/bc02/reads_0.fastq.gz",
        ],
    )
    restored = restore_offline_input_identity(tmp_path, bam_suffix=".bam")
    assert restored is not None
    input_type, input_files = restored
    assert input_type == "fastq"
    assert len(input_files) == 2


def test_restoration_returns_none_without_a_recorded_manifest(tmp_path):
    assert restore_offline_input_identity(tmp_path, bam_suffix=".bam") is None
    assert restore_offline_input_identity(None, bam_suffix=".bam") is None


def test_restoration_declines_mixed_recorded_types(tmp_path):
    """Better to report an unknown identity than to assert a wrong one."""
    _write_resolved_manifest(
        tmp_path,
        [
            "/Volumes/ArchiveDrive/run/a.fastq.gz",
            "/Volumes/ArchiveDrive/run/b.pod5",
        ],
    )
    assert restore_offline_input_identity(tmp_path, bam_suffix=".bam") is None


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("reads.fastq.gz", "fastq"),
        ("reads.fq.xz", "fastq"),
        ("signal.pod5", "pod5"),
        ("signal.fast5", "fast5"),
        ("aln.bam", "bam"),
        ("aln.cram", "cram"),
        ("aln.sam", "sam"),
        ("cache.h5ad", "h5ad"),
        ("notes.txt", "other"),
    ],
)
def test_classifier_agrees_with_directory_discovery(tmp_path, name, kind):
    """Restoration and discovery must classify identically.

    They are the two ways `input_type` is produced -- once from a live directory
    scan and once from recorded paths -- and any disagreement would silently move
    a stage's config hash between an attached and a detached run.
    """
    (tmp_path / name).write_bytes(b"")
    assert input_kind_for_path(tmp_path / name) == kind
    found = discover_input_files(tmp_path)
    bucket = found[f"{kind}_paths"] if kind != "other" else found["other_paths"]
    assert [p.name for p in bucket] == [name]


# --- PSR-12: exact classification via volume identity -----------------------


def _publish_manifest(run_root: Path, source: Path) -> str:
    from smftools.informatics.input_manifest import resolve_input_manifest

    return resolve_input_manifest(output_directory=run_root, input_paths=[source]).digest


def test_exact_check_is_skipped_without_output_directory(tmp_path):
    """`output_directory` omitted must behave exactly like before `PSR-12`."""
    availability = resolve_input_availability(tmp_path / "not-here")
    assert availability.state == INPUT_MISSING


def test_exact_check_defers_to_the_approximation_without_a_published_manifest(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    availability = resolve_input_availability(
        "/Volumes/ArchiveDrive/run/pod5", output_directory=run_root
    )

    assert availability.state == INPUT_OFFLINE
    assert availability.volume == Path("/Volumes/ArchiveDrive")


def test_exact_check_defers_to_the_approximation_without_a_catalog(tmp_path):
    run_root = tmp_path / "run"
    source = tmp_path / "orig" / "sample.fastq"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"@r\nAC\n+\n!!\n")
    _publish_manifest(run_root, source)
    source.unlink()

    availability = resolve_input_availability(source, output_directory=run_root)

    # No catalog exists at all, so this falls back to the structural guess,
    # which does not recognize this path as a volume convention.
    assert availability.state == INPUT_MISSING


def test_exact_offline_with_no_attached_replica(tmp_path):
    from smftools.data.replica_catalog import add_replica, save_catalog

    run_root = tmp_path / "run"
    source = tmp_path / "orig" / "sample.fastq"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"@r\nAC\n+\n!!\n")
    digest = _publish_manifest(run_root, source)
    source.unlink()

    catalog = add_replica({}, digest, volume_id="archive-vol-1", path="orig")
    save_catalog(catalog)  # default location, isolated by SMFTOOLS_CONFIG_DIR above

    availability = resolve_input_availability(source, output_directory=run_root)

    assert availability.state == INPUT_OFFLINE
    assert availability.volume_id == "archive-vol-1"
    assert "archive-vol-1" in availability.detail


def test_exact_offline_beats_the_approximation_for_a_nonstandard_mount(tmp_path):
    """A network share with no recognized mount convention was wrongly `missing`."""
    from smftools.data.replica_catalog import add_replica, save_catalog

    run_root = tmp_path / "run"
    source = tmp_path / "custom-nas-share" / "data" / "sample.fastq"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"@r\nAC\n+\n!!\n")
    digest = _publish_manifest(run_root, source)
    shutil.rmtree(tmp_path / "custom-nas-share")

    catalog = add_replica({}, digest, volume_id="nas-vol", path="data")
    save_catalog(catalog)

    availability = resolve_input_availability(source, output_directory=run_root)

    assert availability.state == INPUT_OFFLINE  # not "missing", despite no mount-root match


def test_exact_present_when_the_volume_reattached_elsewhere(tmp_path, monkeypatch):
    """The dataset's volume moved to a different mount point/name (`PSR-08`/`PSR-09`)."""
    from smftools.data.replica_catalog import add_replica, save_catalog
    from smftools.data.volume_stamp import init_volume

    run_root = tmp_path / "run"
    original_base = tmp_path / "orig_mount" / "run1"
    source = original_base / "sample.fastq"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"@r\nAC\n+\n!!\n")
    digest = _publish_manifest(run_root, source)

    new_mount = tmp_path / "new_mount"
    new_mount.mkdir()
    stamp, _ = init_volume(new_mount, label="renamed-drive", kind="archive")
    shutil.copytree(original_base, new_mount / "run1")
    shutil.rmtree(tmp_path / "orig_mount")  # the original mount point is gone

    catalog = add_replica({}, digest, volume_id=stamp.volume_id, path="run1")
    save_catalog(catalog)
    monkeypatch.setenv("SMFTOOLS_VOLUME_SEARCH_PATHS", str(new_mount))

    availability = resolve_input_availability(source, output_directory=run_root)

    assert availability.state == INPUT_PRESENT
    assert availability.path == new_mount / "run1" / "sample.fastq"
