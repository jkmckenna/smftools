"""Raw input may be archived to a detached volume once a run's outputs exist.

`PSR-01`: config loading used to discover input files eagerly, so an absent path
raised and *every* stage command failed the moment an archive drive was
unplugged -- including stages that read no raw input at all.
"""

from __future__ import annotations

import json
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
