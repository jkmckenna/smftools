from __future__ import annotations

from pathlib import Path

import pytest

from smftools.data.lab_init import scaffold_lab_root

pytestmark = pytest.mark.unit


def test_scaffold_creates_the_documented_layout(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"

    created = scaffold_lab_root(lab_root)

    assert (lab_root / "data").is_dir()
    assert (lab_root / "analyses" / "runs").is_dir()
    assert (lab_root / "analyses" / "projects").is_dir()
    assert set(created) == {
        lab_root / "data",
        lab_root / "analyses" / "runs",
        lab_root / "analyses" / "projects",
    }


def test_scaffold_is_idempotent(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    scaffold_lab_root(lab_root)

    second_pass = scaffold_lab_root(lab_root)

    assert second_pass == []


def test_scaffold_never_touches_existing_data(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    (lab_root / "data" / "run1").mkdir(parents=True)
    marker = lab_root / "data" / "run1" / "pod5_pass" / "reads.pod5"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"signal")

    scaffold_lab_root(lab_root)

    assert marker.read_bytes() == b"signal"


def test_scaffold_fills_in_only_whats_missing(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    (lab_root / "data").mkdir(parents=True)

    created = scaffold_lab_root(lab_root)

    assert created == [lab_root / "analyses" / "runs", lab_root / "analyses" / "projects"]
