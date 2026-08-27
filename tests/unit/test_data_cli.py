from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.data.volume_stamp import STAMP_FILENAME, read_volume_stamp

pytestmark = pytest.mark.unit


def test_init_volume_cli_stamps_a_fresh_mount(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli, ["data", "init-volume", str(tmp_path), "--label", "archive-01", "--kind", "archive"]
    )

    assert result.exit_code == 0, result.output
    assert "Stamped" in result.output
    stamp = read_volume_stamp(tmp_path)
    assert stamp is not None
    assert stamp.label == "archive-01"
    assert stamp.kind == "archive"


def test_init_volume_cli_rerun_reports_existing_identity_and_warns(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["data", "init-volume", str(tmp_path), "--label", "archive-01"])

    result = runner.invoke(
        cli, ["data", "init-volume", str(tmp_path), "--label", "renamed", "--kind", "working"]
    )

    assert result.exit_code == 0, result.output
    assert "already stamped" in result.output
    assert "WARNING" in result.output
    stamp = read_volume_stamp(tmp_path)
    assert stamp is not None
    assert stamp.label == "archive-01"


def test_init_volume_cli_rejects_missing_mount() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "init-volume", "/no/such/mount", "--label", "x"])

    assert result.exit_code != 0


def test_init_volume_cli_requires_label(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "init-volume", str(tmp_path)])

    assert result.exit_code != 0
    assert "--label" in result.output or "Missing option" in result.output
