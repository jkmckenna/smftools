from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.config.roots import ENV_VOLUME_SEARCH_PATHS
from smftools.data import volume_discovery
from smftools.data.volume_stamp import STAMP_FILENAME, init_volume, read_volume_stamp

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_real_mount_roots(monkeypatch):
    """Never let a `data volumes` test see this machine's actual mounted drives."""
    monkeypatch.setattr(volume_discovery, "platform_mount_root_candidates", lambda: [])
    monkeypatch.delenv(ENV_VOLUME_SEARCH_PATHS, raising=False)


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


def test_volumes_cli_reports_nothing_attached() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "volumes"])

    assert result.exit_code == 0, result.output
    assert "No stamped volumes" in result.output


def test_volumes_cli_lists_a_discoverable_volume(tmp_path: Path, monkeypatch) -> None:
    network_mount = tmp_path / "lab-storage"
    network_mount.mkdir()
    init_volume(network_mount, label="lab-storage", kind="working")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(network_mount))
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "volumes"])

    assert result.exit_code == 0, result.output
    assert "lab-storage" in result.output
    assert str(network_mount) in result.output


def test_volumes_cli_json_output(tmp_path: Path, monkeypatch) -> None:
    network_mount = tmp_path / "lab-storage"
    network_mount.mkdir()
    stamp, _ = init_volume(network_mount, label="lab-storage", kind="working")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(network_mount))
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "volumes", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == [{**stamp.to_dict(), "mount_path": str(network_mount)}]
