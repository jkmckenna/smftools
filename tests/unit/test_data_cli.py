from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.config.roots import ENV_VOLUME_SEARCH_PATHS
from smftools.data import volume_discovery
from smftools.data.volume_stamp import STAMP_FILENAME, init_volume, read_volume_stamp
from smftools.informatics.input_manifest import resolve_input_manifest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_real_mount_roots(tmp_path, monkeypatch):
    """Never let a `data` CLI test see this machine's actual mounted drives or config."""
    monkeypatch.setattr(volume_discovery, "platform_mount_root_candidates", lambda: [])
    monkeypatch.delenv(ENV_VOLUME_SEARCH_PATHS, raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))


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


def _publish_run(run_root: Path, *, content: bytes = b"@one\nAC\n+\n!!\n") -> str:
    source = run_root.parent / f"_sources_{run_root.name}" / "sample.fastq"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    return resolve_input_manifest(output_directory=run_root, input_paths=[source]).digest


def test_scan_cli_indexes_an_explicit_mount(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    digest = _publish_run(mount / "exp1")
    catalog_path = tmp_path / "catalog.json"
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(catalog_path)])

    assert result.exit_code == 0, result.output
    assert digest in result.output
    assert catalog_path.is_file()


def test_scan_cli_defaults_to_attached_volumes(tmp_path: Path, monkeypatch) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    _publish_run(mount / "exp1")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    catalog_path = tmp_path / "catalog.json"
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "scan", "--catalog-path", str(catalog_path)])

    assert result.exit_code == 0, result.output
    assert "1 run(s) found" in result.output


def test_scan_cli_rejects_an_unstamped_mount(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        cli, ["data", "scan", str(mount), "--catalog-path", str(tmp_path / "catalog.json")]
    )

    assert result.exit_code != 0
    assert "has not been stamped" in result.output


def test_locate_cli_reports_an_attached_replica(tmp_path: Path, monkeypatch) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    run_root = mount / "exp1"
    _publish_run(run_root)
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(catalog_path)])

    result = runner.invoke(
        cli, ["data", "locate", str(run_root), "--catalog-path", str(catalog_path)]
    )

    assert result.exit_code == 0, result.output
    assert "[attached]" in result.output


def test_locate_cli_reports_no_replicas_for_an_uncatalogued_digest(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "data",
            "locate",
            "0" * 64,
            "--catalog-path",
            str(tmp_path / "catalog.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "no catalogued replicas" in result.output


def test_verify_cli_reports_ok_for_an_intact_replica(tmp_path: Path, monkeypatch) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    run_root = mount / "exp1"
    _publish_run(run_root)
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(catalog_path)])

    result = runner.invoke(
        cli, ["data", "verify", str(run_root), "--catalog-path", str(catalog_path)]
    )

    assert result.exit_code == 0, result.output
    assert "ok" in result.output


def test_verify_cli_fails_and_reports_a_mismatch(tmp_path: Path, monkeypatch) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="archive-01", kind="archive")
    run_root = mount / "exp1"
    _publish_run(run_root)
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(catalog_path)])
    # Corrupt the declared source in place.
    source = next((mount / "_sources_exp1").iterdir())
    source.write_bytes(b"corrupted")

    result = runner.invoke(
        cli, ["data", "verify", str(run_root), "--catalog-path", str(catalog_path)]
    )

    assert result.exit_code != 0
    assert "mismatch" in result.output


def _write_config(tmp_path: Path, **fields: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "experiment_config.csv"
    rows = "\n".join(f"{name},{value}" for name, value in fields.items())
    config.write_text(f"variable,value\n{rows}\n", encoding="utf-8")
    return config


def test_localize_cli_dry_run_does_not_write_anything(tmp_path: Path) -> None:
    fasta = tmp_path / "ref.fasta"
    fasta.write_bytes(b">ref\nACGT\n")
    config = _write_config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "localize", str(config)])

    assert result.exit_code == 0, result.output
    assert "fasta" in result.output
    assert "Dry run" in result.output
    assert not (tmp_path / "out").exists()


def test_localize_cli_apply_copies_and_writes_a_new_config(tmp_path: Path) -> None:
    fasta = tmp_path / "ref.fasta"
    fasta.write_bytes(b">ref\nACGT\n")
    config = _write_config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "localize", str(config), "--apply"])

    assert result.exit_code == 0, result.output
    localized = tmp_path / "out" / "localized_inputs" / "ref.fasta"
    assert localized.is_file()
    new_config = config.with_suffix(".localized.csv")
    assert new_config.is_file()
    assert config.read_bytes() == (
        b"variable,value\n" + f"output_directory,{tmp_path / 'out'}\nfasta,{fasta}\n".encode()
    )


def test_localize_cli_reports_nothing_when_no_fields_declared(tmp_path: Path) -> None:
    config = _write_config(tmp_path, output_directory=str(tmp_path / "out"))
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "localize", str(config)])

    assert result.exit_code == 0, result.output
    assert "Nothing to localize" in result.output
