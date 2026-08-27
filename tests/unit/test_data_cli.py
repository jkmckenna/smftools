from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.config.roots import ENV_PREFIX, ENV_VOLUME_SEARCH_PATHS
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
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)


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


def test_scan_cli_also_registers_an_analysis_location(tmp_path: Path) -> None:
    import json
    import uuid

    from smftools.data.analysis_catalog import default_catalog_path, locations_for
    from smftools.data.analysis_catalog import load_catalog as load_analysis_catalog

    mount = tmp_path / "mount"
    mount.mkdir()
    stamp, _ = init_volume(mount, label="archive-01", kind="archive")
    run_root = mount / "exp1"
    _publish_run(run_root)
    uid = str(uuid.uuid4())
    (run_root / "experiment_manifest.json").write_text(
        json.dumps({"experiment_uid": uid}), encoding="utf-8"
    )
    runner = CliRunner()

    result = runner.invoke(
        cli, ["data", "scan", str(mount), "--catalog-path", str(tmp_path / "catalog.json")]
    )

    assert result.exit_code == 0, result.output
    assert "1 analysis location(s)" in result.output
    analysis_catalog = load_analysis_catalog(default_catalog_path())
    locations = locations_for(analysis_catalog, uid)
    assert len(locations) == 1
    assert locations[0].volume_id == stamp.volume_id
    assert locations[0].path == "exp1"


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
    assert "1 raw dataset(s)" in result.output


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


def test_init_cli_scaffolds_a_fresh_lab_root(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "init", str(lab_root)])

    assert result.exit_code == 0, result.output
    assert (lab_root / "data").is_dir()
    assert (lab_root / "analyses" / "runs").is_dir()
    assert (lab_root / "analyses" / "projects").is_dir()
    assert "Not stamped" in result.output
    assert not (lab_root / STAMP_FILENAME).exists()


def test_init_cli_rerun_reports_nothing_to_create(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    runner = CliRunner()
    runner.invoke(cli, ["data", "init", str(lab_root)])

    result = runner.invoke(cli, ["data", "init", str(lab_root)])

    assert result.exit_code == 0, result.output
    assert "already scaffolded" in result.output


def test_init_cli_with_stamp_volume_scaffolds_and_stamps(tmp_path: Path) -> None:
    lab_root = tmp_path / "lab"
    runner = CliRunner()

    result = runner.invoke(
        cli, ["data", "init", str(lab_root), "--stamp-volume", "--label", "lab-drive"]
    )

    assert result.exit_code == 0, result.output
    assert "Stamped" in result.output
    stamp = read_volume_stamp(lab_root)
    assert stamp is not None
    assert stamp.label == "lab-drive"
    assert stamp.kind == "working"


def _set_identity(run_root: Path, experiment_uid: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "experiment_manifest.json").write_text(
        json.dumps({"experiment_uid": experiment_uid}), encoding="utf-8"
    )


def _publish_generation(
    run_root: Path, stage_dir: str, generation_id: str, *, current: bool = True
) -> None:
    from smftools.informatics.generation_listing import (
        CURRENT_FILENAME,
        GENERATION_MANIFEST,
        GENERATIONS_SUBDIR,
    )

    container = run_root / stage_dir
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / GENERATION_MANIFEST).write_text(
        json.dumps({"schema_version": 2, "status": "complete", "generation_id": generation_id}),
        encoding="utf-8",
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps({"schema_version": 1, "generation_id": generation_id}), encoding="utf-8"
        )


def test_status_cli_reports_nothing_known() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "status"])

    assert result.exit_code == 0, result.output
    assert "No runs known" in result.output


def test_status_cli_reports_an_attached_location(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR

    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="ssd-01", kind="working")
    run_root = mount / "exp1"
    uid = str(uuid.uuid4())
    _set_identity(run_root, uid)
    _publish_generation(run_root, RAW_DIR, "gen-1")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(tmp_path / "cat.json")])

    result = runner.invoke(cli, ["data", "status", "--catalog-path", str(tmp_path / "cat.json")])

    assert result.exit_code == 0, result.output
    assert uid in result.output
    assert "[attached]" in result.output


def test_status_cli_accepts_a_run_root_target(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR

    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="ssd-01", kind="working")
    run_root = mount / "exp1"
    uid = str(uuid.uuid4())
    _set_identity(run_root, uid)
    _publish_generation(run_root, RAW_DIR, "gen-1")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(tmp_path / "cat.json")])

    result = runner.invoke(
        cli, ["data", "status", str(run_root), "--catalog-path", str(tmp_path / "cat.json")]
    )

    assert result.exit_code == 0, result.output
    assert uid in result.output


def test_status_cli_reports_diverged_locality_between_two_attached_copies(
    tmp_path: Path, monkeypatch
) -> None:
    import uuid

    from smftools.constants import RAW_DIR

    uid = str(uuid.uuid4())
    mount_a = tmp_path / "mount_a"
    mount_a.mkdir()
    init_volume(mount_a, label="ssd-a", kind="working")
    run_a = mount_a / "exp1"
    _set_identity(run_a, uid)
    _publish_generation(run_a, RAW_DIR, "gen-a-only")

    mount_b = tmp_path / "mount_b"
    mount_b.mkdir()
    init_volume(mount_b, label="ssd-b", kind="working")
    run_b = mount_b / "exp1"
    _set_identity(run_b, uid)
    _publish_generation(run_b, RAW_DIR, "gen-b-only")

    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, os.pathsep.join([str(mount_a), str(mount_b)]))
    runner = CliRunner()
    catalog_path = tmp_path / "cat.json"
    runner.invoke(cli, ["data", "scan", "--catalog-path", str(catalog_path)])

    result = runner.invoke(cli, ["data", "status", "--catalog-path", str(catalog_path)])

    assert result.exit_code == 0, result.output
    assert "diverged" in result.output


def test_status_cli_json_output(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR

    mount = tmp_path / "mount"
    mount.mkdir()
    init_volume(mount, label="ssd-01", kind="working")
    run_root = mount / "exp1"
    uid = str(uuid.uuid4())
    _set_identity(run_root, uid)
    _publish_generation(run_root, RAW_DIR, "gen-1")
    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, str(mount))
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", str(mount), "--catalog-path", str(tmp_path / "cat.json")])

    result = runner.invoke(
        cli, ["data", "status", "--catalog-path", str(tmp_path / "cat.json"), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runs"][0]["experiment_uid"] == uid
    assert payload["runs"][0]["locations"][0]["attached"] is True


def _two_attached_locations(tmp_path: Path, monkeypatch, uid: str):
    """Two mounts, same experiment_uid, both stamped, scanned, and discoverable."""
    from smftools.constants import RAW_DIR

    mount_a = tmp_path / "mount_a"
    mount_a.mkdir()
    stamp_a, _ = init_volume(mount_a, label="ssd-a", kind="working")
    run_a = mount_a / "exp1"
    _set_identity(run_a, uid)
    _publish_generation(run_a, RAW_DIR, "gen-1")
    _publish_generation(run_a, RAW_DIR, "gen-2")

    mount_b = tmp_path / "mount_b"
    mount_b.mkdir()
    stamp_b, _ = init_volume(mount_b, label="ssd-b", kind="working")
    run_b = mount_b / "exp1"
    _set_identity(run_b, uid)
    _publish_generation(run_b, RAW_DIR, "gen-1")

    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, os.pathsep.join([str(mount_a), str(mount_b)]))
    catalog_path = tmp_path / "cat.json"
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", "--catalog-path", str(catalog_path)])
    return run_a, run_b, stamp_a.volume_id, stamp_b.volume_id, catalog_path


def test_sync_cli_copies_the_missing_generation(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR
    from smftools.informatics.generation_listing import GENERATIONS_SUBDIR

    uid = str(uuid.uuid4())
    run_a, run_b, _, _, catalog_path = _two_attached_locations(tmp_path, monkeypatch, uid)
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "sync", uid])

    assert result.exit_code == 0, result.output
    assert (run_b / RAW_DIR / GENERATIONS_SUBDIR / "gen-2").is_dir()


def test_sync_cli_dry_run_does_not_copy(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR
    from smftools.informatics.generation_listing import GENERATIONS_SUBDIR

    uid = str(uuid.uuid4())
    run_a, run_b, _, _, catalog_path = _two_attached_locations(tmp_path, monkeypatch, uid)
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "sync", uid, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert not (run_b / RAW_DIR / GENERATIONS_SUBDIR / "gen-2").exists()


def test_sync_cli_accepts_explicit_from_to(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR
    from smftools.informatics.generation_listing import GENERATIONS_SUBDIR

    uid = str(uuid.uuid4())
    run_a, run_b, vol_a, vol_b, catalog_path = _two_attached_locations(tmp_path, monkeypatch, uid)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "data",
            "sync",
            uid,
            "--from",
            vol_a,
            "--to",
            vol_b,
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_b / RAW_DIR / GENERATIONS_SUBDIR / "gen-2").is_dir()


def test_sync_cli_refuses_without_exactly_two_attached_locations(tmp_path: Path) -> None:
    import uuid

    runner = CliRunner()

    result = runner.invoke(cli, ["data", "sync", str(uuid.uuid4())])

    assert result.exit_code != 0


def test_sync_cli_diverged_exits_nonzero_and_copies_nothing(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from smftools.constants import RAW_DIR
    from smftools.informatics.generation_listing import GENERATIONS_SUBDIR

    uid = str(uuid.uuid4())
    mount_a = tmp_path / "mount_a"
    mount_a.mkdir()
    init_volume(mount_a, label="ssd-a", kind="working")
    run_a = mount_a / "exp1"
    _set_identity(run_a, uid)
    _publish_generation(run_a, RAW_DIR, "gen-a-only")

    mount_b = tmp_path / "mount_b"
    mount_b.mkdir()
    init_volume(mount_b, label="ssd-b", kind="working")
    run_b = mount_b / "exp1"
    _set_identity(run_b, uid)
    _publish_generation(run_b, RAW_DIR, "gen-b-only")

    monkeypatch.setenv(ENV_VOLUME_SEARCH_PATHS, os.pathsep.join([str(mount_a), str(mount_b)]))
    catalog_path = tmp_path / "cat.json"
    runner = CliRunner()
    runner.invoke(cli, ["data", "scan", "--catalog-path", str(catalog_path)])

    result = runner.invoke(cli, ["data", "sync", uid])

    assert result.exit_code != 0
    assert "diverged" in result.output
    assert not (run_a / RAW_DIR / GENERATIONS_SUBDIR / "gen-b-only").exists()
    assert not (run_b / RAW_DIR / GENERATIONS_SUBDIR / "gen-a-only").exists()


def test_roots_list_cli_reports_nothing_bound() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "roots", "list"])

    assert result.exit_code == 0, result.output
    assert "No named roots" in result.output


def test_roots_list_cli_reports_an_env_bound_root(monkeypatch) -> None:
    from smftools.config.roots import ENV_PREFIX

    monkeypatch.setenv(f"{ENV_PREFIX}DATA", "/tmp/archive-01")
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "roots", "list"])

    assert result.exit_code == 0, result.output
    assert "data" in result.output
    assert "/tmp/archive-01" in result.output
    assert f"{ENV_PREFIX}DATA" in result.output


def test_roots_list_cli_reports_a_file_bound_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "user-config"))
    (tmp_path / "user-config").mkdir()
    (tmp_path / "user-config" / "roots.toml").write_text(
        f'[roots]\nanalyses = "{tmp_path / "analyses"}"\n', encoding="utf-8"
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "roots", "list"])

    assert result.exit_code == 0, result.output
    assert "analyses" in result.output
    assert str(tmp_path / "analyses") in result.output


def test_roots_list_cli_shows_every_candidate_of_a_multi_location_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "user-config"))
    (tmp_path / "user-config").mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()
    (tmp_path / "user-config" / "roots.toml").write_text(
        f'[roots]\nanalyses = ["{first}", "{second}"]\n', encoding="utf-8"
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "roots", "list"])

    assert result.exit_code == 0, result.output
    assert str(first) in result.output
    assert str(second) in result.output


def test_roots_list_cli_json_output(monkeypatch) -> None:
    from smftools.config.roots import ENV_PREFIX

    monkeypatch.setenv(f"{ENV_PREFIX}DATA", "/tmp/archive-01")
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "roots", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == [
        {
            "name": "data",
            "path": "/tmp/archive-01",
            "source": f"{ENV_PREFIX}DATA",
            "all_paths": ["/tmp/archive-01"],
        }
    ]
