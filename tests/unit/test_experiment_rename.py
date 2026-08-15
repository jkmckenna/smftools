import base64
import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import experiment_rename
from smftools.cli.experiment_rename import rename_experiment_id
from smftools.informatics.artifact_paths import serialize_artifact_path

EXPERIMENT_UID = "b91137d2-5559-43c0-9f0c-418270856051"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_experiment(tmp_path: Path) -> Path:
    root = tmp_path / "old-id"
    spine = root / "raw_outputs" / "spine.h5ad"
    spine.parent.mkdir(parents=True)
    spine.write_bytes(b"immutable-spine")
    _write_json(
        root / "experiment_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": "old-id",
            "experiment": "old-id",
            "experiment_uid": EXPERIMENT_UID,
            "stages": {"raw": {"state": "complete"}},
        },
    )
    (root / "experiment_config.csv").write_text(
        "variable,value,help,options,type\n"
        f"output_directory,{root},Directory,,str\n"
        "experiment_name,old-id,Name,,str\n"
        "experiment_id,old-id,ID,,str\n"
        "threads,4,Threads,,int\n",
        encoding="utf-8",
    )
    return root


def _make_project(tmp_path: Path, experiment_root: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    external_catalog = tmp_path / "external" / "catalog.parquet"
    external_catalog.parent.mkdir()
    external_catalog.write_bytes(b"catalog")
    registry = {
        "schema_version": 4,
        "experiments": {
            "old-id": {
                "path": serialize_artifact_path(experiment_root, project),
                "name": "old-id",
                "experiment_uid": EXPERIMENT_UID,
                "spines": {
                    "raw": serialize_artifact_path(
                        experiment_root / "raw_outputs" / "spine.h5ad", project
                    )
                },
                "catalogs": {
                    "internal": serialize_artifact_path(
                        experiment_root / "raw_outputs" / "catalog.parquet", project
                    ),
                    "external": serialize_artifact_path(external_catalog, project),
                },
            }
        },
        "sets": {
            "listed": {"kind": "list", "experiments": ["old-id"]},
            "dynamic": {"kind": "query", "sql": "experiment_id = 'old-id'"},
        },
    }
    _write_json(project / "registry.json", registry)
    pointer = (
        project
        / "project_outputs"
        / "per_sample"
        / "old-id"
        / "ref_top"
        / "sample"
        / "pointer.json"
    )
    _write_json(
        pointer,
        {
            "kind": "cache",
            "experiment_id": "old-id",
            "reference_strand": "ref_top",
            "sample": "sample",
            "cache_path": "cache.h5ad",
        },
    )
    (pointer.parent / "cache.h5ad").write_bytes(b"cached-data")
    return project


def _config_values(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["variable"]: row["value"] for row in csv.DictReader(handle)}


def test_rename_experiment_id_updates_mutable_references_and_preserves_uid(tmp_path):
    old_root = _make_experiment(tmp_path)
    project = _make_project(tmp_path, old_root)

    result = rename_experiment_id(old_root, "new-id", project_dirs=(project,))

    new_root = tmp_path / "new-id"
    assert result.experiment_dir == new_root
    assert result.experiment_uid == EXPERIMENT_UID
    assert result.query_sets_unchanged == (f"{project}:dynamic",)
    assert not old_root.exists()
    manifest = json.loads((new_root / "experiment_manifest.json").read_text())
    assert manifest["experiment_id"] == manifest["experiment"] == "new-id"
    assert manifest["experiment_uid"] == EXPERIMENT_UID
    assert manifest["experiment_id_history"][0]["previous_experiment_id"] == "old-id"
    assert (new_root / "raw_outputs" / "spine.h5ad").read_bytes() == b"immutable-spine"

    config = _config_values(new_root / "experiment_config.csv")
    assert config["experiment_id"] == config["experiment_name"] == "new-id"
    assert config["output_directory"] == str(new_root)
    assert config["threads"] == "4"

    registry = json.loads((project / "registry.json").read_text())
    assert "old-id" not in registry["experiments"]
    entry = registry["experiments"]["new-id"]
    assert entry["experiment_uid"] == EXPERIMENT_UID
    assert (project / entry["path"]).resolve() == new_root
    assert (project / entry["spines"]["raw"]).resolve() == (new_root / "raw_outputs" / "spine.h5ad")
    assert registry["sets"]["listed"]["experiments"] == ["new-id"]
    assert registry["sets"]["dynamic"]["sql"] == "experiment_id = 'old-id'"

    new_pointer = (
        project
        / "project_outputs"
        / "per_sample"
        / "new-id"
        / "ref_top"
        / "sample"
        / "pointer.json"
    )
    assert json.loads(new_pointer.read_text())["experiment_id"] == "new-id"
    assert (new_pointer.parent / "cache.h5ad").read_bytes() == b"cached-data"


def test_rename_experiment_id_preflights_every_project_before_writing(tmp_path):
    old_root = _make_experiment(tmp_path)
    first_project = _make_project(tmp_path, old_root)
    second_project = _make_project(tmp_path / "second", old_root)
    second_registry_path = second_project / "registry.json"
    second_registry = json.loads(second_registry_path.read_text())
    second_registry["experiments"]["new-id"] = {"experiment_uid": "other"}
    _write_json(second_registry_path, second_registry)
    before_manifest = (old_root / "experiment_manifest.json").read_bytes()
    before_registry = (first_project / "registry.json").read_bytes()

    with pytest.raises(ValueError, match="already contains experiment id"):
        rename_experiment_id(
            old_root,
            "new-id",
            project_dirs=(first_project, second_project),
        )

    assert (old_root / "experiment_manifest.json").read_bytes() == before_manifest
    assert (first_project / "registry.json").read_bytes() == before_registry
    assert not (tmp_path / "new-id").exists()


def test_rename_experiment_id_rejects_stale_per_sample_destination(tmp_path):
    old_root = _make_experiment(tmp_path)
    project = _make_project(tmp_path, old_root)
    old_sample_root = project / "project_outputs" / "per_sample" / "old-id"
    stale_destination = project / "project_outputs" / "per_sample" / "new-id"
    old_sample_root.rename(stale_destination)
    before_manifest = (old_root / "experiment_manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="per-sample rename destination"):
        rename_experiment_id(old_root, "new-id", project_dirs=(project,))

    assert (old_root / "experiment_manifest.json").read_bytes() == before_manifest
    assert stale_destination.is_dir()


def test_rename_experiment_id_rolls_back_files_and_per_sample_move(tmp_path, monkeypatch):
    old_root = _make_experiment(tmp_path)
    project = _make_project(tmp_path, old_root)
    tracked = [
        old_root / "experiment_manifest.json",
        old_root / "experiment_config.csv",
        project / "registry.json",
        project
        / "project_outputs"
        / "per_sample"
        / "old-id"
        / "ref_top"
        / "sample"
        / "pointer.json",
    ]
    before = {path: path.read_bytes() for path in tracked}
    real_replace = experiment_rename.os.replace

    def fail_final_move(source, destination):
        if Path(source) == old_root and Path(destination) == tmp_path / "new-id":
            raise OSError("injected final move failure")
        return real_replace(source, destination)

    monkeypatch.setattr(experiment_rename.os, "replace", fail_final_move)

    with pytest.raises(OSError, match="injected final move failure"):
        rename_experiment_id(old_root, "new-id", project_dirs=(project,))

    assert old_root.is_dir()
    assert not (tmp_path / "new-id").exists()
    assert not (project / "project_outputs" / "per_sample" / "new-id").exists()
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not list(tmp_path.glob(".smftools-rename-*.json"))


def test_rename_experiment_id_recovers_a_prepared_journal_before_retry(tmp_path):
    old_root = _make_experiment(tmp_path)
    manifest_path = old_root / "experiment_manifest.json"
    original_manifest = manifest_path.read_bytes()
    interrupted_manifest = json.loads(original_manifest)
    interrupted_manifest["experiment_id"] = "new-id"
    interrupted_manifest["experiment"] = "new-id"
    _write_json(manifest_path, interrupted_manifest)
    journal_path = tmp_path / f".smftools-rename-{EXPERIMENT_UID}.json"
    _write_json(
        journal_path,
        {
            "schema_version": 1,
            "state": "prepared",
            "old_root": str(old_root),
            "new_root": str(tmp_path / "new-id"),
            "files": [
                {
                    "path": str(manifest_path),
                    "before_base64": base64.b64encode(original_manifest).decode("ascii"),
                }
            ],
            "moves": [
                {
                    "source": str(old_root),
                    "destination": str(tmp_path / "new-id"),
                }
            ],
        },
    )

    result = rename_experiment_id(old_root, "new-id")

    assert result.experiment_dir == tmp_path / "new-id"
    assert not journal_path.exists()
    manifest = json.loads((result.experiment_dir / "experiment_manifest.json").read_text())
    assert manifest["experiment_id"] == "new-id"
    assert len(manifest["experiment_id_history"]) == 1


def test_rename_id_cli_reports_unsearched_projects(tmp_path):
    old_root = _make_experiment(tmp_path)

    result = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "rename-id", str(old_root), "new-id"],
    )

    assert result.exit_code == 0, result.output
    assert "Renamed experiment 'old-id' to 'new-id'" in result.output
    assert "external project registries were not searched" in result.output
    assert (tmp_path / "new-id").is_dir()
