from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.informatics.generation import CURRENT_FILENAME, GENERATION_MANIFEST
from smftools.project import embedding_store, sample_analysis
from smftools.project.analysis_inventory import analysis_cache_inventory
from smftools.project.set_store import sets_root

pytestmark = pytest.mark.unit


def _write_periodicity_cache(
    project_dir: Path,
    *,
    algorithm_version: str | None,
    malformed_definition: bool = False,
) -> Path:
    definition = sample_analysis._periodicity_definition(
        layer=None,
        start=None,
        end=None,
        method="direct",
        kwargs={},
    )
    if algorithm_version is None:
        definition.pop("algorithm_version")
        definition.pop("graph_definition_version")
    else:
        definition["algorithm_version"] = algorithm_version
    cache_dir = sample_analysis._analysis_dir(
        project_dir,
        "experiment-a",
        "reference_top",
        "bc01",
        sample_analysis.PERIODICITY_ANALYSIS_NAME,
        sample_analysis._definition_hash(definition),
    )
    cache_dir.mkdir(parents=True)
    definition_path = cache_dir / sample_analysis.DEFINITION_FILENAME
    if malformed_definition:
        definition_path.write_text("{", encoding="utf-8")
    else:
        definition_path.write_text(json.dumps(definition), encoding="utf-8")
    (cache_dir / sample_analysis.RESULT_FILENAME).write_bytes(b"not-loaded")
    return cache_dir


def _write_embedding_cache(project_dir: Path, *, algorithm_version: str) -> Path:
    definition = embedding_store._embedding_definition(
        canonical_reference="reference",
        set_name=None,
        modality=None,
        experiments=None,
        stage=None,
        layer=None,
        start=None,
        end=None,
        feature_kind="raw",
        leiden_resolution=0.5,
        n_neighbors=15,
        min_reads=10,
        random_state=42,
    )
    definition["algorithm_version"] = algorithm_version
    definition_hash = embedding_store._definition_hash(definition)
    cache_dir = sets_root(project_dir) / "reference" / "embeddings" / definition_hash
    generation_id = "generation-a"
    generation_dir = cache_dir / "generations" / generation_id
    generation_dir.mkdir(parents=True)
    # Deliberately invalid for every serialization format. Inventory must only
    # check paths and file metadata, never load these artifacts.
    artifacts = {}
    for filename in embedding_store._ARTIFACT_FILENAMES:
        (generation_dir / filename).write_bytes(b"must-not-be-loaded")
        artifacts[filename] = "unused-by-inventory"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generation_id": generation_id,
        "definition_hash": definition_hash,
        "definition": definition,
        "artifacts": artifacts,
    }
    (generation_dir / GENERATION_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "generation_id": generation_id,
        "generation_path": f"generations/{generation_id}",
    }
    (cache_dir / CURRENT_FILENAME).write_text(json.dumps(pointer), encoding="utf-8")
    return cache_dir


def test_inventory_classifies_current_stale_legacy_and_invalid_caches(tmp_path: Path) -> None:
    current_periodicity = _write_periodicity_cache(
        tmp_path, algorithm_version=sample_analysis.PERIODICITY_ALGORITHM_VERSION
    )
    legacy_periodicity = _write_periodicity_cache(tmp_path, algorithm_version=None)
    invalid_periodicity = _write_periodicity_cache(
        tmp_path,
        algorithm_version="invalid-definition",
        malformed_definition=True,
    )
    current_embedding = _write_embedding_cache(
        tmp_path, algorithm_version=embedding_store.EMBEDDING_ALGORITHM_VERSION
    )
    stale_embedding = _write_embedding_cache(tmp_path, algorithm_version="0")

    inventory = analysis_cache_inventory(tmp_path)
    by_path = {entry["cache_path"]: entry for entry in inventory["entries"]}

    assert inventory["schema_version"] == 1
    assert inventory["counts"] == {"current": 2, "stale": 2, "invalid": 1}
    assert by_path[current_periodicity.relative_to(tmp_path).as_posix()]["status"] == "current"
    legacy = by_path[legacy_periodicity.relative_to(tmp_path).as_posix()]
    assert legacy["status"] == "stale"
    assert legacy["reasons"] == [
        "missing_algorithm_version",
        "missing_graph_definition_version",
    ]
    invalid = by_path[invalid_periodicity.relative_to(tmp_path).as_posix()]
    assert invalid["status"] == "invalid"
    assert invalid["reasons"] == ["definition_unreadable_json"]
    assert by_path[current_embedding.relative_to(tmp_path).as_posix()]["status"] == "current"
    stale = by_path[stale_embedding.relative_to(tmp_path).as_posix()]
    assert stale["status"] == "stale"
    assert stale["reasons"] == ["algorithm_version_mismatch"]
    assert stale["size_bytes"] > 0
    assert stale["generation_count"] == 1


def test_stale_filter_keeps_invalid_entries_and_preserves_full_counts(tmp_path: Path) -> None:
    _write_periodicity_cache(
        tmp_path, algorithm_version=sample_analysis.PERIODICITY_ALGORITHM_VERSION
    )
    _write_periodicity_cache(tmp_path, algorithm_version=None)
    _write_periodicity_cache(
        tmp_path,
        algorithm_version="invalid-definition",
        malformed_definition=True,
    )

    inventory = analysis_cache_inventory(tmp_path, stale_only=True)

    assert inventory["stale_only"] is True
    assert inventory["counts"] == {"current": 1, "stale": 1, "invalid": 1}
    assert {entry["status"] for entry in inventory["entries"]} == {"stale", "invalid"}


def test_embedding_inventory_rejects_escaping_current_pointer(tmp_path: Path) -> None:
    cache_dir = _write_embedding_cache(
        tmp_path, algorithm_version=embedding_store.EMBEDDING_ALGORITHM_VERSION
    )
    pointer = {
        "schema_version": 1,
        "generation_id": "outside",
        "generation_path": "../../outside",
    }
    (cache_dir / CURRENT_FILENAME).write_text(json.dumps(pointer), encoding="utf-8")

    inventory = analysis_cache_inventory(tmp_path)

    assert len(inventory["entries"]) == 1
    entry = inventory["entries"][0]
    assert entry["status"] == "invalid"
    assert "current_pointer_not_portable" in entry["reasons"]


def test_project_analyses_list_cli_supports_json_and_stale_filter(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_periodicity_cache(
        project_dir, algorithm_version=sample_analysis.PERIODICITY_ALGORITHM_VERSION
    )
    stale = _write_periodicity_cache(project_dir, algorithm_version=None)
    runner = CliRunner()

    result = runner.invoke(
        cli_entry.cli,
        ["project", "analyses", "list", str(project_dir), "--stale", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["counts"] == {"current": 1, "stale": 1, "invalid": 0}
    assert [entry["cache_path"] for entry in payload["entries"]] == [
        stale.relative_to(project_dir).as_posix()
    ]

    table = runner.invoke(
        cli_entry.cli,
        ["project", "analyses", "list", str(project_dir), "--stale"],
    )
    assert table.exit_code == 0, table.output
    assert "missing_algorithm_version" in table.output
    assert "current" not in table.output


def test_project_analyses_list_cli_reports_empty_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = CliRunner().invoke(
        cli_entry.cli,
        ["project", "analyses", "list", str(project_dir)],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "No project analysis caches found.\n"
