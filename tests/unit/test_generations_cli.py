from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli.generations import render_json, render_table
from smftools.cli_entry import cli
from smftools.constants import PREPROCESS_DIR, RAW_DIR
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    list_experiment_generations,
)

pytestmark = pytest.mark.unit


def _publish(container: Path, generation_id: str, *, current: bool = False) -> None:
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / GENERATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "generation_id": generation_id,
                "config_hash": "hash0001",
                "artifacts": {"spine": "spine.h5ad"},
            }
        ),
        encoding="utf-8",
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "generation_path": f"{GENERATIONS_SUBDIR}/{generation_id}",
                }
            ),
            encoding="utf-8",
        )


def test_render_table_marks_current_and_summarizes(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "gen-current", current=True)
    _publish(run_root / RAW_DIR, "gen-superseded")

    table = render_table(list_experiment_generations(run_root))

    assert "gen-current" in table
    assert "gen-superseded" in table
    current_line = next(line for line in table.splitlines() if "gen-current" in line)
    superseded_line = next(line for line in table.splitlines() if "gen-superseded" in line)
    assert current_line.startswith("*")
    assert not superseded_line.startswith("*")
    assert "2 generation(s); 1 current, 2 readable, 0 unreadable or missing." in table


def test_render_table_does_not_truncate_generation_ids(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    long_id = "pod5-sup52-provenance-rebuild-2026"
    _publish(run_root / RAW_DIR, long_id, current=True)

    assert long_id in render_table(list_experiment_generations(run_root))


def test_render_table_empty_is_explicit(tmp_path: Path) -> None:
    assert render_table([]) == "No published generations found."


def test_render_json_is_stable_and_versioned(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / PREPROCESS_DIR, "pp-1", current=True)

    payload = json.loads(render_json(list_experiment_generations(run_root)))

    assert payload["schema_version"] == 1
    assert len(payload["generations"]) == 1
    entry = payload["generations"][0]
    assert entry["kind"] == "preprocess"
    assert entry["generation_id"] == "pp-1"
    assert entry["is_current"] is True


def test_cli_experiment_generations_reports_empty_store(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    (run_root / RAW_DIR).mkdir(parents=True)

    result = CliRunner().invoke(cli, ["experiment", "generations", str(run_root)])

    assert result.exit_code == 0
    assert "No published generations found." in result.output


def test_cli_experiment_generations_json(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)

    result = CliRunner().invoke(cli, ["experiment", "generations", str(run_root), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["generations"][0]["generation_id"] == "raw-1"


def test_cli_size_flag_populates_bytes(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)

    without = CliRunner().invoke(cli, ["experiment", "generations", str(run_root), "--json"])
    with_size = CliRunner().invoke(
        cli, ["experiment", "generations", str(run_root), "--json", "--size"]
    )

    assert json.loads(without.output)["generations"][0]["size_bytes"] is None
    assert json.loads(with_size.output)["generations"][0]["size_bytes"] > 0


def test_cli_project_generations_project_only_skips_experiments(tmp_path: Path) -> None:
    from smftools.project.set_store import sets_root

    _publish(sets_root(tmp_path) / "my_set" / "embeddings" / "def0", "emb-1", current=True)

    result = CliRunner().invoke(
        cli, ["project", "generations", str(tmp_path), "--project-only", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [entry["kind"] for entry in payload["generations"]] == ["embedding"]
