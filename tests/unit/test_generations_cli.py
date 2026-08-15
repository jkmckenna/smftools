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
    assert "2 generation(s); 1 current, 0 pinned, 2 readable, 0 unreadable or missing." in table


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

    assert payload["schema_version"] == 2
    assert len(payload["generations"]) == 1
    entry = payload["generations"][0]
    assert entry["kind"] == "preprocess"
    assert entry["generation_id"] == "pp-1"
    assert entry["is_current"] is True
    assert entry["pinned"] is False
    assert entry["retention_reasons"] == []


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


def test_cli_pin_list_and_unpin_reasons(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)
    runner = CliRunner()

    first = runner.invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "pin",
            "raw",
            "raw-1",
            "--reason",
            "paper figure 3",
        ],
    )
    second = runner.invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "pin",
            "raw",
            "raw-1",
            "--reason",
            "SRA:ABC123",
        ],
    )
    listed = runner.invoke(cli, ["experiment", "generations", str(run_root), "--json"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    entry = json.loads(listed.output)["generations"][0]
    assert entry["pinned"] is True
    assert entry["retention_reasons"] == ["paper figure 3", "SRA:ABC123"]

    one = runner.invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "unpin",
            "raw",
            "raw-1",
            "--reason",
            "paper figure 3",
        ],
    )
    all_reasons = runner.invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "unpin",
            "raw",
            "raw-1",
            "--all-reasons",
        ],
    )
    listed_again = runner.invoke(cli, ["experiment", "generations", str(run_root), "--json"])

    assert one.exit_code == 0
    assert "1 reason(s) remain" in one.output
    assert all_reasons.exit_code == 0
    assert json.loads(listed_again.output)["generations"][0]["pinned"] is False


@pytest.mark.parametrize("options", [[], ["--reason", "hold", "--all-reasons"]])
def test_cli_unpin_requires_exactly_one_mode(tmp_path: Path, options: list[str]) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)

    result = CliRunner().invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "unpin",
            "raw",
            "raw-1",
            *options,
        ],
    )

    assert result.exit_code == 2
    assert "choose exactly one" in result.output


def test_cli_prune_is_dry_run_and_never_deletes(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-old")
    _publish(container, "raw-current", current=True)
    paths_before = sorted(path.relative_to(run_root) for path in run_root.rglob("*"))

    result = CliRunner().invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "prune",
            "--stage",
            "raw",
            "--older-than",
            "2100-01-01T00:00:00Z",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["deletion_supported"] is False
    assert payload["reclaimable_bytes"] == 0
    assert {decision["disposition"] for decision in payload["decisions"]} == {
        "keep_current",
        "blocked_reproducibility",
    }
    assert sorted(path.relative_to(run_root) for path in run_root.rglob("*")) == paths_before


def test_cli_prune_table_is_explicitly_blocked(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-old")
    _publish(run_root / RAW_DIR, "raw-current", current=True)

    result = CliRunner().invoke(
        cli,
        [
            "experiment",
            "generations",
            str(run_root),
            "prune",
            "--older-than",
            "2100-01-01T00:00:00Z",
        ],
    )

    assert result.exit_code == 0
    assert "DRY RUN: deletion is not available" in result.output
    assert "BLOCKED" in result.output


def test_cli_project_generations_project_only_skips_experiments(tmp_path: Path) -> None:
    from smftools.project.set_store import sets_root

    _publish(sets_root(tmp_path) / "my_set" / "embeddings" / "def0", "emb-1", current=True)

    result = CliRunner().invoke(
        cli, ["project", "generations", str(tmp_path), "--project-only", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [entry["kind"] for entry in payload["generations"]] == ["embedding"]
