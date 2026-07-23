from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from smftools.cli_entry import cli


def test_batch_full_dispatches_to_full_flow_per_config(tmp_path, monkeypatch):
    import smftools.cli.recipes as recipes_module

    captured = []

    def fake_full_flow(config_path):
        captured.append(config_path)

    monkeypatch.setattr(recipes_module, "full_flow", fake_full_flow)

    config_a = tmp_path / "a.csv"
    config_a.touch()
    config_b = tmp_path / "b.csv"
    config_b.touch()

    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config_a}\n{config_b}\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["experiment", "batch", "full", str(config_table)])

    assert result.exit_code == 0, result.output
    assert captured == [str(config_a), str(config_b)]
    summary = json.loads((tmp_path / "configs.full.batch-summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["completed"] == 2
    assert summary["failed"] == 0


def test_batch_returns_nonzero_and_writes_summary_on_partial_failure(tmp_path, monkeypatch):
    import smftools.cli.recipes as recipes_module

    config_a = tmp_path / "a.csv"
    config_a.touch()
    config_b = tmp_path / "b.csv"
    config_b.touch()
    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config_a}\n{config_b}\n")
    summary_path = tmp_path / "batch-result.json"

    def fake_full_flow(config_path):
        if config_path == str(config_b):
            raise RuntimeError("simulated experiment failure")

    monkeypatch.setattr(recipes_module, "full_flow", fake_full_flow)
    result = CliRunner().invoke(
        cli,
        [
            "experiment",
            "batch",
            "full",
            str(config_table),
            "--summary",
            str(summary_path),
        ],
    )

    assert result.exit_code != 0
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "partial_failure"
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert [item["status"] for item in summary["results"]] == ["completed", "failed"]
    assert summary["results"][1]["exception"] == {
        "type": "RuntimeError",
        "message": "simulated experiment failure",
    }


def test_batch_treats_missing_config_as_scheduler_visible_failure(tmp_path):
    missing = tmp_path / "missing.csv"
    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{missing}\n")

    result = CliRunner().invoke(cli, ["experiment", "batch", "full", str(config_table)])

    assert result.exit_code != 0
    summary = json.loads((tmp_path / "configs.full.batch-summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["results"][0]["exception"]["type"] == "FileNotFoundError"


def test_batch_summary_preserves_explicit_stage_skip_outcome(tmp_path, monkeypatch):
    from smftools.cli import batch as batch_module

    config = tmp_path / "experiment.csv"
    config.touch()
    output = tmp_path / "output"
    monkeypatch.setattr(
        batch_module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(output_directory=output),
    )

    def skipped_raw(_path):
        logs = output / "raw_outputs" / "logs"
        logs.mkdir(parents=True)
        (logs / "run_perf.jsonl").write_text(
            json.dumps({"event": "stage_summary", "outcome": "skipped"}) + "\n"
        )

    summary_path = tmp_path / "summary.json"
    summary = batch_module.run_batch(
        "raw",
        [config],
        skipped_raw,
        config_table=tmp_path / "configs.txt",
        summary_path=summary_path,
        emit=lambda _message: None,
    )

    assert summary["status"] == "complete"
    assert summary["completed"] == 0
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["results"][0]["status"] == "skipped"
    assert json.loads(summary_path.read_text()) == summary
