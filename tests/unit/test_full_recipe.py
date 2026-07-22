from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import helpers, recipes
from smftools.informatics.experiment_manifest import read_experiment_manifest


def test_full_flow_runs_raw_preprocess_spatial_hmm_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(recipes, "raw_adata", lambda path: calls.append(("raw", path)))
    monkeypatch.setattr(
        recipes, "preprocess_adata", lambda path: calls.append(("preprocess", path))
    )
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: calls.append(("spatial", path)))

    def run_hmm(path):
        calls.append(("hmm", path))
        return "adata", "hmm-output"

    monkeypatch.setattr(recipes, "hmm_adata", run_hmm)
    cfg = SimpleNamespace(output_directory=tmp_path)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(hmm_spine=tmp_path / "hmm_adata_outputs" / "spine.h5ad"),
    )

    result = recipes.full_flow("experiment.csv")

    assert calls == [
        ("raw", "experiment.csv"),
        ("preprocess", "experiment.csv"),
        ("spatial", "experiment.csv"),
        ("hmm", "experiment.csv"),
    ]
    assert result == ("adata", "hmm-output")
    assert read_experiment_manifest(tmp_path)["stages"]["full"]["state"] == "complete"


def test_full_cli_invokes_four_stage_recipe(tmp_path, monkeypatch):
    config = tmp_path / "experiment.csv"
    config.write_text("variable,value\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli_entry, "full_flow", lambda path: calls.append(path))

    result = CliRunner().invoke(cli_entry.cli, ["experiment", "full", str(config)])

    assert result.exit_code == 0
    assert calls == [str(config)]


def test_full_flow_records_failure_when_child_stage_raises(tmp_path, monkeypatch):
    cfg = SimpleNamespace(output_directory=tmp_path)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fail_raw(path):
        raise RuntimeError("simulated raw failure")

    monkeypatch.setattr(recipes, "raw_adata", fail_raw)

    with pytest.raises(RuntimeError, match="simulated raw failure"):
        recipes.full_flow("experiment.csv")

    entry = read_experiment_manifest(tmp_path)["stages"]["full"]
    assert entry["state"] == "failed"
    assert "simulated raw failure" in entry["outcome"]


def test_full_flow_rejects_partitioned_result_without_child_completion_records(
    tmp_path, monkeypatch
):
    cfg = SimpleNamespace(output_directory=tmp_path)
    hmm_spine = tmp_path / "hmm_adata_outputs" / "spine.h5ad"
    hmm_spine.parent.mkdir()
    hmm_spine.touch()
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(hmm_spine=hmm_spine),
    )
    monkeypatch.setattr(recipes, "raw_adata", lambda path: None)
    monkeypatch.setattr(recipes, "preprocess_adata", lambda path: None)
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: None)
    monkeypatch.setattr(recipes, "hmm_adata", lambda path: (None, hmm_spine))

    with pytest.raises(RuntimeError, match="incomplete stage record"):
        recipes.full_flow("experiment.csv")

    assert read_experiment_manifest(tmp_path)["stages"]["full"]["state"] == "failed"


def test_stage_config_hash_ignores_machine_resources_but_not_analysis_config():
    cfg = SimpleNamespace(
        output_directory="/machine-a/run",
        threads=32,
        max_memory_gb=128,
        target_task_memory_mb=1024,
        informatics_outputs_path="/machine-a/run/raw_outputs",
        bam_outputs_path="/machine-a/run/raw_outputs/bam_outputs",
        device="cuda",
        autocorr_max_lag=400,
    )
    original = helpers.stage_config_hash(cfg)

    cfg.output_directory = "/machine-b/run"
    cfg.threads = 2
    cfg.max_memory_gb = 8
    cfg.target_task_memory_mb = 128
    cfg.informatics_outputs_path = "/machine-b/run/raw_outputs"
    cfg.bam_outputs_path = "/machine-b/run/raw_outputs/bam_outputs"
    cfg.device = "cpu"
    assert helpers.stage_config_hash(cfg) == original

    cfg.autocorr_max_lag = 800
    assert helpers.stage_config_hash(cfg) != original
