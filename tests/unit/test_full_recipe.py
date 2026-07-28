import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import helpers, recipes
from smftools.informatics.experiment_manifest import read_experiment_manifest


def test_full_flow_runs_raw_preprocess_spatial_hmm_latent_in_order(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        recipes,
        "latent_adata",
        lambda path: calls.append(("latent", path)) or ("adata", "latent-output"),
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        experiment_name="experiment",
        smf_modality="direct",
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(
            hmm_spine=tmp_path / "hmm_adata_outputs" / "spine.h5ad",
            latent_spine=tmp_path / "latent_adata_outputs" / "spine.h5ad",
        ),
    )

    result = recipes.full_flow("experiment.csv")

    assert calls == [
        ("raw", "experiment.csv"),
        ("preprocess", "experiment.csv"),
        ("spatial", "experiment.csv"),
        ("hmm", "experiment.csv"),
        ("latent", "experiment.csv"),
    ]
    assert result == ("adata", "latent-output")
    full_entry = read_experiment_manifest(tmp_path)["stages"]["full"]
    assert full_entry["state"] == "complete"
    assert full_entry["artifacts"]["summary"]["path"] == "full_summary.json"
    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert summary["outcome"] == "completed"
    assert [item["stage"] for item in summary["stages"]] == [
        "raw",
        "preprocess",
        "spatial",
        "hmm",
        "latent",
    ]


def test_full_cli_invokes_recipe(tmp_path, monkeypatch):
    config = tmp_path / "experiment.csv"
    config.write_text("variable,value\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli_entry, "full_flow", lambda path: calls.append(path))

    result = CliRunner().invoke(cli_entry.cli, ["experiment", "full", str(config)])

    assert result.exit_code == 0
    assert calls == [str(config)]


def test_full_summary_links_stage_logs_and_outcomes(tmp_path, monkeypatch):
    cfg = SimpleNamespace(output_directory=tmp_path)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(hmm_spine=None, latent_spine=None),
    )
    directory_names = {
        "raw": "raw_outputs",
        "preprocess": "preprocess_adata_outputs",
        "spatial": "spatial_adata_outputs",
        "hmm": "hmm_adata_outputs",
        "latent": "latent_adata_outputs",
    }

    def stage_runner(stage, outcome):
        def run(_path):
            logs = tmp_path / directory_names[stage] / "logs"
            logs.mkdir(parents=True)
            (logs / "run_log.log").write_text(f"{stage}\n")
            (logs / "run_perf.jsonl").write_text(
                json.dumps({"stage": stage, "event": "stage_summary", "outcome": outcome}) + "\n"
            )
            return (None, None) if stage in {"hmm", "latent"} else None

        return run

    monkeypatch.setattr(recipes, "raw_adata", stage_runner("raw", "completed"))
    monkeypatch.setattr(recipes, "preprocess_adata", stage_runner("preprocess", "skipped"))
    monkeypatch.setattr(recipes, "spatial_adata", stage_runner("spatial", "completed"))
    monkeypatch.setattr(recipes, "hmm_adata", stage_runner("hmm", "completed"))
    monkeypatch.setattr(recipes, "latent_adata", stage_runner("latent", "completed"))

    recipes.full_flow("experiment.csv")

    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert [item["outcome"] for item in summary["stages"]] == [
        "completed",
        "skipped",
        "completed",
        "completed",
        "completed",
    ]
    assert summary["stages"][0]["human_log"] == "raw_outputs/logs/run_log.log"
    assert summary["stages"][1]["performance_log"] == (
        "preprocess_adata_outputs/logs/run_perf.jsonl"
    )


def test_full_flow_records_failure_when_child_stage_raises(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        experiment_name="experiment",
        smf_modality="direct",
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fail_raw(path):
        raise RuntimeError("simulated raw failure")

    monkeypatch.setattr(recipes, "raw_adata", fail_raw)

    with pytest.raises(RuntimeError, match="simulated raw failure"):
        recipes.full_flow("experiment.csv")

    entry = read_experiment_manifest(tmp_path)["stages"]["full"]
    assert entry["state"] == "failed"
    assert "simulated raw failure" in entry["outcome"]
    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert summary["outcome"] == "failed"
    assert summary["exception"] == {
        "type": "RuntimeError",
        "message": "simulated raw failure",
    }


def test_full_flow_rejects_partitioned_result_without_child_completion_records(
    tmp_path, monkeypatch
):
    cfg = SimpleNamespace(output_directory=tmp_path, full_run_latent=False)
    hmm_spine = tmp_path / "hmm_adata_outputs" / "spine.h5ad"
    hmm_spine.parent.mkdir()
    hmm_spine.touch()
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(hmm_spine=hmm_spine, latent_spine=None),
    )
    monkeypatch.setattr(recipes, "raw_adata", lambda path: None)
    monkeypatch.setattr(recipes, "preprocess_adata", lambda path: None)
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: None)
    monkeypatch.setattr(recipes, "hmm_adata", lambda path: (None, hmm_spine))

    with pytest.raises(RuntimeError, match="incomplete stage record"):
        recipes.full_flow("experiment.csv")

    assert read_experiment_manifest(tmp_path)["stages"]["full"]["state"] == "failed"
    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert summary["outcome"] == "failed"
    assert summary["exception"]["type"] == "RuntimeError"


def test_full_flow_can_disable_latent(tmp_path, monkeypatch):
    calls = []
    cfg = SimpleNamespace(output_directory=tmp_path, full_run_latent=False)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        helpers,
        "get_adata_paths",
        lambda _cfg: SimpleNamespace(hmm_spine=None, latent_spine=None),
    )
    monkeypatch.setattr(recipes, "raw_adata", lambda path: calls.append("raw"))
    monkeypatch.setattr(recipes, "preprocess_adata", lambda path: calls.append("preprocess"))
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: calls.append("spatial"))
    monkeypatch.setattr(
        recipes,
        "hmm_adata",
        lambda path: calls.append("hmm") or ("adata", "hmm-output"),
    )
    monkeypatch.setattr(recipes, "latent_adata", lambda path: calls.append("latent"))

    result = recipes.full_flow("experiment.csv")

    assert calls == ["raw", "preprocess", "spatial", "hmm"]
    assert result == ("adata", "hmm-output")
    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["stages"][-1]["outcome"] == "disabled"


def test_full_flow_records_latent_failure(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        experiment_name="experiment",
        smf_modality="direct",
        full_run_latent=True,
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(recipes, "raw_adata", lambda path: None)
    monkeypatch.setattr(recipes, "preprocess_adata", lambda path: None)
    monkeypatch.setattr(recipes, "spatial_adata", lambda path: None)
    monkeypatch.setattr(recipes, "hmm_adata", lambda path: ("adata", "hmm-output"))

    def fail_latent(path):
        raise RuntimeError("simulated latent failure")

    monkeypatch.setattr(recipes, "latent_adata", fail_latent)

    with pytest.raises(RuntimeError, match="simulated latent failure"):
        recipes.full_flow("experiment.csv")

    summary = json.loads((tmp_path / "full_summary.json").read_text())
    assert summary["outcome"] == "failed"
    assert summary["exception"] == {
        "type": "RuntimeError",
        "message": "simulated latent failure",
    }


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
    cfg.max_memory_percent = 40
    cfg.memory_reserve_gb = 2
    cfg.target_task_memory_mb = 128
    cfg.emit_perf_log = False
    cfg.perf_log_sample_interval_seconds = 5
    cfg.informatics_outputs_path = "/machine-b/run/raw_outputs"
    cfg.bam_outputs_path = "/machine-b/run/raw_outputs/bam_outputs"
    cfg.device = "cpu"
    assert helpers.stage_config_hash(cfg) == original

    cfg.autocorr_max_lag = 800
    assert helpers.stage_config_hash(cfg) != original


def test_plot_regions_do_not_invalidate_compute_stage_hashes():
    cfg = SimpleNamespace(
        output_directory="/run",
        analysis_regions_bed="analysis.bed",
        plot_regions_bed="plot-a.bed",
    )
    compute_stages = ("preprocess", "spatial", "hmm", "latent", "full")
    original = {stage: helpers.stage_config_hash(cfg, stage) for stage in compute_stages}
    raw_hash = helpers.stage_config_hash(cfg, "raw")

    cfg.plot_regions_bed = "plot-b.bed"

    assert {stage: helpers.stage_config_hash(cfg, stage) for stage in compute_stages} == original
    assert helpers.stage_config_hash(cfg, "raw") != raw_hash


def test_latent_opt_out_does_not_invalidate_stage_compute_hashes():
    cfg = SimpleNamespace(
        output_directory="/run",
        full_run_latent=True,
        autocorr_max_lag=400,
    )
    original = {
        stage: helpers.stage_config_hash(cfg, stage)
        for stage in ("raw", "preprocess", "spatial", "hmm", "latent")
    }

    cfg.full_run_latent = False

    assert {
        stage: helpers.stage_config_hash(cfg, stage)
        for stage in ("raw", "preprocess", "spatial", "hmm", "latent")
    } == original
