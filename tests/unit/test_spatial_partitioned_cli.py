from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from smftools.cli import helpers
from smftools.cli.spatial_adata import spatial_adata
from smftools.constants import (
    PARTITIONED_STAGE_NONEMPTY_DIRECTORIES,
    PARTITIONED_STAGE_REQUIRED_ARTIFACTS,
)
from smftools.informatics.experiment_manifest import read_experiment_manifest
from smftools.tools import partitioned_spatial


def _cfg(tmp_path, *, mode="auto", force=False):
    return SimpleNamespace(
        output_directory=str(tmp_path),
        experiment_name="experiment",
        smf_modality="conversion",
        spatial_execution_mode=mode,
        force_redo_spatial_analyses=force,
        from_adata_stage=None,
    )


def _spatial_outputs(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "spine": output / "spine.h5ad",
        "task_catalog": output / "task_catalog.parquet",
        "metrics": output / "metrics.parquet",
        "autocorrelation": output / "autocorrelation.parquet",
        "task_store": output / "store",
        "region_catalog": output / "regions.parquet",
        "plot_catalog": output / "plots" / "catalog.parquet",
        "manifest": output / "sidecar_manifest.json",
    }
    ad.AnnData().write_h5ad(paths["spine"])
    pd.DataFrame({"task_id": ["task-1"]}).to_parquet(paths["task_catalog"], index=False)
    for key in ("metrics", "autocorrelation", "region_catalog"):
        pd.DataFrame().to_parquet(paths[key], index=False)
    paths["task_store"].mkdir(exist_ok=True)
    (paths["task_store"] / "task-1").touch()
    paths["plot_catalog"].parent.mkdir(exist_ok=True)
    pd.DataFrame().to_parquet(paths["plot_catalog"], index=False)
    paths["manifest"].write_text("{}\n", encoding="utf-8")
    return paths


def test_spatial_cli_auto_selects_partitioned_preprocess_spine(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fake_execute(source, passed_cfg, output):
        captured.update(source=source, cfg=passed_cfg, output=output)
        return _spatial_outputs(output)

    monkeypatch.setattr(partitioned_spatial, "execute_partitioned_spatial", fake_execute)

    adata, output_path = spatial_adata("config.csv")

    assert adata is None
    assert output_path == paths.spatial_spine
    assert captured["source"] == paths.preprocess_spine
    assert captured["output"] == paths.spatial_spine.parent
    entry = read_experiment_manifest(tmp_path)["stages"]["spatial"]
    assert entry["state"] == "complete"
    assert entry["expected_tasks"] == entry["successful_tasks"] == 1
    assert entry["input_artifact_ids"]

    monkeypatch.setattr(
        partitioned_spatial,
        "execute_partitioned_spatial",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rerun")),
    )
    assert spatial_adata("config.csv") == (None, paths.spatial_spine)


def test_spatial_cli_partitioned_mode_ignores_legacy_spatial_file(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, mode="partitioned")
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    paths.spatial.parent.mkdir(parents=True)
    paths.spatial.touch()

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        partitioned_spatial,
        "execute_partitioned_spatial",
        lambda source, passed_cfg, output: _spatial_outputs(output),
    )

    _, output_path = spatial_adata("config.csv")

    assert output_path == paths.spatial_spine


def test_spatial_cli_reruns_when_completed_artifact_is_missing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    outputs = _spatial_outputs(paths.spatial_spine.parent)
    required = PARTITIONED_STAGE_REQUIRED_ARTIFACTS["spatial"]
    with helpers.stage_lifecycle(cfg, "spatial", paths.preprocess_spine) as lifecycle:
        helpers.publish_stage_outputs(
            lifecycle,
            outputs,
            required=required,
            schema_versions={"spatial": 2},
            nonempty_directory_keys=PARTITIONED_STAGE_NONEMPTY_DIRECTORIES["spatial"],
        )
    outputs["metrics"].unlink()
    calls = []
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def rerun(source, passed_cfg, output):
        calls.append(source)
        return _spatial_outputs(output)

    monkeypatch.setattr(partitioned_spatial, "execute_partitioned_spatial", rerun)

    assert spatial_adata("config.csv") == (None, paths.spatial_spine)
    assert calls == [paths.preprocess_spine]


def test_spatial_cli_reruns_when_semantic_config_changes(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.autocorr_max_lag = 4
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    outputs = _spatial_outputs(paths.spatial_spine.parent)
    required = PARTITIONED_STAGE_REQUIRED_ARTIFACTS["spatial"]
    with helpers.stage_lifecycle(cfg, "spatial", paths.preprocess_spine) as lifecycle:
        helpers.publish_stage_outputs(
            lifecycle,
            outputs,
            required=required,
            nonempty_directory_keys=PARTITIONED_STAGE_NONEMPTY_DIRECTORIES["spatial"],
        )
    cfg.autocorr_max_lag = 8
    calls = []
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def rerun(source, passed_cfg, output):
        calls.append(source)
        return _spatial_outputs(output)

    monkeypatch.setattr(partitioned_spatial, "execute_partitioned_spatial", rerun)

    spatial_adata("config.csv")

    assert calls == [paths.preprocess_spine]


def test_spatial_cli_records_failed_stage_when_executor_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fail(source, passed_cfg, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "spine.h5ad").touch()
        raise RuntimeError("simulated spatial plot failure")

    monkeypatch.setattr(partitioned_spatial, "execute_partitioned_spatial", fail)

    with pytest.raises(RuntimeError, match="simulated spatial plot failure"):
        spatial_adata("config.csv")

    entry = read_experiment_manifest(tmp_path)["stages"]["spatial"]
    assert entry["state"] == "failed"
    assert "simulated spatial plot failure" in entry["outcome"]
