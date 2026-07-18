from pathlib import Path
from types import SimpleNamespace

from smftools.cli import helpers
from smftools.cli.spatial_adata import spatial_adata
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


def test_spatial_cli_auto_selects_partitioned_preprocess_spine(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fake_execute(source, passed_cfg, output):
        captured.update(source=source, cfg=passed_cfg, output=output)
        output_spine = Path(output) / "spine.h5ad"
        return {"spine": output_spine}

    monkeypatch.setattr(partitioned_spatial, "execute_partitioned_spatial", fake_execute)

    adata, output_path = spatial_adata("config.csv")

    assert adata is None
    assert output_path == paths.spatial_spine
    assert captured["source"] == paths.preprocess_spine
    assert captured["output"] == paths.spatial_spine.parent


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
        lambda source, passed_cfg, output: {"spine": Path(output) / "spine.h5ad"},
    )

    _, output_path = spatial_adata("config.csv")

    assert output_path == paths.spatial_spine
