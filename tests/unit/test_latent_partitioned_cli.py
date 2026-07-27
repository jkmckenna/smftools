import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from smftools.cli import helpers
from smftools.cli.latent_adata import _resolve_latent_source, latent_adata
from smftools.cli_entry import cli
from smftools.constants import LATENT_DIR, PREPROCESS_DIR, REFERENCE_STRAND
from smftools.informatics.experiment_manifest import (
    read_experiment_manifest,
    stage_is_complete,
)
from smftools.informatics.experiment_spine import experiment_spine_path
from smftools.informatics.raw_store import write_raw_store
from smftools.latent_resource import LatentResourceDecision, LatentResourceError
from smftools.perf_log import PerfLogger, set_perf_logger
from smftools.readwrite import safe_read_h5ad, safe_write_zarr
from smftools.tools import partitioned_latent


def _cfg(tmp_path, *, mode="auto", force=False):
    return SimpleNamespace(
        output_directory=str(tmp_path),
        experiment_name="experiment",
        smf_modality="conversion",
        latent_execution_mode=mode,
        force_redo_latent_analyses=force,
        from_adata_stage=None,
        emit_log_file=False,
        emit_perf_log=False,
        sample_name_col_for_plotting="Sample",
        umap_layers_to_plot=[],
    )


def _partitioned_source(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "ref",
                REFERENCE_STRAND: "ref_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [0.0, 1.0, 0.0, 1.0],
            }
        ]
    )
    raw = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 4},
        analysis_mode="locus",
    )
    source, _ = safe_read_h5ad(raw["spine"])
    source_path = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    from smftools.readwrite import safe_write_h5ad

    safe_write_h5ad(source, source_path, backup=False, verbose=False)
    return source_path


def _install_fake_latent_unit(monkeypatch, calls):
    def fake_unit(spine_path, unit, cfg, output_dir):
        calls.append(str(unit["analysis_core_id"]))
        path = partitioned_latent._task_path(
            Path(output_dir), unit["reference"], unit["core_start"], unit["core_end"]
        )
        result = ad.AnnData(
            obs=pd.DataFrame(index=list(unit["read_ids"])),
            var=pd.DataFrame(index=["0", "1", "2", "3"]),
        )
        safe_write_zarr(result, path, backup=False, verbose=False, zarr_format=3)
        return {
            "reference": str(unit["reference"]),
            "analysis_mode": str(unit["analysis_mode"]),
            "core_start": int(unit["core_start"]),
            "core_end": int(unit["core_end"]),
            "n_reads": result.n_obs,
            "fit_reads": result.n_obs,
            "group_path": path.relative_to(output_dir).as_posix(),
            "obsm_keys": [],
            "varm_keys": [],
            "obs_columns": [],
            "analysis_core_id": str(unit["analysis_core_id"]),
            "analysis_region_ids": tuple(unit["analysis_region_ids"]),
            "analysis_planner_version": int(unit["analysis_planner_version"]),
        }

    monkeypatch.setattr(partitioned_latent, "execute_latent_unit", fake_unit)


def test_latent_cli_prefers_partitioned_hmm_spine(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    for path in (paths.preprocess_spine, paths.spatial_spine, paths.hmm_spine):
        path.parent.mkdir(parents=True, exist_ok=True)
        ad.AnnData().write_h5ad(path)
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fake_execute(source, passed_cfg, output, **_kwargs):
        captured.update(source=source, cfg=passed_cfg, output=output)
        return {
            "spine": Path(output) / "spine.h5ad",
            "generation_id": "test-generation",
            "task_count": 1,
        }

    monkeypatch.setattr(partitioned_latent, "execute_partitioned_latent", fake_execute)
    monkeypatch.setattr(
        helpers,
        "publish_stage_outputs",
        lambda lifecycle, *_args, **_kwargs: lifecycle.complete(),
    )

    adata, output_path = latent_adata("config.csv")

    assert adata is None
    assert output_path == paths.latent_spine
    assert captured["source"] == paths.hmm_spine
    assert captured["output"] == paths.latent_spine.parent


def test_latent_cli_partitioned_mode_ignores_legacy_latent_file(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, mode="partitioned")
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    ad.AnnData().write_h5ad(paths.preprocess_spine)
    paths.latent.parent.mkdir(parents=True)
    paths.latent.touch()

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        partitioned_latent,
        "execute_partitioned_latent",
        lambda source, passed_cfg, output, **_kwargs: {
            "spine": Path(output) / "spine.h5ad",
            "generation_id": "test-generation",
            "task_count": 1,
        },
    )
    monkeypatch.setattr(
        helpers,
        "publish_stage_outputs",
        lambda lifecycle, *_args, **_kwargs: lifecycle.complete(),
    )

    _, output_path = latent_adata("config.csv")

    assert output_path == paths.latent_spine


def test_latent_cli_auto_mode_ignores_legacy_latent_file(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    ad.AnnData().write_h5ad(paths.preprocess_spine)
    paths.latent.parent.mkdir(parents=True)
    paths.latent.touch()
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fake_execute(source, passed_cfg, output, **_kwargs):
        captured.update(source=source, cfg=passed_cfg, output=output)
        return {
            "spine": Path(output) / "spine.h5ad",
            "generation_id": "test-generation",
            "task_count": 1,
        }

    monkeypatch.setattr(partitioned_latent, "execute_partitioned_latent", fake_execute)
    monkeypatch.setattr(
        helpers,
        "publish_stage_outputs",
        lambda lifecycle, *_args, **_kwargs: lifecycle.complete(),
    )

    _, output_path = latent_adata("config.csv")

    assert captured["source"] == paths.preprocess_spine
    assert output_path == paths.latent_spine


@pytest.mark.parametrize(
    ("stage", "path_attr"),
    [
        ("preprocess", "preprocess_spine"),
        ("spatial", "spatial_spine"),
        ("hmm", "hmm_spine"),
    ],
)
def test_partitioned_latent_explicit_stage_selects_named_spine(tmp_path, stage, path_attr):
    cfg = _cfg(tmp_path, mode="partitioned")
    cfg.from_adata_stage = stage
    paths = helpers.get_adata_paths(cfg)
    source = getattr(paths, path_attr)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch()

    resolved, source_kind, resolved_stage = _resolve_latent_source(cfg, paths)

    assert resolved == source
    assert source_kind == "partitioned"
    assert resolved_stage == stage


def test_auto_latent_explicit_stage_precedes_partitioned_priority(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.from_adata_stage = "spatial"
    paths = helpers.get_adata_paths(cfg)
    for source in (paths.spatial_spine, paths.hmm_spine):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()

    resolved, source_kind, resolved_stage = _resolve_latent_source(cfg, paths)

    assert resolved == paths.spatial_spine
    assert source_kind == "partitioned"
    assert resolved_stage == "spatial"


def test_partitioned_latent_missing_requested_stage_is_precise(tmp_path):
    cfg = _cfg(tmp_path, mode="partitioned")
    cfg.from_adata_stage = "hmm"
    paths = helpers.get_adata_paths(cfg)

    with pytest.raises(FileNotFoundError, match=r"stage 'hmm'.*hmm_adata_outputs/spine.h5ad"):
        _resolve_latent_source(cfg, paths)


def test_partitioned_latent_rejects_unsupported_requested_stage(tmp_path):
    cfg = _cfg(tmp_path, mode="partitioned")
    cfg.from_adata_stage = "variant"
    paths = helpers.get_adata_paths(cfg)

    with pytest.raises(
        ValueError,
        match="Allowed stages: preprocess, spatial, hmm",
    ):
        _resolve_latent_source(cfg, paths)


def test_partitioned_latent_never_falls_back_to_legacy_source(tmp_path):
    cfg = _cfg(tmp_path, mode="partitioned")
    paths = helpers.get_adata_paths(cfg)
    paths.hmm.parent.mkdir(parents=True, exist_ok=True)
    paths.hmm.touch()

    with pytest.raises(FileNotFoundError, match="partitioned source spine"):
        _resolve_latent_source(cfg, paths)


def test_auto_latent_never_falls_back_to_legacy_source(tmp_path):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    paths.hmm.parent.mkdir(parents=True, exist_ok=True)
    paths.hmm.touch()

    with pytest.raises(FileNotFoundError, match="partitioned source spine"):
        _resolve_latent_source(cfg, paths)


def test_legacy_latent_never_selects_partitioned_spine(tmp_path):
    cfg = _cfg(tmp_path, mode="legacy")
    paths = helpers.get_adata_paths(cfg)
    paths.hmm_spine.parent.mkdir(parents=True, exist_ok=True)
    paths.hmm_spine.touch()

    resolved, source_kind, resolved_stage = _resolve_latent_source(cfg, paths)

    assert resolved is None
    assert source_kind == "legacy"
    assert resolved_stage is None


def test_legacy_latent_explicit_stage_selects_monolithic_artifact(tmp_path):
    cfg = _cfg(tmp_path, mode="legacy")
    cfg.from_adata_stage = "hmm"
    paths = helpers.get_adata_paths(cfg)
    paths.hmm.parent.mkdir(parents=True, exist_ok=True)
    paths.hmm.touch()
    paths.hmm_spine.parent.mkdir(parents=True, exist_ok=True)
    paths.hmm_spine.touch()

    resolved, source_kind, resolved_stage = _resolve_latent_source(cfg, paths)

    assert resolved == paths.hmm
    assert source_kind == "legacy"
    assert resolved_stage == "hmm"


def test_legacy_latent_missing_requested_stage_is_precise(tmp_path):
    cfg = _cfg(tmp_path, mode="legacy")
    cfg.from_adata_stage = "spatial"
    paths = helpers.get_adata_paths(cfg)

    with pytest.raises(
        FileNotFoundError,
        match=r"stage 'spatial'.*spatial_adata_outputs/.+_spatial.h5ad.gz",
    ):
        _resolve_latent_source(cfg, paths)


def test_analysis_units_are_reference_or_core_local():
    obs = pd.DataFrame(
        {
            REFERENCE_STRAND: ["locus_top", "genome_top", "genome_top"],
            "reference_start": [0, 0, 6],
            "reference_end": [8, 4, 10],
            "passes_qc": [True, True, True],
        },
        index=["locus_read", "genome_left", "genome_right"],
    )
    spine = ad.AnnData(obs=obs)
    spine.uns["reference_plans"] = {
        "locus_top": {
            "analysis_mode": "locus",
            "reference_length": 8,
            "tile_size": 8,
        },
        "genome_top": {
            "analysis_mode": "genome",
            "reference_length": 10,
            "tile_size": 5,
        },
    }

    units = partitioned_latent._analysis_units(spine, "passes_qc")

    assert [(unit["reference"], unit["core_start"], unit["core_end"]) for unit in units] == [
        ("genome_top", 0, 5),
        ("genome_top", 5, 10),
        ("locus_top", 0, 8),
    ]


def test_batch_latent_dispatches_standalone_stage(tmp_path, monkeypatch):
    import smftools.cli.latent_adata as latent_module

    captured = []
    monkeypatch.setattr(latent_module, "latent_adata", captured.append)
    config = tmp_path / "experiment.csv"
    config.touch()
    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config}\n")

    result = CliRunner().invoke(cli, ["experiment", "batch", "latent", str(config_table)])

    assert result.exit_code == 0, result.output
    assert captured == [str(config)]


def test_fitted_latent_space_transforms_additional_reads():
    fit = ad.AnnData(np.empty((6, 4)))
    fit.layers["signal"] = np.asarray(
        [
            [0.0, 0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.1],
            [0.9, 1.0, 0.9, 1.0],
            [1.0, 0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cfg = SimpleNamespace(
        latent_random_state=0,
        latent_run_pca_umap=True,
        latent_run_nmf=True,
        latent_n_pcs=3,
        latent_knn_neighbors=3,
        latent_leiden_resolution=0.1,
        latent_nmf_components=1,
        latent_nmf_max_iter=500,
        threads=1,
    )

    fitted = partitioned_latent._fit_matrix_representations(
        fit,
        layer="signal",
        mask=np.ones(4, dtype=bool),
        suffix="test",
        cfg=cfg,
        fit_indices=np.arange(fit.n_obs),
    )
    extra = ad.AnnData(np.empty((2, 4)))
    extra.layers["signal"] = np.asarray(
        [[0.05, 0.05, 0.0, 0.1], [0.95, 0.95, 1.0, 0.9]], dtype=np.float32
    )

    transformed = partitioned_latent._transform_matrix_representations(extra, fitted)

    assert transformed["X_pca_test"].shape == (2, 3)
    assert transformed["X_umap_test"].shape == (2, 2)
    assert transformed["X_nmf_test"].shape == (2, 1)
    assert transformed["leiden_test"].shape == (2,)


def test_partitioned_latent_publishes_catalog_and_thin_spine(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "ref",
                REFERENCE_STRAND: "ref_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [0.0, 1.0, 0.0, 1.0],
            }
        ]
    )
    raw = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 4},
        analysis_mode="locus",
    )

    def fake_unit(spine_path, unit, cfg, output_dir):
        path = partitioned_latent._task_path(
            Path(output_dir), unit["reference"], unit["core_start"], unit["core_end"]
        )
        result = ad.AnnData(
            obs=pd.DataFrame(index=["read1"]),
            var=pd.DataFrame(index=["0", "1", "2", "3"]),
        )
        safe_write_zarr(result, path, backup=False, verbose=False, zarr_format=3)
        return {
            "reference": "ref_top",
            "analysis_mode": "locus",
            "core_start": 0,
            "core_end": 4,
            "n_reads": 1,
            "fit_reads": 1,
            "group_path": path.relative_to(output_dir).as_posix(),
            "obsm_keys": [],
            "varm_keys": [],
            "obs_columns": [],
        }

    monkeypatch.setattr(partitioned_latent, "execute_latent_unit", fake_unit)
    cfg = SimpleNamespace(sample_name_col_for_plotting="Sample", umap_layers_to_plot=[])

    outputs = partitioned_latent.execute_partitioned_latent(
        raw["spine"], cfg, tmp_path / "latent_adata_outputs"
    )

    catalog = pd.read_parquet(outputs["task_catalog"])
    resource_plan = json.loads(Path(outputs["resource_plan"]).read_text(encoding="utf-8"))
    generation_manifest = json.loads(
        Path(outputs["generation_manifest"]).read_text(encoding="utf-8")
    )
    latent_spine, _ = safe_read_h5ad(outputs["spine"])
    assert len(catalog) == 1
    assert catalog.loc[0, "resource_estimator_version"] == "1"
    assert catalog.loc[0, "effective_fit_reads"] == catalog.loc[0, "fit_reads"]
    assert resource_plan["task_count"] == 1
    assert resource_plan["estimator_version"] == "1"
    assert resource_plan["units"][0]["resource_envelope_id"]
    assert resource_plan["units"][0]["decisions"][0]["operation"] == "plot"
    assert generation_manifest["resource_plan"] == "resource_plan.json"
    assert generation_manifest["resource_summary"]["requested_fit_reads_ceiling"] == 5000
    assert (
        latent_spine.uns["latent_task_catalog"]
        == Path(outputs["task_catalog"]).relative_to(tmp_path).as_posix()
    )
    assert (
        latent_spine.uns["latent_resource_plan"]
        == Path(outputs["resource_plan"]).relative_to(tmp_path).as_posix()
    )
    assert (
        latent_spine.uns["latent_read_index"]
        == Path(outputs["read_index"]).relative_to(tmp_path).as_posix()
    )
    assert pd.read_parquet(outputs["read_index"])["stage"].tolist() == ["latent"]
    assert latent_spine.uns["latent_coordinate_scope"] == "reference_core"
    assert Path(outputs["generation"]).parent.name == "generations"


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "missing group_row"),
        ("duplicate", "duplicate group_row"),
        ("out_of_range", "out-of-range group_row"),
    ],
)
def test_latent_publication_validation_rejects_invalid_group_rows(
    tmp_path, monkeypatch, damage, message
):
    source_path = _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    _install_fake_latent_unit(monkeypatch, [])
    outputs = partitioned_latent.execute_partitioned_latent(
        source_path,
        cfg,
        tmp_path / LATENT_DIR,
    )
    index_root = Path(outputs["read_index"])
    index = pd.read_parquet(index_root)
    if damage == "missing":
        index.loc[0, "group_row"] = pd.NA
    elif damage == "duplicate":
        index = pd.concat([index, index.iloc[[0]]], ignore_index=True)
    else:
        index.loc[0, "group_row"] = 99

    shutil.rmtree(index_root)
    for bucket, frame in index.groupby("molecule_bucket", dropna=False, observed=True):
        bucket_dir = index_root / f"molecule_bucket={bucket}"
        bucket_dir.mkdir(parents=True)
        frame.drop(columns=["molecule_bucket"]).to_parquet(
            bucket_dir / "damaged.parquet",
            index=False,
        )

    with pytest.raises(RuntimeError, match=message):
        partitioned_latent._validate_latent_generation(
            Path(outputs["generation"]),
            final_dir=Path(outputs["generation"]),
            run_root=tmp_path,
        )


def test_latent_cli_skips_only_compatible_complete_generation(tmp_path, monkeypatch):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    _, first_spine = latent_adata("config.csv")
    first = read_experiment_manifest(tmp_path)["stages"]["latent"]
    _, second_spine = latent_adata("config.csv")

    assert first_spine == second_spine == tmp_path / LATENT_DIR / "spine.h5ad"
    assert len(calls) == 1
    assert first["state"] == "complete"
    assert (
        read_experiment_manifest(tmp_path)["stages"]["latent"]["generation_id"]
        == first["generation_id"]
    )


def test_latent_cli_does_not_trust_unmanifested_canonical_spine(tmp_path, monkeypatch):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    canonical = tmp_path / LATENT_DIR / "spine.h5ad"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData().write_h5ad(canonical)

    latent_adata("config.csv")

    assert len(calls) == 1
    assert read_experiment_manifest(tmp_path)["stages"]["latent"]["state"] == "complete"


def test_latent_cli_plot_only_change_reuses_compute_generation(tmp_path, monkeypatch):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    latent_adata("config.csv")
    first = read_experiment_manifest(tmp_path)["stages"]["latent"]
    cfg.umap_layers_to_plot = ["mapped_length"]
    latent_adata("config.csv")
    second = read_experiment_manifest(tmp_path)["stages"]["latent"]

    assert len(calls) == 1
    assert second["generation_id"] != first["generation_id"]
    assert second["reused_compute_generation"] == first["generation_id"]


def test_latent_cli_compute_or_source_change_creates_fresh_generation(tmp_path, monkeypatch):
    source_path = _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.latent_n_pcs = 3
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    latent_adata("config.csv")
    first = read_experiment_manifest(tmp_path)["stages"]["latent"]
    cfg.latent_n_pcs = 4
    latent_adata("config.csv")
    second = read_experiment_manifest(tmp_path)["stages"]["latent"]
    source, _ = safe_read_h5ad(source_path)
    source.uns["source_revision"] = "changed"
    from smftools.readwrite import safe_write_h5ad

    safe_write_h5ad(source, source_path, backup=False, verbose=False)
    latent_adata("config.csv")
    third = read_experiment_manifest(tmp_path)["stages"]["latent"]

    assert len(calls) == 3
    assert len({first["generation_id"], second["generation_id"], third["generation_id"]}) == 3
    assert second["reused_compute_generation"] is None
    assert third["reused_compute_generation"] is None


def test_latent_cli_force_redo_creates_clean_generation(tmp_path, monkeypatch):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    latent_adata("config.csv")
    first = read_experiment_manifest(tmp_path)["stages"]["latent"]
    cfg.force_redo_latent_analyses = True
    latent_adata("config.csv")
    second = read_experiment_manifest(tmp_path)["stages"]["latent"]

    assert len(calls) == 2
    assert second["generation_id"] != first["generation_id"]
    assert second["reused_compute_generation"] is None
    generations = list((tmp_path / LATENT_DIR / "generations").iterdir())
    assert {path.name for path in generations} == {
        first["generation_id"],
        second["generation_id"],
    }


def test_latent_stage_completion_rejects_task_store_checksum_change(tmp_path, monkeypatch):
    source_path = _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    latent_adata("config.csv")
    entry = read_experiment_manifest(tmp_path)["stages"]["latent"]
    store = tmp_path / entry["artifacts"]["store"]["path"]
    victim = next(path for path in store.rglob("*") if path.is_file())
    victim.write_bytes(victim.read_bytes() + b"corrupt")

    assert not stage_is_complete(
        tmp_path,
        "latent",
        config_hash=entry["config_hash"],
        required_artifacts=("store",),
    )
    latent_adata("config.csv")
    assert len(calls) == 2


def test_failed_replacement_preserves_prior_complete_generation(tmp_path, monkeypatch):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    _install_fake_latent_unit(monkeypatch, calls)
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    latent_adata("config.csv")
    first = read_experiment_manifest(tmp_path)["stages"]["latent"]
    first_generation = tmp_path / first["artifacts"]["generation"]["path"]
    first_spine, _ = safe_read_h5ad(first_generation / "spine.h5ad")
    cfg.force_redo_latent_analyses = True
    monkeypatch.setattr(
        partitioned_latent,
        "_plot_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plot failure")),
    )

    with pytest.raises(RuntimeError, match="plot failure"):
        latent_adata("config.csv")

    failed = read_experiment_manifest(tmp_path)["stages"]["latent"]
    restored, _ = safe_read_h5ad(first_generation / "spine.h5ad")
    assert failed["state"] == "failed"
    assert failed["previous_complete"]["generation_id"] == first["generation_id"]
    assert first_generation.is_dir()
    assert restored.uns["latent_generation_id"] == first_spine.uns["latent_generation_id"]
    assert not any((tmp_path / LATENT_DIR / ".staging").iterdir())


@pytest.mark.parametrize("failure_task", [1, 2])
def test_latent_task_failure_never_publishes_partial_generation(
    tmp_path, monkeypatch, failure_task
):
    source_path = _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    source, _ = safe_read_h5ad(source_path)
    unit = partitioned_latent._analysis_units(source, None, spine_path=source_path)[0]
    units = [dict(unit), dict(unit)]
    units[1]["analysis_core_id"] = "second-core"
    units[1]["core_start"] = 1
    units[1]["core_end"] = 4
    monkeypatch.setattr(
        partitioned_latent,
        "_analysis_units",
        lambda *_args, **_kwargs: units,
    )
    calls = []

    def fail_unit(spine_path, current, passed_cfg, output_dir):
        calls.append(current["analysis_core_id"])
        if len(calls) == failure_task:
            raise RuntimeError(f"task {failure_task} failure")
        path = partitioned_latent._task_path(
            Path(output_dir),
            current["reference"],
            current["core_start"],
            current["core_end"],
        )
        result = ad.AnnData(
            obs=pd.DataFrame(index=current["read_ids"]),
            var=pd.DataFrame(index=["0", "1", "2", "3"]),
        )
        safe_write_zarr(result, path, backup=False, verbose=False, zarr_format=3)
        return {
            "reference": current["reference"],
            "analysis_mode": current["analysis_mode"],
            "core_start": current["core_start"],
            "core_end": current["core_end"],
            "n_reads": result.n_obs,
            "fit_reads": result.n_obs,
            "group_path": path.relative_to(output_dir).as_posix(),
            "obsm_keys": [],
            "varm_keys": [],
            "obs_columns": [],
            "analysis_core_id": current["analysis_core_id"],
            "analysis_region_ids": (),
            "analysis_planner_version": 1,
        }

    monkeypatch.setattr(partitioned_latent, "execute_latent_unit", fail_unit)

    with pytest.raises(RuntimeError, match=f"task {failure_task} failure"):
        partitioned_latent.execute_partitioned_latent(source_path, cfg, tmp_path / LATENT_DIR)

    assert not (tmp_path / LATENT_DIR / "spine.h5ad").exists()
    assert not list((tmp_path / LATENT_DIR / "generations").iterdir())
    assert not any((tmp_path / LATENT_DIR / ".staging").iterdir())
    experiment_spine, _ = safe_read_h5ad(experiment_spine_path(tmp_path))
    assert "latent_task_catalog" not in experiment_spine.uns


@pytest.mark.parametrize(
    ("failure_target", "message"),
    [
        ("catalog", "after catalog"),
        ("plot", "during plot"),
        ("validation", "before publication"),
    ],
)
def test_latent_post_compute_failure_never_publishes_generation(
    tmp_path, monkeypatch, failure_target, message
):
    source_path = _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    _install_fake_latent_unit(monkeypatch, [])
    if failure_target == "catalog":
        monkeypatch.setattr(
            partitioned_latent,
            "prepare_analysis_plot_layout",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )
    elif failure_target == "plot":
        monkeypatch.setattr(
            partitioned_latent,
            "_plot_task",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )
    else:
        monkeypatch.setattr(
            partitioned_latent,
            "_validate_latent_generation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )

    with pytest.raises(RuntimeError, match=message):
        partitioned_latent.execute_partitioned_latent(source_path, cfg, tmp_path / LATENT_DIR)

    assert not (tmp_path / LATENT_DIR / "spine.h5ad").exists()
    assert not list((tmp_path / LATENT_DIR / "generations").iterdir())
    assert not any((tmp_path / LATENT_DIR / ".staging").iterdir())


@pytest.mark.parametrize("damage", ["missing", "unreadable", "checksum"])
def test_latent_completion_rejects_damaged_task_store(tmp_path, monkeypatch, damage):
    _partitioned_source(tmp_path)
    cfg = _cfg(tmp_path)
    _install_fake_latent_unit(monkeypatch, [])
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    latent_adata("config.csv")
    entry = read_experiment_manifest(tmp_path)["stages"]["latent"]
    store = tmp_path / entry["artifacts"]["store"]["path"]
    group = next(path for path in store.rglob("core=*") if path.is_dir())

    if damage == "missing":
        shutil.rmtree(group)
    elif damage == "unreadable":
        (group / "zarr.json").write_text("{", encoding="utf-8")
    else:
        (group / "unexpected.bin").write_bytes(b"checksum mismatch")

    assert not stage_is_complete(
        tmp_path,
        "latent",
        config_hash=entry["config_hash"],
        required_artifacts=("store",),
    )


def test_latent_unit_without_representations_is_skipped(tmp_path, monkeypatch):
    result = ad.AnnData(
        obs=pd.DataFrame(index=["read1", "read2", "read3"]),
        var=pd.DataFrame(index=["0", "1", "2", "3"]),
    )
    result.layers["nan_half"] = np.zeros((3, 4), dtype=np.float32)
    result.layers["sequence_integer_encoding"] = np.zeros((3, 4), dtype=np.float32)
    monkeypatch.setattr(partitioned_latent, "materialize", lambda *args, **kwargs: result.copy())
    monkeypatch.setattr(
        partitioned_latent,
        "_build_mod_sites_var_filter_mask",
        lambda *args, **kwargs: np.zeros(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_shared_valid_non_mod_sites_mask",
        lambda *args, **kwargs: np.zeros(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_reference_position_mask",
        lambda *args, **kwargs: np.zeros(4, dtype=bool),
    )
    cfg = SimpleNamespace(
        latent_min_reads=3,
        latent_max_fit_reads=5000,
        latent_random_state=0,
        latent_run_pca_umap=True,
        latent_run_nmf=True,
        latent_run_cp=False,
        layer_for_umap_plotting="nan_half",
        smf_modality="conversion",
    )
    unit = {
        "reference": "ref_top",
        "analysis_mode": "locus",
        "core_start": 0,
        "core_end": 4,
        "read_ids": ["read1", "read2", "read3"],
    }

    record = partitioned_latent.execute_latent_unit("spine.h5ad", unit, cfg, tmp_path)

    assert record is None
    assert not (tmp_path / partitioned_latent.LATENT_STORE_SUBDIR).exists()


def test_latent_plot_colors_drop_constants_and_include_matching_leiden():
    result = ad.AnnData(
        obs=pd.DataFrame(
            {
                "Sample": ["sample1", "sample2"],
                REFERENCE_STRAND: ["ref_top", "ref_top"],
                "mapped_length": [100, 200],
                "leiden_signal": ["0", "1"],
                "leiden_sequence": ["0", "0"],
            },
            index=["read1", "read2"],
        )
    )
    cfg = SimpleNamespace(
        sample_name_col_for_plotting="Sample",
        umap_layers_to_plot=[REFERENCE_STRAND, "mapped_length"],
    )

    colors = partitioned_latent._plot_colors(result, "umap_signal", cfg)

    assert colors == ["Sample", "mapped_length", "leiden_signal"]


def test_latent_plot_subset_is_deterministic_and_bounded(tmp_path):
    result = ad.AnnData(
        obs=pd.DataFrame(index=[f"read{index}" for index in range(20)]),
        var=pd.DataFrame(index=["0", "1"]),
    )
    path = tmp_path / "result.zarr"
    safe_write_zarr(result, path, backup=False, verbose=False, zarr_format=3)

    first = partitioned_latent._read_plot_subset(path, max_reads=5, seed=17)
    second = partitioned_latent._read_plot_subset(path, max_reads=5, seed=17)

    assert first.n_obs == 5
    assert list(first.obs_names) == list(second.obs_names)


@pytest.mark.parametrize(
    ("policy", "raises"),
    [("skip", False), ("fail", True)],
)
def test_latent_cp_memory_policy_is_deterministic(tmp_path, monkeypatch, policy, raises):
    materialized = ad.AnnData(
        obs=pd.DataFrame(index=["read1", "read2", "read3"]),
        var=pd.DataFrame(index=["0", "1", "2", "3"]),
    )
    materialized.layers["nan_half"] = np.zeros((3, 4), dtype=np.float32)
    materialized.layers["sequence_integer_encoding"] = np.zeros((3, 4), dtype=np.float32)
    monkeypatch.setattr(
        partitioned_latent,
        "materialize",
        lambda *_args, **_kwargs: materialized.copy(),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_mod_sites_var_filter_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_shared_valid_non_mod_sites_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_reference_position_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )

    def fake_fit(adata, *, suffix, **_kwargs):
        adata.obsm[f"X_pca_{suffix}"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        return {"suffix": suffix}

    monkeypatch.setattr(partitioned_latent, "_fit_matrix_representations", fake_fit)
    cp_calls = []
    monkeypatch.setattr(
        partitioned_latent,
        "_fit_cp_representations",
        lambda *_args, **_kwargs: cp_calls.append(True),
    )

    def fake_decision(
        _cfg,
        operation,
        *,
        requested_reads,
        n_positions,
        minimum_reads,
        **_kwargs,
    ):
        effective = 1 if operation == "cp" else requested_reads
        return LatentResourceDecision(
            operation=operation,
            estimator_version="1",
            requested_reads=requested_reads,
            effective_reads=effective,
            minimum_reads=minimum_reads,
            n_positions=n_positions,
            usable_headroom_bytes=1024,
            predicted_peak_bytes=512,
            limiting_operation=operation if effective < requested_reads else None,
            pool_budget={},
            estimate={},
        )

    monkeypatch.setattr(partitioned_latent, "resolve_latent_operation", fake_decision)
    cfg = SimpleNamespace(
        latent_min_reads=3,
        latent_max_fit_reads=3,
        latent_transform_chunk_reads=2,
        latent_random_state=0,
        latent_run_pca_umap=True,
        latent_run_nmf=False,
        latent_run_cp=True,
        latent_cp_memory_policy=policy,
        latent_plot_max_reads=10,
        layer_for_umap_plotting="nan_half",
        smf_modality="conversion",
    )
    unit = {
        "reference": "ref_top",
        "analysis_mode": "locus",
        "core_start": 0,
        "core_end": 4,
        "read_ids": ["read1", "read2", "read3"],
    }

    if raises:
        with pytest.raises(LatentResourceError, match="minimum_unit_exceeds_memory"):
            partitioned_latent.execute_latent_unit("spine.h5ad", unit, cfg, tmp_path)
    else:
        record = partitioned_latent.execute_latent_unit("spine.h5ad", unit, cfg, tmp_path)
        assert record["cp_skip_reason"] == "minimum_unit_exceeds_memory"
        assert cp_calls == []


def test_latent_unit_applies_effective_fit_and_transform_counts(tmp_path, monkeypatch):
    read_ids = [f"read{index}" for index in range(6)]

    def fake_materialize(
        _spine_path,
        *,
        read_ids,
        **_kwargs,
    ):
        result = ad.AnnData(
            obs=pd.DataFrame(index=list(read_ids)),
            var=pd.DataFrame(index=["0", "1", "2", "3"]),
        )
        result.layers["nan_half"] = np.zeros((len(read_ids), 4), dtype=np.float32)
        result.layers["sequence_integer_encoding"] = np.zeros((len(read_ids), 4), dtype=np.float32)
        return result

    monkeypatch.setattr(partitioned_latent, "materialize", fake_materialize)
    monkeypatch.setattr(
        partitioned_latent,
        "load_spine",
        lambda *_args, **_kwargs: ad.AnnData(obs=pd.DataFrame(index=read_ids)),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_mod_sites_var_filter_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_shared_valid_non_mod_sites_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_reference_position_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )

    def fake_fit(adata, *, suffix, **_kwargs):
        key = f"X_pca_{suffix}"
        adata.obsm[key] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        return {"suffix": suffix}

    monkeypatch.setattr(partitioned_latent, "_fit_matrix_representations", fake_fit)
    monkeypatch.setattr(
        partitioned_latent,
        "_transform_matrix_representations",
        lambda chunk, fitted: {
            f"X_pca_{fitted['suffix']}": np.ones((chunk.n_obs, 2), dtype=np.float32)
        },
    )
    operations = []

    def fake_decision(
        _cfg,
        operation,
        *,
        requested_reads,
        n_positions,
        minimum_reads,
        **_kwargs,
    ):
        operations.append(operation)
        effective = {"fit": 3, "transform": 2}.get(operation, requested_reads)
        return LatentResourceDecision(
            operation=operation,
            estimator_version="1",
            requested_reads=requested_reads,
            effective_reads=effective,
            minimum_reads=minimum_reads,
            n_positions=n_positions,
            usable_headroom_bytes=1024,
            predicted_peak_bytes=512,
            limiting_operation=operation if effective < requested_reads else None,
            pool_budget={},
            estimate={},
        )

    monkeypatch.setattr(partitioned_latent, "resolve_latent_operation", fake_decision)
    cfg = SimpleNamespace(
        latent_min_reads=3,
        latent_max_fit_reads=5,
        latent_transform_chunk_reads=4,
        latent_random_state=0,
        latent_run_pca_umap=True,
        latent_run_nmf=False,
        latent_run_cp=False,
        latent_plot_max_reads=10,
        layer_for_umap_plotting="nan_half",
        smf_modality="conversion",
    )
    unit = {
        "reference": "ref_top",
        "analysis_mode": "locus",
        "core_start": 0,
        "core_end": 4,
        "read_ids": read_ids,
    }

    record = partitioned_latent.execute_latent_unit("spine.h5ad", unit, cfg, tmp_path)

    assert record["n_reads"] == 6
    assert record["requested_fit_reads"] == 5
    assert record["effective_fit_reads"] == 3
    assert record["requested_transform_chunk_reads"] == 4
    assert record["effective_transform_chunk_reads"] == 2
    assert operations == ["fit", "result", "transform", "write"]


def test_latent_sequential_memory_sample_updates_perf_summary(tmp_path):
    path = tmp_path / "latent_perf.jsonl"
    perf = PerfLogger(path, "latent")
    set_perf_logger(perf)
    try:
        partitioned_latent._record_memory_sample("unit_complete:ref:0-4")
    finally:
        set_perf_logger(None)
        perf.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    sample = next(record for record in records if record["event"] == "sample")
    assert sample["sample_label"] == "unit_complete:ref:0-4"
    assert sample["tree_rss_gb"] > 0
    assert records[-1]["peak_tree_rss_gb"] > 0


def test_latent_growth_reuses_model_and_transforms_only_new_rows(tmp_path, monkeypatch):
    initial_ids = [f"read-{index}" for index in range(5)]
    selection_obs = pd.DataFrame(
        {"experiment_uid": "injected", "molecule_uid": initial_ids},
        index=initial_ids,
    )
    _, initial_fit = partitioned_latent.deterministic_fit_membership(
        selection_obs,
        initial_ids,
        limit=3,
        random_state=0,
        coordinate_owner="core-a",
    )
    new_id = None
    for index in range(100, 200):
        candidate = f"read-{index}"
        expanded_ids = [*initial_ids, candidate]
        expanded_obs = pd.DataFrame(
            {"experiment_uid": "injected", "molecule_uid": expanded_ids},
            index=expanded_ids,
        )
        _, expanded_fit = partitioned_latent.deterministic_fit_membership(
            expanded_obs,
            expanded_ids,
            limit=3,
            random_state=0,
            coordinate_owner="core-a",
        )
        if expanded_fit == initial_fit:
            new_id = candidate
            break
    assert new_id is not None

    def fake_materialize(_spine_path, *, read_ids, **_kwargs):
        read_ids = list(read_ids)
        result = ad.AnnData(
            obs=pd.DataFrame(
                {"experiment_uid": "injected", "molecule_uid": read_ids},
                index=read_ids,
            ),
            var=pd.DataFrame(index=["0", "1", "2", "3"]),
        )
        values = np.asarray(
            [[float(read_id.split("-")[-1])] * 4 for read_id in read_ids],
            dtype=np.float32,
        )
        result.layers["nan_half"] = values
        result.layers["sequence_integer_encoding"] = values
        return result

    monkeypatch.setattr(partitioned_latent, "materialize", fake_materialize)
    monkeypatch.setattr(
        partitioned_latent,
        "_build_mod_sites_var_filter_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_shared_valid_non_mod_sites_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    monkeypatch.setattr(
        partitioned_latent,
        "_build_reference_position_mask",
        lambda *_args, **_kwargs: np.ones(4, dtype=bool),
    )
    fit_calls = []

    def fake_fit(adata, *, layer, mask, suffix, **_kwargs):
        fit_calls.append(suffix)
        adata.obsm[f"X_pca_{suffix}"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        return {"suffix": suffix, "layer": layer, "mask": mask, "pca": "fitted"}

    transform_calls = []

    def fake_transform(chunk, fitted):
        transform_calls.extend(chunk.obs_names.astype(str).tolist())
        return {f"X_pca_{fitted['suffix']}": np.ones((chunk.n_obs, 2), dtype=np.float32)}

    monkeypatch.setattr(partitioned_latent, "_fit_matrix_representations", fake_fit)
    monkeypatch.setattr(partitioned_latent, "_transform_matrix_representations", fake_transform)

    def fake_decision(
        _cfg,
        operation,
        *,
        requested_reads,
        n_positions,
        minimum_reads,
        **_kwargs,
    ):
        effective = 3 if operation == "fit" else requested_reads
        return LatentResourceDecision(
            operation=operation,
            estimator_version="1",
            requested_reads=requested_reads,
            effective_reads=effective,
            minimum_reads=minimum_reads,
            n_positions=n_positions,
            usable_headroom_bytes=1024,
            predicted_peak_bytes=512,
            limiting_operation=None,
            pool_budget={},
            estimate={},
        )

    monkeypatch.setattr(partitioned_latent, "resolve_latent_operation", fake_decision)
    cfg = SimpleNamespace(
        latent_min_reads=3,
        latent_max_fit_reads=3,
        latent_transform_chunk_reads=2,
        latent_random_state=0,
        latent_run_pca_umap=True,
        latent_run_nmf=False,
        latent_run_cp=False,
        latent_plot_max_reads=10,
        layer_for_umap_plotting="nan_half",
        smf_modality="conversion",
    )
    first_dir = tmp_path / "first"
    first_unit = {
        "reference": "ref_top",
        "analysis_mode": "locus",
        "analysis_core_id": "core-a",
        "core_start": 0,
        "core_end": 4,
        "read_ids": initial_ids,
        "_source_identity": {"source_stage_generation_id": "source-a"},
    }
    first = partitioned_latent.execute_latent_unit(
        "injected-spine.h5ad", first_unit, cfg, first_dir
    )
    first_result, _ = partitioned_latent.safe_read_zarr(
        first_dir / first["group_path"], verbose=False
    )
    old_coordinates = {key: np.asarray(value).copy() for key, value in first_result.obsm.items()}

    fit_calls.clear()
    transform_calls.clear()
    second_dir = tmp_path / "second"
    second_unit = {
        **first_unit,
        "read_ids": [*initial_ids, new_id],
        "_source_identity": {"source_stage_generation_id": "source-b"},
        "_prior_generation": first_dir.as_posix(),
        "_prior_record": first,
    }
    second = partitioned_latent.execute_latent_unit(
        "injected-spine.h5ad", second_unit, cfg, second_dir
    )
    second_result, _ = partitioned_latent.safe_read_zarr(
        second_dir / second["group_path"], verbose=False
    )

    assert second["model_id"] == first["model_id"]
    assert second["model_checksum"] == first["model_checksum"]
    assert not fit_calls
    assert transform_calls == [new_id, new_id]
    for key, values in old_coordinates.items():
        np.testing.assert_array_equal(
            np.asarray(second_result.obsm[key])[: len(initial_ids)], values
        )
