import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from click.testing import CliRunner

from smftools.cli import helpers
from smftools.cli.latent_adata import latent_adata
from smftools.cli_entry import cli
from smftools.constants import REFERENCE_STRAND
from smftools.informatics.raw_store import write_raw_store
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
    )


def test_latent_cli_prefers_partitioned_hmm_spine(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = helpers.get_adata_paths(cfg)
    for path in (paths.preprocess_spine, paths.spatial_spine, paths.hmm_spine):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    def fake_execute(source, passed_cfg, output):
        captured.update(source=source, cfg=passed_cfg, output=output)
        return {"spine": Path(output) / "spine.h5ad"}

    monkeypatch.setattr(partitioned_latent, "execute_partitioned_latent", fake_execute)

    adata, output_path = latent_adata("config.csv")

    assert adata is None
    assert output_path == paths.latent_spine
    assert captured["source"] == paths.hmm_spine
    assert captured["output"] == paths.latent_spine.parent


def test_latent_cli_partitioned_mode_ignores_legacy_latent_file(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, mode="partitioned")
    paths = helpers.get_adata_paths(cfg)
    paths.preprocess_spine.parent.mkdir(parents=True)
    paths.preprocess_spine.touch()
    paths.latent.parent.mkdir(parents=True)
    paths.latent.touch()

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(
        partitioned_latent,
        "execute_partitioned_latent",
        lambda source, passed_cfg, output: {"spine": Path(output) / "spine.h5ad"},
    )

    _, output_path = latent_adata("config.csv")

    assert output_path == paths.latent_spine


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
    latent_spine, _ = safe_read_h5ad(outputs["spine"])
    assert len(catalog) == 1
    assert latent_spine.uns["latent_task_catalog"] == ("latent_adata_outputs/task_catalog.parquet")
    assert latent_spine.uns["latent_coordinate_scope"] == "reference_core"


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
