from types import SimpleNamespace

import numpy as np
import pandas as pd
from click.testing import CliRunner

from smftools.cli.raw_adata import _attach_direct_signals, _conversion_signal
from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT, PREPROCESS_DIR


def test_conversion_signal_matches_existing_binarization_maps():
    record = {
        "sequence": [
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["C"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["T"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
        ],
        "dataset": "5mC",
        "strand": "top",
        "Read_mismatch_trend": "C->T",
    }
    signal = _conversion_signal(record, deaminase=False)
    np.testing.assert_array_equal(signal, [1.0, 0.0, np.nan])

    record["sequence"] = [
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
    ]
    record["Read_mismatch_trend"] = "G->A"
    signal = _conversion_signal(record, deaminase=True)
    np.testing.assert_array_equal(signal, [0.0, 1.0])


def test_attach_direct_signals_converts_reference_to_query_coordinates(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "chr1",
                "reference_start": 5,
                "cigar": "2M1I2M",
            }
        ]
    )
    calls = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "ref_position": [5, 7],
            "modified_primary_base": ["A", "A"],
            "ref_strand": ["+", "+"],
            "read_id": ["read1", "read1"],
            "call_code": ["a", "-"],
            "call_prob": [0.8, 0.7],
        }
    )
    calls.to_csv(tmp_path / "calls.tsv", sep="\t", index=False)

    result, columns = _attach_direct_signals(frame, tmp_path)
    assert columns == ["modification_signal_A_plus"]
    np.testing.assert_allclose(
        result.at[0, "modification_signal"],
        [0.8, np.nan, np.nan, 0.3, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.at[0, "modification_signal_A_plus"],
        [0.8, np.nan, np.nan, 0.3, np.nan],
        equal_nan=True,
    )


def test_artifact_paths_include_raw_and_dense_outputs(tmp_path):
    from smftools.cli.helpers import get_artifact_paths

    cfg = SimpleNamespace(
        output_directory=tmp_path,
        split_path=tmp_path / "split",
        bam_suffix=".bam",
        input_type="bam",
        input_data_path=tmp_path / "input.bam",
        smf_modality="conversion",
        experiment_name="experiment",
    )
    paths = get_artifact_paths(cfg)
    assert paths.raw_directory == tmp_path / "raw_outputs"
    assert paths.spine == paths.load_directory / "spine.h5ad"
    assert paths.dense_store == paths.load_directory / "store"
    assert paths.dense_catalog == paths.load_directory / "catalog.parquet"
    assert paths.sidecar_manifest.parent == paths.raw_directory


def test_cli_exposes_raw_and_optional_load_commands():
    from smftools.cli_entry import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "raw" in result.output
    assert "Optionally pre-build the dense zarr cache" in result.output


def test_raw_wrapper_stops_legacy_pipeline_before_dense_loading(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.cli import load_adata as load_module
    from smftools.cli.raw_adata import raw_adata

    cfg = SimpleNamespace(output_directory=tmp_path, force_redo_load_adata=False)
    paths = SimpleNamespace(raw_spine=tmp_path / "raw_outputs" / "spine.h5ad")
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    def fake_core(core_cfg, core_paths, config_path=None, *, raw_only=False):
        captured.update(
            cfg=core_cfg,
            paths=core_paths,
            config_path=config_path,
            raw_only=raw_only,
        )
        return "spine", paths.raw_spine, core_cfg

    monkeypatch.setattr(load_module, "load_adata_core", fake_core)
    result = raw_adata("experiment.csv")

    assert result == ("spine", paths.raw_spine, cfg)
    assert captured["raw_only"] is True


def test_load_dense_cache_runs_raw_then_cache_builder(tmp_path, monkeypatch):
    from smftools import readwrite
    from smftools.cli import raw_adata as raw_module
    from smftools.cli.load_adata import load_dense_cache
    from smftools.informatics import partition_store

    cfg = SimpleNamespace(output_directory=tmp_path)
    source_spine = tmp_path / "spine.h5ad"
    monkeypatch.setattr(
        raw_module,
        "raw_adata",
        lambda _path: ("raw-spine", source_spine, cfg),
    )
    monkeypatch.setattr(
        partition_store,
        "write_dense_cache_from_spine",
        lambda path, output_dir=None: {"spine": path},
    )
    monkeypatch.setattr(readwrite, "safe_read_h5ad", lambda path: ("cached-spine", None))

    assert load_dense_cache("experiment.csv") == ("cached-spine", source_spine, cfg)


def test_preprocess_wrapper_accepts_raw_spine_source(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.informatics import partition_read

    raw_spine_path = tmp_path / "raw_outputs" / "spine.h5ad"
    raw_spine_path.parent.mkdir()
    raw_spine_path.touch()
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "load_adata_outputs" / "spine.h5ad",
        raw_spine=raw_spine_path,
        pp=tmp_path / "pp.h5ad.gz",
        pp_dedup=tmp_path / "pp_dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    captured = {}
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)
    monkeypatch.setattr(partition_read, "materialize", lambda path: "materialized")

    def fake_core(**kwargs):
        captured.update(kwargs)
        return paths.pp, paths.pp_dedup

    monkeypatch.setattr(preprocess_module, "preprocess_adata_core", fake_core)
    result = preprocess_module.preprocess_adata("experiment.csv")

    assert result == (paths.pp, paths.pp_dedup)
    assert captured["adata"] == "materialized"
    assert captured["source_adata_path"] == raw_spine_path


def test_preprocess_wrapper_dispatches_planned_spine(tmp_path, monkeypatch):
    import anndata as ad

    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.preprocessing import partitioned_executor

    raw_spine_path = tmp_path / "raw_outputs" / "spine.h5ad"
    raw_spine_path.parent.mkdir()
    spine = ad.AnnData(obs=pd.DataFrame(index=["read1"]))
    spine.uns["reference_plans"] = {"locus_top": {"analysis_mode": "locus"}}
    spine.write_h5ad(raw_spine_path)
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "load_adata_outputs" / "spine.h5ad",
        raw_spine=raw_spine_path,
        pp=tmp_path / "pp.h5ad.gz",
        pp_dedup=tmp_path / "pp_dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        preprocess_execution_mode="auto",
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    output_spine = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)
    monkeypatch.setattr(
        partitioned_executor,
        "execute_partitioned_preprocessing",
        lambda source, config, output: {"spine": output_spine},
    )

    assert preprocess_module.preprocess_adata("experiment.csv") == (output_spine, None)


def test_preprocess_wrapper_returns_existing_partitioned_spine(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module

    output_spine = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    output_spine.parent.mkdir()
    output_spine.touch()
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "missing-load-spine.h5ad",
        raw_spine=tmp_path / "missing-raw-spine.h5ad",
        pp=tmp_path / "missing-pp.h5ad.gz",
        pp_dedup=tmp_path / "missing-dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    assert preprocess_module.preprocess_adata("experiment.csv") == (output_spine, None)
