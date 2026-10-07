import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.cli.hmm_adata import (
    _feature_ranges_for_merged_layer,
    _resolve_pos_mask_for_methbase,
    hmm_adata,
)
from smftools.hmm.HMM import SingleBernoulliHMM
from smftools.informatics.experiment_manifest import read_experiment_manifest
from smftools.informatics.partition_read import materialize, relative_uns_path
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.readwrite import safe_read_h5ad, safe_read_zarr
from smftools.tools.partitioned_hmm import (
    _apply_merges,
    _matching_hmm_layers,
    execute_partitioned_hmm,
)


def _frame():
    return pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [1.0, np.nan, 0.0, 1.0],
                "read_length": 4,
                "mapped_length": 4,
                "reference_length": 12,
                "read_quality": 30,
                "mapping_quality": 60,
                "read_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_read_length_ratio": 1.0,
            },
            {
                "read_id": "read2",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 5,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [31, 31, 31, 31],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [0.0, 1.0, 1.0, 0.0],
                "read_length": 4,
                "mapped_length": 4,
                "reference_length": 12,
                "read_quality": 31,
                "mapping_quality": 50,
                "read_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_read_length_ratio": 1.0,
            },
        ]
    )


def _preprocess_cfg():
    return SimpleNamespace(
        smf_modality="conversion",
        output_binary_layer_name="binarized_methylation",
        bypass_clean_nan=False,
        clean_nan_layers=["nan0_0minus1", "nan_half"],
        reference_column="Reference_strand",
        mod_target_bases=["GpC", "CpG"],
        bypass_append_base_context=False,
        target_task_memory_mb=1,
        position_max_nan_threshold=0.6,
        read_len_filter_thresholds=[None, None],
        mapped_len_filter_thresholds=[None, None],
        read_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_read_len_ratio_filter_thresholds=[None, None],
        read_quality_filter_thresholds=[None, None],
        read_mapping_quality_filter_thresholds=[None, None],
        bypass_filter_reads_on_length_quality_mapping=False,
        read_mod_filtering_gpc_thresholds=None,
        read_mod_filtering_cpg_thresholds=None,
        read_mod_filtering_c_thresholds=None,
        read_mod_filtering_a_thresholds=None,
        read_mod_filtering_use_other_c_as_background=False,
        min_valid_fraction_positions_in_read_vs_ref=None,
        bypass_filter_reads_on_modification_thresholds=False,
        bypass_flag_duplicate_reads=True,
        sample_name_col_for_plotting="Sample",
    )


def test_hmm_wrapper_dispatches_partitioned_spatial_spine(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.tools import partitioned_hmm

    spatial_spine = tmp_path / "spatial_adata_outputs" / "spine.h5ad"
    spatial_spine.parent.mkdir()
    spatial_spine.touch()
    paths = SimpleNamespace(
        hmm=tmp_path / "missing_hmm.h5ad.gz",
        hmm_spine=tmp_path / "hmm_adata_outputs" / "spine.h5ad",
        spatial_spine=spatial_spine,
        preprocess_spine=None,
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        hmm_execution_mode="auto",
        force_redo_hmm_fit=False,
        force_redo_hmm_apply=False,
        force_redo_hmm_plots=False,
        from_adata_stage=None,
    )
    captured = {}
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    def execute(source, executor_cfg, output_dir):
        captured.update(source=source, cfg=executor_cfg, output_dir=output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ad.AnnData().write_h5ad(paths.hmm_spine)
        task_catalog = output_dir / "task_catalog.parquet"
        pd.DataFrame({"task_id": ["task-1"]}).to_parquet(task_catalog, index=False)
        store = output_dir / "store"
        read_index = output_dir / "read_index"
        models = output_dir / "models"
        store.mkdir()
        read_index.mkdir()
        models.mkdir()
        (store / "task-1").touch()
        (models / "model-1.json").write_text("{}\n", encoding="utf-8")
        plot_catalog = output_dir / "plots" / "catalog.parquet"
        plot_catalog.parent.mkdir()
        pd.DataFrame().to_parquet(plot_catalog, index=False)
        manifest = output_dir / "sidecar_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return {
            "spine": paths.hmm_spine,
            "task_catalog": task_catalog,
            "read_index": read_index,
            "store": store,
            "models": models,
            "plot_catalog": plot_catalog,
            "manifest": manifest,
        }

    monkeypatch.setattr(partitioned_hmm, "execute_partitioned_hmm", execute)

    assert hmm_adata("experiment.csv") == (None, paths.hmm_spine)
    assert captured["source"] == spatial_spine
    # The executor now runs inside a staging directory under the stage root; the
    # tree moves to generations/<id> only once complete, so a failure never
    # leaves a partial stage output where readers would find it.
    staged_output = captured["output_dir"]
    assert staged_output.parent == tmp_path / "hmm_adata_outputs" / ".staging"
    entry = read_experiment_manifest(tmp_path)["stages"]["hmm"]
    assert entry["state"] == "complete"
    assert entry["expected_tasks"] == entry["successful_tasks"] == 1

    monkeypatch.setattr(
        partitioned_hmm,
        "execute_partitioned_hmm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rerun")),
    )
    assert hmm_adata("experiment.csv") == (None, paths.hmm_spine)


def test_hmm_position_mask_normalizes_nullable_boolean_values():
    adata = ad.AnnData(
        X=np.zeros((1, 3)),
        var=pd.DataFrame(
            {"ref_top_C_site": pd.array([True, pd.NA, False], dtype="boolean")},
            index=["0", "1", "2"],
        ),
    )

    mask = _resolve_pos_mask_for_methbase(adata, "ref_top", "C")

    assert mask.dtype == bool
    assert mask.tolist() == [True, False, False]


def test_partitioned_hmm_resolves_feature_and_footprint_length_layers():
    records = [
        {
            "layers": [
                "C_all_accessible_features",
                "C_all_accessible_features_lengths",
                "C_all_footprint_features_lengths",
            ]
        }
    ]

    assert _matching_hmm_layers(records, ["all_accessible_features"]) == [
        "C_all_accessible_features"
    ]
    assert _matching_hmm_layers(records, ["all_footprint_features"], lengths=True) == [
        "C_all_footprint_features_lengths"
    ]


def test_footprint_merge_writes_binary_and_derived_length_layers():
    values = np.zeros((1, 14), dtype=np.uint8)
    values[0, :2] = 1
    values[0, 12:] = 1
    adata = ad.AnnData(
        obs=pd.DataFrame({"reference_start": [0], "reference_end": [13]}, index=["read1"]),
        var=pd.DataFrame(index=pd.Index(map(str, range(14)))),
        layers={"C_all_footprint_features": values},
    )
    feature_sets = {
        "accessible": {"features": {"small_accessible_patch": [3, 20]}},
        "footprint": {"features": {"small_bound_stretch": [6, 30]}},
    }
    cfg = SimpleNamespace(
        hmm_merged_suffix="_merged",
        hmm_merge_layer_features=[("all_footprint_features", 10)],
    )
    adata.uns["hmm_appended_layers"] = np.asarray(["C_all_footprint_features"])

    _apply_merges(adata, SingleBernoulliHMM(), "C", feature_sets, cfg)

    merged = np.asarray(adata.layers["C_all_footprint_features_merged"])
    lengths = np.asarray(adata.layers["C_all_footprint_features_merged_lengths"])
    assert np.all(merged == 1)
    assert np.all(lengths == 14)
    assert "C_small_bound_stretch_merged" in adata.layers
    assert "C_small_accessible_patch_merged" not in adata.layers
    assert "C_all_footprint_features_merged" in adata.uns["hmm_appended_layers"]
    assert _feature_ranges_for_merged_layer("all_footprint_features", feature_sets) == {
        "small_bound_stretch": [6, 30]
    }


def test_partitioned_hmm_writes_task_store_and_rematerializes_layers(tmp_path, monkeypatch):
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        adata.uns["hmm_model_artifacts"] = [
            {
                "model_id": "hmm-0123456789abcdef",
                "checkpoint_sha256": "a" * 64,
                "checkpoint": "models/ef/hmm-0123456789abcdef.pt",
                "metadata": "models/ef/hmm-0123456789abcdef.json",
                "model_key": {"label": "GpC", "fit_kind": "PER"},
                "layers": ["GpC_test_feature"],
            }
        ]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    cfg = SimpleNamespace(target_task_memory_mb=1)
    outputs = execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    catalog = pd.read_parquet(outputs["task_catalog"])
    assert len(catalog) == 1
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / catalog.iloc[0]["group_path"])
    assert set(task.layers) == {"GpC_test_feature"}
    task_models = task.uns["hmm_model_artifacts"]
    assert task_models[0]["model_id"] == "hmm-0123456789abcdef"
    assert task.uns["hmm_layer_model_map"] == {"GpC_test_feature": "hmm-0123456789abcdef"}
    assert catalog.iloc[0]["hmm_model_ids"][0] == "hmm-0123456789abcdef"
    assert catalog.iloc[0]["hmm_model_checksums"][0] == "a" * 64
    read_index = pd.concat(
        [pd.read_parquet(path) for path in outputs["read_index"].glob("*.parquet")],
        ignore_index=True,
    )
    assert set(read_index["model_id"]) == {"hmm-0123456789abcdef"}
    assert set(read_index["read_id"]) == {"read1", "read2"}
    spine, _ = safe_read_h5ad(outputs["spine"])
    assert spine.uns["hmm_catalog"] == relative_uns_path(outputs["task_catalog"], tmp_path)
    assert spine.uns["hmm_source_spine"] == relative_uns_path(preprocess["spine"], tmp_path)
    restored = materialize(
        outputs["spine"],
        references="ref_top",
        read_ids=["read1", "read2"],
        start=0,
        end=12,
        layers=["GpC_test_feature"],
    )
    assert np.all(restored.layers["GpC_test_feature"] == 1)
    plot_types = set(pd.read_parquet(outputs["plot_catalog"])["plot_type"])
    assert "barcode_hmm_feature_fraction" in plot_types


def test_partitioned_hmm_masks_read_span_for_single_channel_signals(tmp_path, monkeypatch):
    # Regression test: annotate_adata's read-span masking (NaN outside each
    # read's own reference_start/reference_end, so clustermaps grey those
    # positions out and _plot_feature_fractions' isfinite-based counts
    # exclude them) used to only run for multi-channel signals
    # (mask_to_read_span=len(signals) > 1 in _annotate_task) -- single-channel
    # signals (e.g. deaminase modality's single "C" channel) never got masked
    # at all, even though _apply_merges always masks its own merged layers
    # unconditionally. Read-span masking is a per-read positional concern,
    # unrelated to channel count, so it must run either way.
    import smftools.hmm.HMM as hmm_module

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    real_mask = hmm_module.mask_layers_outside_read_span
    captured_calls = []

    def spy_mask(adata, layers, **kwargs):
        captured_calls.append(list(layers))
        return real_mask(adata, layers, **kwargs)

    monkeypatch.setattr(hmm_module, "mask_layers_outside_read_span", spy_mask)

    cfg = _hmm_cfg(hmm_methbases=["C"], hmm_max_iter=2, target_task_memory_mb=1)
    outputs = execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    # A single-channel ("C") signal was used -- with the bug, this call never
    # happened at all for the raw (non-merged) layers.
    assert captured_calls, (
        "mask_layers_outside_read_span was never called for a single-channel signal"
    )
    masked_layers = {layer for call in captured_calls for layer in call}
    assert any(layer.endswith("_all_accessible_features") for layer in masked_layers)
    assert any(layer.endswith("_all_accessible_features_lengths") for layer in masked_layers)
    catalog = pd.read_parquet(outputs["task_catalog"])
    model_id = catalog.iloc[0]["hmm_model_ids"][0]
    artifact_ref = catalog.iloc[0]["hmm_model_artifact_refs"][0]
    assert model_id.startswith("hmm-")
    assert (outputs["task_catalog"].parent / artifact_ref).is_file()
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / catalog.iloc[0]["group_path"])
    layer_model_map = task.uns["hmm_layer_model_map"]
    assert layer_model_map
    assert set(layer_model_map.values()) == {model_id}


def test_partitioned_hmm_fits_once_before_chunk_apply_and_is_order_invariant(tmp_path, monkeypatch):
    from dataclasses import replace

    from smftools import memory_guard
    from smftools.preprocessing.dispatch_plan import plan_preprocess_tasks as real_plan
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    def split_plan(spine, **kwargs):
        original = real_plan(spine, **kwargs)
        assert len(original) == 1 and len(original[0].read_ids) == 2
        task = original[0]
        return [
            replace(
                task,
                task_id=f"{task.reference}|{task.barcode}|0-12|{index:05d}",
                chunk_index=index,
                n_reads=1,
                estimated_memory_bytes=max(1, task.estimated_memory_bytes // 2),
                read_ids=(read_id,),
            )
            for index, read_id in enumerate(task.read_ids)
        ]

    def ordered_dispatch(worker, task_args_list, *, cfg, **kwargs):
        order = list(range(len(task_args_list)))
        if bool(getattr(cfg, "reverse_dispatch", False)):
            order.reverse()
        results = [None] * len(task_args_list)
        for index in order:
            results[index] = worker(*task_args_list[index])
        return results

    monkeypatch.setattr(partitioned_hmm, "plan_preprocess_tasks", split_plan)
    monkeypatch.setattr(memory_guard, "run_tasks_parallel", ordered_dispatch)
    monkeypatch.setattr(partitioned_hmm, "_plot_feature_fractions", lambda *args: None)
    monkeypatch.setattr(
        partitioned_hmm, "_plot_feature_count_size_histograms", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(partitioned_hmm, "_plot_molecule_fractions", lambda *args, **kwargs: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_hmm_parameters_across_barcodes", lambda *args: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_hmm_fit_history", lambda *args: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_feature_clustermaps", lambda *args: None)

    cfg_forward = _hmm_cfg(
        hmm_methbases=["C"],
        hmm_max_iter=2,
        hmm_max_fit_reads=1,
        hmm_fit_selection_seed=7,
        target_task_memory_mb=1,
        reverse_dispatch=False,
    )
    cfg_reverse = _hmm_cfg(**vars(cfg_forward))
    cfg_reverse.reverse_dispatch = True
    forward = execute_partitioned_hmm(preprocess["spine"], cfg_forward, tmp_path / "hmm_forward")
    reverse = execute_partitioned_hmm(preprocess["spine"], cfg_reverse, tmp_path / "hmm_reverse")
    cfg_forced = _hmm_cfg(**vars(cfg_forward))
    cfg_forced.force_redo_hmm_fit = True
    forced = execute_partitioned_hmm(preprocess["spine"], cfg_forced, tmp_path / "hmm_forced")

    forward_fits = pd.read_parquet(forward["fit_catalog"])
    reverse_fits = pd.read_parquet(reverse["fit_catalog"])
    forced_fits = pd.read_parquet(forced["fit_catalog"])
    assert list(forward_fits["fit_state"]) == ["complete"]
    assert list(forward_fits["candidate_n_reads"]) == [2]
    assert list(forward_fits["selected_n_reads"]) == [1]
    assert list(forward_fits["model_id"]) == list(reverse_fits["model_id"])
    assert list(forward_fits["model_checksum"]) == list(reverse_fits["model_checksum"])
    assert list(forward_fits["model_id"]) != list(forced_fits["model_id"])
    assert len(pd.read_parquet(forward["fit_selection"])) == 1
    assert len(list(forward["models"].rglob("*.pt"))) == 1

    forward_tasks = pd.read_parquet(forward["task_catalog"]).sort_values("task_id")
    reverse_tasks = pd.read_parquet(reverse["task_catalog"]).sort_values("task_id")
    assert forward_tasks["hmm_model_ids"].tolist() == reverse_tasks["hmm_model_ids"].tolist()
    for forward_record, reverse_record in zip(
        forward_tasks.to_dict("records"), reverse_tasks.to_dict("records"), strict=True
    ):
        forward_task, _ = safe_read_zarr(
            forward["task_catalog"].parent / forward_record["group_path"]
        )
        reverse_task, _ = safe_read_zarr(
            reverse["task_catalog"].parent / reverse_record["group_path"]
        )
        assert set(forward_task.layers) == set(reverse_task.layers)
        for layer in forward_task.layers:
            np.testing.assert_array_equal(
                np.asarray(forward_task.layers[layer]),
                np.asarray(reverse_task.layers[layer]),
            )


def test_partitioned_hmm_shared_transitions_fit_before_barcode_adaptation(tmp_path, monkeypatch):
    from smftools import memory_guard
    from smftools.tools import partitioned_hmm

    monkeypatch.setattr(
        memory_guard,
        "run_tasks_parallel",
        lambda worker, task_args_list, **kwargs: [worker(*args) for args in task_args_list],
    )
    frame = _frame()
    frame.loc[frame["read_id"] == "read2", ["barcode", "sample"]] = "bc2"
    raw = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )
    monkeypatch.setattr(partitioned_hmm, "_plot_feature_fractions", lambda *args: None)
    monkeypatch.setattr(
        partitioned_hmm, "_plot_feature_count_size_histograms", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(partitioned_hmm, "_plot_molecule_fractions", lambda *args, **kwargs: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_hmm_parameters_across_barcodes", lambda *args: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_hmm_fit_history", lambda *args: None)
    monkeypatch.setattr(partitioned_hmm, "_plot_feature_clustermaps", lambda *args: None)

    result = execute_partitioned_hmm(
        preprocess["spine"],
        _hmm_cfg(
            hmm_methbases=["C"],
            hmm_fit_strategy="shared_transitions",
            hmm_max_iter=2,
            hmm_emission_adapt_iters=1,
            target_task_memory_mb=1,
        ),
        tmp_path / "hmm_outputs",
    )

    fits = pd.read_parquet(result["fit_catalog"])
    base = fits.loc[fits["fit_kind"] == "GLOBAL"].iloc[0]
    adaptations = fits.loc[fits["fit_kind"] == "ADAPT"]
    assert base["fit_state"] == "complete"
    assert len(adaptations) == 2
    assert set(adaptations["barcode"]) == {"bc1", "bc2"}
    assert set(adaptations["parent_fit_id"]) == {base["fit_id"]}

    tasks = pd.read_parquet(result["task_catalog"])
    assigned_ids = {ids[0] for ids in tasks["hmm_model_ids"]}
    assert assigned_ids == set(adaptations["model_id"])
    assert base["model_id"] not in assigned_ids


def test_partitioned_hmm_excludes_reads_failing_qc(tmp_path, monkeypatch):
    # End-to-end regression test for the passes_dedup/passes_qc gate: a read
    # that fails read QC (here, read2's length is below read_len_filter_thresholds)
    # must never reach the HMM task catalog, mirroring the equivalent check
    # already covered for the spatial stage
    # (test_partitioned_executor_writes_derived_layers_context_and_reduced_coverage).
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess_cfg = _preprocess_cfg()
    preprocess_cfg.read_mapping_quality_filter_thresholds = [55, None]
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], preprocess_cfg, tmp_path / "preprocess_outputs"
    )
    preprocess_obs = pd.read_parquet(preprocess["obs"]).set_index("read_id")
    assert preprocess_obs["passes_dedup"].to_dict() == {"read1": True, "read2": False}

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    cfg = SimpleNamespace(target_task_memory_mb=1)
    outputs = execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    catalog = pd.read_parquet(outputs["task_catalog"])
    assert catalog["n_reads"].sum() == 1
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / catalog.iloc[0]["group_path"])
    assert list(task.obs_names) == ["read1"]
    spine, _ = safe_read_h5ad(outputs["spine"])
    assert spine.uns["hmm_filter_mask"] == "passes_dedup"


def test_partitioned_hmm_clustermaps_parallelize_using_cfg_threads(tmp_path, monkeypatch):
    # Regression test: combined_hmm_raw_clustermap/combined_hmm_length_clustermap
    # already parallelize across (reference, sample) groups internally via
    # n_jobs (see plotting/hmm_plotting.py, same pattern partitioned_spatial.py
    # already uses), but _plot_feature_clustermaps hardcoded n_jobs=1, leaving
    # that dispatch unused and making clustermap generation the slow, purely
    # sequential tail of an otherwise-parallel HMM run.
    import smftools.plotting as plotting_pkg
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)

    captured_n_jobs = []

    def fake_raw_clustermap(*args, n_jobs=1, **kwargs):
        captured_n_jobs.append(n_jobs)

    monkeypatch.setattr(plotting_pkg, "combined_hmm_raw_clustermap", fake_raw_clustermap)

    cfg = SimpleNamespace(
        target_task_memory_mb=1,
        threads=4,
        hmm_clustermap_feature_layers=["test_feature"],
        hmm_clustermap_length_layers=[],
    )
    execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    assert captured_n_jobs == [4]


def test_partitioned_hmm_clustermaps_apply_reindexing_offsets(tmp_path, monkeypatch):
    # reindex_references_adata (previously only wired into the legacy,
    # non-partitioned pipeline -- see preprocessing/reindex_references_adata.py)
    # should now run before HMM clustermap plotting: it's purely additive
    # (writes a new var display column, never touches X/layers), so it's safe
    # to run per task-window materialization.
    import smftools.plotting as plotting_pkg
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)

    captured = {}

    def fake_raw_clustermap(adata, *args, index_col_suffix=None, **kwargs):
        captured["var"] = adata.var.copy()
        captured["index_col_suffix"] = index_col_suffix

    monkeypatch.setattr(plotting_pkg, "combined_hmm_raw_clustermap", fake_raw_clustermap)

    cfg = SimpleNamespace(
        target_task_memory_mb=1,
        hmm_clustermap_feature_layers=["test_feature"],
        hmm_clustermap_length_layers=[],
        reindexing_offsets={"ref_top": 1000},
        reindexed_var_suffix="reindexed",
    )
    execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    assert captured["index_col_suffix"] == "reindexed"
    reindexed_col = captured["var"]["ref_top_reindexed"]
    var_coords = captured["var"].index.astype(int)
    assert (reindexed_col.astype(int) == var_coords + 1000).all()


def test_partitioned_hmm_prefers_hmm_device_over_device(tmp_path, monkeypatch):
    # hmm_device (config default "cpu", see experiment_config.py) overrides the
    # general `device` setting for HMM specifically -- GPU is measurably worse
    # for this workload (small-state sequential loop). Confirm the precedence:
    # hmm_device wins when set, cfg.device is only a fallback when it's unset.
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    captured_devices = []
    real_resolve = partitioned_hmm.resolve_torch_device

    def spy_resolve(device_str):
        captured_devices.append(device_str)
        return real_resolve(device_str)

    monkeypatch.setattr(partitioned_hmm, "resolve_torch_device", spy_resolve)

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.uns["hmm_appended_layers"] = []
        return []

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)

    cfg = SimpleNamespace(target_task_memory_mb=1, hmm_device="cpu", device="mps")
    execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    assert captured_devices and all(d == "cpu" for d in captured_devices)


def test_partitioned_hmm_forces_sequential_execution_on_gpu_device(tmp_path, monkeypatch):
    # Regression test: multiple worker *processes* concurrently initializing
    # the same GPU context (confirmed via real-data testing: MPS on Apple
    # Silicon reliably crashed the whole pool with BrokenProcessPool) isn't
    # safe the way CPU-bound task parallelism is. execute_partitioned_hmm
    # must force run_tasks_parallel's force_sequential=True whenever the
    # resolved device isn't "cpu", regardless of how many tasks/threads/
    # memory would otherwise justify a pool.
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )

    def annotate(adata, task, cfg, models_dir, model_assignments):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    monkeypatch.setattr(partitioned_hmm, "resolve_torch_device", lambda device: "mps")

    captured = {}
    from smftools import memory_guard

    real_run_tasks_parallel = memory_guard.run_tasks_parallel

    def spying_run_tasks_parallel(
        worker,
        task_args_list,
        *,
        cfg,
        force_sequential=False,
        pool_label=None,
        **budget,
    ):
        captured["force_sequential"] = force_sequential
        return real_run_tasks_parallel(
            worker,
            task_args_list,
            cfg=cfg,
            force_sequential=force_sequential,
            **budget,
        )

    # execute_partitioned_hmm does `from ..memory_guard import run_tasks_parallel`
    # as a local import inside its own body, so it must be patched on the
    # memory_guard module itself (where that import resolves at call time),
    # not on partitioned_hmm's own namespace.
    monkeypatch.setattr(memory_guard, "run_tasks_parallel", spying_run_tasks_parallel)

    cfg = SimpleNamespace(target_task_memory_mb=1, threads=8, device="auto")
    execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    assert captured["force_sequential"] is True


def _hmm_cfg(**overrides):
    defaults = dict(
        hmm_methbases=["GpC"],
        cpg=False,
        hmm_feature_sets={
            "footprint": {"state": "Non-Modified", "features": {"small_bound_stretch": [6, 40]}},
            "accessible": {
                "state": "Modified",
                "features": {"small_accessible_patch": [3, 20]},
            },
        },
        hmm_fit_scope="per_sample",
        hmm_distance_aware=False,
        hmm_n_states=2,
        device="cpu",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_feature_run_lengths_extracts_contiguous_run_sizes():
    from smftools.tools.partitioned_hmm import _feature_run_lengths

    row = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=float)
    assert list(_feature_run_lengths(row)) == [2, 1]

    # No features at all.
    assert list(_feature_run_lengths(np.zeros(5))) == []

    # NaN (masked outside a read's own span) breaks a run just like a 0 does,
    # and is never itself counted as part of a run.
    row_with_mask = np.array([1, 1, np.nan, np.nan, 1, 0], dtype=float)
    assert list(_feature_run_lengths(row_with_mask)) == [2, 1]


def test_plot_feature_count_size_histograms_writes_per_barcode_grids(tmp_path):
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.readwrite import safe_write_zarr
    from smftools.tools.partitioned_hmm import _plot_feature_count_size_histograms

    output_dir = tmp_path / "hmm_outputs"
    output_dir.mkdir()

    def _write_task(name: str, rows: list[list[float]]) -> str:
        arr = np.asarray(rows, dtype=float)
        adata = ad.AnnData(
            X=np.zeros(arr.shape),
            obs=pd.DataFrame(index=[f"read{i}" for i in range(arr.shape[0])]),
            var=pd.DataFrame(index=[str(i) for i in range(arr.shape[1])]),
            layers={"C_all_accessible_features": arr},
        )
        path = output_dir / name
        safe_write_zarr(adata, path, backup=False, verbose=False, zarr_format=3)
        return name

    group_bc1 = _write_task(
        "task_bc1",
        [
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 2 features: sizes 2, 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 features
        ],
    )
    group_bc2 = _write_task(
        "task_bc2",
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 1 feature: size 4
        ],
    )
    records = [
        {
            "reference": "ref_top",
            "barcode": "bc1",
            "core_start": 0,
            "core_end": 10,
            "group_path": group_bc1,
            "layers": ["C_all_accessible_features"],
        },
        {
            "reference": "ref_top",
            "barcode": "bc2",
            "core_start": 0,
            "core_end": 10,
            "group_path": group_bc2,
            "layers": ["C_all_accessible_features"],
        },
    ]
    layout = prepare_analysis_plot_layout(output_dir, stage="hmm")

    _plot_feature_count_size_histograms(records, output_dir, layout)

    catalog = pd.read_parquet(layout.catalog)
    count_rows = catalog[catalog["plot_type"] == "hmm_feature_count_histogram"]
    size_rows = catalog[catalog["plot_type"] == "hmm_feature_size_histogram"]
    assert len(count_rows) == 1
    assert len(size_rows) == 1
    assert count_rows.iloc[0]["category"] == "features"
    count_path = layout.root.parent / count_rows.iloc[0]["path"]
    size_path = layout.root.parent / size_rows.iloc[0]["path"]
    assert count_path.exists() and count_path.stat().st_size > 0
    assert size_path.exists() and size_path.stat().st_size > 0


def test_plot_hmm_parameters_across_barcodes_compares_saved_models(tmp_path):
    import torch

    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.hmm.HMM import create_hmm
    from smftools.tools.partitioned_hmm import _plot_hmm_parameters_across_barcodes

    cfg = _hmm_cfg()
    models_dir = tmp_path / "models"
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)

    model_reference = "ref_top__0_100"
    label = "GpC"
    # Two barcodes with deliberately different emission probabilities, so the
    # comparison plot has something real to show, not just identical bars.
    for barcode, emission_prob in (("bc1", 0.2), ("bc2", 0.8)):
        model = create_hmm(cfg, arch="single", device="cpu")
        with torch.no_grad():
            model.emission.data = torch.tensor([1.0 - emission_prob, emission_prob])
        path = trainer._path("PER", barcode, model_reference, label)
        trainer._save(model, path)

    records = [
        {"reference": "ref_top", "barcode": "bc1", "core_start": 0, "core_end": 100},
        {"reference": "ref_top", "barcode": "bc2", "core_start": 0, "core_end": 100},
    ]
    layout = prepare_analysis_plot_layout(tmp_path / "hmm_outputs", stage="hmm")

    _plot_hmm_parameters_across_barcodes(records, models_dir, cfg, layout)

    catalog = pd.read_parquet(layout.catalog)
    matching = catalog[catalog["plot_type"] == "hmm_parameters_across_barcodes"]
    assert len(matching) == 1
    assert matching.iloc[0]["category"] == "emissions"
    assert matching.iloc[0]["reference"] == "ref_top"
    plot_path = layout.root.parent / matching.iloc[0]["path"]
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_hmm_parameters_across_barcodes_noop_for_global_scope(tmp_path):
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.tools.partitioned_hmm import _plot_hmm_parameters_across_barcodes

    cfg = _hmm_cfg(hmm_fit_scope="global")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    records = [
        {"reference": "ref_top", "barcode": "bc1", "core_start": 0, "core_end": 100},
        {"reference": "ref_top", "barcode": "bc2", "core_start": 0, "core_end": 100},
    ]
    layout = prepare_analysis_plot_layout(tmp_path / "hmm_outputs", stage="hmm")

    _plot_hmm_parameters_across_barcodes(records, models_dir, cfg, layout)

    catalog = pd.read_parquet(layout.catalog)
    assert catalog.empty


def test_hmm_trainer_save_persists_fit_history(tmp_path):
    import torch

    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.hmm.HMM import create_hmm

    cfg = _hmm_cfg()
    trainer = HMMTrainer(cfg=cfg, models_dir=tmp_path / "models")
    model = create_hmm(cfg, arch="single", device="cpu")
    path = trainer._path("PER", "bc1", "ref_top__0_100", "GpC")

    trainer._save(model, path, hist=[1.0, 0.5, 0.5000001])

    payload = torch.load(path, map_location="cpu")
    assert payload["fit_history"] == [1.0, 0.5, 0.5000001]

    # hist=None (the default) must not add the key at all -- distinguishes
    # "no history recorded" from "history was an empty list".
    other_path = trainer._path("PER", "bc2", "ref_top__0_100", "GpC")
    trainer._save(model, other_path)
    assert "fit_history" not in torch.load(other_path, map_location="cpu")


def test_hmm_trainer_fit_or_load_records_fit_history_for_new_fits(tmp_path):
    import torch

    from smftools.cli.hmm_adata import HMMTrainer

    cfg = _hmm_cfg(hmm_max_iter=3, hmm_tol=0.0)
    trainer = HMMTrainer(cfg=cfg, models_dir=tmp_path / "models")
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(20, 10)).astype(float)

    trainer.fit_or_load(
        sample="bc1",
        ref="ref_top__0_10",
        label="GpC",
        arch="single",
        X=X,
        coords=None,
        device="cpu",
    )

    path = trainer.models_dir / trainer.last_artifact["checkpoint"]
    payload = torch.load(path, map_location="cpu")
    assert "fit_history" in payload
    assert len(payload["fit_history"]) >= 1
    assert all(isinstance(v, float) for v in payload["fit_history"])


def test_hmm_trainer_fit_or_load_default_tol_stops_before_max_iter(tmp_path):
    # Regression test for the default hmm_tol (1e-5, relative -- see
    # BaseHMM.fit's docstring) actually being used when cfg doesn't set
    # hmm_tol at all, and for it meaningfully early-stopping a real fit_or_load
    # call rather than always running to hmm_max_iter (the old absolute
    # default of 1e-4 fired at iteration 2 on real-scale log-likelihoods,
    # i.e. essentially never let EM run, while being irrelevantly loose here).
    import torch

    from smftools.cli.hmm_adata import HMMTrainer

    cfg = _hmm_cfg(hmm_max_iter=200)  # no hmm_tol override -- exercise the default
    trainer = HMMTrainer(cfg=cfg, models_dir=tmp_path / "models")
    rng = np.random.default_rng(4)
    X = np.zeros((30, 12))
    half = 15
    X[:half, :6] = 1
    X[half:, 6:] = 1
    X = np.logical_xor(X.astype(bool), rng.random(X.shape) < 0.05).astype(float)

    trainer.fit_or_load(
        sample="bc1",
        ref="ref_top__0_12",
        label="GpC",
        arch="single",
        X=X,
        coords=None,
        device="cpu",
    )

    payload = torch.load(
        trainer.models_dir / trainer.last_artifact["checkpoint"], map_location="cpu"
    )
    assert len(payload["fit_history"]) < 200


def test_plot_hmm_fit_history_reads_checkpoints_and_registers_plot(tmp_path):
    import torch

    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.hmm.HMM import create_hmm
    from smftools.tools.partitioned_hmm import _plot_hmm_fit_history

    cfg = _hmm_cfg()
    models_dir = tmp_path / "models"
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)
    model = create_hmm(cfg, arch="single", device="cpu")
    trainer._save(
        model, trainer._path("PER", "bc1", "ref_top__0_100", "GpC"), hist=[1.0, 0.6, 0.55]
    )
    trainer._save(model, trainer._path("PER", "bc2", "ref_top__0_100", "GpC"), hist=[1.2, 0.9])
    layout = prepare_analysis_plot_layout(tmp_path / "hmm_outputs", stage="hmm")

    _plot_hmm_fit_history(models_dir, layout)

    catalog = pd.read_parquet(layout.catalog)
    matching = catalog[catalog["plot_type"] == "hmm_fit_history"]
    assert len(matching) == 1
    assert matching.iloc[0]["category"] == "training"
    plot_path = layout.root.parent / matching.iloc[0]["path"]
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_hmm_fit_history_noop_when_no_checkpoints_have_history(tmp_path):
    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.hmm.HMM import create_hmm
    from smftools.tools.partitioned_hmm import _plot_hmm_fit_history

    cfg = _hmm_cfg()
    models_dir = tmp_path / "models"
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)
    model = create_hmm(cfg, arch="single", device="cpu")
    trainer._save(model, trainer._path("PER", "bc1", "ref_top__0_100", "GpC"))  # no hist
    layout = prepare_analysis_plot_layout(tmp_path / "hmm_outputs", stage="hmm")

    _plot_hmm_fit_history(models_dir, layout)

    assert pd.read_parquet(layout.catalog).empty


def test_plot_hmm_parameters_across_barcodes_skips_single_barcode_windows(tmp_path):
    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.hmm.HMM import create_hmm
    from smftools.tools.partitioned_hmm import _plot_hmm_parameters_across_barcodes

    cfg = _hmm_cfg()
    models_dir = tmp_path / "models"
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)
    model = create_hmm(cfg, arch="single", device="cpu")
    trainer._save(model, trainer._path("PER", "bc1", "ref_top__0_100", "GpC"))

    records = [{"reference": "ref_top", "barcode": "bc1", "core_start": 0, "core_end": 100}]
    layout = prepare_analysis_plot_layout(tmp_path / "hmm_outputs", stage="hmm")

    _plot_hmm_parameters_across_barcodes(records, models_dir, cfg, layout)

    catalog = pd.read_parquet(layout.catalog)
    assert catalog.empty


_LINEAGE_PROVENANCE = {
    "lineage_id": "a" * 64,
    "origin_experiment_uid": "uid-a",
    "parent_raw_generation_id": "parent-a",
    "parent_preprocess_generation_id": None,
    "selection_id": "b" * 64,
    "source_resolution_digest": None,
    "basecall_id": "c" * 64,
    "generation_kind": "selected_cohort",
    "identity_map": None,
}


def test_a_descendant_hmm_generation_does_not_take_the_canonical_spine(tmp_path, monkeypatch):
    """The stage-root spine belongs to whatever generation is current."""
    from smftools.cli import helpers
    from smftools.tools import partitioned_hmm

    spatial_spine = tmp_path / "spatial_adata_outputs" / "spine.h5ad"
    spatial_spine.parent.mkdir()
    spatial_spine.touch()
    hmm_root = tmp_path / "hmm_adata_outputs"
    paths = SimpleNamespace(
        hmm=tmp_path / "missing_hmm.h5ad.gz",
        hmm_spine=hmm_root / "spine.h5ad",
        spatial_spine=spatial_spine,
        preprocess_spine=None,
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        hmm_execution_mode="auto",
        force_redo_hmm_fit=False,
        force_redo_hmm_apply=False,
        force_redo_hmm_plots=False,
        from_adata_stage=None,
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg, **_kwargs: paths)

    def execute(_source, _cfg, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        # The real executor writes its spine inside the staging directory, which
        # is what lets publication remap it into the generation.
        staged_spine = output_dir / "spine.h5ad"
        ad.AnnData().write_h5ad(staged_spine)
        task_catalog = output_dir / "task_catalog.parquet"
        pd.DataFrame({"task_id": ["task-1"]}).to_parquet(task_catalog, index=False)
        for name in ("store", "read_index", "models"):
            (output_dir / name).mkdir()
        (output_dir / "store" / "task-1").touch()
        (output_dir / "models" / "model-1.json").write_text("{}\n", encoding="utf-8")
        plot_catalog = output_dir / "plots" / "catalog.parquet"
        plot_catalog.parent.mkdir()
        pd.DataFrame().to_parquet(plot_catalog, index=False)
        (output_dir / "sidecar_manifest.json").write_text("{}\n", encoding="utf-8")
        return {
            "spine": staged_spine,
            "task_catalog": task_catalog,
            "read_index": output_dir / "read_index",
            "store": output_dir / "store",
            "models": output_dir / "models",
            "plot_catalog": plot_catalog,
            "manifest": output_dir / "sidecar_manifest.json",
        }

    monkeypatch.setattr(partitioned_hmm, "execute_partitioned_hmm", execute)

    hmm_adata("experiment.csv")
    parent_pointer = json.loads((hmm_root / "current.json").read_text(encoding="utf-8"))

    _, descendant_spine = hmm_adata(
        "experiment.csv",
        lineage_provenance=dict(_LINEAGE_PROVENANCE),
    )

    # The descendant published its own generation without taking the selector.
    assert descendant_spine != paths.hmm_spine
    assert descendant_spine.parent.parent.name == "generations"
    manifest = json.loads(
        (descendant_spine.parent / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["lineage"] == _LINEAGE_PROVENANCE
    assert json.loads((hmm_root / "current.json").read_text(encoding="utf-8")) == parent_pointer


# --- HCE-04: sequence-context-aware emissions in the partitioned stage --------


def _context_run(tmp_path, **overrides):
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )
    cfg = _hmm_cfg(hmm_methbases=["C"], hmm_max_iter=3, target_task_memory_mb=1, **overrides)
    outputs = execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")
    catalog = pd.read_parquet(outputs["task_catalog"])
    artifacts = json.loads(catalog.iloc[0]["hmm_model_artifacts_json"])
    return cfg, outputs, artifacts


def _load_model(cfg, outputs, artifact):
    from smftools.cli.hmm_adata import HMMTrainer

    trainer = HMMTrainer(cfg=cfg, models_dir=outputs["task_catalog"].parent / "models")
    return trainer.load_artifact(artifact, device="cpu")


def test_context_learned_mode_runs_end_to_end(tmp_path, monkeypatch):
    cfg, outputs, artifacts = _context_run(
        tmp_path, hmm_context_model="learned", hmm_context_k=3, hmm_context_shrinkage=5.0
    )
    assert artifacts[0]["model_key"]["architecture"] == "context_single"
    model = _load_model(cfg, outputs, artifacts[0])
    assert model.learn and model.position_codes.size == 12
    # Codes follow the reference: C at 1, 3, 8, 11 (forward, top strand) where windows fit.
    assert (model.position_codes >= 0).sum() >= 3
    assert model.log_weights.numel() == 16


def test_context_table_mode_reads_the_groups_weights(tmp_path, monkeypatch):
    from smftools.analysis.compute.site_context_bias import context_kmers, write_weight_table

    table = pd.DataFrame(
        {
            "group": "bc1",
            "k": 3,
            "kmer": context_kmers(3),
            "weight": np.linspace(0.5, 2.0, 16),
            "n_sites": 1,
            "observed": 10,
            "source": "naked_dna",
        }
    )
    path = tmp_path / "weights.parquet"
    write_weight_table(table, path)
    cfg, outputs, artifacts = _context_run(
        tmp_path,
        hmm_context_model="table",
        hmm_context_table=str(path),
        hmm_context_table_group="Sample",
    )
    model = _load_model(cfg, outputs, artifacts[0])
    assert not model.learn
    np.testing.assert_allclose(model.log_weights.numpy(), np.log(np.linspace(0.5, 2.0, 16)))


def test_context_model_refuses_distance_aware_hmms():
    from smftools.tools.partitioned_hmm import _configured_model_specs

    cfg = _hmm_cfg(hmm_methbases=["C"], hmm_context_model="learned", hmm_distance_aware=True)
    with pytest.raises(ValueError, match="distance_aware"):
        _configured_model_specs(cfg)
    plain = _configured_model_specs(_hmm_cfg(hmm_methbases=["C"]))
    assert {spec.architecture for spec in plain} == {"single"}


def test_context_setup_errors():
    import anndata as ad

    from smftools.tools.partitioned_hmm import context_setup

    adata = ad.AnnData(obs=pd.DataFrame({"Sample": ["a", "b"]}, index=["r1", "r2"]))
    adata.uns["References"] = {"ref_FASTA_sequence": "ACGCGTACGTAC"}
    learned = context_setup(adata, "ref_top", _hmm_cfg(hmm_context_model="learned"))
    assert learned["position_codes"].size == 12 and not learned["log_weights"].any()
    with pytest.raises(KeyError, match="no reference sequence"):
        context_setup(adata, "other_top", _hmm_cfg(hmm_context_model="learned"))
    with pytest.raises(ValueError, match="needs hmm_context_table"):
        context_setup(adata, "ref_top", _hmm_cfg(hmm_context_model="table"))
    table_cfg = _hmm_cfg(
        hmm_context_model="table", hmm_context_table="x.parquet", hmm_context_table_group="Sample"
    )
    with pytest.raises(ValueError, match="span"):
        context_setup(adata, "ref_top", table_cfg)
    with pytest.raises(KeyError, match="column"):
        context_setup(
            adata, "ref_top", _hmm_cfg(hmm_context_model="table", hmm_context_table="x.parquet")
        )


def test_context_checkpoint_round_trips_through_the_trainer(tmp_path):
    import torch

    from smftools.cli.hmm_adata import HMMTrainer
    from smftools.hmm.HMM import ContextBernoulliHMM

    model = ContextBernoulliHMM(
        init_emission=[0.1, 0.6],
        log_weights=np.log(np.linspace(0.5, 2.0, 16)),
        position_codes=np.arange(20) % 16,
        cpg_codes=[1, 5],
        cpg="exclude",
        learn=True,
        shrinkage=7.0,
    )
    trainer = HMMTrainer(cfg=_hmm_cfg(), models_dir=tmp_path)
    path = tmp_path / "model.pt"
    torch.save(trainer._payload(model), path)
    loaded = trainer._load(path, arch="context_single", device="cpu")
    np.testing.assert_array_equal(loaded.position_codes, model.position_codes)
    np.testing.assert_array_equal(loaded.cpg_codes, [1, 5])
    torch.testing.assert_close(loaded.log_weights, model.log_weights)
    assert loaded.learn and loaded.cpg == "exclude" and loaded.shrinkage == 7.0


def test_context_settings_leave_fingerprints_alone_until_used(tmp_path):
    from smftools.cli.helpers import resolved_stage_config
    from smftools.config.experiment_config import ExperimentConfig
    from smftools.hmm.model_artifacts import hmm_fit_config, hmm_fit_config_hash

    base = ExperimentConfig()
    stage = resolved_stage_config(base, "hmm")
    assert not any(key.startswith("hmm_context_") for key in stage)
    assert not any(key.startswith("hmm_context_") for key in hmm_fit_config(base))
    learned = ExperimentConfig(hmm_context_model="learned")
    assert resolved_stage_config(learned, "hmm")["hmm_context_model"] == "learned"
    assert hmm_fit_config_hash(learned) != hmm_fit_config_hash(base)

    table = tmp_path / "weights.csv"
    table.write_text("a")
    tabled = ExperimentConfig(hmm_context_model="table", hmm_context_table=str(table))
    before = (resolved_stage_config(tabled, "hmm"), hmm_fit_config_hash(tabled))
    table.write_text("b")  # edited in place, same path
    after = (resolved_stage_config(tabled, "hmm"), hmm_fit_config_hash(tabled))
    assert before[0]["hmm_context_table_sha256"] != after[0]["hmm_context_table_sha256"]
    assert before[1] != after[1]


# --- HCE-06: emission variants in one HMM run ----------------------------------


def test_variant_config_parses_and_validates():
    from smftools.config.experiment_config import _parse_hmm_variants
    from smftools.tools.partitioned_hmm import hmm_variants

    assert _parse_hmm_variants(None) == {} and _parse_hmm_variants("{}") == {}
    parsed = _parse_hmm_variants('{"learned": {"hmm_context_model": "learned"}}')
    assert parsed == {"learned": {"hmm_context_model": "learned"}}
    for variants, message in (
        ({"all": {}}, "clashes"),
        ({"small": {}}, "clashes"),  # leading word of small_bound_stretch
        ({"bad-name": {}}, "identifier"),
        ({"v": {"hmm_max_iter": 3}}, "may only set"),
    ):
        with pytest.raises(ValueError, match=message):
            hmm_variants(_hmm_cfg(hmm_variants=variants))


def test_variants_expand_specs_and_keep_the_default_fit_identity():
    from smftools.hmm.model_artifacts import hmm_fit_config_hash
    from smftools.tools.partitioned_hmm import _configured_model_specs

    plain = _configured_model_specs(_hmm_cfg(hmm_methbases=["C"]))
    cfg = _hmm_cfg(
        hmm_methbases=["C"],
        hmm_variants={"learned": {"hmm_context_model": "learned", "hmm_context_shrinkage": 5}},
    )
    specs = _configured_model_specs(cfg)
    assert [(s.label, s.variant, s.architecture) for s in specs] == [
        ("C", "", "single"),
        ("C_learned", "learned", "context_single"),
    ]
    assert specs[0] == plain[0]  # the default spec is untouched
    assert hmm_fit_config_hash(specs[0].config(cfg)) == hmm_fit_config_hash(
        _hmm_cfg(hmm_methbases=["C"])
    )
    assert hmm_fit_config_hash(specs[1].config(cfg)) != hmm_fit_config_hash(cfg)
    assert specs[1].config(cfg).hmm_context_shrinkage == 5 and not hasattr(
        cfg, "hmm_context_shrinkage"
    )


def test_variant_layer_groups_pair_variant_layers_with_the_default():
    from smftools.tools.partitioned_hmm import _configured_model_specs, variant_layer_groups

    cfg = _hmm_cfg(hmm_methbases=["C"], hmm_variants={"learned": {"hmm_context_model": "learned"}})
    groups = dict(
        variant_layer_groups(
            [
                "C_all_accessible_features",
                "C_learned_all_accessible_features",
                "C_all_footprint_features_lengths",
                "C_learned_all_footprint_features_lengths",
                "unrelated",
            ],
            _configured_model_specs(cfg),
        )
    )
    assert groups["C_all_accessible_features"] == [
        ("", "C_all_accessible_features"),
        ("learned", "C_learned_all_accessible_features"),
    ]
    assert [v for v, _ in groups["C_all_footprint_features_lengths"]] == ["", "learned"]
    assert groups["unrelated"] == [("", "unrelated")]


def test_variants_run_end_to_end_with_comparison_plots(tmp_path, monkeypatch):
    cfg, outputs, artifacts = _context_run(
        tmp_path, hmm_variants={"learned": {"hmm_context_model": "learned"}}
    )
    catalog = pd.read_parquet(outputs["task_catalog"])
    layers = set(catalog.iloc[0]["layers"])
    assert {"C_all_accessible_features", "C_learned_all_accessible_features"} <= layers
    architectures = {a["model_key"]["architecture"] for a in artifacts}
    assert architectures == {"single", "context_single"}
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / catalog.iloc[0]["group_path"])
    for column in (
        "C_all_accessible_features_fraction",
        "C_learned_all_accessible_features_fraction",
        "C_learned_all_footprint_features_fraction",
    ):
        values = task.obs[column].to_numpy(dtype=float)
        assert np.all((values >= 0) & (values <= 1) | np.isnan(values)), column
    pngs = [p.as_posix() for p in (tmp_path / "hmm_outputs").rglob("*.png")]
    assert any(p.endswith("__molecule_fractions.png") for p in pngs)
    # Comparison, not duplication: no figure directory is named for a variant layer.
    assert not any("/C_learned_" in p for p in pngs)


def test_no_variants_leave_the_stage_hash_alone():
    from smftools.cli.helpers import resolved_stage_config, stage_config_hash
    from smftools.config.experiment_config import ExperimentConfig

    base = ExperimentConfig()
    assert "hmm_variants" not in resolved_stage_config(base, "hmm")
    learned = ExperimentConfig(hmm_variants={"learned": {"hmm_context_model": "learned"}})
    assert stage_config_hash(learned, "hmm") != stage_config_hash(base, "hmm")


# --- SCQ-02: sequence-context QC of state calls -------------------------------


def test_context_qc_tallies_every_variant_and_match_the_decoded_layers(tmp_path):
    from smftools.analysis.compute.site_context_bias import read_weight_table
    from smftools.informatics.partition_read import materialize
    from smftools.tools.partitioned_hmm import _configured_model_specs, _prepare_model_input

    cfg, outputs, _ = _context_run(
        tmp_path, hmm_variants={"learned": {"hmm_context_model": "learned"}}
    )
    target = outputs["task_catalog"].parent / "context_qc"
    counts = pd.read_parquet(target / "site_counts.parquet")
    assert set(counts["variant"]) == {"default", "learned"}
    assert not (target / "partials").exists()
    weights = read_weight_table(target / "accessible_weights_C_learned_k3.parquet")
    assert set(weights["source"]) == {"accessible"}

    # Equal to a direct count from the stored decoded layer and the input calls.
    catalog = pd.read_parquet(outputs["task_catalog"])
    record = catalog.iloc[0]
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / record["group_path"])
    spine = outputs["task_catalog"].parent / "spine.h5ad"
    adata = materialize(
        spine,
        references=record["reference"],
        read_ids=list(task.obs_names),
        start=int(record["core_start"]),
        end=int(record["core_end"]),
    )
    spec = _configured_model_specs(cfg)[0]
    values, coords, _ = _prepare_model_input(adata, record["reference"], spec, cfg)
    values = np.asarray(values, dtype=float)
    columns = np.searchsorted(np.asarray(task.var_names, dtype=np.int64), np.asarray(coords))
    state = np.asarray(task.layers["C_all_accessible_features"], dtype=float)[:, columns] > 0
    observed = ~np.isnan(values)
    expected = pd.DataFrame(
        {
            "position": np.asarray(coords, dtype=np.int64),
            "observed": observed.sum(0),
            "accessible": (observed & state).sum(0),
        }
    )
    expected = expected[expected["observed"] > 0].reset_index(drop=True)
    got = (
        counts[counts["variant"] == "default"][["position", "observed", "accessible"]]
        .sort_values("position")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_context_qc_without_variants_reports_the_default_alone(tmp_path):
    _, outputs, _ = _context_run(tmp_path)
    counts = pd.read_parquet(outputs["task_catalog"].parent / "context_qc" / "site_counts.parquet")
    assert set(counts["variant"]) == {"default"}


def test_context_qc_off_writes_nothing(tmp_path):
    _, outputs, _ = _context_run(tmp_path, stage_context_qc=False)
    assert not (outputs["task_catalog"].parent / "context_qc").exists()


def test_context_qc_figure_overlays_every_variant(tmp_path, monkeypatch):
    from smftools.analysis.plot import site_context_bias as plots
    from smftools.cli.stage_artifacts import prepare_analysis_plot_layout
    from smftools.tools.hmm_context_qc import context_tables, plot_hmm_context_qc

    rng = np.random.default_rng(0)
    rows = []
    for variant in ("default", "learned", "cells"):
        for position in range(3, 40):
            rows.append(("SQK_barcode01", "ref_top", "C", variant, position, 50, 20, 30, 15))
    tallies = pd.DataFrame(
        rows,
        columns=[
            "barcode",
            "physical_reference",
            "model",
            "variant",
            "position",
            "observed",
            "modified",
            "accessible",
            "accessible_modified",
        ],
    )
    sequence = "".join(rng.choice(list("ACGT"), 60))
    sequence = "".join("C" if 3 <= i < 40 else base for i, base in enumerate(sequence))
    _, rates = context_tables(tallies, {"ref": sequence}, flank=3, kmers=[1, 3])
    calls = []
    monkeypatch.setattr(
        plots, "plot_kmer_rate_series", lambda rates, path, **kwargs: calls.append(kwargs)
    )
    layout = prepare_analysis_plot_layout(tmp_path, stage="hmm")
    plot_hmm_context_qc(rates, layout, min_calls=0)
    assert {call["scale"] for call in calls} == {"relative", "absolute"}
    for call in calls:
        labels = [item["label"] for items in call["panels"].values() for item in items]
        assert labels[:3] == ["default", "cells", "learned"]  # default first
