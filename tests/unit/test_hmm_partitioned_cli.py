from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from smftools.cli.hmm_adata import (
    _feature_ranges_for_merged_layer,
    _resolve_pos_mask_for_methbase,
    hmm_adata,
)
from smftools.hmm.HMM import SingleBernoulliHMM
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
        return {"spine": paths.hmm_spine}

    monkeypatch.setattr(partitioned_hmm, "execute_partitioned_hmm", execute)

    assert hmm_adata("experiment.csv") == (None, paths.hmm_spine)
    assert captured["source"] == spatial_spine
    assert captured["output_dir"] == tmp_path / "hmm_adata_outputs"


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

    def annotate(adata, task, cfg, models_dir):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    cfg = SimpleNamespace(target_task_memory_mb=1)
    outputs = execute_partitioned_hmm(preprocess["spine"], cfg, tmp_path / "hmm_outputs")

    catalog = pd.read_parquet(outputs["task_catalog"])
    assert len(catalog) == 1
    task, _ = safe_read_zarr(outputs["task_catalog"].parent / catalog.iloc[0]["group_path"])
    assert set(task.layers) == {"GpC_test_feature"}
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

    def annotate(adata, task, cfg, models_dir):
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

    def annotate(adata, task, cfg, models_dir):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    monkeypatch.setattr(partitioned_hmm, "resolve_torch_device", lambda device: "mps")

    captured = {}
    from smftools import memory_guard

    real_run_tasks_parallel = memory_guard.run_tasks_parallel

    def spying_run_tasks_parallel(worker, task_args_list, *, cfg, force_sequential=False):
        captured["force_sequential"] = force_sequential
        return real_run_tasks_parallel(
            worker, task_args_list, cfg=cfg, force_sequential=force_sequential
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
