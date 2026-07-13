from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import (
    execute_partitioned_preprocessing,
    fit_direct_modality_youden_thresholds,
)
from smftools.readwrite import safe_read_h5ad, safe_read_zarr
from smftools.tools.partitioned_spatial import (
    _compute_read_spatial_statistics,
    execute_partitioned_spatial,
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
                "read_length": 3,
                "mapped_length": 3,
                "reference_length": 12,
                "read_quality": 31,
                "mapping_quality": 50,
                "read_length_to_reference_length_ratio": 3 / 12,
                "mapped_length_to_reference_length_ratio": 3 / 12,
                "mapped_length_to_read_length_ratio": 1.0,
            },
        ]
    )


def _cfg():
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
        read_len_filter_thresholds=[4, None],
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
        duplicate_detection_site_types=["GpC", "CpG"],
        duplicate_detection_distance_threshold=0.07,
        duplicate_detection_keep_best_metric="read_quality",
        duplicate_detection_window_size_for_hamming_neighbors=50,
        duplicate_detection_min_overlapping_positions=3,
        duplicate_detection_do_hierarchical=False,
        duplicate_detection_hierarchical_linkage="average",
        duplicate_detection_do_pca=False,
        duplicate_detection_demux_types_to_use=[],
        duplicate_detection_max_reads_per_window=1000,
        sample_name_col_for_plotting="Sample",
    )


def _deaminase_frame():
    frame = _frame()
    # Strand-switch metrics carried from raw extraction (see ragged_store).
    # read1 is a clean C->T/G->A chimera; read2 is one-sided (pure top).
    frame["ct_event_count"] = [8, 9]
    frame["ga_event_count"] = [8, 0]
    frame["strand_segment_purity"] = [1.0, 1.0]
    frame["strand_switch_position"] = [6, -1]
    return frame


def _deaminase_cfg():
    cfg = _cfg()
    cfg.smf_modality = "deaminase"
    cfg.bypass_label_deaminase_pcr_chimeras = False
    cfg.deaminase_chimera_min_events_per_span = 3
    cfg.deaminase_chimera_min_segment_purity = 0.9
    cfg.deaminase_chimera_max_single_strand_fraction = 0.8
    return cfg


def _direct_youden_frame():
    def _read(read_id, sample, signal):
        return {
            "read_id": read_id,
            "reference": "ref",
            "Reference_strand": "ref_top",
            "barcode": "bc1",
            "sample": sample,
            "reference_start": 0,
            "cigar": "4M",
            "aligned_length": 4,
            "sequence": [0, 1, 2, 3],
            "quality": [30, 30, 30, 30],
            "mismatch": [4, 4, 4, 4],
            "modification_signal": signal,
            "read_length": 4,
            "mapped_length": 4,
            "reference_length": 12,
            "read_quality": 30,
            "mapping_quality": 60,
            "read_length_to_reference_length_ratio": 4 / 12,
            "mapped_length_to_reference_length_ratio": 4 / 12,
            "mapped_length_to_read_length_ratio": 1.0,
        }

    # Positions 0-3 are covered by both control samples with a clean
    # separation (positive=1.0, negative=0.0) so Youden fitting has an
    # unambiguous threshold to find. "high1" is a non-control read used only
    # to check threshold *application* -- the fitted threshold for perfectly
    # separable data lands exactly on the smallest positive-control value
    # (1.0 here), and binarize_on_Youden classifies with strict ``>``, so a
    # control read sitting exactly at that value is a boundary tie, not a
    # meaningful check of which side of the threshold it falls on.
    return pd.DataFrame(
        [
            _read("pos1", "positive_ctrl", [1.0, 1.0, 1.0, 1.0]),
            _read("pos2", "positive_ctrl", [1.0, 1.0, 1.0, 1.0]),
            _read("neg1", "negative_ctrl", [0.0, 0.0, 0.0, 0.0]),
            _read("neg2", "negative_ctrl", [0.0, 0.0, 0.0, 0.0]),
            _read("high1", "experimental", [5.0, 5.0, 5.0, 5.0]),
        ]
    )


def _direct_youden_cfg():
    cfg = _cfg()
    cfg.smf_modality = "direct"
    cfg.sample_column = "Sample"
    cfg.fit_position_methylation_thresholds = True
    cfg.positive_control_sample_methylation_fitting = "positive_ctrl"
    cfg.negative_control_sample_methylation_fitting = "negative_ctrl"
    cfg.infer_on_percentile_sample_methylation_fitting = False
    cfg.inference_variable_sample_methylation_fitting = "Raw_modification_signal"
    cfg.fit_j_threshold = 0.5
    cfg.binarize_on_fixed_methlyation_threshold = 0.5
    return cfg


def test_partitioned_executor_fits_youden_thresholds_for_direct_modality(tmp_path):
    pytest.importorskip("pyarrow")
    raw = write_raw_store(
        _direct_youden_frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    output_dir = tmp_path / "preprocess_outputs"

    outputs = execute_partitioned_preprocessing(raw["spine"], _direct_youden_cfg(), output_dir)

    # The fit pre-pass ran once per reference (not raising the old
    # "not yet supported" error) and left the same ROC diagnostics the
    # legacy path saves.
    roc_dir = output_dir / "02B_Position_wide_Youden_threshold_performance"
    assert roc_dir.is_dir()
    assert any(roc_dir.iterdir())

    derived = materialize(outputs["spine"], layers=["binarized_methylation"])
    by_read = dict(zip(derived.obs_names, derived.layers["binarized_methylation"]))
    assert list(by_read["high1"][:4]) == [1.0, 1.0, 1.0, 1.0]
    for read_id in ("neg1", "neg2"):
        assert list(by_read[read_id][:4]) == [0.0, 0.0, 0.0, 0.0]


def test_youden_fit_loads_spine_once_and_reuses_it(tmp_path, monkeypatch):
    spine = SimpleNamespace(
        obs=pd.DataFrame(
            {
                "Reference_strand": pd.Categorical(["ref_top", "ref_top"]),
                "Sample": ["positive_ctrl", "negative_ctrl"],
                "Raw_modification_signal": [2.0, 0.0],
            },
            index=["read1", "read2"],
        ),
        var=pd.DataFrame(index=pd.Index([0, 1], name="position")),
        var_names=pd.Index([0, 1]),
        uns={},
    )
    cfg = SimpleNamespace(
        reference_column="Reference_strand",
        positive_control_sample_methylation_fitting="positive_ctrl",
        negative_control_sample_methylation_fitting="negative_ctrl",
        fit_j_threshold=0.5,
        sample_column="Sample",
        infer_on_percentile_sample_methylation_fitting=False,
        inference_variable_sample_methylation_fitting="Raw_modification_signal",
    )
    materialize_calls = []
    load_spine_calls = []

    def fake_load_spine(path, *, verbose=True):
        load_spine_calls.append((path, verbose))
        return spine

    def fake_materialize(spine_arg, *, references):
        materialize_calls.append((spine_arg, references))
        return SimpleNamespace(
            obs=spine.obs.loc[spine.obs["Reference_strand"] == references].copy(),
            var=pd.DataFrame(
                {
                    f"{references}_position_methylation_thresholding_Youden_stats": [
                        (0.5, 0.9),
                        (0.5, 0.9),
                    ],
                    f"{references}_position_passed_Youden_thresholding_QC": [True, True],
                },
                index=spine.var.index,
            ),
            var_names=spine.var_names,
            uns={},
        )

    def fake_calculate_position_Youden(ref_adata, **kwargs):
        return None

    import importlib

    youden_module = importlib.import_module("smftools.preprocessing.calculate_position_Youden")
    monkeypatch.setattr(
        "smftools.preprocessing.partitioned_executor.load_spine", fake_load_spine
    )
    monkeypatch.setattr(
        "smftools.preprocessing.partitioned_executor.materialize", fake_materialize
    )
    monkeypatch.setattr(youden_module, "calculate_position_Youden", fake_calculate_position_Youden)

    fit_direct_modality_youden_thresholds(
        tmp_path / "spine.h5ad",
        cfg,
        ["ref_top"],
        tmp_path / "out",
    )

    assert load_spine_calls == [(tmp_path / "spine.h5ad", False)]
    assert materialize_calls == [(spine, "ref_top")]


def test_partitioned_executor_labels_deaminase_pcr_chimeras(tmp_path):
    raw = write_raw_store(
        _deaminase_frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )

    outputs = execute_partitioned_preprocessing(
        raw["spine"], _deaminase_cfg(), tmp_path / "preprocess_outputs"
    )

    # Label lands on the obs sidecar parquet ...
    obs = pd.read_parquet(outputs["obs"]).set_index("read_id")
    assert obs["deaminase_PCR_chimera"].to_dict() == {"read1": True, "read2": False}

    # ... and propagates onto the derived spine obs.
    spine, _ = safe_read_h5ad(outputs["spine"])
    assert spine.obs["deaminase_PCR_chimera"].to_dict() == {"read1": True, "read2": False}


def test_partitioned_executor_writes_derived_layers_context_and_reduced_coverage(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )

    outputs = execute_partitioned_preprocessing(
        raw["spine"], _cfg(), tmp_path / "preprocess_outputs"
    )

    catalog = pd.read_parquet(outputs["catalog"])
    assert len(catalog) == 1
    result, _ = safe_read_zarr(outputs["catalog"].parent / catalog.iloc[0]["group_path"])
    assert set(result.layers) == {"nan0_0minus1", "nan_half"}
    assert "ref_top_GpC_site" in result.var
    assert list(result.var_names) == [str(position) for position in range(12)]

    coverage = pd.read_parquet(outputs["var"]).set_index("position")
    assert coverage.loc[0, "valid_count"] == 1
    assert coverage.loc[0, "valid_fraction"] == 0.5
    assert bool(coverage.loc[0, "position_valid"])
    assert coverage.loc[1, "valid_count"] == 0
    assert not bool(coverage.loc[1, "position_valid"])

    spine, _ = safe_read_h5ad(outputs["spine"])
    assert spine.uns["preprocess_catalog"] == str(outputs["catalog"].resolve())
    assert spine.uns["preprocess_var"] == str(outputs["var"].resolve())
    assert spine.obs["passes_read_qc"].to_dict() == {"read1": True, "read2": False}
    assert outputs["plots"].is_dir()
    assert set(path.name for path in outputs["plots"].iterdir() if path.is_dir()) == {
        "barcode_summary",
        "read_qc",
        "modification_qc",
        "duplicate_qc",
        "library_complexity",
        "read_span_quality",
        "coverage",
        "task_diagnostics",
    }
    plot_catalog = pd.read_parquet(outputs["plot_catalog"])
    assert {
        "read_qc_metric_dashboard",
        "barcode_reference_overview",
        "barcode_reference_read_metric_distributions",
        "barcode_reference_modification_distributions",
        "modification_qc_metric_dashboard",
        "duplicate_rate_by_sample_reference",
        "valid_fraction_by_position",
        "read_span_base_quality",
    }.issubset(set(plot_catalog["plot_type"]))
    barcode_summary = pd.read_parquet(
        outputs["plots"] / "barcode_summary" / "barcode_reference_summary.parquet"
    ).set_index(["barcode", "reference"])
    assert barcode_summary.loc[("bc1", "ref_top"), "n_reads"] == 2
    assert barcode_summary.loc[("bc1", "ref_top"), "n_read_qc_pass"] == 1
    assert barcode_summary.loc[("bc1", "ref_top"), "read_length_median"] == 3.5
    obs = pd.read_parquet(outputs["obs"])
    assert set(obs["read_id"]) == {"read1", "read2"}
    assert obs.set_index("read_id")["passes_read_qc"].to_dict() == {
        "read1": True,
        "read2": False,
    }
    obs_by_read = obs.set_index("read_id")
    assert obs_by_read["Raw_modification_signal"].to_dict() == {
        "read1": 2.0,
        "read2": 2.0,
    }
    assert obs_by_read.loc["read2", "Fraction_CpG_site_modified"] == 1.0

    spatial_cfg = _cfg()
    spatial_cfg.autocorr_site_types = ["GpC", "CpG", "C"]
    spatial_cfg.autocorr_max_lag = 4
    spatial_cfg.autocorr_normalization_method = "pearson"
    spatial = execute_partitioned_spatial(
        outputs["spine"], spatial_cfg, tmp_path / "spatial_outputs"
    )
    spatial_spine, _ = safe_read_h5ad(spatial["spine"])
    assert spatial_spine.uns["spatial_source_spine"] == str(outputs["spine"].resolve())
    spatial_tasks = pd.read_parquet(spatial["task_catalog"])
    assert spatial_tasks["n_reads"].sum() == 1
    metrics = pd.read_parquet(spatial["metrics"])
    assert set(metrics["reference"]) == {"ref_top"}
    assert set(metrics["barcode"]) == {"bc1"}
    assert metrics["n_reads"].max() == 1
    assert spatial["position_store"].is_dir()
    spatial_plot_types = set(pd.read_parquet(spatial["plot_catalog"])["plot_type"])
    assert "barcode_position_modification_profile" in spatial_plot_types
    assert obs["passes_modification_qc"].all()
    assert obs_by_read["passes_qc"].to_dict() == {"read1": True, "read2": False}
    assert obs_by_read["passes_dedup"].to_dict() == {"read1": True, "read2": False}

    derived = materialize(outputs["spine"], layers=["nan_half"])
    assert set(derived.layers) == {"nan_half"}
    assert derived.layers["nan_half"][0, 0] == 1.0
    assert derived.layers["nan_half"][0, 1] == 0.5
    assert "ref_top_GpC_site" in derived.var
    assert derived.var.loc["0", "ref_top_valid_count"] == 1
    assert bool(derived.var.loc["0", "position_in_ref_top"])


def test_genome_derived_layers_stitch_across_cores_with_absent_read_fill(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="genome",
        genome_tile_size=4,
        genome_tile_halo=1,
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    outputs = execute_partitioned_preprocessing(
        raw["spine"], _cfg(), tmp_path / "preprocess_outputs"
    )

    catalog = pd.read_parquet(outputs["catalog"])
    assert set(catalog["core_start"]) == {0, 4, 8}
    derived = materialize(
        outputs["spine"], references="ref_top", start=3, end=9, layers=["nan_half"]
    )

    assert list(derived.var_names) == ["3", "4", "5", "6", "7", "8"]
    assert derived.layers["nan_half"][0].tolist() == [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert derived.layers["nan_half"][1].tolist() == [0.5, 0.5, 0.0, 1.0, 1.0, 0.0]


def test_duplicate_reduction_reconciles_clusters_across_genome_core_boundary(tmp_path):
    frame = pd.DataFrame(
        [
            {
                **_frame().iloc[0].to_dict(),
                "read_id": "lower_quality",
                "reference_start": 2,
                "cigar": "6M",
                "aligned_length": 6,
                "sequence": [1] * 6,
                "quality": [20] * 6,
                "mismatch": [4] * 6,
                "modification_signal": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "read_length": 6,
                "mapped_length": 6,
                "read_quality": 20,
            },
            {
                **_frame().iloc[0].to_dict(),
                "read_id": "higher_quality",
                "reference_start": 3,
                "cigar": "6M",
                "aligned_length": 6,
                "sequence": [1] * 6,
                "quality": [40] * 6,
                "mismatch": [4] * 6,
                "modification_signal": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "read_length": 6,
                "mapped_length": 6,
                "read_quality": 40,
            },
            {
                **_frame().iloc[0].to_dict(),
                "read_id": "distinct",
                "reference_start": 2,
                "cigar": "6M",
                "aligned_length": 6,
                "sequence": [1] * 6,
                "quality": [30] * 6,
                "mismatch": [4] * 6,
                "modification_signal": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "read_length": 6,
                "mapped_length": 6,
                "read_quality": 30,
            },
        ]
    )
    raw = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="genome",
        genome_tile_size=4,
        genome_tile_halo=1,
        extra_uns={"References": {"ref_FASTA_sequence": "CCCCCCCCCCCC"}},
    )
    cfg = _cfg()
    cfg.mod_target_bases = ["C"]
    cfg.duplicate_detection_site_types = ["C"]
    cfg.duplicate_detection_distance_threshold = 0.1
    cfg.bypass_flag_duplicate_reads = False

    outputs = execute_partitioned_preprocessing(raw["spine"], cfg, tmp_path / "preprocess_outputs")
    obs = pd.read_parquet(outputs["obs"]).set_index("read_id")

    assert bool(obs.loc["lower_quality", "is_duplicate"])
    assert not bool(obs.loc["higher_quality", "is_duplicate"])
    assert not bool(obs.loc["distinct", "is_duplicate"])
    assert (
        obs.loc["lower_quality", "duplicate_cluster_id"]
        == obs.loc["higher_quality", "duplicate_cluster_id"]
    )
    assert obs.loc["lower_quality", "duplicate_cluster_size"] == 2
    assert {
        "fwd_hamming_to_next",
        "rev_hamming_to_prev",
        "sequence__hier_hamming_to_pair",
        "sequence__min_hamming_to_pair",
    }.issubset(obs.columns)
    plot_types = set(pd.read_parquet(outputs["plot_catalog"])["plot_type"])
    assert "hamming_distance_by_reference" in plot_types
    assert "duplicate_cluster_size_histogram" in plot_types


def test_partitioned_spatial_writes_locus_clustermaps_and_position_matrices(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    cfg = _cfg()
    cfg.read_len_filter_thresholds = [None, None]
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], cfg, tmp_path / "preprocess_outputs"
    )
    cfg.autocorr_site_types = ["C"]
    cfg.autocorr_max_lag = 4
    cfg.autocorr_normalization_method = "pearson"
    cfg.spatial_generate_clustermaps = True
    cfg.spatial_generate_position_matrices = True
    cfg.spatial_matrix_min_reads = 2
    cfg.spatial_clustermap_sortby = "none"
    cfg.layer_for_clustermap_plotting = "nan0_0minus1"
    cfg.clustermap_cmap_c = "coolwarm"
    cfg.clustermap_cmap_gpc = "coolwarm"
    cfg.clustermap_cmap_cpg = "viridis"
    cfg.clustermap_cmap_a = "coolwarm"
    cfg.correlation_matrix_types = ["binary_covariance"]
    cfg.correlation_matrix_site_types = ["GpC_site"]
    cfg.correlation_matrix_cmaps = ["viridis"]
    cfg.correlation_matrix_flip_axes = True
    cfg.correlation_matrix_n_ticks = 4
    cfg.correlation_matrix_tick_fontsize = 6
    cfg.correlation_matrix_tick_rotation = 90
    cfg.rows_per_qc_autocorr_grid = 4
    cfg.threads = 1

    spatial = execute_partitioned_spatial(preprocess["spine"], cfg, tmp_path / "spatial_outputs")

    regions = pd.read_parquet(spatial["region_catalog"])
    assert regions[["reference", "start", "end", "source"]].to_dict("records") == [
        {"reference": "ref_top", "start": 0, "end": 12, "source": "locus"}
    ]
    assert list(spatial["matrix_store"].rglob("*.parquet"))
    matrix_plots = list(
        (tmp_path / "spatial_outputs" / "plots" / "position_correlation").rglob("method=*/*.png")
    )
    assert len(matrix_plots) == len(list(spatial["matrix_store"].rglob("*.parquet")))
    assert all("page" not in path.name for path in matrix_plots)
    plot_catalog = pd.read_parquet(spatial["plot_catalog"])
    plot_types = set(plot_catalog["plot_type"])
    assert "barcode_region_clustermap" in plot_types
    assert "barcode_region_position_matrix" in plot_types
    assert "read_lomb_scargle_metrics" in plot_types
    assert "read_autocorrelation_clustermap" in plot_types
    matrix_catalog = plot_catalog.loc[plot_catalog["plot_type"] == "barcode_region_position_matrix"]
    assert matrix_catalog["sample"].notna().all()

    read_metric_paths = list(spatial["task_store"].rglob("read_metrics.zarr"))
    assert len(read_metric_paths) == 1
    read_metrics, _ = safe_read_zarr(read_metric_paths[0])
    assert set(read_metrics.obs_names) == {"read1", "read2"}
    assert read_metrics.obsm["C_spatial_autocorr"].shape == (2, cfg.autocorr_max_lag + 1)
    assert read_metrics.obsm["C_spatial_autocorr_counts"].shape == (
        2,
        cfg.autocorr_max_lag + 1,
    )
    assert read_metrics.obsm["C_lomb_scargle_power"].shape == (2, 321)
    assert set(read_metrics.obs["C_ls_status"]) == {"insufficient_sites_or_signal"}
    assert list(read_metrics.uns["C_spatial_autocorr_lags"]) == list(
        range(cfg.autocorr_max_lag + 1)
    )
    assert pd.read_parquet(spatial["read_autocorrelation_axis"])["lag_bp"].tolist() == list(
        range(cfg.autocorr_max_lag + 1)
    )
    spatial_spine, _ = safe_read_h5ad(spatial["spine"])
    assert spatial_spine.uns["spatial_task_store"] == str(spatial["task_store"].resolve())


def test_read_spatial_statistics_saves_known_direct_periodicity():
    rng = np.random.default_rng(42)
    positions = np.sort(rng.choice(4000, size=300, replace=False)).astype(float)
    probability = 0.5 + 0.4 * np.cos(2 * np.pi * positions / 185.0)
    values = (rng.random((1, len(positions))) < probability).astype(float)
    cfg = SimpleNamespace(
        autocorr_max_lag=300,
        autocorr_normalization_method="pearson",
        spatial_save_read_autocorrelation=True,
        spatial_compute_read_lomb_scargle=True,
        spatial_lomb_scargle_period_range_bp=[80, 400],
        spatial_lomb_scargle_peak_range_bp=[150, 250],
        spatial_lomb_scargle_poly_degree=2,
        spatial_lomb_scargle_min_sites=40,
    )

    result = _compute_read_spatial_statistics(values, positions, cfg)

    assert result["autocorrelation"].shape == (1, 301)
    assert result["pair_counts"].shape == (1, 301)
    assert result["status"].tolist() == ["ok"]
    assert 150 <= result["ls_nrl_bp"][0] <= 250
    assert result["ls_peak_power"][0] > 0
    assert result["periodogram_power"].shape == (1, 321)
