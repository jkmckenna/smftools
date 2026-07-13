from types import SimpleNamespace

import numpy as np
import pandas as pd

from smftools.informatics.partition_read import materialize
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.readwrite import safe_read_h5ad, safe_read_zarr


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
        "sample_retention_heatmap",
        "modification_qc_metric_dashboard",
        "duplicate_rate_by_sample_reference",
        "valid_fraction_by_position",
        "read_span_base_quality",
    }.issubset(set(plot_catalog["plot_type"]))
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
