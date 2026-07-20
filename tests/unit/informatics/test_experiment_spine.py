from types import SimpleNamespace

import numpy as np
import pandas as pd

from smftools.constants import HMM_DIR, LATENT_DIR, PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR
from smftools.informatics.experiment_spine import (
    experiment_spine_path,
    write_experiment_spine,
)
from smftools.informatics.partition_read import materialize
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad
from smftools.tools.partitioned_hmm import execute_partitioned_hmm
from smftools.tools.partitioned_spatial import execute_partitioned_spatial


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
        emit_automated_plots=False,
    )


def test_write_experiment_spine_returns_none_without_raw_obs_parquet(tmp_path):
    assert write_experiment_spine(tmp_path) is None
    assert not experiment_spine_path(tmp_path).exists()


def test_write_experiment_spine_raw_only(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / RAW_DIR,
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )

    path = experiment_spine_path(tmp_path)
    assert path == tmp_path / "experiment_spine_outputs" / "spine.h5ad"
    assert path.exists()  # written automatically by write_raw_store

    experiment_spine, _ = safe_read_h5ad(path)
    raw_spine, _ = safe_read_h5ad(raw["spine"])
    assert list(experiment_spine.obs_names) == list(raw_spine.obs_names)
    assert set(experiment_spine.obs.columns) == set(raw_spine.obs.columns)
    assert bool(experiment_spine.uns["is_spine"])
    assert "preprocess_catalog" not in experiment_spine.uns
    assert experiment_spine.uns["source_base_dir"] == "raw_outputs"


def test_write_experiment_spine_joins_preprocess_columns(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / RAW_DIR,
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    execute_partitioned_preprocessing(raw["spine"], _preprocess_cfg(), tmp_path / PREPROCESS_DIR)

    experiment_spine, _ = safe_read_h5ad(experiment_spine_path(tmp_path))
    assert "passes_read_qc" in experiment_spine.obs.columns
    # raw-only columns still present -- the join adds preprocess's columns, it
    # doesn't replace raw's.
    assert "Reference_strand" in experiment_spine.obs.columns
    assert experiment_spine.uns["preprocess_catalog"]


def test_write_experiment_spine_unions_latent_catalog_pointer(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / RAW_DIR,
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
    )
    raw_spine, _ = safe_read_h5ad(raw["spine"])
    latent_spine = raw_spine.copy()
    latent_spine.uns["latent_task_catalog"] = "latent_adata_outputs/task_catalog.parquet"
    latent_path = tmp_path / LATENT_DIR / "spine.h5ad"
    safe_write_h5ad(latent_spine, latent_path, backup=False, verbose=False)

    write_experiment_spine(tmp_path)

    experiment_spine, _ = safe_read_h5ad(experiment_spine_path(tmp_path))
    assert experiment_spine.uns["latent_task_catalog"] == (
        "latent_adata_outputs/task_catalog.parquet"
    )


def test_experiment_spine_resolves_sibling_branch_layers_together(tmp_path, monkeypatch):
    """The concrete case a single per-stage spine can't do: spatial and hmm are
    sibling branches off preprocess, so materializing off either one alone only
    ever carries that one branch's uns catalog pointer forward. The consolidated
    experiment_spine.h5ad unions both, so one materialize() call can pull an hmm
    derived layer *and* a spatial read-metric together.
    """
    from smftools.tools import partitioned_hmm

    raw = write_raw_store(
        _frame(),
        tmp_path / RAW_DIR,
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / PREPROCESS_DIR
    )

    spatial_cfg = _preprocess_cfg()
    spatial_cfg.autocorr_site_types = ["GpC", "CpG", "C"]
    spatial_cfg.autocorr_max_lag = 4
    spatial_cfg.autocorr_normalization_method = "pearson"
    execute_partitioned_spatial(preprocess["spine"], spatial_cfg, tmp_path / SPATIAL_DIR)

    def annotate(adata, task, cfg, models_dir):
        adata.layers["GpC_test_feature"] = np.ones(adata.shape, dtype=np.int8)
        adata.uns["hmm_appended_layers"] = ["GpC_test_feature"]
        return ["GpC_test_feature"]

    monkeypatch.setattr(partitioned_hmm, "_annotate_task", annotate)
    hmm_cfg = SimpleNamespace(target_task_memory_mb=1)
    execute_partitioned_hmm(preprocess["spine"], hmm_cfg, tmp_path / HMM_DIR)

    experiment_spine, _ = safe_read_h5ad(experiment_spine_path(tmp_path))
    assert experiment_spine.uns["spatial_task_catalog"]
    assert experiment_spine.uns["hmm_catalog"]

    restored = materialize(
        experiment_spine_path(tmp_path),
        references="ref_top",
        read_ids=["read1", "read2"],
        start=0,
        end=12,
        layers=["GpC_test_feature"],
        read_metrics=True,
    )
    assert np.all(restored.layers["GpC_test_feature"] == 1)
    assert restored.obsm  # spatial's per-read outputs attached alongside hmm's layer
