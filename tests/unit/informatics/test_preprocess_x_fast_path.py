from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import _load_preprocess_x_selection, materialize
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing


def _frame(barcodes=("bc1", "bc1")):
    rows = []
    for offset, barcode in enumerate(barcodes):
        rows.append(
            {
                "read_id": f"read{offset + 1}",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "barcode": barcode,
                "sample": barcode,
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30 + offset, 30 + offset, 30 + offset, 30 + offset],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [float(offset), np.nan, 0.0, 1.0],
                "read_length": 4,
                "mapped_length": 4,
                "reference_length": 12,
                "read_quality": 30,
                "mapping_quality": 60,
                "read_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_reference_length_ratio": 4 / 12,
                "mapped_length_to_read_length_ratio": 1.0,
            }
        )
    return pd.DataFrame(rows)


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


def _build(tmp_path, barcodes=("bc1", "bc1")):
    raw = write_raw_store(
        _frame(barcodes),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _cfg(), tmp_path / "preprocess_outputs"
    )
    return raw, preprocess


def test_preprocess_shard_carries_x_and_has_x_marker(tmp_path):
    _, preprocess = _build(tmp_path)
    catalog = pd.read_parquet(preprocess["catalog"])
    assert len(catalog) == 1  # single barcode -> single task
    assert catalog["has_x"].all()


def test_fast_path_matches_raw_reconstruction_for_single_barcode_selection(tmp_path, monkeypatch):
    raw, preprocess = _build(tmp_path)

    # Ground truth computed before patching -- this call legitimately needs the
    # raw ragged path (raw's own spine has no preprocess_catalog).
    reference = materialize(raw["spine"], references="ref_top", start=0, end=12)

    from smftools.informatics import partition_read

    def _boom(*args, **kwargs):
        raise AssertionError("raw ragged/tiled reconstruction should not run")

    monkeypatch.setattr(partition_read, "_load_ragged_selection", _boom)
    monkeypatch.setattr(partition_read, "_load_tiled_cache_selection", _boom)

    fast = materialize(
        preprocess["spine"],
        references="ref_top",
        start=0,
        end=12,
        layers=["nan_half"],
    )
    reference = reference[list(fast.obs_names)]
    np.testing.assert_array_equal(np.asarray(fast.X), np.asarray(reference.X))


def test_fast_path_skipped_for_layers_none_and_still_correct(tmp_path):
    _, preprocess = _build(tmp_path)

    result = materialize(preprocess["spine"], references="ref_top", start=0, end=12)

    # layers=None ("give me everything") must never use the preprocess-only fast
    # path -- it can't provide raw-sourced layers like read_span_mask.
    assert "read_span_mask" in result.layers
    assert "nan_half" in result.layers


def test_fast_path_falls_back_for_multi_barcode_selection_spanning_shards(tmp_path):
    raw, preprocess = _build(tmp_path, barcodes=("bc1", "bc2"))
    catalog = pd.read_parquet(preprocess["catalog"])
    assert len(catalog) == 2  # two barcodes -> two separate tasks

    fast = materialize(
        preprocess["spine"],
        references="ref_top",
        start=0,
        end=12,
        layers=["nan_half"],
    )
    reference = materialize(raw["spine"], references="ref_top", start=0, end=12)
    reference = reference[list(fast.obs_names)]
    np.testing.assert_array_equal(np.asarray(fast.X), np.asarray(reference.X))
    assert set(fast.obs_names) == {"read1", "read2"}


def test_fast_path_falls_back_when_catalog_lacks_has_x_column(tmp_path):
    _, preprocess = _build(tmp_path)
    catalog = pd.read_parquet(preprocess["catalog"])
    catalog = catalog.drop(columns=["has_x"])
    catalog.to_parquet(preprocess["catalog"], index=False)

    result = materialize(
        preprocess["spine"],
        references="ref_top",
        start=0,
        end=12,
        layers=["nan_half"],
    )
    assert result.n_obs == 2


def test_fast_path_falls_back_for_window_not_covered_by_one_task(tmp_path):
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 24},
        analysis_mode="genome",
        genome_tile_size=8,
        genome_tile_halo=0,
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTACACGCGTACGTAC"}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _cfg(), tmp_path / "preprocess_outputs"
    )
    catalog = pd.read_parquet(preprocess["catalog"])
    # Reads only span position 0-4, so only the first 8-wide tile has any data --
    # its core window is [0, 8), which does not cover the requested [0, 16).
    assert list(catalog["core_start"]) == [0]
    assert list(catalog["core_end"]) == [8]

    # Requested window extends past what the single covering task's core spans.
    result = materialize(
        preprocess["spine"],
        references="ref_top",
        start=0,
        end=16,
        layers=["nan_half"],
    )
    assert result.n_vars == 16


def test_load_preprocess_x_selection_returns_none_without_run_root():
    assert _load_preprocess_x_selection(None, pd.DataFrame(), None, [], None, None) is None
