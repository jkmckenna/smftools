from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.ragged_store import cigar_max_indel_runs
from smftools.preprocessing.filter_reads_on_cigar_indels import (
    filter_reads_on_cigar_indels,
)


def _make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "max_insertion_length": [0, 5, 15, 8],
            "max_deletion_length": [3, 12, 0, 9],
        },
        index=[f"r{i}" for i in range(4)],
    )
    return ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)


def test_cigar_max_indel_runs_reports_longest_internal_runs():
    assert cigar_max_indel_runs("10M5I10M") == (5, 0)
    assert cigar_max_indel_runs("5S10M12D8M3H") == (0, 12)
    # longest run across multiple ops wins
    assert cigar_max_indel_runs("10M3I5M4I2M") == (4, 0)
    # no internal indels
    assert cigar_max_indel_runs("100M") == (0, 0)


def test_cigar_max_indel_runs_tolerates_absent_cigar():
    assert cigar_max_indel_runs("*") == (0, 0)
    assert cigar_max_indel_runs("") == (0, 0)


def test_default_thresholds_remove_large_indels():
    adata = _make_adata()
    out = filter_reads_on_cigar_indels(adata, max_insertion_length=10, max_deletion_length=10)
    # r1 deletion=12>10, r2 insertion=15>10 -> removed
    assert set(out.obs_names) == {"r0", "r3"}


def test_none_thresholds_disable_each_check():
    adata = _make_adata()
    # only enforce deletion; r1 (del=12) drops, r2 (ins=15) survives
    out = filter_reads_on_cigar_indels(adata, max_insertion_length=None, max_deletion_length=10)
    assert set(out.obs_names) == {"r0", "r2", "r3"}

    # both disabled -> keep all
    out_all = filter_reads_on_cigar_indels(
        adata, max_insertion_length=None, max_deletion_length=None
    )
    assert set(out_all.obs_names) == {"r0", "r1", "r2", "r3"}


def test_bypass_returns_input_unchanged():
    adata = _make_adata()
    out = filter_reads_on_cigar_indels(adata, bypass=True)
    assert out is adata
    assert out.n_obs == 4


def test_missing_columns_skips_filter_gracefully():
    adata = ad.AnnData(X=np.zeros((2, 1), dtype=float), obs=pd.DataFrame(index=["a", "b"]))
    out = filter_reads_on_cigar_indels(adata, max_insertion_length=10, max_deletion_length=10)
    assert set(out.obs_names) == {"a", "b"}


def test_partitioned_read_qc_sidecar_applies_cigar_indel_filter(tmp_path):
    from types import SimpleNamespace

    from smftools.preprocessing.partitioned_executor import write_read_qc_sidecar

    obs = pd.DataFrame(
        {
            "read_length": [100.0, 100.0, 100.0],
            "mapped_length": [100.0, 100.0, 100.0],
            "max_insertion_length": [2, 15, 3],
            "max_deletion_length": [1, 0, 12],
        },
        index=["keep", "big_ins", "big_del"],
    )
    spine = ad.AnnData(obs=obs)
    cfg = SimpleNamespace(
        read_len_filter_thresholds=[None, None],
        mapped_len_filter_thresholds=[None, None],
        read_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_ref_ratio_filter_thresholds=[None, None],
        mapped_len_to_read_len_ratio_filter_thresholds=[None, None],
        read_quality_filter_thresholds=[None, None],
        read_mapping_quality_filter_thresholds=[None, None],
        bypass_filter_reads_on_length_quality_mapping=False,
        max_internal_insertion_length=10,
        max_internal_deletion_length=10,
        bypass_filter_reads_on_cigar_indels=False,
    )
    path = write_read_qc_sidecar(spine, cfg, tmp_path / "read_qc.parquet")
    passes = pd.read_parquet(path).set_index("read_id")["passes_read_qc"].to_dict()
    assert passes == {"keep": True, "big_ins": False, "big_del": False}
