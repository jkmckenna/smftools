from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.filter_reads_on_length_quality_mapping import (
    filter_reads_on_length_quality_mapping,
)


def _make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "read_length": [2800.0, 3000.0, 3200.0, 2900.0],
            "mapped_length": [2600.0, 3000.0, 3100.0, 3050.0],
            "read_length_to_reference_length_ratio": [0.85, 0.95, 1.05, 1.10],
            "mapped_length_to_reference_length_ratio": [0.80, 0.92, 1.00, 1.08],
            "mapped_length_to_read_length_ratio": [
                2600.0 / 2800.0,
                1.0,
                3100.0 / 3200.0,
                3050.0 / 2900.0,
            ],
            "read_quality": [20.0, 20.0, 20.0, 20.0],
            "mapping_quality": [60.0, 60.0, 60.0, 60.0],
        },
        index=[f"r{i}" for i in range(4)],
    )
    return ad.AnnData(X=np.zeros((4, 1), dtype=float), obs=obs)


def test_read_and_mapped_thresholds_apply_independently():
    adata = _make_adata()
    out = filter_reads_on_length_quality_mapping(
        adata,
        read_length=[2800, 3100],  # keeps r0,r1,r3
        mapped_length=[2700, 3050],  # keeps r1,r3
        length_ratio=[0.9, 1.2],  # keeps r1,r2,r3
        mapped_length_ratio=[0.9, 1.05],  # keeps r1,r2
        read_quality=[10, None],
        mapping_quality=[30, None],
    )
    # intersection -> r1 only
    assert list(out.obs_names) == ["r1"]


def test_read_length_thresholds_target_read_length_column():
    adata = _make_adata()
    out = filter_reads_on_length_quality_mapping(
        adata,
        read_length=[None, 2900],  # filters by read_length, keeps r0,r3
        mapped_length=[None, None],
        read_quality=[None, None],
        mapping_quality=[None, None],
    )
    assert set(out.obs_names) == {"r0", "r3"}


def test_mapped_to_read_ratio_thresholds_apply():
    adata = _make_adata()
    # mapped_length_to_read_length_ratio values are:
    # r0=0.9286, r1=1.0, r2=0.96875, r3=1.0517
    out = filter_reads_on_length_quality_mapping(
        adata,
        mapped_to_read_ratio=[0.95, 1.02],
        read_quality=[None, None],
        mapping_quality=[None, None],
    )
    assert set(out.obs_names) == {"r1", "r2"}
