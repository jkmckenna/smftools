"""Partition-store projection baseline for future ML data adapters."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.informatics.partition_store import write_experiment_store
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

pytestmark = pytest.mark.integration


def test_partition_projection_preserves_requested_groups_layer_and_positions(tmp_path):
    values = np.arange(36, dtype=np.float32).reshape(6, 6)
    samples = np.repeat(["sample-a", "sample-b", "sample-c"], 2)
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["ref_top"] * 6),
            "Sample": pd.Categorical(samples),
            "experiment_uid": ["experiment-1"] * 6,
            "reference_start": np.zeros(6, dtype=np.int64),
            "reference_end": np.full(6, 6, dtype=np.int64),
        },
        index=[f"read-{index}" for index in range(6)],
    )
    adata = ad.AnnData(
        X=np.zeros_like(values),
        obs=obs,
        layers={
            "C_site_binary": values,
            "unused_layer": np.full_like(values, -1),
        },
    )
    adata.var_names = [str(position) for position in range(6)]
    paths = write_experiment_store(
        adata,
        tmp_path,
        experiment="experiment-1",
        modality="deaminase",
    )
    spine, _ = safe_read_h5ad(paths["spine"], verbose=False)
    spine.obs["reference_start"] = np.zeros(spine.n_obs, dtype=np.int64)
    spine.obs["reference_end"] = np.full(spine.n_obs, 6, dtype=np.int64)
    safe_write_h5ad(spine, paths["spine"], backup=False, verbose=False)

    projected = materialize(
        paths["spine"],
        references="ref_top",
        samples=["sample-a", "sample-c"],
        layers=["C_site_binary"],
        start=1,
        end=4,
    )

    expected_rows = np.array([0, 1, 4, 5])
    assert set(projected.obs["Sample"].astype(str)) == {"sample-a", "sample-c"}
    assert list(projected.var_names) == ["1", "2", "3"]
    assert set(projected.layers) == {"C_site_binary"}
    np.testing.assert_array_equal(
        projected.layers["C_site_binary"],
        values[expected_rows, 1:4],
    )
