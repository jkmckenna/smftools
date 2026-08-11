from __future__ import annotations

import anndata as ad
import numpy as np

from smftools.hmm.HMM import mask_layers_outside_read_span
from smftools.tools.partitioned_hmm import _mask_uncovered_model_input


def test_mask_layers_outside_read_span_uses_var_names() -> None:
    obs = {"reference_start": [2, 1], "reference_end": [3, 4]}
    var_names = ["1", "2", "3", "4"]
    adata = ad.AnnData(X=np.zeros((2, 4)), obs=obs, var={"idx": var_names})
    adata.var_names = var_names

    adata.layers["hmm_test"] = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=int)

    mask_layers_outside_read_span(adata, ["hmm_test"])

    masked = adata.layers["hmm_test"]
    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[0, 3])
    assert masked[0, 1] == 2
    assert masked[0, 2] == 3
    assert np.all(~np.isnan(masked[1, :]))


def test_mask_layers_outside_read_span_uses_original_var_names() -> None:
    obs = {"reference_start": [2], "reference_end": [3]}
    var_names = ["4", "3", "2", "1"]
    adata = ad.AnnData(
        X=np.zeros((1, 4)), obs=obs, var={"Original_var_names": ["1", "2", "3", "4"]}
    )
    adata.var_names = var_names

    adata.layers["hmm_test"] = np.array([[10, 11, 12, 13]], dtype=int)

    mask_layers_outside_read_span(adata, ["hmm_test"])

    masked = adata.layers["hmm_test"]
    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[0, 3])
    assert masked[0, 1] == 11
    assert masked[0, 2] == 12


def test_mask_layers_prefers_observed_base_coverage_for_paired_gap() -> None:
    adata = ad.AnnData(
        X=np.zeros((1, 5)),
        obs={"reference_start": [0], "reference_end": [4]},
    )
    adata.var_names = ["0", "1", "2", "3", "4"]
    adata.layers["covered_base_mask"] = np.array([[1, 1, 0, 1, 1]], dtype=np.int8)
    adata.layers["hmm_test"] = np.ones((1, 5), dtype=float)

    mask_layers_outside_read_span(adata, ["hmm_test"])

    np.testing.assert_array_equal(adata.layers["hmm_test"], [[1.0, 1.0, np.nan, 1.0, 1.0]])


def test_hmm_model_input_keeps_paired_gap_missing() -> None:
    adata = ad.AnnData(X=np.zeros((1, 5)))
    adata.var_names = ["10", "11", "12", "13", "14"]
    adata.layers["covered_base_mask"] = np.array([[1, 1, 0, 1, 1]], dtype=np.int8)

    result = _mask_uncovered_model_input(
        adata,
        np.ones((1, 3, 2), dtype=float),
        np.array([11, 12, 13]),
    )

    assert np.isnan(result[0, 1]).all()
    assert np.isfinite(result[0, [0, 2]]).all()
