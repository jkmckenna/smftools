from __future__ import annotations

import numpy as np
import anndata as ad

from smftools.hmm.HMM import mask_layers_outside_read_span


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
    adata = ad.AnnData(X=np.zeros((1, 4)), obs=obs, var={"Original_var_names": ["1", "2", "3", "4"]})
    adata.var_names = var_names

    adata.layers["hmm_test"] = np.array([[10, 11, 12, 13]], dtype=int)

    mask_layers_outside_read_span(adata, ["hmm_test"])

    masked = adata.layers["hmm_test"]
    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[0, 3])
    assert masked[0, 1] == 11
    assert masked[0, 2] == 12
