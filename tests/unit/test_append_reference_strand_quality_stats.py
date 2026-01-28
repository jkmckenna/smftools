import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.h5ad_functions import append_reference_strand_quality_stats


def test_append_reference_strand_quality_stats_adds_expected_vars():
    quality_matrix = np.array(
        [
            [30, 20, 10],
            [20, -1, 10],
            [40, 40, 40],
            [10, 10, 10],
        ],
        dtype=np.int16,
    )
    read_span_mask = np.ones_like(quality_matrix, dtype=np.int16)

    obs = pd.DataFrame({"Reference_strand": ["ref1", "ref1", "ref2", "ref2"]})
    obs["Reference_strand"] = obs["Reference_strand"].astype("category")
    var = pd.DataFrame(index=["p0", "p1", "p2"])
    var["position_in_ref1"] = [True, True, False]
    var["position_in_ref2"] = [True, True, True]

    adata = ad.AnnData(X=np.zeros(quality_matrix.shape), obs=obs, var=var)
    adata.layers["base_quality_scores"] = quality_matrix
    adata.layers["read_span_mask"] = read_span_mask

    append_reference_strand_quality_stats(adata)

    assert "ref1_mean_base_quality" in adata.var
    assert "ref1_std_base_quality" in adata.var
    assert "ref1_mean_error_rate" in adata.var
    assert "ref1_std_error_rate" in adata.var
    assert "ref2_mean_base_quality" in adata.var

    np.testing.assert_allclose(
        adata.var["ref1_mean_base_quality"].values,
        np.array([25.0, 20.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref1_std_base_quality"].values,
        np.array([5.0, 0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref2_mean_base_quality"].values,
        np.array([25.0, 25.0, 25.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref2_std_base_quality"].values,
        np.array([15.0, 15.0, 15.0]),
        equal_nan=True,
    )

    np.testing.assert_allclose(
        adata.var["ref1_mean_error_rate"].values,
        np.array([0.0055, 0.01, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref1_std_error_rate"].values,
        np.array([0.0045, 0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref2_mean_error_rate"].values,
        np.array([0.05005, 0.05005, 0.05005]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        adata.var["ref2_std_error_rate"].values,
        np.array([0.04995, 0.04995, 0.04995]),
        rtol=1e-6,
    )
