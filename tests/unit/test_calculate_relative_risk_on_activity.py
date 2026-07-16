import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from smftools.tools.position_stats import calculate_relative_risk_on_activity  # noqa: E402


def _adata(n_positions=4):
    var_names = [str(100 + i) for i in range(n_positions)]
    # 4 reads: 2 active, 2 silent; methylation alternates per read at every
    # position so every position yields a valid, non-degenerate 2x2 table.
    X = np.tile(np.array([[1.0], [0.0], [1.0], [0.0]]), (1, n_positions))
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.var["refA_GpC_site"] = [True] * n_positions
    adata.var["refA_CpG_site"] = [False] * n_positions
    adata.obs["Reference_strand"] = pd.Categorical(["refA"] * 4)
    adata.obs["activity_status"] = ["Active", "Active", "Silent", "Silent"]
    var_coords = np.array([100 + i for i in range(n_positions)])
    adata.var["refA_reindexed"] = -(var_coords - 100)  # inverted, anchored at 100
    return adata


def test_genomic_position_defaults_to_var_names():
    adata = _adata()
    results = calculate_relative_risk_on_activity(adata, sites=["GpC_site"])
    results_df, _ = results["refA"]["all"]
    np.testing.assert_array_equal(results_df["Genomic_Position"].to_numpy(), [100, 101, 102, 103])


def test_genomic_position_uses_reindexed_column_when_configured():
    adata = _adata()
    results = calculate_relative_risk_on_activity(
        adata, sites=["GpC_site"], index_col_suffix="reindexed"
    )
    results_df, _ = results["refA"]["all"]
    np.testing.assert_array_equal(results_df["Genomic_Position"].to_numpy(), [0, -1, -2, -3])


def test_genomic_position_falls_back_when_reindexed_column_missing():
    adata = _adata()
    del adata.var["refA_reindexed"]
    results = calculate_relative_risk_on_activity(
        adata, sites=["GpC_site"], index_col_suffix="reindexed"
    )
    results_df, _ = results["refA"]["all"]
    np.testing.assert_array_equal(results_df["Genomic_Position"].to_numpy(), [100, 101, 102, 103])
