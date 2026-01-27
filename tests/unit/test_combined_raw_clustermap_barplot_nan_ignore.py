import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import combined_raw_clustermap


def test_combined_raw_clustermap_barplot_ignores_fill(tmp_path, monkeypatch):
    matrix = np.array(
        [
            [1.0, np.nan, 0.0],
            [np.nan, 1.0, 1.0],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2"]
    adata.var["R1_C_site"] = [True, True, True]
    adata.var["R1_GpC_site"] = [True, True, True]
    adata.var["R1_CpG_site"] = [True, True, True]
    adata.var["R1_A_site"] = [False, False, False]

    captured = {}

    def fake_clean_barplot(ax, mean_values, title):
        captured[title] = np.array(mean_values, copy=True)

    monkeypatch.setattr("smftools.plotting.general_plotting.clean_barplot", fake_clean_barplot)

    combined_raw_clustermap(
        adata,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        sort_by="gpc",
        fill_nan_strategy="value",
        fill_nan_value=-1.0,
    )

    expected = np.ones(3)
    np.testing.assert_allclose(captured["GpC Modification Signal"], expected)


def test_combined_raw_clustermap_barplot_counts_zeros_for_raw_layer(tmp_path, monkeypatch):
    matrix = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 1.0],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["raw"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2"]
    adata.var["R1_C_site"] = [True, True, True]
    adata.var["R1_GpC_site"] = [True, True, True]
    adata.var["R1_CpG_site"] = [True, True, True]
    adata.var["R1_A_site"] = [False, False, False]

    captured = {}

    def fake_clean_barplot(ax, mean_values, title):
        captured[title] = np.array(mean_values, copy=True)

    monkeypatch.setattr("smftools.plotting.general_plotting.clean_barplot", fake_clean_barplot)

    combined_raw_clustermap(
        adata,
        layer_c="raw",
        layer_gpc="raw",
        layer_cpg="raw",
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        sort_by="gpc",
        fill_nan_strategy="none",
    )

    expected = np.array([0.5, 0.5, 1.0])
    np.testing.assert_allclose(captured["GpC Modification Signal"], expected)
