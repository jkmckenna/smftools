from types import SimpleNamespace

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smftools.plotting.general_plotting import plot_rolling_nn_and_layer


def test_plot_rolling_nn_and_layer_filters_nan_fraction(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((3, 4)))
    adata.obs["Sample"] = pd.Categorical(["S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1", "R1"])
    adata.var_names = [f"pos{i}" for i in range(4)]
    adata.layers["nan0_0minus1"] = np.array(
        [
            [0.0, 1.0, np.nan, 0.0],
            [1.0, np.nan, np.nan, 1.0],
            [0.0, 1.0, 1.0, np.nan],
        ]
    )
    adata.obsm["rolling_nn_dist"] = np.array(
        [
            [0.1, np.nan],
            [0.2, 0.3],
            [np.nan, 0.4],
        ]
    )
    adata.uns["rolling_nn_dist_starts"] = np.array([0, 2])
    adata.var["R1_valid_fraction"] = [1.0, 0.5, 0.2, 0.8]

    captured = []

    def fake_heatmap(data, ax=None, **kwargs):
        captured.append(np.asarray(data))
        return ax

    def fake_clustermap(data, **kwargs):
        return SimpleNamespace(
            dendrogram_row=SimpleNamespace(reordered_ind=list(range(data.shape[0]))),
            fig=plt.figure(),
        )

    monkeypatch.setattr("smftools.plotting.general_plotting.sns.heatmap", fake_heatmap)
    monkeypatch.setattr("smftools.plotting.general_plotting.sns.clustermap", fake_clustermap)

    plot_rolling_nn_and_layer(
        adata,
        obsm_key="rolling_nn_dist",
        layer_key="nan0_0minus1",
        max_nan_fraction=0.4,
        var_valid_fraction_col="R1_valid_fraction",
        save_name=tmp_path / "plot.png",
    )

    assert len(captured) == 2
    assert captured[1].shape[1] == 2
