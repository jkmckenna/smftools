from types import SimpleNamespace

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smftools.plotting.chimeric_plotting import plot_rolling_nn_and_two_layers


def test_plot_rolling_nn_and_two_layers_masks_read_span(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.obs["Sample"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = [f"pos{i}" for i in range(3)]
    adata.layers["layer_one"] = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, 0.0, 1.0],
        ]
    )
    adata.layers["layer_two"] = np.array(
        [
            [1.0, np.nan, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    adata.layers["read_span_mask"] = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    adata.obsm["rolling_nn_dist"] = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )
    adata.uns["rolling_nn_dist_starts"] = np.array([0, 1])

    captured = []

    def fake_heatmap(data, ax=None, **kwargs):
        captured.append(np.asarray(data))
        return ax

    def fake_clustermap(data, **kwargs):
        return SimpleNamespace(
            dendrogram_row=SimpleNamespace(reordered_ind=[0, 1]),
            fig=plt.figure(),
        )

    monkeypatch.setattr("smftools.plotting.chimeric_plotting.sns.heatmap", fake_heatmap)
    monkeypatch.setattr("smftools.plotting.chimeric_plotting.sns.clustermap", fake_clustermap)

    plot_rolling_nn_and_two_layers(
        adata,
        obsm_key="rolling_nn_dist",
        layer_keys=("layer_one", "layer_two"),
        save_name=tmp_path / "plot.png",
    )

    assert len(captured) == 3
    layer_one_plot = captured[1]
    layer_two_plot = captured[2]
    assert np.isnan(layer_one_plot[0, 1])
    assert np.isnan(layer_one_plot[1, 0])
    assert np.isnan(layer_two_plot[0, 1])
    assert np.isnan(layer_two_plot[1, 0])
