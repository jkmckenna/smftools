import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smftools.plotting.general_plotting import plot_zero_hamming_pair_counts


def test_plot_zero_hamming_pair_counts_builds_counts(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((3, 4)))
    adata.obs["Sample"] = pd.Categorical(["S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1", "R1"])
    adata.var_names = ["1", "2", "3", "4"]
    zero_pairs_key = "rolling_zero_pairs"
    adata.uns[zero_pairs_key] = [
        np.array([[0, 1], [1, 2]]),
        np.empty((0, 2), dtype=int),
    ]
    adata.uns[f"{zero_pairs_key}_starts"] = np.array([0, 2])
    adata.uns[f"{zero_pairs_key}_window"] = 2

    captured = {}

    def fake_clustermap(data, **kwargs):
        captured["data"] = np.asarray(data)
        fig, ax = plt.subplots()
        return type(
            "Dummy",
            (),
            {"ax_heatmap": ax, "fig": fig},
        )()

    monkeypatch.setattr(
        "smftools.plotting.spatial_plotting.sns.clustermap", fake_clustermap
    )

    plot_zero_hamming_pair_counts(
        adata,
        zero_pairs_uns_key=zero_pairs_key,
        save_name=tmp_path / "plot.png",
    )

    assert np.array_equal(captured["data"][:, 0], np.array([1, 2, 1]))
    assert np.array_equal(captured["data"][:, 1], np.array([0, 0, 0]))
