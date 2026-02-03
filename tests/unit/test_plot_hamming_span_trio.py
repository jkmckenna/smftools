from types import SimpleNamespace

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smftools.plotting.chimeric_plotting import plot_hamming_span_trio


def test_plot_hamming_span_trio_adds_classification_strip(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.obs_names = ["r1", "r2"]
    adata.var_names = ["0", "1", "2", "3"]
    adata.obs["chimeric_by_mod_hamming_distance"] = [True, False]

    adata.layers["zero_hamming_distance_spans"] = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )
    adata.layers["cross_sample_zero_hamming_distance_spans"] = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    adata.layers["delta_zero_hamming_distance_spans"] = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=float,
    )

    captured = []

    def fake_heatmap(data, ax=None, **kwargs):
        captured.append(np.asarray(data))
        return ax

    def fake_clustermap(data, **kwargs):
        return SimpleNamespace(
            dendrogram_row=SimpleNamespace(reordered_ind=[1, 0]),
            fig=plt.figure(),
        )

    monkeypatch.setattr("smftools.plotting.chimeric_plotting.sns.heatmap", fake_heatmap)
    monkeypatch.setattr("smftools.plotting.chimeric_plotting.sns.clustermap", fake_clustermap)

    plot_hamming_span_trio(
        adata,
        save_name=tmp_path / "plot.png",
    )

    # self, cross, delta, classification
    assert len(captured) == 4
    classification = captured[3]
    assert classification.shape == (2, 1)
    np.testing.assert_array_equal(classification[:, 0], np.array([0, 1]))
