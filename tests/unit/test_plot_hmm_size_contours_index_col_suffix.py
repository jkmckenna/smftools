import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.plotting import plot_hmm_size_contours


def _adata(n_positions=3):
    var_names = [str(i) for i in range(n_positions)]
    adata = ad.AnnData(X=np.zeros((2, n_positions)))
    adata.layers["hmm_lengths"] = np.array(
        [
            [5.0, 10.0, 5.0],
            [3.0, 4.0, 3.0],
        ]
    )
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = var_names
    var_coords = np.arange(n_positions)
    adata.var["R1_reindexed"] = -var_coords  # inverted, anchored at 0
    return adata


def _capture_x(monkeypatch, tmp_path, **kwargs):
    import matplotlib.axes

    captured = {}
    real_pcolormesh = matplotlib.axes.Axes.pcolormesh

    def fake_pcolormesh(self, x_edges, *args, **fake_kwargs):
        if "x_edges" not in captured:
            captured["x_edges"] = np.asarray(x_edges)
        return real_pcolormesh(self, x_edges, *args, **fake_kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", fake_pcolormesh)
    plot_hmm_size_contours(
        _adata(),
        length_layer="hmm_lengths",
        sample_col="Sample_Names",
        ref_obs_col="Reference_strand",
        save_path=tmp_path,
        save_each_page=True,
        save_pdf=False,
        **kwargs,
    )
    return captured["x_edges"]


def test_no_index_col_suffix_uses_raw_var_names(monkeypatch, tmp_path):
    x_edges = _capture_x(monkeypatch, tmp_path)
    assert x_edges[0] < x_edges[-1]
    np.testing.assert_allclose(x_edges, [-0.5, 0.5, 1.5, 2.5])


def test_index_col_suffix_uses_reindexed_column(monkeypatch, tmp_path):
    x_edges = _capture_x(monkeypatch, tmp_path, index_col_suffix="reindexed")
    # Reindexed coords are [0, -1, -2] -- descending; edges should follow.
    np.testing.assert_allclose(x_edges, [0.5, -0.5, -1.5, -2.5])


def test_index_col_suffix_falls_back_when_column_missing(monkeypatch, tmp_path):
    adata = _adata()
    del adata.var["R1_reindexed"]

    captured = {}
    import matplotlib.axes

    real_pcolormesh = matplotlib.axes.Axes.pcolormesh

    def fake_pcolormesh(self, x_edges, *args, **fake_kwargs):
        if "x_edges" not in captured:
            captured["x_edges"] = np.asarray(x_edges)
        return real_pcolormesh(self, x_edges, *args, **fake_kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", fake_pcolormesh)
    plot_hmm_size_contours(
        adata,
        length_layer="hmm_lengths",
        sample_col="Sample_Names",
        ref_obs_col="Reference_strand",
        save_path=tmp_path,
        save_each_page=True,
        save_pdf=False,
        index_col_suffix="reindexed",
    )
    np.testing.assert_allclose(captured["x_edges"], [-0.5, 0.5, 1.5, 2.5])
