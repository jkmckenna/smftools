from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from smftools.plotting import plot_read_span_quality_clustermaps, preprocess_plotting


def test_plot_read_span_quality_clustermaps_filters_nan_positions(tmp_path):
    matplotlib.use("Agg")

    quality = np.array([[10.0, np.nan, 5.0], [12.0, np.nan, 7.0]])
    span = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["base_quality_scores"] = quality
    adata.layers["read_span_mask"] = span
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["pos0", "pos1", "pos2"]

    results = plot_read_span_quality_clustermaps(
        adata,
        save_path=tmp_path,
        max_nan_fraction=0.4,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
    )

    assert len(results) == 1
    assert results[0]["n_positions"] == 2
    assert Path(results[0]["output_path"]).is_file()


def test_plot_read_span_quality_clustermaps_single_read(tmp_path):
    matplotlib.use("Agg")

    quality = np.array([[10.0, 8.0, 5.0]])
    span = np.array([[1.0, 0.0, 1.0]])

    adata = ad.AnnData(X=np.zeros((1, 3)))
    adata.layers["base_quality_scores"] = quality
    adata.layers["read_span_mask"] = span
    adata.obs["Sample_Names"] = pd.Categorical(["S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1"])
    adata.var_names = ["pos0", "pos1", "pos2"]

    results = plot_read_span_quality_clustermaps(
        adata,
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
    )

    assert len(results) == 1
    assert results[0]["n_positions"] == 3
    assert Path(results[0]["output_path"]).is_file()


def test_plot_read_span_quality_clustermaps_reorders_for_inverted_reference(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    # Column j's quality value equals its genomic position, so column order
    # is directly observable from the captured matrix.
    var_coords = np.array([10, 11, 12, 13])
    quality = np.tile(var_coords.astype(float), (2, 1))
    span = np.ones((2, 4))

    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.layers["base_quality_scores"] = quality
    adata.layers["read_span_mask"] = span
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = [str(c) for c in var_coords]
    adata.var["R1_reindexed"] = -var_coords  # inverted, anchored at 0

    captured = {}
    real_plot_one_group = preprocess_plotting._plot_one_group

    def fake_plot_one_group(args):
        if "quality_matrix" not in captured:
            captured["quality_matrix"] = args["quality_matrix"].copy()
            captured["var_names"] = args["var_names"].copy()
        return real_plot_one_group(args)

    monkeypatch.setattr(preprocess_plotting, "_plot_one_group", fake_plot_one_group)

    plot_read_span_quality_clustermaps(
        adata,
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        index_col_suffix="reindexed",
    )

    np.testing.assert_array_equal(captured["quality_matrix"][0], [13, 12, 11, 10])
    np.testing.assert_array_equal(captured["var_names"].astype(str), ["-13", "-12", "-11", "-10"])


def test_plot_read_span_quality_clustermaps_pca_sort(tmp_path):
    matplotlib.use("Agg")

    quality = np.array([[10.0, 8.0, 5.0], [9.0, 7.0, 6.0], [11.0, 6.0, 4.0]])
    span = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    adata = ad.AnnData(X=np.zeros((3, 3)))
    adata.layers["base_quality_scores"] = quality
    adata.layers["read_span_mask"] = span
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1", "R1"])
    adata.var_names = ["pos0", "pos1", "pos2"]

    results = plot_read_span_quality_clustermaps(
        adata,
        save_path=tmp_path,
        sort_method="pca",
        pca_n_components=20,
        pca_sort_component=0,
    )

    assert len(results) == 1
    assert results[0]["n_positions"] == 3
    assert Path(results[0]["output_path"]).is_file()
