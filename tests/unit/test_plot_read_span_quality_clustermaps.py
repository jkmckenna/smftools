from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from smftools.plotting import plot_read_span_quality_clustermaps


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
