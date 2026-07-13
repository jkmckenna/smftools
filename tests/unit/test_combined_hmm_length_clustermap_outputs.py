import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import combined_hmm_length_clustermap, hmm_plotting


def test_combined_hmm_length_clustermap_outputs(tmp_path):
    matrix = np.array(
        [
            [0.0, 10.0, 25.0],
            [15.0, np.nan, 35.0],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["GpC_all_accessible_features_lengths"] = matrix
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2"]

    combined_hmm_length_clustermap(
        adata,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        length_layer="GpC_all_accessible_features_lengths",
        length_feature_ranges=[(3, 20, "#A5D6A7"), (20, 40, "#2E7D32")],
        save_path=tmp_path,
        sort_by="hmm",
        fill_nan_strategy="value",
        fill_nan_value=0.0,
    )

    out_file = tmp_path / "R1__S1.png"
    assert out_file.is_file()


def test_combined_hmm_length_clustermap_keeps_nan_panels_visible(tmp_path):
    matrix = np.array(
        [
            [0.0, np.nan, 25.0],
            [15.0, np.nan, 35.0],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["GpC_all_accessible_features_lengths"] = matrix
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2"]
    adata.var["R1_C_site"] = [True, True, True]
    adata.var["R1_GpC_site"] = [True, True, True]
    adata.var["R1_CpG_site"] = [True, True, True]
    adata.var["R1_A_site"] = [False, False, False]

    captured = []
    original_heatmap = hmm_plotting.sns.heatmap

    def fake_heatmap(matrix, *args, **kwargs):
        captured.append(np.asarray(matrix))
        return None

    hmm_plotting.sns.heatmap = fake_heatmap
    try:
        combined_hmm_length_clustermap(
            adata,
            min_quality=None,
            min_length=None,
            min_mapped_length_to_reference_length_ratio=None,
            min_position_valid_fraction=None,
            length_layer="GpC_all_accessible_features_lengths",
            save_path=tmp_path,
            sort_by="hmm",
            fill_nan_strategy="value",
            fill_nan_value=0.0,
        )
    finally:
        hmm_plotting.sns.heatmap = original_heatmap

    assert any(np.isnan(matrix).any() for matrix in captured[1:])
