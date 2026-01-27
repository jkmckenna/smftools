import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import combined_hmm_length_clustermap


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
