import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import combined_hmm_raw_clustermap, hmm_plotting


def test_combined_hmm_raw_clustermap_nan_fill(tmp_path):
    matrix = np.array(
        [
            [0.0, np.nan, 1.0],
            [1.0, 0.0, np.nan],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["hmm_combined"] = matrix
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
        combined_hmm_raw_clustermap(
            adata,
            min_quality=None,
            min_length=None,
            min_mapped_length_to_reference_length_ratio=None,
            min_position_valid_fraction=None,
            save_path=tmp_path,
            sort_by="hmm",
            fill_nan_strategy="value",
            fill_nan_value=0.0,
        )
    finally:
        hmm_plotting.sns.heatmap = original_heatmap

    assert any(np.isnan(matrix).any() for matrix in captured[1:])

    out_file = tmp_path / "R1__S1.png"
    assert out_file.is_file()
