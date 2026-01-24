import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import plot_hmm_size_contours


def test_plot_hmm_size_contours_feature_ranges(tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.layers["hmm_lengths"] = np.array(
        [
            [1.0, 4.0, 2.0, 6.0],
            [2.0, 5.0, 3.0, 0.0],
        ]
    )
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2", "3"]

    feature_ranges = [
        (1, 3, "#2E7D32"),
        (4, 10, "#6A1B9A"),
    ]

    figs = plot_hmm_size_contours(
        adata,
        length_layer="hmm_lengths",
        sample_col="Sample_Names",
        ref_obs_col="Reference_strand",
        save_path=tmp_path,
        save_each_page=True,
        save_pdf=False,
        feature_ranges=feature_ranges,
    )

    assert len(figs) == 1
    assert (tmp_path / "hmm_size_page_001.png").is_file()
