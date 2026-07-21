import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import combined_raw_clustermap


def test_combined_raw_clustermap_nan_fill(tmp_path, monkeypatch):
    matrix = np.array(
        [
            [0.0, np.nan, 1.0],
            [1.0, 0.0, np.nan],
        ]
    )
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["0", "1", "2"]
    adata.var["R1_C_site"] = [True, True, True]
    adata.var["R1_GpC_site"] = [True, True, True]
    adata.var["R1_CpG_site"] = [True, True, True]
    adata.var["R1_A_site"] = [False, False, False]

    captured_heatmaps = []

    from smftools.optional_imports import require as real_require

    class SeabornProxy:
        @staticmethod
        def heatmap(data, **_kwargs):
            captured_heatmaps.append(np.asarray(data))

    def capture_seaborn(module, **kwargs):
        if module == "seaborn":
            return SeabornProxy()
        return real_require(module, **kwargs)

    monkeypatch.setattr("smftools.optional_imports.require", capture_seaborn)

    results = combined_raw_clustermap(
        adata,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        sort_by="gpc",
        fill_nan_strategy="value",
        fill_nan_value=0.0,
    )

    assert len(results) == 1
    assert captured_heatmaps
    assert all(not np.isnan(matrix).any() for matrix in captured_heatmaps)
