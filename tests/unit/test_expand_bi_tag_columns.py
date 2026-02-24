import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.h5ad_functions import expand_bi_tag_columns


def test_expand_bi_tag_columns_handles_array_like_and_null_values():
    adata = ad.AnnData(X=np.zeros((4, 1)))
    adata.obs["bi"] = pd.Series(
        [
            [1.0, 2.0, 3.0, 0.8, 5.0, 6.0, 0.9],
            np.array([10.0, 20.0, 30.0, 0.7, 50.0, 60.0, 0.6], dtype=float),
            None,
            np.nan,
        ],
        index=adata.obs_names,
        dtype=object,
    )

    expand_bi_tag_columns(adata, bi_column="bi")

    assert adata.obs.loc[adata.obs_names[0], "bi_overall_score"] == 1.0
    assert adata.obs.loc[adata.obs_names[0], "bi_top_score"] == 0.8
    assert adata.obs.loc[adata.obs_names[0], "bi_bottom_score"] == 0.9

    assert adata.obs.loc[adata.obs_names[1], "bi_overall_score"] == 10.0
    assert adata.obs.loc[adata.obs_names[1], "bi_top_score"] == 0.7
    assert adata.obs.loc[adata.obs_names[1], "bi_bottom_score"] == 0.6

    assert np.isnan(adata.obs.loc[adata.obs_names[2], "bi_overall_score"])
    assert np.isnan(adata.obs.loc[adata.obs_names[3], "bi_overall_score"])
