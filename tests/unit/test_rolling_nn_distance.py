import anndata as ad
import numpy as np

from smftools.tools.rolling_nn_distance import rolling_window_nn_distance


def test_rolling_window_nn_distance_basic():
    X = np.array(
        [
            [1.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 0.0, 1.0],
            [np.nan, 1.0, 1.0, 1.0],
        ]
    )
    adata = ad.AnnData(X)

    distances, starts = rolling_window_nn_distance(
        adata,
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm="rolling_nn_dist",
    )

    assert starts.tolist() == [0]
    assert distances.shape == (3, 1)

    expected = np.array([[1.0 / 3.0], [1.0 / 3.0], [0.5]])
    np.testing.assert_allclose(distances, expected, rtol=1e-6)

    assert "rolling_nn_dist" in adata.obsm
    assert "rolling_nn_dist_starts" in adata.uns
    assert adata.uns["rolling_nn_dist_window"] == 4
    assert adata.uns["rolling_nn_dist_min_overlap"] == 2
