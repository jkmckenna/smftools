import anndata as ad
import matplotlib
import numpy as np

from smftools.plotting import plot_pca_components


def test_plot_pca_components_writes_files(tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((4, 6)))
    adata.var_names = [f"pos{i}" for i in range(6)]
    adata.varm["PCs"] = np.array(
        [
            [0.2, -0.1],
            [-0.3, 0.0],
            [0.0, 0.4],
            [0.1, -0.2],
            [-0.05, 0.3],
            [0.15, -0.25],
        ]
    )

    outputs = plot_pca_components(adata, output_dir=tmp_path, max_features=10)

    assert outputs["heatmap"].exists()
    assert outputs["lineplot"].exists()
