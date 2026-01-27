import anndata as ad
import matplotlib
import numpy as np

from smftools.plotting import plot_nmf_components


def test_plot_nmf_components_writes_files(tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((3, 5)))
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.varm["H_nmf"] = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.1],
            [0.0, 0.3],
            [0.5, 0.4],
            [0.0, 0.2],
        ]
    )

    outputs = plot_nmf_components(adata, output_dir=tmp_path, max_features=10)

    assert outputs["heatmap"].exists()
    assert outputs["lineplot"].exists()
