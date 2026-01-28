import anndata as ad
import matplotlib
import numpy as np

from smftools.plotting import plot_cp_sequence_components


def test_plot_cp_sequence_components_writes_files(tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((3, 5)))
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.varm["H_cp_sequence"] = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.1],
            [0.0, 0.3],
            [0.5, 0.4],
            [0.0, 0.2],
        ]
    )
    adata.uns["cp_sequence"] = {
        "base_factors": np.array(
            [
                [0.2, 0.1],
                [0.1, 0.2],
                [0.3, 0.0],
                [0.4, 0.5],
            ]
        ),
        "base_labels": ["A", "C", "G", "T"],
    }

    outputs = plot_cp_sequence_components(adata, output_dir=tmp_path, max_positions=10)

    assert outputs["heatmap"].exists()
    assert outputs["lineplot"].exists()
    assert outputs["base_factors"].exists()


def test_plot_cp_sequence_components_skips_nan_rows(tmp_path):
    matplotlib.use("Agg")

    adata = ad.AnnData(X=np.zeros((2, 12)))
    adata.var_names = [f"pos{i}" for i in range(12)]
    components = np.full((12, 2), np.nan)
    components[1] = [0.2, 0.3]
    components[5] = [0.1, 0.4]
    adata.varm["H_cp_sequence"] = components
    adata.uns["cp_sequence"] = {
        "base_factors": np.array(
            [
                [0.2, 0.1],
                [0.1, 0.2],
                [0.3, 0.0],
                [0.4, 0.5],
            ]
        ),
        "base_labels": ["A", "C", "G", "T"],
    }

    outputs = plot_cp_sequence_components(adata, output_dir=tmp_path, max_positions=10)

    assert outputs["heatmap"].exists()
    assert outputs["lineplot"].exists()
    assert outputs["base_factors"].exists()
