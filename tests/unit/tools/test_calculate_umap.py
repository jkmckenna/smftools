from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.tools import calculate_pca, calculate_umap


@pytest.mark.unit
def test_calculate_umap_outputs() -> None:
    pytest.importorskip("umap")
    pytest.importorskip("pynndescent")

    rng = np.random.default_rng(0)
    data = rng.random((12, 6))
    adata = ad.AnnData(data)
    adata.layers["nan_half"] = data.copy()

    calculate_pca(adata, n_pcs=3)
    calculate_umap(adata, obsm="X_pca")

    assert adata.obsm["X_pca"].shape == (12, 3)
    assert adata.obsm["X_umap"].shape == (12, 2)
    assert adata.obsp["connectivities_X_pca"].shape == (12, 12)
    assert adata.varm["PCs"].shape == (6, 3)


@pytest.mark.unit
def test_plot_embedding_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    from smftools.plotting import plot_embedding

    rng = np.random.default_rng(1)
    data = rng.random((8, 4))
    adata = ad.AnnData(data)
    adata.obsm["X_umap"] = rng.random((8, 2))
    adata.obs["group"] = pd.Categorical(["a", "b", "a", "b", "a", "b", "a", "b"])

    outputs = plot_embedding(adata, basis="umap", color="group", output_dir=tmp_path)
    assert outputs["group"].exists()


@pytest.mark.unit
def test_plot_embedding_handles_ndarray_palette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    import smftools.plotting.latent_plotting as latent_plotting

    rng = np.random.default_rng(2)
    data = rng.random((6, 4))
    adata = ad.AnnData(data)
    adata.obsm["X_umap"] = rng.random((6, 2))
    adata.obs["group"] = pd.Categorical(["a", "b", "a", "b", "a", "b"])

    original_palette = latent_plotting.sns.color_palette

    def _ndarray_palette(name: str, n_colors: int) -> np.ndarray:
        return np.array(original_palette(name, n_colors=n_colors))

    monkeypatch.setattr(latent_plotting.sns, "color_palette", _ndarray_palette)

    outputs = latent_plotting.plot_embedding(
        adata, basis="umap", color="group", output_dir=tmp_path
    )
    assert outputs["group"].exists()


@pytest.mark.unit
def test_plot_embedding_grid_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    from smftools.plotting import plot_embedding_grid

    rng = np.random.default_rng(3)
    data = rng.random((10, 4))
    adata = ad.AnnData(data)
    adata.obsm["X_umap"] = rng.random((10, 2))
    adata.obs["group"] = pd.Categorical(["a", "b"] * 5)
    adata.obs["score"] = rng.random(10)

    output = plot_embedding_grid(
        adata,
        basis="umap",
        color=["group", "score"],
        output_dir=tmp_path,
    )
    assert output is not None
    assert output.exists()


@pytest.mark.unit
def test_plot_pca_explained_variance_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    from smftools.plotting import plot_pca_explained_variance

    rng = np.random.default_rng(4)
    data = rng.random((10, 6))
    adata = ad.AnnData(data)
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.3, 0.2, 0.1])}

    output = plot_pca_explained_variance(adata, output_dir=tmp_path)
    assert output is not None
    assert output.exists()
