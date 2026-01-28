from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from smftools.tools.calculate_nmf import calculate_nmf


@pytest.mark.unit
def test_calculate_nmf_adds_embedding() -> None:
    pytest.importorskip("sklearn")
    data = np.array([[0.0, 1.0], [0.5, 0.2], [1.0, 0.3]], dtype=float)
    adata = ad.AnnData(data)
    adata.layers["nan_half"] = data.copy()

    result = calculate_nmf(
        adata,
        layer="nan_half",
        n_components=2,
        max_iter=50,
        random_state=0,
    )

    assert "X_nmf" in result.obsm
    assert result.obsm["X_nmf"].shape == (3, 2)
    assert "H_nmf" in result.varm
    assert result.varm["H_nmf"].shape == (2, 2)
    assert result.uns["nmf"]["n_components"] == 2


@pytest.mark.unit
def test_calculate_nmf_with_var_mask_subsets_features() -> None:
    pytest.importorskip("sklearn")
    data = np.array([[0.2, 0.8, 0.1], [0.4, 0.2, 0.3], [0.1, 0.6, 0.5]], dtype=float)
    adata = ad.AnnData(data)
    adata.layers["nan_half"] = data.copy()

    var_mask = np.array([True, False, True])
    result = calculate_nmf(
        adata,
        layer="nan_half",
        var_mask=var_mask,
        n_components=2,
        max_iter=50,
        random_state=0,
    )

    assert "H_nmf" in result.varm
    assert result.varm["H_nmf"].shape == (3, 2)
    assert np.allclose(result.varm["H_nmf"][1, :], 0.0)
