from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad


def binarize_adata(
    adata: "ad.AnnData",
    source: str = "X",
    target_layer: str = "binary",
    threshold: float = 0.8,
) -> None:
    """Binarize a dense matrix and preserve NaNs.

    Args:
        adata: AnnData object with input matrix or layer.
        source: ``"X"`` to use the main matrix or a layer name.
        target_layer: Layer name to store the binarized values.
        threshold: Threshold above which values are set to 1.
    """
    X = adata.X if source == "X" else adata.layers[source]

    # Copy to avoid modifying original in-place
    X_bin = X.copy()

    # Where not NaN: apply threshold
    mask = ~np.isnan(X_bin)
    X_bin[mask] = (X_bin[mask] > threshold).astype(np.int8)

    adata.layers[target_layer] = X_bin
