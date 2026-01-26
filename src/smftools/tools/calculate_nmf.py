from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_nmf(
    adata: "ad.AnnData",
    layer: str | None = "nan_half",
    var_filters: Sequence[str] | None = None,
    n_components: int = 2,
    max_iter: int = 200,
    random_state: int = 0,
    overwrite: bool = True,
) -> "ad.AnnData":
    """Compute a low-dimensional NMF embedding.

    Args:
        adata: AnnData object to update.
        layer: Layer name to use for NMF (``None`` uses ``adata.X``).
        var_filters: Optional list of var masks to subset features.
        n_components: Number of NMF components to compute.
        max_iter: Maximum number of NMF iterations.
        random_state: Random seed for the NMF initializer.
        overwrite: Whether to recompute if the embedding already exists.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    from scipy.sparse import issparse

    require("sklearn", extra="ml-base", purpose="NMF calculation")
    from sklearn.decomposition import NMF

    if "X_nmf" in adata.obsm and not overwrite:
        logger.info("NMF embedding already present; skipping recomputation.")
        return adata

    if var_filters:
        subset_mask = np.logical_or.reduce([adata.var[f].values for f in var_filters])
        adata_subset = adata[:, subset_mask].copy()
        logger.info(
            "Subsetting adata: retained %s features based on filters %s",
            adata_subset.shape[1],
            var_filters,
        )
    else:
        adata_subset = adata.copy()
        logger.info("No var filters provided. Using all features.")

    data = adata_subset.layers[layer] if layer else adata_subset.X
    if issparse(data):
        data = data.copy()
        if data.data.size and np.isnan(data.data).any():
            logger.warning("NaNs detected in sparse data, filling with 0.5 before NMF.")
            data.data = np.nan_to_num(data.data, nan=0.5)
        if data.data.size and (data.data < 0).any():
            logger.warning("Negative values detected in sparse data, clipping to 0 for NMF.")
            data.data[data.data < 0] = 0
    else:
        if np.isnan(data).any():
            logger.warning("NaNs detected, filling with 0.5 before NMF.")
            data = np.nan_to_num(data, nan=0.5)
        if (data < 0).any():
            logger.warning("Negative values detected, clipping to 0 for NMF.")
            data = np.clip(data, a_min=0, a_max=None)

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=random_state,
    )
    embedding = model.fit_transform(data)

    adata.obsm["X_nmf"] = embedding
    adata.uns["nmf"] = {
        "n_components": n_components,
        "max_iter": max_iter,
        "random_state": random_state,
        "layer": layer,
        "var_filters": list(var_filters) if var_filters else None,
    }

    logger.info("Stored: adata.obsm['X_nmf']")
    return adata
