from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np

logger = get_logger(__name__)


def calculate_nmf(
    adata: "ad.AnnData",
    layer: str | None = "nan_half",
    var_mask: "np.ndarray | Sequence[bool] | None" = None,
    n_components: int = 2,
    max_iter: int = 200,
    random_state: int = 0,
    overwrite: bool = True,
    embedding_key: str = "X_nmf",
    components_key: str = "H_nmf",
    uns_key: str = "nmf",
    suffix: str | None = None,
) -> "ad.AnnData":
    """Compute a low-dimensional NMF embedding.

    Args:
        adata: AnnData object to update.
        layer: Layer name to use for NMF (``None`` uses ``adata.X``).
        var_mask: Optional boolean mask to subset features.
        n_components: Number of NMF components to compute.
        max_iter: Maximum number of NMF iterations.
        random_state: Random seed for the NMF initializer.
        overwrite: Whether to recompute if the embedding already exists.
        embedding_key: Key for the embedding in ``adata.obsm``.
        components_key: Key for the components matrix in ``adata.varm``.
        uns_key: Key for metadata stored in ``adata.uns``.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    from scipy.sparse import issparse

    require("sklearn", extra="ml-base", purpose="NMF calculation")
    from sklearn.decomposition import NMF

    if suffix:
        embedding_key = f"{embedding_key}_{suffix}"
        components_key = f"{components_key}_{suffix}"
        uns_key = f"{uns_key}_{suffix}"

    has_embedding = embedding_key in adata.obsm
    has_components = components_key in adata.varm
    if has_embedding and has_components and not overwrite:
        logger.info("NMF embedding and components already present; skipping recomputation.")
        return adata
    if has_embedding and not has_components and not overwrite:
        logger.info("NMF embedding present without components; recomputing to store components.")

    subset_mask = None
    if var_mask is not None:
        subset_mask = np.asarray(var_mask, dtype=bool)
        if subset_mask.ndim != 1 or subset_mask.shape[0] != adata.n_vars:
            raise ValueError(
                "var_mask must be a 1D boolean array with length matching adata.n_vars."
            )
        adata_subset = adata[:, subset_mask].copy()
        logger.info(
            "Subsetting adata: retained %s features based on filters %s",
            adata_subset.shape[1],
            "var_mask",
        )
    else:
        adata_subset = adata.copy()
        logger.info("No var_mask provided. Using all features.")

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
    components = model.components_.T

    if subset_mask is not None:
        components_matrix = np.zeros((adata.shape[1], components.shape[1]))
        components_matrix[subset_mask, :] = components
    else:
        components_matrix = components

    adata.obsm[embedding_key] = embedding
    adata.varm[components_key] = components_matrix
    adata.uns[uns_key] = {
        "n_components": n_components,
        "max_iter": max_iter,
        "random_state": random_state,
        "layer": layer,
        "var_mask_provided": var_mask is not None,
        "components_key": components_key,
    }

    logger.info(
        "Stored: adata.obsm['%s'] and adata.varm['%s']",
        embedding_key,
        components_key,
    )
    return adata
