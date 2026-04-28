## invert_adata

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def invert_adata(
    adata: "ad.AnnData",
    uns_flag: str = "invert_adata_performed",
    force_redo: bool = False,
) -> "ad.AnnData":
    """Invert the AnnData object along the column axis in-place.

    Flips each layer and X one at a time to avoid materialising the full
    AnnData twice (which would peak at 2× total size).

    Args:
        adata: AnnData object.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: The same AnnData object with inverted column ordering.
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        logger.info("Inversion already performed")
        return adata

    logger.info("Inverting AnnData along the column axis...")

    # Store original var_names before flipping
    original_var_names = adata.var_names.copy()

    # Flip X one matrix at a time
    if sp.issparse(adata.X):
        adata.X = adata.X[:, ::-1]
    else:
        adata.X = np.ascontiguousarray(adata.X[:, ::-1])

    # Flip each layer one at a time so only one extra layer-sized array exists at peak
    for key in list(adata.layers.keys()):
        arr = adata.layers[key]
        if sp.issparse(arr):
            adata.layers[key] = arr[:, ::-1]
        else:
            adata.layers[key] = np.ascontiguousarray(arr[:, ::-1])

    # Flip var DataFrame and restore var_names in original order
    adata.var["Original_var_names"] = original_var_names[::-1]
    adata.var = adata.var.iloc[::-1]
    adata.var.index = original_var_names

    # mark as done
    adata.uns[uns_flag] = True

    logger.info("Inversion complete!")
    return adata
