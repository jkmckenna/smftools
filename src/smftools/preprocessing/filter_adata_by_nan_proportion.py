## filter_adata_by_nan_proportion

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def filter_adata_by_nan_proportion(
    adata: "ad.AnnData", threshold: float, axis: str = "obs"
) -> "ad.AnnData":
    """Filter an AnnData object on NaN proportion in a matrix axis.

    Args:
        adata: AnnData object to filter.
        threshold: Maximum allowed NaN proportion.
        axis: Whether to filter based on ``"obs"`` or ``"var"`` NaN content.

    Returns:
        anndata.AnnData: Filtered AnnData object.

    Raises:
        ValueError: If ``axis`` is not ``"obs"`` or ``"var"``.
    """
    import numpy as np

    if axis == "obs":
        # Calculate the proportion of NaN values in each read
        nan_proportion = np.isnan(adata.X).mean(axis=1)
        # Filter reads to keep reads with less than a certain NaN proportion
        filtered_indices = np.where(nan_proportion <= threshold)[0]
        filtered_adata = adata[filtered_indices, :].copy()
    elif axis == "var":
        # Calculate the proportion of NaN values at a given position
        nan_proportion = np.isnan(adata.X).mean(axis=0)
        # Filter positions to keep positions with less than a certain NaN proportion
        filtered_indices = np.where(nan_proportion <= threshold)[0]
        filtered_adata = adata[:, filtered_indices].copy()
    else:
        raise ValueError("Axis must be either 'obs' or 'var'")
    return filtered_adata
