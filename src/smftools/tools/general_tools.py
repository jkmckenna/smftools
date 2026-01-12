from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def create_nan_mask_from_X(adata: "ad.AnnData", new_layer_name: str = "nan_mask") -> "ad.AnnData":
    """Generate a NaN mask layer from ``adata.X``.

    Args:
        adata: AnnData object.
        new_layer_name: Name of the output mask layer.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    import numpy as np

    nan_mask = np.isnan(adata.X).astype(int)
    adata.layers[new_layer_name] = nan_mask
    logger.info("Created '%s' layer based on NaNs in adata.X", new_layer_name)
    return adata


def create_nan_or_non_gpc_mask(
    adata: "ad.AnnData",
    obs_column: str,
    new_layer_name: str = "nan_or_non_gpc_mask",
) -> "ad.AnnData":
    """Generate a mask layer combining NaNs and non-GpC positions.

    Args:
        adata: AnnData object.
        obs_column: Obs column used to derive reference-specific GpC masks.
        new_layer_name: Name of the output mask layer.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    import numpy as np

    nan_mask = np.isnan(adata.X).astype(int)
    combined_mask = np.zeros_like(nan_mask)

    for idx, row in enumerate(adata.obs.itertuples()):
        ref = getattr(row, obs_column)
        gpc_mask = adata.var[f"{ref}_GpC_site"].astype(int).values
        combined_mask[idx, :] = 1 - gpc_mask  # non-GpC is 1

    mask = np.maximum(nan_mask, combined_mask)
    adata.layers[new_layer_name] = mask

    logger.info(
        "Created '%s' layer based on NaNs in adata.X and non-GpC regions using %s",
        new_layer_name,
        obs_column,
    )
    return adata


def combine_layers(
    adata: "ad.AnnData",
    input_layers: Sequence[str],
    output_layer: str,
    negative_mask: str | None = None,
    values: Sequence[int] | None = None,
    binary_mode: bool = False,
) -> "ad.AnnData":
    """Combine layers into a single coded layer.

    Args:
        adata: AnnData object.
        input_layers: Input layer names.
        output_layer: Name of the output layer.
        negative_mask: Optional binary mask layer to enforce zeros.
        values: Values assigned to each input layer when ``binary_mode`` is ``False``.
        binary_mode: Whether to build a simple 0/1 mask.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    import numpy as np

    combined = np.zeros_like(adata.layers[input_layers[0]])

    if binary_mode:
        for layer in input_layers:
            combined = np.logical_or(combined, adata.layers[layer] > 0)
        combined = combined.astype(int)
    else:
        if values is None:
            values = list(range(1, len(input_layers) + 1))
        for i, layer in enumerate(input_layers):
            arr = adata.layers[layer]
            combined[arr > 0] = values[i]

    if negative_mask:
        mask = adata.layers[negative_mask]
        combined[mask == 0] = 0

    adata.layers[output_layer] = combined
    logger.info(
        "Combined layers into %s %s",
        output_layer,
        "(binary)" if binary_mode else f"with values {values}",
    )

    return adata
