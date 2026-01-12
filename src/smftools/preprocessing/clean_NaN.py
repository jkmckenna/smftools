from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def clean_NaN(
    adata: "ad.AnnData",
    layer: str | None = None,
    uns_flag: str = "clean_NaN_performed",
    bypass: bool = False,
    force_redo: bool = True,
) -> None:
    """Append layers to ``adata`` that contain NaN-cleaning strategies.

    Args:
        adata: AnnData object.
        layer: Layer to fill NaN values in. If ``None``, uses ``adata.X``.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
    """

    from ..readwrite import adata_to_df

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

    # Ensure the specified layer exists
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Convert to DataFrame
    df = adata_to_df(adata, layer=layer)

    # Fill NaN with closest SMF value (forward then backward fill)
    logger.info("Making layer: fill_nans_closest")
    adata.layers["fill_nans_closest"] = df.ffill(axis=1).bfill(axis=1).values

    # Replace NaN with 0, and 0 with -1
    logger.info("Making layer: nan0_0minus1")
    df_nan0_0minus1 = df.replace(0, -1).fillna(0)
    adata.layers["nan0_0minus1"] = df_nan0_0minus1.values

    # Replace NaN with 1, and 1 with 2
    logger.info("Making layer: nan1_12")
    df_nan1_12 = df.replace(1, 2).fillna(1)
    adata.layers["nan1_12"] = df_nan1_12.values

    # Replace NaN with -1
    logger.info("Making layer: nan_minus_1")
    df_nan_minus_1 = df.fillna(-1)
    adata.layers["nan_minus_1"] = df_nan_minus_1.values

    # Replace NaN with -1
    logger.info("Making layer: nan_half")
    df_nan_half = df.fillna(0.5)
    adata.layers["nan_half"] = df_nan_half.values

    # mark as done
    adata.uns[uns_flag] = True

    return None
