## invert_adata

from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def invert_adata(
    adata: "ad.AnnData",
    uns_flag: str = "invert_adata_performed",
    force_redo: bool = False,
) -> "ad.AnnData":
    """Invert the AnnData object along the column axis.

    Args:
        adata: AnnData object.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: New AnnData object with inverted column ordering.
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        # QC already performed; nothing to do
        return adata

    logger.info("Inverting AnnData along the column axis...")

    # Reverse the order of columns (variables)
    inverted_adata = adata[:, ::-1].copy()

    # Reassign var_names with new order
    inverted_adata.var_names = adata.var_names

    # Optional: Store original coordinates for reference
    inverted_adata.var["Original_var_names"] = adata.var_names[::-1]

    # mark as done
    inverted_adata.uns[uns_flag] = True

    logger.info("Inversion complete!")
    return inverted_adata
