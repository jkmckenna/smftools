from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_coverage(
    adata: "ad.AnnData",
    ref_column: str = "Reference_strand",
    position_nan_threshold: float = 0.01,
    smf_modality: str = "deaminase",
    target_layer: str = "binarized_methylation",
    uns_flag: str = "calculate_coverage_performed",
    force_redo: bool = False,
) -> None:
    """Append position-level coverage metadata per reference category.

    Args:
        adata: AnnData object.
        ref_column: Obs column used to define reference/strand categories.
        position_nan_threshold: Minimum fraction of coverage to mark a position as valid.
        smf_modality: SMF modality. Use ``adata.X`` for conversion/deaminase or ``target_layer`` for direct.
        target_layer: Layer used for direct SMF coverage calculations.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
    """
    import numpy as np
    import pandas as pd

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        # QC already performed; nothing to do
        return

    references = adata.obs[ref_column].cat.categories
    n_categories_with_position = np.zeros(adata.shape[1])

    # Loop over references
    for ref in references:
        logger.info("Assessing positional coverage across samples for %s reference", ref)

        # Subset to current category
        ref_mask = adata.obs[ref_column] == ref
        temp_ref_adata = adata[ref_mask]

        if smf_modality == "direct":
            matrix = temp_ref_adata.layers[target_layer]
        else:
            matrix = temp_ref_adata.X

        # Compute fraction of valid coverage
        ref_valid_coverage = np.sum(~np.isnan(matrix), axis=0)
        ref_valid_fraction = ref_valid_coverage / temp_ref_adata.shape[0]  # Avoid extra computation

        # Store coverage stats
        adata.var[f"{ref}_valid_count"] = pd.Series(ref_valid_coverage, index=adata.var.index)
        adata.var[f"{ref}_valid_fraction"] = pd.Series(ref_valid_fraction, index=adata.var.index)

        # Assign whether the position is covered based on threshold
        adata.var[f"position_in_{ref}"] = ref_valid_fraction >= position_nan_threshold

        # Sum the number of categories covering each position
        n_categories_with_position += adata.var[f"position_in_{ref}"].values

    # Store final category count
    adata.var[f"N_{ref_column}_with_position"] = n_categories_with_position.astype(int)

    # mark as done
    adata.uns[uns_flag] = True
