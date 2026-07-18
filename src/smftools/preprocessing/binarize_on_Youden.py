from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def binarize_on_Youden(
    adata: "ad.AnnData",
    ref_column: str = "Reference_strand",
    output_layer_name: str = "binarized_methylation",
    mask_failed_positions: bool = True,
) -> None:
    """Binarize SMF values using thresholds from ``calculate_position_Youden``.

    Args:
        adata: AnnData object to binarize.
        ref_column: Obs column denoting reference/strand categories.
        output_layer_name: Layer in which to store the binarized matrix.
        mask_failed_positions: If ``True``, positions that failed Youden QC are set to NaN;
            otherwise all positions are binarized.
    """

    import numpy as np

    # Extract dense X once
    X = adata.X
    if hasattr(X, "toarray"):  # sparse → dense
        X = X.toarray()

    n_obs, n_var = X.shape
    # Kept float32 (via the default float dtype) deliberately: this layer is
    # {0, 1, NaN}, so it can't be int8 (no NaN) and must NOT be float16 -- numpy
    # keeps a float16 accumulator, so downstream sum/nansum/nanmean over the read
    # axis overflows to inf past ~2048 reads. Making this a bool methylation layer
    # paired with a separate bool "observed" mask (dropping the NaN sentinel) is
    # the memory win, but that is a cross-repo change tracked on the layer-audit
    # branch, not a contained dtype swap.
    binarized = np.full((n_obs, n_var), np.nan, dtype=np.float32)

    references = adata.obs[ref_column].cat.categories
    ref_labels = adata.obs[ref_column].to_numpy()

    for ref in references:
        logger.info("Binarizing on Youden statistics for %s", ref)

        ref_mask = ref_labels == ref
        if not np.any(ref_mask):
            continue

        X_block = X[ref_mask, :].astype(np.float32, copy=True)

        # thresholds: list of (threshold, J)
        youden_stats = adata.var[f"{ref}_position_methylation_thresholding_Youden_stats"].to_numpy()

        thresholds = np.array(
            [t[0] if isinstance(t, (tuple, list)) else np.nan for t in youden_stats],
            dtype=float,
        )

        # QC mask
        qc_mask = adata.var[f"{ref}_position_passed_Youden_thresholding_QC"].to_numpy().astype(bool)

        if mask_failed_positions:
            # Only binarize positions passing QC
            cols_to_binarize = np.where(qc_mask)[0]
        else:
            # Binarize all positions
            cols_to_binarize = np.arange(n_var)

        # Prepare result block (float32: holds 0/1/NaN exactly; float32 reductions
        # are safe -- unlike float16, which overflows in numpy's float16 accumulator)
        block_out = np.full_like(X_block, np.nan, dtype=np.float32)

        if len(cols_to_binarize) > 0:
            sub_X = X_block[:, cols_to_binarize]
            sub_thresh = thresholds[cols_to_binarize]

            nan_mask = np.isnan(sub_X)

            bin_sub = (sub_X > sub_thresh[None, :]).astype(np.float32)
            bin_sub[nan_mask] = np.nan

            block_out[:, cols_to_binarize] = bin_sub

        # Write into full output matrix
        binarized[ref_mask, :] = block_out

    adata.layers[output_layer_name] = binarized
    logger.info(
        "Finished binarization → stored in adata.layers['%s'] (mask_failed_positions=%s)",
        output_layer_name,
        mask_failed_positions,
    )


# def binarize_on_Youden(adata,
#                        ref_column='Reference_strand',
#                        output_layer_name='binarized_methylation'):
#     """
#     Binarize SMF values based on position thresholds determined by calculate_position_Youden.

#     Parameters:
#         adata (AnnData): The anndata object to binarize. `calculate_position_Youden` must have been run first.
#         obs_column (str): The obs column to stratify on. Needs to match what was passed in `calculate_position_Youden`.

#     Modifies:
#         Adds a new layer to `adata.layers['binarized_methylation']` containing the binarized methylation matrix.
#     """
#     import numpy as np
#     import anndata as ad

#     # Initialize an empty matrix to store the binarized methylation values
#     binarized_methylation = np.full_like(adata.X, np.nan, dtype=float)  # Keeps same shape as adata.X

#     # Get unique categories
#     references = adata.obs[ref_column].cat.categories

#     for ref in references:
#         print(f"Binarizing adata on Youden statistics for {ref}")
#         # Select subset for this category
#         ref_mask = adata.obs[ref_column] == ref
#         ref_subset = adata[ref_mask]

#         # Extract the probability matrix
#         original_matrix = ref_subset.X.copy()

#         # Extract the thresholds for each position efficiently
#         thresholds = np.array(ref_subset.var[f'{ref}_position_methylation_thresholding_Youden_stats'].apply(lambda x: x[0]))

#         # Identify NaN values
#         nan_mask = np.isnan(original_matrix)

#         # Binarize based on threshold
#         binarized_matrix = (original_matrix > thresholds).astype(float)

#         # Restore NaN values
#         binarized_matrix[nan_mask] = np.nan

#         # Assign the binarized values back into the preallocated storage
#         binarized_methylation[ref_subset, :] = binarized_matrix

#     # Store the binarized matrix in a new layer
#     adata.layers[output_layer_name] = binarized_methylation

#     print(f"Finished binarizing adata on Youden statistics")
