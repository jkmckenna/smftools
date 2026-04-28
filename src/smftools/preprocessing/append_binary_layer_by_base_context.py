from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def append_binary_layer_by_base_context(
    adata: "ad.AnnData",
    reference_column: str,
    smf_modality: str = "conversion",
    verbose: bool = True,
    uns_flag: str = "append_binary_layer_by_base_context_performed",
    bypass: bool = False,
    force_redo: bool = False,
    from_valid_sites_only: bool = False,
    valid_site_col_suffix: str = "_valid_coverage",
) -> "ad.AnnData":
    """Build per-reference masked layers for base-context sites.

    Args:
        adata: AnnData object to annotate.
        reference_column: Obs column containing reference identifiers.
        smf_modality: SMF modality identifier.
        verbose: Whether to log layer summary information.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        from_valid_sites_only: Whether to use valid-coverage site masks only.
        valid_site_col_suffix: Suffix for valid-coverage site columns.

    Returns:
        anndata.AnnData: AnnData object with new masked layers.
    """
    if not from_valid_sites_only:
        valid_site_col_suffix = ""

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass or ("append_base_context_performed" not in adata.uns):
        # QC already performed; nothing to do
        return adata

    # check inputs
    if reference_column not in adata.obs.columns:
        raise KeyError(f"reference_column '{reference_column}' not found in adata.obs")

    # modality flag (kept for your potential use)
    if smf_modality != "direct":
        if smf_modality == "conversion":
            deaminase = False
        else:
            deaminase = True
    else:
        deaminase = None  # unused but preserved

    # expected per-reference var column names
    references = adata.obs[reference_column].astype("category").cat.categories
    reference_to_gpc_column = {ref: f"{ref}_GpC_site{valid_site_col_suffix}" for ref in references}
    reference_to_cpg_column = {ref: f"{ref}_CpG_site{valid_site_col_suffix}" for ref in references}
    reference_to_c_column = {ref: f"{ref}_C_site{valid_site_col_suffix}" for ref in references}
    reference_to_other_c_column = {
        ref: f"{ref}_other_C_site{valid_site_col_suffix}" for ref in references
    }
    reference_to_a_column = {ref: f"{ref}_A_site{valid_site_col_suffix}" for ref in references}

    # verify var columns exist and build boolean masks per ref (len = n_vars)
    n_obs, n_vars = adata.shape

    def _col_mask_or_warn(colname):
        """Return a boolean mask for a var column, or all-False if missing."""
        if colname not in adata.var.columns:
            if verbose:
                logger.warning(
                    "Var column '%s' not found; treating as all-False mask.",
                    colname,
                )
            return np.zeros(n_vars, dtype=bool)
        vals = adata.var[colname].values
        # coerce truthiness
        try:
            return vals.astype(bool)
        except Exception:
            return np.array([bool(v) for v in vals], dtype=bool)

    gpc_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_gpc_column.items()}
    cpg_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_cpg_column.items()}
    c_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_c_column.items()}
    other_c_var_masks = {
        ref: _col_mask_or_warn(col) for ref, col in reference_to_other_c_column.items()
    }
    a_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_a_column.items()}

    # prepare X as dense float32 for layer filling (we leave adata.X untouched)
    X = adata.X
    if sp.issparse(X):
        if verbose:
            logger.info("Converting sparse X to dense array for layer construction (temporary).")
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    obs_ref_series = adata.obs[reference_column]

    def _build_masked(var_masks: dict) -> np.ndarray:
        """Allocate one NaN-filled layer and fill row-blocks per reference."""
        arr = np.full((n_obs, n_vars), np.nan, dtype=np.float32)
        for ref in references:
            cols = var_masks.get(ref, np.zeros(n_vars, dtype=bool))
            if not cols.any():
                continue
            rows_mask = obs_ref_series.values == ref
            if not rows_mask.any():
                continue
            row_idx = np.nonzero(rows_mask)[0]
            arr[np.ix_(row_idx, cols)] = X[np.ix_(row_idx, cols)]
        return arr

    # Build and store each layer sequentially — only one working array lives in
    # memory at a time alongside X, keeping peak at ~2× X instead of ~7× X.

    masked_gpc = _build_masked(gpc_var_masks)
    adata.layers["GpC_site_binary"] = masked_gpc
    if verbose:
        logger.info("  GpC non-NaN cells: %s", int(np.sum(~np.isnan(masked_gpc))))
    del masked_gpc

    masked_cpg = _build_masked(cpg_var_masks)
    adata.layers["CpG_site_binary"] = masked_cpg
    if verbose:
        logger.info("  CpG non-NaN cells: %s", int(np.sum(~np.isnan(masked_cpg))))
    del masked_cpg

    # Build combined GpC+CpG from the already-stored layers (no extra full arrays)
    gpc_stored = adata.layers["GpC_site_binary"]
    cpg_stored = adata.layers["CpG_site_binary"]
    both_nan = np.isnan(gpc_stored) & np.isnan(cpg_stored)
    combined_sum = np.nan_to_num(gpc_stored, nan=0.0) + np.nan_to_num(cpg_stored, nan=0.0)
    combined_sum[both_nan] = np.nan
    adata.layers["GpC_CpG_combined_site_binary"] = combined_sum
    if verbose:
        logger.info("  GpC+CpG combined non-NaN cells: %s", int(np.sum(~np.isnan(combined_sum))))
    del combined_sum, both_nan, gpc_stored, cpg_stored

    masked_any_c = _build_masked(c_var_masks)
    adata.layers["C_site_binary"] = masked_any_c
    if verbose:
        logger.info("  C non-NaN cells: %s", int(np.sum(~np.isnan(masked_any_c))))
    del masked_any_c

    masked_other_c = _build_masked(other_c_var_masks)
    adata.layers["other_C_site_binary"] = masked_other_c
    if verbose:
        logger.info("  other_C non-NaN cells: %s", int(np.sum(~np.isnan(masked_other_c))))
    del masked_other_c

    masked_a = _build_masked(a_var_masks)
    adata.layers["A_site_binary"] = masked_a
    if verbose:
        logger.info("  A non-NaN cells: %s", int(np.sum(~np.isnan(masked_a))))
    del masked_a

    # Release dense X copy now that all layers are built
    del X

    # mark as done
    adata.uns[uns_flag] = True

    return adata
