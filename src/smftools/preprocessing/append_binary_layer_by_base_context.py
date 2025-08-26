import numpy as np
import scipy.sparse as sp

def append_binary_layer_by_base_context(
    adata,
    reference_column: str,
    smf_modality: str = "conversion",
    verbose: bool = True,
    uns_flag: str = "binary_layers_by_base_context_added",
    bypass: bool = False,
    force_redo: bool = False
):
    """
    Build per-reference C/G-site masked layers:
      - GpC_site_binary
      - CpG_site_binary
      - GpC_CpG_combined_site_binary (numeric sum where present; NaN where neither present)
      - any_C_site_binary
      - other_C_site_binary

    Behavior:
      - If X is sparse it will be converted to dense for these layers (keeps original adata.X untouched).
      - Missing var columns are warned about but do not crash.
      - Masked positions are filled with np.nan to make masked vs unmasked explicit.
      - Requires append_base_context to be run first
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass or ("base_context_added" not in adata.uns):
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
    reference_to_gpc_column = {ref: f"{ref}_GpC_site" for ref in references}
    reference_to_cpg_column = {ref: f"{ref}_CpG_site" for ref in references}
    reference_to_c_column = {ref: f"{ref}_any_C_site" for ref in references}
    reference_to_other_c_column = {ref: f"{ref}_other_C_site" for ref in references}

    # verify var columns exist and build boolean masks per ref (len = n_vars)
    n_obs, n_vars = adata.shape
    def _col_mask_or_warn(colname):
        if colname not in adata.var.columns:
            if verbose:
                print(f"Warning: var column '{colname}' not found; treating as all-False mask.")
            return np.zeros(n_vars, dtype=bool)
        vals = adata.var[colname].values
        # coerce truthiness
        try:
            return vals.astype(bool)
        except Exception:
            return np.array([bool(v) for v in vals], dtype=bool)

    gpc_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_gpc_column.items()}
    cpg_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_cpg_column.items()}
    c_var_masks =   {ref: _col_mask_or_warn(col) for ref, col in reference_to_c_column.items()}
    other_c_var_masks = {ref: _col_mask_or_warn(col) for ref, col in reference_to_other_c_column.items()}

    # prepare X as dense float32 for layer filling (we leave adata.X untouched)
    X = adata.X
    if sp.issparse(X):
        if verbose:
            print("Converting sparse X to dense array for layer construction (temporary).")
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # initialize masked arrays filled with NaN
    masked_gpc = np.full((n_obs, n_vars), np.nan, dtype=np.float32)
    masked_cpg = np.full((n_obs, n_vars), np.nan, dtype=np.float32)
    masked_any_c = np.full((n_obs, n_vars), np.nan, dtype=np.float32)
    masked_other_c = np.full((n_obs, n_vars), np.nan, dtype=np.float32)

    # fill row-blocks per reference (this avoids creating a full row×var boolean mask)
    obs_ref_series = adata.obs[reference_column]
    for ref in references:
        rows_mask = (obs_ref_series.values == ref)
        if not rows_mask.any():
            continue
        row_idx = np.nonzero(rows_mask)[0]  # integer indices of rows for this ref

        # column masks for this ref
        gpc_cols = gpc_var_masks.get(ref, np.zeros(n_vars, dtype=bool))
        cpg_cols = cpg_var_masks.get(ref, np.zeros(n_vars, dtype=bool))
        c_cols   = c_var_masks.get(ref, np.zeros(n_vars, dtype=bool))
        other_c_cols = other_c_var_masks.get(ref, np.zeros(n_vars, dtype=bool))

        if gpc_cols.any():
            # assign only the submatrix (rows x selected cols)
            masked_gpc[np.ix_(row_idx, gpc_cols)] = X[np.ix_(row_idx, gpc_cols)]
        if cpg_cols.any():
            masked_cpg[np.ix_(row_idx, cpg_cols)] = X[np.ix_(row_idx, cpg_cols)]
        if c_cols.any():
            masked_any_c[np.ix_(row_idx, c_cols)] = X[np.ix_(row_idx, c_cols)]
        if other_c_cols.any():
            masked_other_c[np.ix_(row_idx, other_c_cols)] = X[np.ix_(row_idx, other_c_cols)]

    # Build combined layer:
    # - numeric_sum: sum where either exists, NaN where neither exists
    #   we compute numeric sum but preserve NaN where both are NaN
    gpc_nan = np.isnan(masked_gpc)
    cpg_nan = np.isnan(masked_cpg)
    combined_sum = np.nan_to_num(masked_gpc, nan=0.0) + np.nan_to_num(masked_cpg, nan=0.0)
    both_nan = gpc_nan & cpg_nan
    combined_sum[both_nan] = np.nan

    # Alternative: if you prefer a boolean OR combined layer, uncomment:
    # combined_bool = (~gpc_nan & (masked_gpc != 0)) | (~cpg_nan & (masked_cpg != 0))
    # combined_layer = combined_bool.astype(np.float32)

    adata.layers['GpC_site_binary'] = masked_gpc
    adata.layers['CpG_site_binary'] = masked_cpg
    adata.layers['GpC_CpG_combined_site_binary'] = combined_sum
    adata.layers['any_C_site_binary'] = masked_any_c
    adata.layers['other_C_site_binary'] = masked_other_c

    if verbose:
        def _filled_positions(arr):
            return int(np.sum(~np.isnan(arr)))
        print("Layer build summary (non-NaN cell counts):")
        print(f"  GpC: {_filled_positions(masked_gpc)}")
        print(f"  CpG: {_filled_positions(masked_cpg)}")
        print(f"  GpC+CpG combined: {_filled_positions(combined_sum)}")
        print(f"  any_C: {_filled_positions(masked_any_c)}")
        print(f"  other_C: {_filled_positions(masked_other_c)}")

    # mark as done
    adata.uns[uns_flag] = True

    return adata
