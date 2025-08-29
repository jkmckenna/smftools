import math
import gc
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Sequence, List

def filter_reads_on_modification_thresholds(
    adata: ad.AnnData,
    smf_modality: str,
    mod_target_bases: List[str] = [],
    gpc_thresholds: Optional[Sequence[float]] = None,
    cpg_thresholds: Optional[Sequence[float]] = None,
    any_c_thresholds: Optional[Sequence[float]] = None,
    a_thresholds: Optional[Sequence[float]] = None,
    use_other_c_as_background: bool = False,
    min_valid_fraction_positions_in_read_vs_ref: Optional[float] = None,
    uns_flag: str = 'reads_filtered_on_modification_thresholds',
    bypass: bool = False,
    force_redo: bool = False,
    reference_column: str = 'Reference_strand',
    # memory-control options:
    batch_size: int = 200,
    compute_obs_if_missing: bool = True,
    treat_zero_as_invalid: bool = False
) -> ad.AnnData:
    """
    Memory-efficient filtering by per-read modification thresholds.

    - If required obs columns exist, uses them directly (fast).
    - Otherwise, computes the relevant per-read metrics per-reference in batches
      and writes them into adata.obs before filtering.

    Parameters of interest (same semantics as your original function):
      - gpc_thresholds, cpg_thresholds, any_c_thresholds, a_thresholds:
          each should be [min, max] (floats 0..1) or None.
      - use_other_c_as_background: require GpC/CpG > other_C background (if present).
      - min_valid_fraction_positions_in_read_vs_ref: minimum fraction of valid sites
          in the read vs reference (0..1). If None, this check is skipped.
      - compute_obs_if_missing: if True, compute Fraction_* and Valid_* obs columns
          if they are not already present, using a low-memory per-ref strategy.
      - treat_zero_as_invalid: if True, a zero in X counts as invalid (non-site).
          If False, zeros are considered valid positions (adjust to your data semantics).
    """

    # quick exit flags:
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        return adata

    # helper: check whether obs columns exist for a particular mod type
    def obs_has_columns_for(mod_type):
        col_pref = {
            "GpC": ("Fraction_GpC_site_modified", f"Valid_GpC_site_in_read_vs_reference"),
            "CpG": ("Fraction_CpG_site_modified", f"Valid_CpG_site_in_read_vs_reference"),
            "C": ("Fraction_any_C_site_modified", f"Valid_any_C_site_in_read_vs_reference"),
            "A": ("Fraction_A_site_modified", f"Valid_A_site_in_read_vs_reference"),
        }.get(mod_type, (None, None))
        return (col_pref[0] in adata.obs.columns) and (col_pref[1] in adata.obs.columns)

    # if all required obs columns are present, use them directly (fast path)
    required_present = True
    for mt, thr in (("GpC", gpc_thresholds), ("CpG", cpg_thresholds), ("C", any_c_thresholds), ("A", a_thresholds)):
        if thr is not None and mt in mod_target_bases:
            if not obs_has_columns_for(mt):
                required_present = False
                break

    # If required obs columns are not present and compute_obs_if_missing is False => error
    if not required_present and not compute_obs_if_missing:
        raise RuntimeError(
            "Required per-read summary columns not found in adata.obs and compute_obs_if_missing is False."
        )

    # Build mapping from reference -> var column names (expected pattern)
    # e.g. var column names: "{ref}_GpC_site", "{ref}_CpG_site", "{ref}_any_C_site", "{ref}_other_C_site", "{ref}_A_site"
    # If your var column naming differs, adjust these suffixes.
    refs = list(adata.obs[reference_column].astype('category').cat.categories)

    def _find_var_col_for(ref, suffix):
        name = f"{ref}_{suffix}"
        if name in adata.var.columns:
            return name
        return None

    # If we need to compute obs summaries: do so per-reference in batches
    if not required_present and compute_obs_if_missing:
        n_obs = adata.n_obs
        # prepare empty columns in obs if they don't exist; fill later
        # We'll create only columns that are relevant to mod_target_bases
        create_cols = {}
        if "GpC" in mod_target_bases:
            create_cols["Fraction_GpC_site_modified"] = np.full((n_obs,), np.nan)
            create_cols["Valid_GpC_site_in_read_vs_reference"] = np.full((n_obs,), np.nan)
            # optional background ratio if other_C exists
            create_cols["GpC_to_other_C_mod_ratio"] = np.full((n_obs,), np.nan)
        if "CpG" in mod_target_bases:
            create_cols["Fraction_CpG_site_modified"] = np.full((n_obs,), np.nan)
            create_cols["Valid_CpG_site_in_read_vs_reference"] = np.full((n_obs,), np.nan)
            create_cols["CpG_to_other_C_mod_ratio"] = np.full((n_obs,), np.nan)
        if "C" in mod_target_bases:
            create_cols["Fraction_any_C_site_modified"] = np.full((n_obs,), np.nan)
            create_cols["Valid_any_C_site_in_read_vs_reference"] = np.full((n_obs,), np.nan)
        if "A" in mod_target_bases:
            create_cols["Fraction_A_site_modified"] = np.full((n_obs,), np.nan)
            create_cols["Valid_A_site_in_read_vs_reference"] = np.full((n_obs,), np.nan)

        # helper to compute for one reference and one suffix
        def _compute_for_ref_and_suffix(ref, suffix, out_frac_arr, out_valid_arr):
            """
            Compute fraction modified and valid fraction for reads mapping to 'ref'
            using var column named f"{ref}_{suffix}" to select var columns.
            """
            var_colname = _find_var_col_for(ref, suffix)
            if var_colname is None:
                # nothing to compute
                return

            # var boolean mask (which var columns belong to this suffix for the ref)
            try:
                var_mask_bool = np.asarray(adata.var[var_colname].values).astype(bool)
            except Exception:
                # if var has values not boolean, attempt coercion
                var_mask_bool = np.asarray(pd.to_numeric(adata.var[var_colname], errors='coerce').fillna(0).astype(bool))

            if not var_mask_bool.any():
                return
            col_indices = np.where(var_mask_bool)[0]
            n_cols_for_ref = len(col_indices)
            if n_cols_for_ref == 0:
                return

            # rows that belong to this reference
            row_indices_all = np.where(adata.obs[reference_column].values == ref)[0]
            if len(row_indices_all) == 0:
                return

            # process rows for this reference in batches to avoid allocating huge slices
            for start in range(0, len(row_indices_all), batch_size):
                block_rows_idx = row_indices_all[start : start + batch_size]
                # slice rows x selected columns
                X_block = adata.X[block_rows_idx, :][:, col_indices]

                # If sparse, sum(axis=1) returns a (nrows,1) sparse/dense object -> coerce to 1d array
                # If dense, this will be a dense array but limited to batch_size * n_cols_for_ref
                # Count modified (assume numeric values where >0 indicate modification)
                try:
                    # use vectorized sums; works for sparse/dense
                    # "modified_count" - count of entries > 0 (or > 0.5 if binary probabilities)
                    if hasattr(X_block, "toarray") and not isinstance(X_block, np.ndarray):
                        # sparse or matrix-like: convert sums carefully
                        # We compute:
                        #   modified_count = (X_block > 0).sum(axis=1)
                        #   valid_count = (non-nan if float data else non-zero) per row
                        # For sparse, .data are only stored nonzeros, so (X_block > 0).sum is fine
                        modified_count = np.asarray((X_block > 0).sum(axis=1)).ravel()
                        if np.isnan(X_block.data).any() if hasattr(X_block, 'data') else False:
                            # if sparse with stored NaNs (!) handle differently - unlikely
                            valid_count = np.asarray(~np.isnan(X_block.toarray()).sum(axis=1)).ravel()
                        else:
                            if treat_zero_as_invalid:
                                # valid = number of non-zero entries
                                valid_count = np.asarray((X_block != 0).sum(axis=1)).ravel()
                            else:
                                # treat all positions as valid positions (they exist in reference) -> denominator = n_cols_for_ref
                                valid_count = np.full_like(modified_count, n_cols_for_ref, dtype=float)
                    else:
                        # dense numpy
                        Xb = np.asarray(X_block)
                        if np.isnan(Xb).any():
                            valid_count = np.sum(~np.isnan(Xb), axis=1).astype(float)
                        else:
                            if treat_zero_as_invalid:
                                valid_count = np.sum(Xb != 0, axis=1).astype(float)
                            else:
                                valid_count = np.full((Xb.shape[0],), float(n_cols_for_ref))
                        modified_count = np.sum(Xb > 0, axis=1).astype(float)
                except Exception:
                    # fallback to safe dense conversion per-row (shouldn't be needed usually)
                    Xb = np.asarray(X_block.toarray() if hasattr(X_block, "toarray") else X_block)
                    if Xb.size == 0:
                        modified_count = np.zeros(len(block_rows_idx), dtype=float)
                        valid_count = np.zeros(len(block_rows_idx), dtype=float)
                    else:
                        if np.isnan(Xb).any():
                            valid_count = np.sum(~np.isnan(Xb), axis=1).astype(float)
                        else:
                            if treat_zero_as_invalid:
                                valid_count = np.sum(Xb != 0, axis=1).astype(float)
                            else:
                                valid_count = np.full((Xb.shape[0],), float(n_cols_for_ref))
                        modified_count = np.sum(Xb > 0, axis=1).astype(float)

                # fraction modified = modified_count / valid_count (guard divide-by-zero)
                frac = np.zeros_like(modified_count, dtype=float)
                mask_valid_nonzero = (valid_count > 0)
                frac[mask_valid_nonzero] = modified_count[mask_valid_nonzero] / valid_count[mask_valid_nonzero]

                # write to out arrays
                out_frac_arr[block_rows_idx] = frac
                # valid fraction relative to reference = valid_count / n_cols_for_ref
                out_valid_arr[block_rows_idx] = np.zeros_like(valid_count, dtype=float)
                out_valid_arr[block_rows_idx][mask_valid_nonzero] = (valid_count[mask_valid_nonzero] / float(n_cols_for_ref))

                # free block memory ASAP
                del X_block, modified_count, valid_count, frac
                gc.collect()

        # compute for each reference and required suffixes
        # GpC
        if "GpC" in mod_target_bases:
            for ref in refs:
                _compute_for_ref_and_suffix(ref, "GpC_site", create_cols["Fraction_GpC_site_modified"], create_cols["Valid_GpC_site_in_read_vs_reference"])
        # other_C (for background)
        # We'll also compute 'other_C' per reference if it exists
        other_c_per_ref = {}
        for ref in refs:
            other_col = _find_var_col_for(ref, "other_C_site")
            if other_col:
                other_c_per_ref[ref] = np.where(np.asarray(adata.var[other_col].values).astype(bool))[0]

        # CpG
        if "CpG" in mod_target_bases:
            for ref in refs:
                _compute_for_ref_and_suffix(ref, "CpG_site", create_cols["Fraction_CpG_site_modified"], create_cols["Valid_CpG_site_in_read_vs_reference"])

        # any C
        if "C" in mod_target_bases:
            for ref in refs:
                _compute_for_ref_and_suffix(ref, "any_C_site", create_cols["Fraction_any_C_site_modified"], create_cols["Valid_any_C_site_in_read_vs_reference"])

        # A
        if "A" in mod_target_bases:
            for ref in refs:
                _compute_for_ref_and_suffix(ref, "A_site", create_cols["Fraction_A_site_modified"], create_cols["Valid_A_site_in_read_vs_reference"])

        # write created arrays into adata.obs
        for cname, arr in create_cols.items():
            adata.obs[cname] = arr

        # optionally compute GpC_to_other_C_mod_ratio and CpG_to_other_C_mod_ratio (if other_C masks exist)
        if "GpC" in mod_target_bases and use_other_c_as_background:
            # compute per-ref background ratio if both exist
            # Simplest approach: if 'Fraction_GpC_site_modified' and 'Fraction_other_C_site_modified' exist, compute ratio
            if "Fraction_other_C_site_modified" in adata.obs.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = adata.obs["Fraction_GpC_site_modified"].astype(float) / adata.obs["Fraction_other_C_site_modified"].astype(float)
                adata.obs["GpC_to_other_C_mod_ratio"] = ratio.fillna(0.0)
            else:
                adata.obs["GpC_to_other_C_mod_ratio"] = np.nan

        if "CpG" in mod_target_bases and use_other_c_as_background:
            if "Fraction_other_C_site_modified" in adata.obs.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = adata.obs["Fraction_CpG_site_modified"].astype(float) / adata.obs["Fraction_other_C_site_modified"].astype(float)
                adata.obs["CpG_to_other_C_mod_ratio"] = ratio.fillna(0.0)
            else:
                adata.obs["CpG_to_other_C_mod_ratio"] = np.nan

        # free memory
        del create_cols
        gc.collect()

    # --- Now apply the filters using adata.obs columns (this part is identical to your previous code but memory-friendly) ---
    filtered = adata  # we'll chain subset operations

    # helper to get min/max from param like [min, max] or tuple(None,..)
    def _unpack_minmax(thr):
        if thr is None:
            return None, None
        try:
            lo, hi = float(thr[0]) if thr[0] is not None else None, float(thr[1]) if thr[1] is not None else None
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            return lo, hi
        except Exception:
            return None, None

    # GpC thresholds
    if gpc_thresholds and 'GpC' in mod_target_bases:
        lo, hi = _unpack_minmax(gpc_thresholds)
        if use_other_c_as_background and smf_modality != 'deaminase' and "GpC_to_other_C_mod_ratio" in filtered.obs.columns:
            filtered = filtered[filtered.obs["GpC_to_other_C_mod_ratio"].astype(float) > 1]
        if lo is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_GpC_site_modified"].astype(float) > lo]
            print(f"Removed {s0 - filtered.n_obs} reads below min GpC fraction {lo}")
        if hi is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_GpC_site_modified"].astype(float) < hi]
            print(f"Removed {s0 - filtered.n_obs} reads above max GpC fraction {hi}")
        if (min_valid_fraction_positions_in_read_vs_ref is not None) and ("Valid_GpC_site_in_read_vs_reference" in filtered.obs.columns):
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Valid_GpC_site_in_read_vs_reference"].astype(float) > float(min_valid_fraction_positions_in_read_vs_ref)]
            print(f"Removed {s0 - filtered.n_obs} reads with insufficient valid GpC site fraction vs ref")

    # CpG thresholds
    if cpg_thresholds and 'CpG' in mod_target_bases:
        lo, hi = _unpack_minmax(cpg_thresholds)
        if use_other_c_as_background and smf_modality != 'deaminase' and "CpG_to_other_C_mod_ratio" in filtered.obs.columns:
            filtered = filtered[filtered.obs["CpG_to_other_C_mod_ratio"].astype(float) > 1]
        if lo is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_CpG_site_modified"].astype(float) > lo]
            print(f"Removed {s0 - filtered.n_obs} reads below min CpG fraction {lo}")
        if hi is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_CpG_site_modified"].astype(float) < hi]
            print(f"Removed {s0 - filtered.n_obs} reads above max CpG fraction {hi}")
        if (min_valid_fraction_positions_in_read_vs_ref is not None) and ("Valid_CpG_site_in_read_vs_reference" in filtered.obs.columns):
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Valid_CpG_site_in_read_vs_reference"].astype(float) > float(min_valid_fraction_positions_in_read_vs_ref)]
            print(f"Removed {s0 - filtered.n_obs} reads with insufficient valid CpG site fraction vs ref")

    # any C thresholds
    if any_c_thresholds and 'C' in mod_target_bases:
        lo, hi = _unpack_minmax(any_c_thresholds)
        if lo is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_any_C_site_modified"].astype(float) > lo]
            print(f"Removed {s0 - filtered.n_obs} reads below min any-C fraction {lo}")
        if hi is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_any_C_site_modified"].astype(float) < hi]
            print(f"Removed {s0 - filtered.n_obs} reads above max any-C fraction {hi}")
        if (min_valid_fraction_positions_in_read_vs_ref is not None) and ("Valid_any_C_site_in_read_vs_reference" in filtered.obs.columns):
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Valid_any_C_site_in_read_vs_reference"].astype(float) > float(min_valid_fraction_positions_in_read_vs_ref)]
            print(f"Removed {s0 - filtered.n_obs} reads with insufficient valid any-C site fraction vs ref")

    # A thresholds
    if a_thresholds and 'A' in mod_target_bases:
        lo, hi = _unpack_minmax(a_thresholds)
        if lo is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_A_site_modified"].astype(float) > lo]
            print(f"Removed {s0 - filtered.n_obs} reads below min A fraction {lo}")
        if hi is not None:
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Fraction_A_site_modified"].astype(float) < hi]
            print(f"Removed {s0 - filtered.n_obs} reads above max A fraction {hi}")
        if (min_valid_fraction_positions_in_read_vs_ref is not None) and ("Valid_A_site_in_read_vs_reference" in filtered.obs.columns):
            s0 = filtered.n_obs
            filtered = filtered[filtered.obs["Valid_A_site_in_read_vs_reference"].astype(float) > float(min_valid_fraction_positions_in_read_vs_ref)]
            print(f"Removed {s0 - filtered.n_obs} reads with insufficient valid A site fraction vs ref")

    filtered = filtered.copy()

    # mark as done
    filtered.uns[uns_flag] = True

    return filtered