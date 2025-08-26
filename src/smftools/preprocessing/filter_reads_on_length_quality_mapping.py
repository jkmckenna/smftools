from typing import Optional, Union, Sequence
import numpy as np
import pandas as pd
import anndata as ad

def filter_reads_on_length_quality_mapping(
    adata: ad.AnnData,
    filter_on_coordinates: Union[bool, Sequence] = False,
    # New single-range params (preferred):
    read_length: Optional[Sequence[float]] = None,          # e.g. [min, max]
    length_ratio: Optional[Sequence[float]] = None,         # e.g. [min, max]
    read_quality: Optional[Sequence[float]] = None,         # e.g. [min, max]  (commonly min only)
    mapping_quality: Optional[Sequence[float]] = None,      # e.g. [min, max]  (commonly min only)
    uns_flag: str = "reads_removed_failing_length_quality_mapping_qc",
    bypass: bool = False,
    force_redo: bool = True
) -> ad.AnnData:
    """
    Filter AnnData by coordinate window, read length, length ratios, read quality and mapping quality.

    New: you may pass `read_length=[min, max]` (or tuple) to set both min/max in one argument.
    If `read_length` is given it overrides scalar min/max variants (which are not present in this signature).
    Same behavior supported for `length_ratio`, `read_quality`, `mapping_quality`.

    Returns a filtered copy of the input AnnData and marks adata.uns[uns_flag] = True.
    """
    # early exit
    already = bool(adata.uns.get(uns_flag, False))
    if bypass or (already and not force_redo):
        return adata

    adata_work = adata
    start_n = adata_work.n_obs

    # --- coordinate filtering (unchanged) ---
    if filter_on_coordinates:
        try:
            low, high = tuple(filter_on_coordinates)
        except Exception:
            raise ValueError("filter_on_coordinates must be False or an iterable of two numbers (low, high).")
        try:
            var_coords = np.array([float(v) for v in adata_work.var_names])
            if low > high:
                low, high = high, low
            col_mask_bool = (var_coords >= float(low)) & (var_coords <= float(high))
            if not col_mask_bool.any():
                start_idx = int(np.argmin(np.abs(var_coords - float(low))))
                end_idx = int(np.argmin(np.abs(var_coords - float(high))))
                lo_idx, hi_idx = min(start_idx, end_idx), max(start_idx, end_idx)
                selected_cols = list(adata_work.var_names[lo_idx : hi_idx + 1])
            else:
                selected_cols = list(adata_work.var_names[col_mask_bool])
            print(f"Subsetting adata to coordinates between {low} and {high}: keeping {len(selected_cols)} variables.")
            adata_work = adata_work[:, selected_cols].copy()
        except Exception:
            print("Warning: could not interpret adata.var_names as numeric coordinates — skipping coordinate filtering.")

    # --- helper to coerce range inputs ---
    def _coerce_range(range_arg):
        """
        Given range_arg which may be None or a 2-seq [min,max], return (min_or_None, max_or_None).
        If both present and min>max they are swapped.
        """
        if range_arg is None:
            return None, None
        if not isinstance(range_arg, (list, tuple, np.ndarray)) or len(range_arg) != 2:
            # not a 2-element range -> treat as no restriction (or you could raise)
            return None, None
        lo_raw, hi_raw = range_arg[0], range_arg[1]
        lo = None if lo_raw is None else float(lo_raw)
        hi = None if hi_raw is None else float(hi_raw)
        if (lo is not None) and (hi is not None) and lo > hi:
            lo, hi = hi, lo
        return lo, hi

    # Resolve ranges using only the provided range arguments
    rl_min, rl_max = _coerce_range(read_length)
    lr_min, lr_max = _coerce_range(length_ratio)
    rq_min, rq_max = _coerce_range(read_quality)
    mq_min, mq_max = _coerce_range(mapping_quality)

    # --- build combined mask ---
    combined_mask = pd.Series(True, index=adata_work.obs.index)

    # read length filter
    if (rl_min is not None) or (rl_max is not None):
        if "mapped_length" not in adata_work.obs.columns:
            print("Warning: 'mapped_length' not found in adata.obs — skipping read_length filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["mapped_length"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if rl_min is not None:
                mask &= (vals >= rl_min)
            if rl_max is not None:
                mask &= (vals <= rl_max)
            mask &= vals.notna()
            combined_mask &= mask
            print(f"Planned read_length filter: min={rl_min}, max={rl_max}")

    # length ratio filter
    if (lr_min is not None) or (lr_max is not None):
        if "mapped_length_to_reference_length_ratio" not in adata_work.obs.columns:
            print("Warning: 'mapped_length_to_reference_length_ratio' not found in adata.obs — skipping length_ratio filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["mapped_length_to_reference_length_ratio"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if lr_min is not None:
                mask &= (vals >= lr_min)
            if lr_max is not None:
                mask &= (vals <= lr_max)
            mask &= vals.notna()
            combined_mask &= mask
            print(f"Planned length_ratio filter: min={lr_min}, max={lr_max}")

    # read quality filter (supporting optional range but typically min only)
    if (rq_min is not None) or (rq_max is not None):
        if "read_quality" not in adata_work.obs.columns:
            print("Warning: 'read_quality' not found in adata.obs — skipping read_quality filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["read_quality"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if rq_min is not None:
                mask &= (vals >= rq_min)
            if rq_max is not None:
                mask &= (vals <= rq_max)
            mask &= vals.notna()
            combined_mask &= mask
            print(f"Planned read_quality filter: min={rq_min}, max={rq_max}")

    # mapping quality filter (supporting optional range but typically min only)
    if (mq_min is not None) or (mq_max is not None):
        if "mapping_quality" not in adata_work.obs.columns:
            print("Warning: 'mapping_quality' not found in adata.obs — skipping mapping_quality filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["mapping_quality"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if mq_min is not None:
                mask &= (vals >= mq_min)
            if mq_max is not None:
                mask &= (vals <= mq_max)
            mask &= vals.notna()
            combined_mask &= mask
            print(f"Planned mapping_quality filter: min={mq_min}, max={mq_max}")

    # Apply combined mask and report
    s0 = adata_work.n_obs
    combined_mask_bool = combined_mask.astype(bool).values
    adata_work = adata_work[combined_mask_bool].copy()
    s1 = adata_work.n_obs
    print(f"Combined filters applied: kept {s1} / {s0} reads (removed {s0 - s1})")

    final_n = adata_work.n_obs
    print(f"Filtering complete: start={start_n}, final={final_n}, removed={start_n - final_n}")

    # mark as done
    adata_work.uns[uns_flag] = True

    return adata_work
