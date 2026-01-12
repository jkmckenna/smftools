from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def filter_reads_on_length_quality_mapping(
    adata: ad.AnnData,
    filter_on_coordinates: Union[bool, Sequence] = False,
    # New single-range params (preferred):
    read_length: Optional[Sequence[float]] = None,  # e.g. [min, max]
    length_ratio: Optional[Sequence[float]] = None,  # e.g. [min, max]
    read_quality: Optional[Sequence[float]] = None,  # e.g. [min, max]  (commonly min only)
    mapping_quality: Optional[Sequence[float]] = None,  # e.g. [min, max]  (commonly min only)
    uns_flag: str = "filter_reads_on_length_quality_mapping_performed",
    bypass: bool = False,
    force_redo: bool = True,
) -> ad.AnnData:
    """Filter AnnData by coordinates, read length, quality, and mapping metrics.

    Args:
        adata: AnnData object to filter.
        filter_on_coordinates: Optional coordinate window as a two-value sequence.
        read_length: Read length range as ``[min, max]``.
        length_ratio: Length ratio range as ``[min, max]``.
        read_quality: Read quality range as ``[min, max]``.
        mapping_quality: Mapping quality range as ``[min, max]``.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: Filtered copy of the input AnnData.
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
            raise ValueError(
                "filter_on_coordinates must be False or an iterable of two numbers (low, high)."
            )
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
            logger.info(
                "Subsetting adata to coordinates between %s and %s: keeping %s variables.",
                low,
                high,
                len(selected_cols),
            )
            adata_work = adata_work[:, selected_cols].copy()
        except Exception:
            logger.warning(
                "Could not interpret adata.var_names as numeric coordinates — skipping coordinate filtering."
            )

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
            logger.warning("'mapped_length' not found in adata.obs — skipping read_length filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["mapped_length"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if rl_min is not None:
                mask &= vals >= rl_min
            if rl_max is not None:
                mask &= vals <= rl_max
            mask &= vals.notna()
            combined_mask &= mask
            logger.info("Planned read_length filter: min=%s, max=%s", rl_min, rl_max)

    # length ratio filter
    if (lr_min is not None) or (lr_max is not None):
        if "mapped_length_to_reference_length_ratio" not in adata_work.obs.columns:
            logger.warning(
                "'mapped_length_to_reference_length_ratio' not found in adata.obs — skipping length_ratio filter."
            )
        else:
            vals = pd.to_numeric(
                adata_work.obs["mapped_length_to_reference_length_ratio"], errors="coerce"
            )
            mask = pd.Series(True, index=adata_work.obs.index)
            if lr_min is not None:
                mask &= vals >= lr_min
            if lr_max is not None:
                mask &= vals <= lr_max
            mask &= vals.notna()
            combined_mask &= mask
            logger.info("Planned length_ratio filter: min=%s, max=%s", lr_min, lr_max)

    # read quality filter (supporting optional range but typically min only)
    if (rq_min is not None) or (rq_max is not None):
        if "read_quality" not in adata_work.obs.columns:
            logger.warning("'read_quality' not found in adata.obs — skipping read_quality filter.")
        else:
            vals = pd.to_numeric(adata_work.obs["read_quality"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if rq_min is not None:
                mask &= vals >= rq_min
            if rq_max is not None:
                mask &= vals <= rq_max
            mask &= vals.notna()
            combined_mask &= mask
            logger.info("Planned read_quality filter: min=%s, max=%s", rq_min, rq_max)

    # mapping quality filter (supporting optional range but typically min only)
    if (mq_min is not None) or (mq_max is not None):
        if "mapping_quality" not in adata_work.obs.columns:
            logger.warning(
                "'mapping_quality' not found in adata.obs — skipping mapping_quality filter."
            )
        else:
            vals = pd.to_numeric(adata_work.obs["mapping_quality"], errors="coerce")
            mask = pd.Series(True, index=adata_work.obs.index)
            if mq_min is not None:
                mask &= vals >= mq_min
            if mq_max is not None:
                mask &= vals <= mq_max
            mask &= vals.notna()
            combined_mask &= mask
            logger.info("Planned mapping_quality filter: min=%s, max=%s", mq_min, mq_max)

    # Apply combined mask and report
    s0 = adata_work.n_obs
    combined_mask_bool = combined_mask.astype(bool).values
    adata_work = adata_work[combined_mask_bool].copy()
    s1 = adata_work.n_obs
    logger.info("Combined filters applied: kept %s / %s reads (removed %s)", s1, s0, s0 - s1)

    final_n = adata_work.n_obs
    logger.info(
        "Filtering complete: start=%s, final=%s, removed=%s",
        start_n,
        final_n,
        start_n - final_n,
    )

    # mark as done
    adata_work.uns[uns_flag] = True

    return adata_work
