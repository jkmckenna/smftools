from __future__ import annotations

from typing import Optional

import anndata as ad
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def filter_reads_on_cigar_indels(
    adata: ad.AnnData,
    max_insertion_length: Optional[float] = 10,
    max_deletion_length: Optional[float] = 10,
    insertion_col: str = "max_insertion_length",
    deletion_col: str = "max_deletion_length",
    uns_flag: str = "filter_reads_on_cigar_indels_performed",
    bypass: bool = False,
    force_redo: bool = True,
) -> ad.AnnData:
    """Filter reads whose longest internal indel exceeds a threshold.

    The per-read longest internal insertion/deletion run lengths are derived from
    the alignment CIGAR at raw extraction time and stored in ``adata.obs``. Reads
    whose ``max_insertion_length`` or ``max_deletion_length`` exceeds the supplied
    threshold are removed. A threshold of ``None`` disables that check.

    Args:
        adata: AnnData object to filter.
        max_insertion_length: Maximum allowed internal insertion run (bp). ``None``
            disables the insertion check.
        max_deletion_length: Maximum allowed internal deletion run (bp). ``None``
            disables the deletion check.
        insertion_col: ``obs`` column holding the per-read longest insertion run.
        deletion_col: ``obs`` column holding the per-read longest deletion run.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: Filtered copy of the input AnnData (or the input itself
        when nothing is filtered).
    """
    already = bool(adata.uns.get(uns_flag, False))
    if bypass or (already and not force_redo):
        return adata

    if max_insertion_length is None and max_deletion_length is None:
        logger.info(
            "CIGAR indel filter: both thresholds are None — skipping (no reads removed)."
        )
        adata.uns[uns_flag] = True
        return adata

    start_n = adata.n_obs
    combined_mask = pd.Series(True, index=adata.obs.index)

    checks = (
        ("insertion", max_insertion_length, insertion_col),
        ("deletion", max_deletion_length, deletion_col),
    )
    for label, threshold, column in checks:
        if threshold is None:
            continue
        if column not in adata.obs.columns:
            logger.warning(
                "'%s' not found in adata.obs — skipping max_%s_length filter. "
                "Re-run raw extraction to populate CIGAR indel metrics.",
                column,
                label,
            )
            continue
        vals = pd.to_numeric(adata.obs[column], errors="coerce")
        mask = (vals <= float(threshold)) & vals.notna()
        combined_mask &= mask
        logger.info("Planned max_%s_length filter: max=%s bp", label, threshold)

    combined_mask_bool = combined_mask.astype(bool).values
    adata_work = adata[combined_mask_bool].copy()
    final_n = adata_work.n_obs
    logger.info(
        "CIGAR indel filter applied: kept %s / %s reads (removed %s).",
        final_n,
        start_n,
        start_n - final_n,
    )

    adata_work.uns[uns_flag] = True
    return adata_work
