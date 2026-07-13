from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

CHIMERA_OBS_COLUMN = "deaminase_PCR_chimera"
_REQUIRED_COLUMNS = ("ct_event_count", "ga_event_count", "strand_segment_purity")


def label_deaminase_pcr_chimeras(
    adata: ad.AnnData,
    min_events_per_span: int = 3,
    min_segment_purity: float = 0.9,
    max_single_strand_fraction: float = 0.8,
    obs_column: str = CHIMERA_OBS_COLUMN,
    uns_flag: str = "label_deaminase_pcr_chimeras_performed",
    bypass: bool = False,
    force_redo: bool = True,
) -> ad.AnnData:
    """Flag deaminase PCR-chimeric reads from per-read strand-switch metrics.

    A deaminase PCR chimera is a molecule stitched from two templates of opposite
    deamination state, so along the read it shows a span of C->T (top-consistent)
    events and a span of G->A (bottom-consistent) events. The per-read metrics
    ``ct_event_count``, ``ga_event_count`` and ``strand_segment_purity`` are computed
    at raw extraction (see ``smftools.informatics.ragged_store.strand_switch_metrics``)
    and carried on ``adata.obs``. This step thresholds them into a boolean
    ``obs[obs_column]`` label; reads are **not** removed.

    A read is labeled a chimera when both strand signatures are present in sufficient
    number, the best two-segment (single-switch) model is highly pure, and the read is
    not overwhelmingly one-sided:

    - ``ct_event_count >= min_events_per_span`` and ``ga_event_count >= min_events_per_span``
    - ``strand_segment_purity >= min_segment_purity``
    - ``single_strand_fraction <= max_single_strand_fraction`` where
      ``single_strand_fraction = max(ct, ga) / (ct + ga)``

    Args:
        adata: AnnData object (typically deaminase modality).
        min_events_per_span: Minimum C->T and G->A events required on each side.
        min_segment_purity: Minimum best two-segment purity to accept a clean switch.
        max_single_strand_fraction: Maximum one-sidedness for a read to qualify.
        obs_column: Name of the boolean obs column to write.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: The input AnnData with ``obs[obs_column]`` set (label only).
    """
    already = bool(adata.uns.get(uns_flag, False))
    if bypass or (already and not force_redo):
        return adata

    missing = [column for column in _REQUIRED_COLUMNS if column not in adata.obs.columns]
    if missing:
        logger.warning(
            "Deaminase chimera labeling skipped: obs is missing %s. "
            "Re-run raw extraction to populate strand-switch metrics.",
            missing,
        )
        adata.obs[obs_column] = False
        adata.uns[uns_flag] = True
        return adata

    ct = pd.to_numeric(adata.obs["ct_event_count"], errors="coerce").to_numpy(dtype=float)
    ga = pd.to_numeric(adata.obs["ga_event_count"], errors="coerce").to_numpy(dtype=float)
    purity = pd.to_numeric(adata.obs["strand_segment_purity"], errors="coerce").to_numpy(dtype=float)

    total = ct + ga
    with np.errstate(invalid="ignore", divide="ignore"):
        single_strand_fraction = np.where(total > 0, np.maximum(ct, ga) / total, 1.0)

    # Comparisons against NaN metrics evaluate False, so unusable rows are not flagged.
    chimera = (
        (ct >= float(min_events_per_span))
        & (ga >= float(min_events_per_span))
        & (purity >= float(min_segment_purity))
        & (single_strand_fraction <= float(max_single_strand_fraction))
    )

    adata.obs[obs_column] = chimera
    adata.uns[uns_flag] = True
    logger.info(
        "Deaminase chimera labeling: flagged %s / %s reads as %s.",
        int(chimera.sum()),
        adata.n_obs,
        obs_column,
    )
    return adata
