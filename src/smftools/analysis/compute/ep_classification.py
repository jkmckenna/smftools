"""
hmm_ep_classification.py — Enhancer/promoter NDR state classification per read.

Classifies each read's chromatin state at two genomic anchor positions (typically
TSS = 0 and enhancer peak) by checking HMM feature layers in priority order.

Priority order (highest to lowest):
    GpC_nucleosome_depleted_region_merged_lengths  → NDR
    GpC_large_accessible_patch_merged_lengths      → Large
    GpC_mid_accessible_patch_merged_lengths        → Mid
    GpC_small_accessible_patch_merged_lengths      → Small
    (none of the above)                            → None

Functions
---------
classify_position      Classify all reads at one genomic anchor for one reference.
add_ep_obs_columns     Add Promoter/Enhancer class and open/closed bool columns to adata.obs.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

DEFAULT_PATCH_LAYERS = [
    ("GpC_nucleosome_depleted_region_merged_lengths", "NDR"),
    ("GpC_large_accessible_patch_merged_lengths", "Large"),
    ("GpC_mid_accessible_patch_merged_lengths", "Mid"),
    ("GpC_small_accessible_patch_merged_lengths", "Small"),
]
CATEGORIES = ["NDR", "Large", "Mid", "Small", "None"]


def classify_position(
    adata: ad.AnnData,
    ref_strand: str,
    target_bp: float,
    patch_layers: list[tuple[str, str]] = DEFAULT_PATCH_LAYERS,
    ref_obs_col: str = "Reference_strand",
) -> np.ndarray:
    """
    Classify each read's patch state at the position nearest target_bp.

    Parameters
    ----------
    adata       : AnnData with layers named in patch_layers and var column
                  f"{ref_strand}_reindexed".
    ref_strand  : e.g. "6B6_top" — used to find the reindexed coord column.
    target_bp   : TSS-centred bp coordinate of the anchor (e.g. 0 for TSS, -1690 for enhancer).
    patch_layers: priority-ordered list of (layer_name, class_label) tuples.
    ref_obs_col : obs column that identifies which reference a read mapped to.

    Returns
    -------
    classes : object array of length n_obs; values in CATEGORIES.
              Reads not assigned to this ref_strand remain "None".
    """
    reindex = pd.to_numeric(adata.var[f"{ref_strand}_reindexed"], errors="coerce").to_numpy()
    pos_idx = int(np.argmin(np.abs(reindex - target_bp)))

    ref_mask = adata.obs[ref_obs_col].values == ref_strand
    classes = np.full(adata.n_obs, "None", dtype=object)
    unassigned = ref_mask.copy()

    for layer_name, label in patch_layers:
        if layer_name not in adata.layers:
            continue
        col = np.asarray(adata.layers[layer_name][:, pos_idx], dtype=float)
        assign = unassigned & (col > 0) & np.isfinite(col)
        classes[assign] = label
        unassigned[assign] = False

    return classes


def add_ep_obs_columns(
    adata: ad.AnnData,
    references: list[str],
    tss_position: float = 0.0,
    enhancer_position: float = -1690.0,
    patch_layers: list[tuple[str, str]] = DEFAULT_PATCH_LAYERS,
    ref_obs_col: str = "Reference_strand",
) -> ad.AnnData:
    """
    Add Promoter_HMM_class, Enhancer_HMM_class, Promoter_open, Enhancer_open to adata.obs.

    Parameters
    ----------
    adata             : AnnData to annotate (modified in-place).
    references        : list of ref_strand values to classify (e.g. ["6B6_top", "6BALB_cJ_top"]).
    tss_position      : TSS anchor in TSS-centred bp coordinates (default 0).
    enhancer_position : enhancer anchor in TSS-centred bp (default -1690).

    Returns
    -------
    adata with four new obs columns added.
    """
    n = adata.n_obs
    promoter_class = np.full(n, "None", dtype=object)
    enhancer_class = np.full(n, "None", dtype=object)

    for ref in references:
        ref_mask = adata.obs[ref_obs_col].values == ref

        prom = classify_position(adata, ref, tss_position, patch_layers, ref_obs_col)
        promoter_class[ref_mask] = prom[ref_mask]

        enh = classify_position(adata, ref, enhancer_position, patch_layers, ref_obs_col)
        enhancer_class[ref_mask] = enh[ref_mask]

    adata.obs["Promoter_HMM_class"] = pd.Categorical(promoter_class, categories=CATEGORIES)
    adata.obs["Enhancer_HMM_class"] = pd.Categorical(enhancer_class, categories=CATEGORIES)
    adata.obs["Promoter_open"] = promoter_class == "NDR"
    adata.obs["Enhancer_open"] = enhancer_class == "NDR"

    return adata
