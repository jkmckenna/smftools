from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
gridspec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
patches = require("matplotlib.patches", extra="plotting", purpose="plot rendering")
plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
sns = require("seaborn", extra="plotting", purpose="plot styling")

logger = get_logger(__name__)

DNA_5COLOR_PALETTE = {
    "A": "#00A000",  # green
    "C": "#0000FF",  # blue
    "G": "#FF7F00",  # orange
    "T": "#FF0000",  # red
    "OTHER": "#808080",  # gray (N, PAD, unknown)
}


def _fixed_tick_positions(n_positions: int, n_ticks: int) -> np.ndarray:
    """
    Return indices for ~n_ticks evenly spaced labels across [0, n_positions-1].
    Always includes 0 and n_positions-1 when possible.
    """
    n_ticks = int(max(2, n_ticks))
    if n_positions <= n_ticks:
        return np.arange(n_positions)

    # linspace gives fixed count
    pos = np.linspace(0, n_positions - 1, n_ticks)
    return np.unique(np.round(pos).astype(int))


def _select_labels(subset, sites: np.ndarray, reference: str, index_col_suffix: str | None):
    """
    Select tick labels for the heatmap axis.

    Parameters
    ----------
    subset : AnnData view
        The per-bin subset of the AnnData.
    sites : np.ndarray[int]
        Indices of the subset.var positions to annotate.
    reference : str
        Reference name (e.g., '6B6_top').
    index_col_suffix : None or str
        If None → use subset.var_names
        Else     → use subset.var[f"{reference}_{index_col_suffix}"]

    Returns
    -------
    np.ndarray[str]
        The labels to use for tick positions.
    """
    if sites.size == 0:
        return np.array([])

    # Default behavior: use var_names
    if index_col_suffix is None:
        return subset.var_names[sites].astype(str)

    # Otherwise: use a computed column adata.var[f"{reference}_{suffix}"]
    colname = f"{reference}_{index_col_suffix}"

    if colname not in subset.var:
        raise KeyError(
            f"index_col_suffix='{index_col_suffix}' requires var column '{colname}', "
            f"but it is not present in adata.var."
        )

    labels = subset.var[colname].astype(str).values
    return labels[sites]


def normalized_mean(matrix: np.ndarray) -> np.ndarray:
    """Compute normalized column means for a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        1D array of normalized means.
    """
    mean = np.nanmean(matrix, axis=0)
    denom = (mean.max() - mean.min()) + 1e-9
    return (mean - mean.min()) / denom


def methylation_fraction(matrix: np.ndarray) -> np.ndarray:
    """
    Fraction methylated per column.
    Methylated = 1
    Valid = finite AND not 0
    """
    matrix = np.asarray(matrix)
    valid_mask = np.isfinite(matrix) & (matrix != 0)
    methyl_mask = (matrix == 1) & np.isfinite(matrix)

    methylated = methyl_mask.sum(axis=0)
    valid = valid_mask.sum(axis=0)

    return np.divide(
        methylated, valid, out=np.zeros_like(methylated, dtype=float), where=valid != 0
    )


def clean_barplot(ax, mean_values, title):
    """Format a barplot with consistent axes and labels.

    Args:
        ax: Matplotlib axes.
        mean_values: Values to plot.
        title: Plot title.
    """
    x = np.arange(len(mean_values))
    ax.bar(x, mean_values, color="gray", width=1.0, align="edge")
    ax.set_xlim(0, len(mean_values))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Mean")
    ax.set_title(title, fontsize=12, pad=2)

    # Hide all spines except left
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "left")

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)


def combined_hmm_raw_clustermap(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    hmm_feature_layer: str = "hmm_combined",
    layer_gpc: str = "nan0_0minus1",
    layer_cpg: str = "nan0_0minus1",
    layer_c: str = "nan0_0minus1",
    layer_a: str = "nan0_0minus1",
    cmap_hmm: str = "tab10",
    cmap_gpc: str = "coolwarm",
    cmap_cpg: str = "viridis",
    cmap_c: str = "coolwarm",
    cmap_a: str = "coolwarm",
    min_quality: int = 20,
    min_length: int = 200,
    min_mapped_length_to_reference_length_ratio: float = 0.8,
    min_position_valid_fraction: float = 0.5,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sample_mapping: Optional[Mapping[str, str]] = None,
    save_path: str | Path | None = None,
    normalize_hmm: bool = False,
    sort_by: str = "gpc",
    bins: Optional[Dict[str, Any]] = None,
    deaminase: bool = False,
    min_signal: float = 0.0,
    # ---- fixed tick label controls (counts, not spacing)
    n_xticks_hmm: int = 10,
    n_xticks_any_c: int = 8,
    n_xticks_gpc: int = 8,
    n_xticks_cpg: int = 8,
    n_xticks_a: int = 8,
    index_col_suffix: str | None = None,
):
    """
    Makes a multi-panel clustermap per (sample, reference):
      HMM panel (always) + optional raw panels for C, GpC, CpG, and A sites.

    Panels are added only if the corresponding site mask exists AND has >0 sites.

    sort_by options:
      'gpc', 'cpg', 'c', 'a', 'gpc_cpg', 'none', 'hmm', or 'obs:<col>'
    """

    def pick_xticks(labels: np.ndarray, n_ticks: int):
        """Pick tick indices/labels from an array."""
        if labels.size == 0:
            return [], []
        idx = np.linspace(0, len(labels) - 1, n_ticks).round().astype(int)
        idx = np.unique(idx)
        return idx.tolist(), labels[idx].tolist()

    # Helper: build a True mask if filter is inactive or column missing
    def _mask_or_true(series_name: str, predicate):
        """Return a mask from predicate or an all-True mask."""
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            # Fallback: all True if bad dtype / predicate failure
            return pd.Series(True, index=adata.obs.index)

    results = []
    signal_type = "deamination" if deaminase else "methylation"

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            # Optionally remap sample label for display
            display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample
            # Row-level masks (obs)
            qmask = _mask_or_true(
                "read_quality",
                (lambda s: s >= float(min_quality))
                if (min_quality is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lm_mask = _mask_or_true(
                "mapped_length",
                (lambda s: s >= float(min_length))
                if (min_length is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lrr_mask = _mask_or_true(
                "mapped_length_to_reference_length_ratio",
                (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                if (min_mapped_length_to_reference_length_ratio is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            demux_mask = _mask_or_true(
                "demux_type",
                (lambda s: s.astype("string").isin(list(demux_types)))
                if (demux_types is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            ref_mask = adata.obs[reference_col] == ref
            sample_mask = adata.obs[sample_col] == sample

            row_mask = ref_mask & sample_mask & qmask & lm_mask & lrr_mask & demux_mask

            if not bool(row_mask.any()):
                print(
                    f"No reads for {display_sample} - {ref} after read quality and length filtering"
                )
                continue

            try:
                # ---- subset reads ----
                subset = adata[row_mask, :].copy()

                # Column-level mask (var)
                if min_position_valid_fraction is not None:
                    valid_key = f"{ref}_valid_fraction"
                    if valid_key in subset.var:
                        v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                        col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                        if col_mask.any():
                            subset = subset[:, col_mask].copy()
                        else:
                            print(
                                f"No positions left after valid_fraction filter for {display_sample} - {ref}"
                            )
                            continue

                if subset.shape[0] == 0:
                    print(f"No reads left after filtering for {display_sample} - {ref}")
                    continue

                # ---- bins ----
                if bins is None:
                    bins_temp = {"All": np.ones(subset.n_obs, dtype=bool)}
                else:
                    bins_temp = bins

                # ---- site masks (robust) ----
                def _sites(*keys):
                    """Return indices for the first matching site key."""
                    for k in keys:
                        if k in subset.var:
                            return np.where(subset.var[k].values)[0]
                    return np.array([], dtype=int)

                gpc_sites = _sites(f"{ref}_GpC_site")
                cpg_sites = _sites(f"{ref}_CpG_site")
                any_c_sites = _sites(f"{ref}_any_C_site", f"{ref}_C_site")
                any_a_sites = _sites(f"{ref}_A_site", f"{ref}_any_A_site")

                # ---- labels via _select_labels ----
                # HMM uses *all* columns
                hmm_sites = np.arange(subset.n_vars, dtype=int)
                hmm_labels = _select_labels(subset, hmm_sites, ref, index_col_suffix)
                gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)
                any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                # storage
                stacked_hmm = []
                stacked_any_c = []
                stacked_gpc = []
                stacked_cpg = []
                stacked_any_a = []

                row_labels, bin_labels, bin_boundaries = [], [], []
                total_reads = subset.n_obs
                percentages = {}
                last_idx = 0

                # ---------------- process bins ----------------
                for bin_label, bin_filter in bins_temp.items():
                    sb = subset[bin_filter].copy()
                    n = sb.n_obs
                    if n == 0:
                        continue

                    pct = (n / total_reads) * 100 if total_reads else 0
                    percentages[bin_label] = pct

                    # ---- sorting ----
                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        order = np.argsort(sb.obs[colname].values)

                    elif sort_by == "gpc" and gpc_sites.size:
                        linkage = sch.linkage(sb[:, gpc_sites].layers[layer_gpc], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "cpg" and cpg_sites.size:
                        linkage = sch.linkage(sb[:, cpg_sites].layers[layer_cpg], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "c" and any_c_sites.size:
                        linkage = sch.linkage(sb[:, any_c_sites].layers[layer_c], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "a" and any_a_sites.size:
                        linkage = sch.linkage(sb[:, any_a_sites].layers[layer_a], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "gpc_cpg" and gpc_sites.size and cpg_sites.size:
                        linkage = sch.linkage(sb.layers[layer_gpc], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "hmm" and hmm_sites.size:
                        linkage = sch.linkage(
                            sb[:, hmm_sites].layers[hmm_feature_layer], method="ward"
                        )
                        order = sch.leaves_list(linkage)

                    else:
                        order = np.arange(n)

                    sb = sb[order]

                    # ---- collect matrices ----
                    stacked_hmm.append(sb.layers[hmm_feature_layer])
                    if any_c_sites.size:
                        stacked_any_c.append(sb[:, any_c_sites].layers[layer_c])
                    if gpc_sites.size:
                        stacked_gpc.append(sb[:, gpc_sites].layers[layer_gpc])
                    if cpg_sites.size:
                        stacked_cpg.append(sb[:, cpg_sites].layers[layer_cpg])
                    if any_a_sites.size:
                        stacked_any_a.append(sb[:, any_a_sites].layers[layer_a])

                    row_labels.extend([bin_label] * n)
                    bin_labels.append(f"{bin_label}: {n} reads ({pct:.1f}%)")
                    last_idx += n
                    bin_boundaries.append(last_idx)

                # ---------------- stack ----------------
                hmm_matrix = np.vstack(stacked_hmm)
                mean_hmm = (
                    normalized_mean(hmm_matrix) if normalize_hmm else np.nanmean(hmm_matrix, axis=0)
                )

                panels = [
                    (
                        f"HMM - {hmm_feature_layer}",
                        hmm_matrix,
                        hmm_labels,
                        cmap_hmm,
                        mean_hmm,
                        n_xticks_hmm,
                    ),
                ]

                if stacked_any_c:
                    m = np.vstack(stacked_any_c)
                    panels.append(
                        ("C", m, any_c_labels, cmap_c, methylation_fraction(m), n_xticks_any_c)
                    )

                if stacked_gpc:
                    m = np.vstack(stacked_gpc)
                    panels.append(
                        ("GpC", m, gpc_labels, cmap_gpc, methylation_fraction(m), n_xticks_gpc)
                    )

                if stacked_cpg:
                    m = np.vstack(stacked_cpg)
                    panels.append(
                        ("CpG", m, cpg_labels, cmap_cpg, methylation_fraction(m), n_xticks_cpg)
                    )

                if stacked_any_a:
                    m = np.vstack(stacked_any_a)
                    panels.append(
                        ("A", m, any_a_labels, cmap_a, methylation_fraction(m), n_xticks_a)
                    )

                # ---------------- plotting ----------------
                n_panels = len(panels)
                fig = plt.figure(figsize=(4.5 * n_panels, 10))
                gs = gridspec.GridSpec(2, n_panels, height_ratios=[1, 6], hspace=0.01)
                fig.suptitle(
                    f"{sample} — {ref} — {total_reads} reads ({signal_type})", fontsize=14, y=0.98
                )

                axes_heat = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]
                axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(n_panels)]

                for i, (name, matrix, labels, cmap, mean_vec, n_ticks) in enumerate(panels):
                    # ---- your clean barplot ----
                    clean_barplot(axes_bar[i], mean_vec, name)

                    # ---- heatmap ----
                    sns.heatmap(matrix, cmap=cmap, ax=axes_heat[i], yticklabels=False, cbar=False)

                    # ---- xticks ----
                    xtick_pos, xtick_labels = pick_xticks(np.asarray(labels), n_ticks)
                    axes_heat[i].set_xticks(xtick_pos)
                    axes_heat[i].set_xticklabels(xtick_labels, rotation=90, fontsize=8)

                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=1.2)

                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    safe_name = f"{ref}__{sample}".replace("/", "_")
                    out_file = save_path / f"{safe_name}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close(fig)
                else:
                    plt.show()

            except Exception:
                import traceback

                traceback.print_exc()
                continue


def combined_raw_clustermap(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    mod_target_bases: Sequence[str] = ("GpC", "CpG"),
    layer_c: str = "nan0_0minus1",
    layer_gpc: str = "nan0_0minus1",
    layer_cpg: str = "nan0_0minus1",
    layer_a: str = "nan0_0minus1",
    cmap_c: str = "coolwarm",
    cmap_gpc: str = "coolwarm",
    cmap_cpg: str = "viridis",
    cmap_a: str = "coolwarm",
    min_quality: float | None = 20,
    min_length: int | None = 200,
    min_mapped_length_to_reference_length_ratio: float | None = 0,
    min_position_valid_fraction: float | None = 0,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sample_mapping: Optional[Mapping[str, str]] = None,
    save_path: str | Path | None = None,
    sort_by: str = "gpc",  # 'gpc','cpg','c','gpc_cpg','a','none','obs:<col>'
    bins: Optional[Dict[str, Any]] = None,
    deaminase: bool = False,
    min_signal: float = 0,
    n_xticks_any_c: int = 10,
    n_xticks_gpc: int = 10,
    n_xticks_cpg: int = 10,
    n_xticks_any_a: int = 10,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    index_col_suffix: str | None = None,
):
    """
    Plot stacked heatmaps + per-position mean barplots for C, GpC, CpG, and optional A.

    Key fixes vs old version:
      - order computed ONCE per bin, applied to all matrices
      - no hard-coded axes indices
      - NaNs excluded from methylation denominators
      - var_names not forced to int
      - fixed count of x tick labels per block (controllable)
      - adata.uns updated once at end

    Returns
    -------
    results : list[dict]
        One entry per (sample, ref) plot with matrices + bin metadata.
    """

    # Helper: build a True mask if filter is inactive or column missing
    def _mask_or_true(series_name: str, predicate):
        """Return a mask from predicate or an all-True mask."""
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            # Fallback: all True if bad dtype / predicate failure
            return pd.Series(True, index=adata.obs.index)

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    # Ensure categorical
    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype("category")

    base_set = set(mod_target_bases)
    include_any_c = any(b in {"C", "CpG", "GpC"} for b in base_set)
    include_any_a = "A" in base_set

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            # Optionally remap sample label for display
            display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample

            # Row-level masks (obs)
            qmask = _mask_or_true(
                "read_quality",
                (lambda s: s >= float(min_quality))
                if (min_quality is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lm_mask = _mask_or_true(
                "mapped_length",
                (lambda s: s >= float(min_length))
                if (min_length is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lrr_mask = _mask_or_true(
                "mapped_length_to_reference_length_ratio",
                (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                if (min_mapped_length_to_reference_length_ratio is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            demux_mask = _mask_or_true(
                "demux_type",
                (lambda s: s.astype("string").isin(list(demux_types)))
                if (demux_types is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            ref_mask = adata.obs[reference_col] == ref
            sample_mask = adata.obs[sample_col] == sample

            row_mask = ref_mask & sample_mask & qmask & lm_mask & lrr_mask & demux_mask

            if not bool(row_mask.any()):
                print(
                    f"No reads for {display_sample} - {ref} after read quality and length filtering"
                )
                continue

            try:
                subset = adata[row_mask, :].copy()

                # Column-level mask (var)
                if min_position_valid_fraction is not None:
                    valid_key = f"{ref}_valid_fraction"
                    if valid_key in subset.var:
                        v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                        col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                        if col_mask.any():
                            subset = subset[:, col_mask].copy()
                        else:
                            print(
                                f"No positions left after valid_fraction filter for {display_sample} - {ref}"
                            )
                            continue

                if subset.shape[0] == 0:
                    print(f"No reads left after filtering for {display_sample} - {ref}")
                    continue

                # bins mode
                if bins is None:
                    bins_temp = {"All": (subset.obs[reference_col] == ref)}
                else:
                    bins_temp = bins

                # find sites (positions)
                any_c_sites = gpc_sites = cpg_sites = np.array([], dtype=int)
                any_a_sites = np.array([], dtype=int)

                num_any_c = num_gpc = num_cpg = num_any_a = 0

                if include_any_c:
                    any_c_sites = np.where(subset.var.get(f"{ref}_C_site", False).values)[0]
                    gpc_sites = np.where(subset.var.get(f"{ref}_GpC_site", False).values)[0]
                    cpg_sites = np.where(subset.var.get(f"{ref}_CpG_site", False).values)[0]

                    num_any_c, num_gpc, num_cpg = len(any_c_sites), len(gpc_sites), len(cpg_sites)

                    any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                    gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                    cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)

                if include_any_a:
                    any_a_sites = np.where(subset.var.get(f"{ref}_A_site", False).values)[0]
                    num_any_a = len(any_a_sites)
                    any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                stacked_any_c, stacked_gpc, stacked_cpg, stacked_any_a = [], [], [], []
                row_labels, bin_labels, bin_boundaries = [], [], []
                percentages = {}
                last_idx = 0
                total_reads = subset.shape[0]

                # ----------------------------
                # per-bin stacking
                # ----------------------------
                for bin_label, bin_filter in bins_temp.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    if num_reads == 0:
                        percentages[bin_label] = 0.0
                        continue

                    percent_reads = (num_reads / total_reads) * 100
                    percentages[bin_label] = percent_reads

                    # compute order ONCE
                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        order = np.argsort(subset_bin.obs[colname].values)

                    elif sort_by == "gpc" and num_gpc > 0:
                        linkage = sch.linkage(
                            subset_bin[:, gpc_sites].layers[layer_gpc], method="ward"
                        )
                        order = sch.leaves_list(linkage)

                    elif sort_by == "cpg" and num_cpg > 0:
                        linkage = sch.linkage(
                            subset_bin[:, cpg_sites].layers[layer_cpg], method="ward"
                        )
                        order = sch.leaves_list(linkage)

                    elif sort_by == "c" and num_any_c > 0:
                        linkage = sch.linkage(
                            subset_bin[:, any_c_sites].layers[layer_c], method="ward"
                        )
                        order = sch.leaves_list(linkage)

                    elif sort_by == "gpc_cpg":
                        linkage = sch.linkage(subset_bin.layers[layer_gpc], method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "a" and num_any_a > 0:
                        linkage = sch.linkage(
                            subset_bin[:, any_a_sites].layers[layer_a], method="ward"
                        )
                        order = sch.leaves_list(linkage)

                    elif sort_by == "none":
                        order = np.arange(num_reads)

                    else:
                        order = np.arange(num_reads)

                    subset_bin = subset_bin[order]

                    # stack consistently
                    if include_any_c and num_any_c > 0:
                        stacked_any_c.append(subset_bin[:, any_c_sites].layers[layer_c])
                    if include_any_c and num_gpc > 0:
                        stacked_gpc.append(subset_bin[:, gpc_sites].layers[layer_gpc])
                    if include_any_c and num_cpg > 0:
                        stacked_cpg.append(subset_bin[:, cpg_sites].layers[layer_cpg])
                    if include_any_a and num_any_a > 0:
                        stacked_any_a.append(subset_bin[:, any_a_sites].layers[layer_a])

                    row_labels.extend([bin_label] * num_reads)
                    bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
                    last_idx += num_reads
                    bin_boundaries.append(last_idx)

                # ----------------------------
                # build matrices + means
                # ----------------------------
                blocks = []  # list of dicts describing what to plot in order

                if include_any_c and stacked_any_c:
                    any_c_matrix = np.vstack(stacked_any_c)
                    gpc_matrix = np.vstack(stacked_gpc) if stacked_gpc else np.empty((0, 0))
                    cpg_matrix = np.vstack(stacked_cpg) if stacked_cpg else np.empty((0, 0))

                    mean_any_c = methylation_fraction(any_c_matrix) if any_c_matrix.size else None
                    mean_gpc = methylation_fraction(gpc_matrix) if gpc_matrix.size else None
                    mean_cpg = methylation_fraction(cpg_matrix) if cpg_matrix.size else None

                    if any_c_matrix.size:
                        blocks.append(
                            dict(
                                name="c",
                                matrix=any_c_matrix,
                                mean=mean_any_c,
                                labels=any_c_labels,
                                cmap=cmap_c,
                                n_xticks=n_xticks_any_c,
                                title="any C site Modification Signal",
                            )
                        )
                    if gpc_matrix.size:
                        blocks.append(
                            dict(
                                name="gpc",
                                matrix=gpc_matrix,
                                mean=mean_gpc,
                                labels=gpc_labels,
                                cmap=cmap_gpc,
                                n_xticks=n_xticks_gpc,
                                title="GpC Modification Signal",
                            )
                        )
                    if cpg_matrix.size:
                        blocks.append(
                            dict(
                                name="cpg",
                                matrix=cpg_matrix,
                                mean=mean_cpg,
                                labels=cpg_labels,
                                cmap=cmap_cpg,
                                n_xticks=n_xticks_cpg,
                                title="CpG Modification Signal",
                            )
                        )

                if include_any_a and stacked_any_a:
                    any_a_matrix = np.vstack(stacked_any_a)
                    mean_any_a = methylation_fraction(any_a_matrix) if any_a_matrix.size else None
                    if any_a_matrix.size:
                        blocks.append(
                            dict(
                                name="a",
                                matrix=any_a_matrix,
                                mean=mean_any_a,
                                labels=any_a_labels,
                                cmap=cmap_a,
                                n_xticks=n_xticks_any_a,
                                title="any A site Modification Signal",
                            )
                        )

                if not blocks:
                    print(f"No matrices to plot for {display_sample} - {ref}")
                    continue

                gs_dim = len(blocks)
                fig = plt.figure(figsize=(5.5 * gs_dim, 11))
                gs = gridspec.GridSpec(2, gs_dim, height_ratios=[1, 6], hspace=0.02)
                fig.suptitle(f"{display_sample} - {ref} - {total_reads} reads", fontsize=14, y=0.97)

                axes_heat = [fig.add_subplot(gs[1, i]) for i in range(gs_dim)]
                axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(gs_dim)]

                # ----------------------------
                # plot blocks
                # ----------------------------
                for i, blk in enumerate(blocks):
                    mat = blk["matrix"]
                    mean = blk["mean"]
                    labels = np.asarray(blk["labels"], dtype=str)
                    n_xticks = blk["n_xticks"]

                    # barplot
                    clean_barplot(axes_bar[i], mean, blk["title"])

                    # heatmap
                    sns.heatmap(
                        mat, cmap=blk["cmap"], ax=axes_heat[i], yticklabels=False, cbar=False
                    )

                    # fixed tick labels
                    tick_pos = _fixed_tick_positions(len(labels), n_xticks)
                    axes_heat[i].set_xticks(tick_pos)
                    axes_heat[i].set_xticklabels(
                        labels[tick_pos], rotation=xtick_rotation, fontsize=xtick_fontsize
                    )

                    # bin separators
                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=2)

                    axes_heat[i].set_xlabel("Position", fontsize=9)

                plt.tight_layout()

                # save or show
                if save_path is not None:
                    safe_name = (
                        f"{ref}__{display_sample}".replace("=", "")
                        .replace("__", "_")
                        .replace(",", "_")
                        .replace(" ", "_")
                    )
                    out_file = save_path / f"{safe_name}.png"
                    fig.savefig(out_file, dpi=300)
                    plt.close(fig)
                    print(f"Saved: {out_file}")
                else:
                    plt.show()

                # record results
                rec = {
                    "sample": str(sample),
                    "ref": str(ref),
                    "row_labels": row_labels,
                    "bin_labels": bin_labels,
                    "bin_boundaries": bin_boundaries,
                    "percentages": percentages,
                }
                for blk in blocks:
                    rec[f"{blk['name']}_matrix"] = blk["matrix"]
                    rec[f"{blk['name']}_labels"] = list(map(str, blk["labels"]))
                results.append(rec)

                print(f"Summary for {display_sample} - {ref}:")
                for bin_label, percent in percentages.items():
                    print(f"  - {bin_label}: {percent:.1f}%")

            except Exception:
                import traceback

                traceback.print_exc()
                continue

    return results


def make_row_colors(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert metadata columns to RGB colors without invoking pandas Categorical.map
    (MultiIndex-safe, category-safe).
    """
    row_colors = pd.DataFrame(index=meta.index)

    for col in meta.columns:
        # Force plain python objects to avoid ExtensionArray/Categorical behavior
        s = meta[col].astype("object")

        def _to_label(x):
            if x is None:
                return "NA"
            if isinstance(x, float) and np.isnan(x):
                return "NA"
            # If a MultiIndex object is stored in a cell (rare), bucket it
            if isinstance(x, pd.MultiIndex):
                return "MultiIndex"
            # Tuples are common when MultiIndex-ish things get stored as values
            if isinstance(x, tuple):
                return "|".join(map(str, x))
            return str(x)

        labels = np.array([_to_label(x) for x in s.to_numpy()], dtype=object)
        uniq = pd.unique(labels)
        palette = dict(zip(uniq, sns.color_palette(n_colors=len(uniq))))

        # Map via python loop -> no pandas map machinery
        colors = [palette.get(lbl, (0.7, 0.7, 0.7)) for lbl in labels]
        row_colors[col] = colors

    return row_colors


def plot_rolling_nn_and_layer(
    subset,
    obsm_key: str = "rolling_nn_dist",
    layer_key: str = "nan0_0minus1",
    meta_cols=("Reference_strand", "Sample"),
    col_cluster: bool = False,
    fill_nn_with_colmax: bool = True,
    fill_layer_value: float = 0.0,
    drop_all_nan_windows: bool = True,
    figsize=(14, 10),
    right_panel_var_mask=None,  # optional boolean mask over subset.var to reduce width
    robust=True,
    save_name=None,
):
    """
    1) Cluster rows by subset.obsm[obsm_key] (rolling NN distances)
    2) Plot two heatmaps side-by-side in the SAME row order:
         - left: rolling NN distance matrix
         - right: subset.layers[layer_key] matrix

    Handles categorical/MultiIndex issues in metadata coloring.
    """

    # --- rolling NN distances
    X = subset.obsm[obsm_key]
    valid = ~np.all(np.isnan(X), axis=1)

    X_df = pd.DataFrame(X[valid], index=subset.obs_names[valid])

    if drop_all_nan_windows:
        X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]

    X_df_filled = X_df.copy()
    if fill_nn_with_colmax:
        col_max = X_df_filled.max(axis=0, skipna=True)
        X_df_filled = X_df_filled.fillna(col_max)

    # Ensure non-MultiIndex index for seaborn
    X_df_filled.index = X_df_filled.index.astype(str)

    # --- row colors from metadata (MultiIndex-safe)
    meta = subset.obs.loc[X_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    # --- get row order via clustermap
    g = sns.clustermap(
        X_df_filled,
        cmap="viridis",
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = X_df_filled.index[row_order]
    plt.close(g.fig)

    # reorder rolling NN matrix
    X_ord = X_df_filled.loc[ordered_index]

    # --- layer matrix
    L = subset.layers[layer_key]
    L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)

    L_df = pd.DataFrame(L[valid], index=subset.obs_names[valid], columns=subset.var_names)
    L_df.index = L_df.index.astype(str)

    if right_panel_var_mask is not None:
        # right_panel_var_mask must be boolean array/Series aligned to subset.var_names
        if hasattr(right_panel_var_mask, "values"):
            right_panel_var_mask = right_panel_var_mask.values
        L_df = L_df.loc[:, right_panel_var_mask]

    L_ord = L_df.loc[ordered_index]
    L_plot = L_ord.fillna(fill_layer_value)

    # --- plot side-by-side
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.heatmap(X_ord, ax=ax1, cmap="viridis", xticklabels=False, yticklabels=False, robust=robust)
    ax1.set_title(f"{obsm_key} (row-clustered)")

    sns.heatmap(
        L_plot, ax=ax2, cmap="coolwarm", xticklabels=False, yticklabels=False, robust=robust
    )
    ax2.set_title(f"{layer_key} (same row order)")

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")

    else:
        plt.show()

    return ordered_index


def plot_sequence_integer_encoding_clustermaps(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    layer: str = "sequence_integer_encoding",
    min_quality: float | None = 20,
    min_length: int | None = 200,
    min_mapped_length_to_reference_length_ratio: float | None = 0,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sort_by: str = "none",  # "none", "hierarchical", "obs:<col>"
    cmap: str = "viridis",
    max_unknown_fraction: float | None = None,
    unknown_values: Sequence[int] = (4, 5),
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    max_reads: int | None = None,
    save_path: str | Path | None = None,
    use_dna_5color_palette: bool = True,
    show_numeric_colorbar: bool = False,
):
    """Plot integer-encoded sequence clustermaps per sample/reference.

    Args:
        adata: AnnData with a ``sequence_integer_encoding`` layer.
        sample_col: Column in ``adata.obs`` that identifies samples.
        reference_col: Column in ``adata.obs`` that identifies references.
        layer: Layer name containing integer-encoded sequences.
        min_quality: Optional minimum read quality filter.
        min_length: Optional minimum mapped length filter.
        min_mapped_length_to_reference_length_ratio: Optional min length ratio filter.
        demux_types: Allowed ``demux_type`` values, if present in ``adata.obs``.
        sort_by: Row sorting strategy: ``none``, ``hierarchical``, or ``obs:<col>``.
        cmap: Matplotlib colormap for the heatmap when ``use_dna_5color_palette`` is False.
        max_unknown_fraction: Optional maximum fraction of ``unknown_values`` allowed per
            position; positions above this threshold are excluded.
        unknown_values: Integer values to treat as unknown/padding.
        xtick_step: Spacing between x-axis tick labels (None = no labels).
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        max_reads: Optional maximum number of reads to plot per sample/reference.
        save_path: Optional output directory for saving plots.
        use_dna_5color_palette: Whether to use a fixed A/C/G/T/Other palette.
        show_numeric_colorbar: If False, use a legend instead of a numeric colorbar.

    Returns:
        List of dictionaries with per-plot metadata and output paths.
    """

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    if max_unknown_fraction is not None and not (0 <= max_unknown_fraction <= 1):
        raise ValueError("max_unknown_fraction must be between 0 and 1.")

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    int_to_base = adata.uns.get("sequence_integer_decoding_map", {}) or {}
    if not int_to_base:
        encoding_map = adata.uns.get("sequence_integer_encoding_map", {}) or {}
        int_to_base = {int(v): str(k) for k, v in encoding_map.items()} if encoding_map else {}

    coerced_int_to_base = {}
    for key, value in int_to_base.items():
        try:
            coerced_key = int(key)
        except Exception:
            continue
        coerced_int_to_base[coerced_key] = str(value)
    int_to_base = coerced_int_to_base

    def normalize_base(base: str) -> str:
        return base if base in {"A", "C", "G", "T"} else "OTHER"

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            qmask = _mask_or_true(
                "read_quality",
                (lambda s: s >= float(min_quality))
                if (min_quality is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lm_mask = _mask_or_true(
                "mapped_length",
                (lambda s: s >= float(min_length))
                if (min_length is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lrr_mask = _mask_or_true(
                "mapped_length_to_reference_length_ratio",
                (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                if (min_mapped_length_to_reference_length_ratio is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            demux_mask = _mask_or_true(
                "demux_type",
                (lambda s: s.astype("string").isin(list(demux_types)))
                if (demux_types is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            row_mask = (
                (adata.obs[reference_col] == ref)
                & (adata.obs[sample_col] == sample)
                & qmask
                & lm_mask
                & lrr_mask
                & demux_mask
            )
            if not bool(row_mask.any()):
                continue

            subset = adata[row_mask, :].copy()
            matrix = np.asarray(subset.layers[layer])

            if max_unknown_fraction is not None:
                unknown_mask = np.isin(matrix, np.asarray(unknown_values))
                unknown_fraction = unknown_mask.mean(axis=0)
                keep_columns = unknown_fraction <= max_unknown_fraction
                if not np.any(keep_columns):
                    continue
                matrix = matrix[:, keep_columns]
                subset = subset[:, keep_columns].copy()

            if max_reads is not None and matrix.shape[0] > max_reads:
                matrix = matrix[:max_reads]
                subset = subset[:max_reads, :].copy()

            if matrix.size == 0:
                continue

            if use_dna_5color_palette and not int_to_base:
                uniq_vals = np.unique(matrix[~pd.isna(matrix)])
                guess = {}
                for val in uniq_vals:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    guess[int_val] = {0: "A", 1: "C", 2: "G", 3: "T"}.get(int_val, "OTHER")
                int_to_base_local = guess
            else:
                int_to_base_local = int_to_base

            order = None
            if sort_by.startswith("obs:"):
                colname = sort_by.split("obs:")[1]
                order = np.argsort(subset.obs[colname].values)
            elif sort_by == "hierarchical":
                linkage = sch.linkage(np.nan_to_num(matrix), method="ward")
                order = sch.leaves_list(linkage)
            elif sort_by != "none":
                raise ValueError("sort_by must be 'none', 'hierarchical', or 'obs:<col>'")

            if order is not None:
                matrix = matrix[order]

            fig, ax = plt.subplots(figsize=(12, 6))

            if use_dna_5color_palette and int_to_base_local:
                int_to_color = {
                    int(int_val): DNA_5COLOR_PALETTE[normalize_base(str(base))]
                    for int_val, base in int_to_base_local.items()
                }
                uniq_matrix = np.unique(matrix[~pd.isna(matrix)])
                for val in uniq_matrix:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    if int_val not in int_to_color:
                        int_to_color[int_val] = DNA_5COLOR_PALETTE["OTHER"]

                ordered = sorted(int_to_color.items(), key=lambda x: x[0])
                colors_list = [color for _, color in ordered]
                bounds = [int_val - 0.5 for int_val, _ in ordered]
                bounds.append(ordered[-1][0] + 0.5)

                cmap_obj = colors.ListedColormap(colors_list)
                norm = colors.BoundaryNorm(bounds, cmap_obj.N)

                sns.heatmap(
                    matrix,
                    cmap=cmap_obj,
                    norm=norm,
                    ax=ax,
                    yticklabels=False,
                    cbar=show_numeric_colorbar,
                )

                legend_handles = [
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["A"], label="A"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["C"], label="C"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["G"], label="G"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["T"], label="T"),
                    patches.Patch(
                        facecolor=DNA_5COLOR_PALETTE["OTHER"],
                        label="Other (N / PAD / unknown)",
                    ),
                ]
                ax.legend(
                    handles=legend_handles,
                    title="Base",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )
            else:
                sns.heatmap(matrix, cmap=cmap, ax=ax, yticklabels=False, cbar=True)

            ax.set_title(f"{sample} - {ref} ({layer})")

            if xtick_step is not None and xtick_step > 0:
                sites = np.arange(0, matrix.shape[1], xtick_step)
                ax.set_xticks(sites)
                ax.set_xticklabels(
                    subset.var_names[sites].astype(str),
                    rotation=xtick_rotation,
                    fontsize=xtick_fontsize,
                )
            else:
                ax.set_xticks([])

            out_file = None
            if save_path is not None:
                safe_name = f"{ref}__{sample}__{layer}".replace("=", "").replace(",", "_")
                out_file = save_path / f"{safe_name}.png"
                fig.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            results.append(
                {
                    "reference": str(ref),
                    "sample": str(sample),
                    "layer": layer,
                    "n_positions": int(matrix.shape[1]),
                    "output_path": str(out_file) if out_file is not None else None,
                }
            )

    return results


def plot_hmm_layers_rolling_by_sample_ref(
    adata,
    layers: Optional[Sequence[str]] = None,
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    samples: Optional[Sequence[str]] = None,
    references: Optional[Sequence[str]] = None,
    window: int = 51,
    min_periods: int = 1,
    center: bool = True,
    rows_per_page: int = 6,
    figsize_per_cell: Tuple[float, float] = (4.0, 2.5),
    dpi: int = 160,
    output_dir: Optional[str] = None,
    save: bool = True,
    show_raw: bool = False,
    cmap: str = "tab20",
    use_var_coords: bool = True,
):
    """
    For each sample (row) and reference (col) plot the rolling average of the
    positional mean (mean across reads) for each layer listed.

    Parameters
    ----------
    adata : AnnData
        Input annotated data (expects obs columns sample_col and ref_col).
    layers : list[str] | None
        Which adata.layers to plot. If None, attempts to autodetect layers whose
        matrices look like "HMM" outputs (else will error). If None and layers
        cannot be found, user must pass a list.
    sample_col, ref_col : str
        obs columns used to group rows.
    samples, references : optional lists
        explicit ordering of samples / references. If None, categories in adata.obs are used.
    window : int
        rolling window size (odd recommended). If window <= 1, no smoothing applied.
    min_periods : int
        min periods param for pd.Series.rolling.
    center : bool
        center the rolling window.
    rows_per_page : int
        paginate rows per page into multiple figures if needed.
    figsize_per_cell : (w,h)
        per-subplot size in inches.
    dpi : int
        figure dpi when saving.
    output_dir : str | None
        directory to save pages; created if necessary. If None and save=True, uses cwd.
    save : bool
        whether to save PNG files.
    show_raw : bool
        draw unsmoothed mean as faint line under smoothed curve.
    cmap : str
        matplotlib colormap for layer lines.
    use_var_coords : bool
        if True, tries to use adata.var_names (coerced to int) as x-axis coordinates; otherwise uses 0..n-1.

    Returns
    -------
    saved_files : list[str]
        list of saved filenames (may be empty if save=False).
    """

    # --- basic checks / defaults ---
    if sample_col not in adata.obs.columns or ref_col not in adata.obs.columns:
        raise ValueError(
            f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs"
        )

    # canonicalize samples / refs
    if samples is None:
        sseries = adata.obs[sample_col]
        if not pd.api.types.is_categorical_dtype(sseries):
            sseries = sseries.astype("category")
        samples_all = list(sseries.cat.categories)
    else:
        samples_all = list(samples)

    if references is None:
        rseries = adata.obs[ref_col]
        if not pd.api.types.is_categorical_dtype(rseries):
            rseries = rseries.astype("category")
        refs_all = list(rseries.cat.categories)
    else:
        refs_all = list(references)

    # choose layers: if not provided, try a sensible default: all layers
    if layers is None:
        layers = list(adata.layers.keys())
        if len(layers) == 0:
            raise ValueError(
                "No adata.layers found. Please pass `layers=[...]` of the HMM layers to plot."
            )
    layers = list(layers)

    # x coordinates (positions)
    try:
        if use_var_coords:
            x_coords = np.array([int(v) for v in adata.var_names])
        else:
            raise Exception("user disabled var coords")
    except Exception:
        # fallback to 0..n_vars-1
        x_coords = np.arange(adata.shape[1], dtype=int)

    # make output dir
    if save:
        outdir = output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None

    n_samples = len(samples_all)
    n_refs = len(refs_all)
    total_pages = math.ceil(n_samples / rows_per_page)
    saved_files = []

    # color cycle for layers
    cmap_obj = plt.get_cmap(cmap)
    n_layers = max(1, len(layers))
    colors = [cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)]

    for page in range(total_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n_samples)
        chunk = samples_all[start:end]
        nrows = len(chunk)
        ncols = n_refs

        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
        )

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(refs_all):
                ax = axes[r_idx][c_idx]

                # subset adata
                mask = (adata.obs[sample_col].values == sample_name) & (
                    adata.obs[ref_col].values == ref_name
                )
                sub = adata[mask]
                if sub.n_obs == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No reads",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="gray",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r_idx == 0:
                        ax.set_title(str(ref_name), fontsize=9)
                    if c_idx == 0:
                        total_reads = int((adata.obs[sample_col] == sample_name).sum())
                        ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                    continue

                # for each layer, compute positional mean across reads (ignore NaNs)
                plotted_any = False
                for li, layer in enumerate(layers):
                    if layer in sub.layers:
                        mat = sub.layers[layer]
                    else:
                        # fallback: try .X only for the first layer if layer not present
                        if layer == layers[0] and getattr(sub, "X", None) is not None:
                            mat = sub.X
                        else:
                            # layer not present for this subset
                            continue

                    # convert matrix to numpy 2D
                    if hasattr(mat, "toarray"):
                        try:
                            arr = mat.toarray()
                        except Exception:
                            arr = np.asarray(mat)
                    else:
                        arr = np.asarray(mat)

                    if arr.size == 0 or arr.shape[1] == 0:
                        continue

                    # compute column-wise mean ignoring NaNs
                    # if arr is boolean or int, convert to float to support NaN
                    arr = arr.astype(float)
                    with np.errstate(all="ignore"):
                        col_mean = np.nanmean(arr, axis=0)

                    # If all-NaN, skip
                    if np.all(np.isnan(col_mean)):
                        continue

                    # smooth via pandas rolling (centered)
                    if (window is None) or (window <= 1):
                        smoothed = col_mean
                    else:
                        ser = pd.Series(col_mean)
                        smoothed = (
                            ser.rolling(window=window, min_periods=min_periods, center=center)
                            .mean()
                            .to_numpy()
                        )

                    # x axis: x_coords (trim/pad to match length)
                    L = len(col_mean)
                    x = x_coords[:L]

                    # optionally plot raw faint line first
                    if show_raw:
                        ax.plot(x, col_mean[:L], linewidth=0.7, alpha=0.25, zorder=1)

                    ax.plot(
                        x,
                        smoothed[:L],
                        label=layer,
                        color=colors[li],
                        linewidth=1.2,
                        alpha=0.95,
                        zorder=2,
                    )
                    plotted_any = True

                # labels / titles
                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=9)
                if c_idx == 0:
                    total_reads = int((adata.obs[sample_col] == sample_name).sum())
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                if r_idx == nrows - 1:
                    ax.set_xlabel("position", fontsize=8)

                # legend (only show in top-left plot to reduce clutter)
                if (r_idx == 0 and c_idx == 0) and plotted_any:
                    ax.legend(fontsize=7, loc="upper right")

                ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"Rolling mean of layer positional means (window={window}) — page {page + 1}/{total_pages}",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            fname = os.path.join(outdir, f"hmm_layers_rolling_page{page + 1}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=dpi)
            saved_files.append(fname)
        else:
            plt.show()
        plt.close(fig)

    return saved_files
