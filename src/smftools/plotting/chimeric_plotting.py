from __future__ import annotations

import os
from math import floor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require
from smftools.plotting.plotting_utils import (
    _methylation_fraction_for_layer,
    clean_barplot,
    make_row_colors,
)

plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
grid_spec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
sns = require("seaborn", extra="plotting", purpose="plot styling")

logger = get_logger(__name__)


def plot_rolling_nn_and_layer(
    subset,
    obsm_key: str = "rolling_nn_dist",
    layer_key: str = "nan0_0minus1",
    meta_cols: tuple[str, ...] = ("Reference_strand", "Sample"),
    col_cluster: bool = False,
    fill_nn_with_colmax: bool = True,
    fill_layer_value: float = 0.0,
    drop_all_nan_windows: bool = True,
    max_nan_fraction: float | None = None,
    var_valid_fraction_col: str | None = None,
    var_nan_fraction_col: str | None = None,
    read_span_layer: str | None = "read_span_mask",
    outside_read_color: str = "#bdbdbd",
    nn_nan_color: str = "#bdbdbd",
    figsize: tuple[float, float] = (14, 10),
    right_panel_var_mask=None,  # optional boolean mask over subset.var to reduce width
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name: str | None = None,
):
    """
    1) Cluster rows by subset.obsm[obsm_key] (rolling NN distances)
    2) Plot two heatmaps side-by-side in the SAME row order, with mean barplots above:
         - left: rolling NN distance matrix
         - right: subset.layers[layer_key] matrix

    Handles categorical/MultiIndex issues in metadata coloring.

    Args:
        subset: AnnData subset with rolling NN distances stored in ``obsm``.
        obsm_key: Key in ``subset.obsm`` containing rolling NN distances.
        layer_key: Layer name to plot alongside rolling NN distances.
        meta_cols: Obs columns used for row color annotations.
        col_cluster: Whether to cluster columns in the rolling NN clustermap.
        fill_nn_with_colmax: Fill NaNs in rolling NN distances with per-column max values.
        fill_layer_value: Fill NaNs in the layer heatmap with this value.
        drop_all_nan_windows: Drop rolling windows that are all NaN.
        max_nan_fraction: Maximum allowed NaN fraction per position (filtering columns).
        var_valid_fraction_col: ``subset.var`` column with valid fractions (1 - NaN fraction).
        var_nan_fraction_col: ``subset.var`` column with NaN fractions.
        read_span_layer: Layer name with read span mask; 0 values are treated as outside read.
        outside_read_color: Color used to show positions outside each read.
        nn_nan_color: Color used for NaNs in the rolling NN heatmap.
        figsize: Figure size for the combined plot.
        right_panel_var_mask: Optional boolean mask over ``subset.var`` for the right panel.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")

    logger.info("Plotting rolling NN distances with layer '%s'.", layer_key)

    def _apply_xticks(ax, labels, step):
        if labels is None or len(labels) == 0:
            ax.set_xticks([])
            return
        if step is None or step <= 0:
            step = max(1, len(labels) // 10)
        ticks = np.arange(0, len(labels), step)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(
            [labels[i] for i in ticks],
            rotation=xtick_rotation,
            fontsize=xtick_fontsize,
        )

    def _format_labels(values):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.number):
            if np.all(np.isfinite(values)) and np.all(np.isclose(values, np.round(values))):
                values = np.round(values).astype(int)
        return [str(v) for v in values]

    X = subset.obsm[obsm_key]
    valid = ~np.all(np.isnan(X), axis=1)

    X_df = pd.DataFrame(X[valid], index=subset.obs_names[valid])

    if drop_all_nan_windows:
        X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]

    col_max = X_df.max(axis=0, skipna=True).fillna(0)
    X_df_cluster = X_df.fillna(col_max)
    X_df_cluster.index = X_df_cluster.index.astype(str)
    if fill_nn_with_colmax:
        X_df_display = X_df_cluster
    else:
        X_df_display = X_df.copy()
        X_df_display.index = X_df_display.index.astype(str)

    meta = subset.obs.loc[X_df_cluster.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    g = sns.clustermap(
        X_df_cluster,
        cmap="viridis",
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = X_df_cluster.index[row_order]
    plt.close(g.fig)

    X_ord = X_df_display.loc[ordered_index]

    L = subset.layers[layer_key]
    L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)

    L_df = pd.DataFrame(L[valid], index=subset.obs_names[valid], columns=subset.var_names)
    L_df.index = L_df.index.astype(str)

    if right_panel_var_mask is not None:
        if hasattr(right_panel_var_mask, "values"):
            right_panel_var_mask = right_panel_var_mask.values
        right_panel_var_mask = np.asarray(right_panel_var_mask, dtype=bool)

    if max_nan_fraction is not None:
        nan_fraction = None
        if var_nan_fraction_col and var_nan_fraction_col in subset.var:
            nan_fraction = pd.to_numeric(
                subset.var[var_nan_fraction_col], errors="coerce"
            ).to_numpy()
        elif var_valid_fraction_col and var_valid_fraction_col in subset.var:
            valid_fraction = pd.to_numeric(
                subset.var[var_valid_fraction_col], errors="coerce"
            ).to_numpy()
            nan_fraction = 1 - valid_fraction
        if nan_fraction is not None:
            nan_mask = nan_fraction <= max_nan_fraction
            if right_panel_var_mask is None:
                right_panel_var_mask = nan_mask
            else:
                right_panel_var_mask = right_panel_var_mask & nan_mask

    if right_panel_var_mask is not None:
        if right_panel_var_mask.size != L_df.shape[1]:
            raise ValueError("right_panel_var_mask must align with subset.var_names.")
        L_df = L_df.loc[:, right_panel_var_mask]

    read_span_mask = None
    if read_span_layer and read_span_layer in subset.layers:
        span = subset.layers[read_span_layer]
        span = span.toarray() if hasattr(span, "toarray") else np.asarray(span)
        span_df = pd.DataFrame(span[valid], index=subset.obs_names[valid], columns=subset.var_names)
        span_df.index = span_df.index.astype(str)
        if right_panel_var_mask is not None:
            span_df = span_df.loc[:, right_panel_var_mask]
        read_span_mask = span_df.loc[ordered_index].to_numpy() == 0

    L_ord = L_df.loc[ordered_index]
    L_plot = L_ord.fillna(fill_layer_value)
    if read_span_mask is not None:
        L_plot = L_plot.mask(read_span_mask)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 6],
        wspace=0.2,
        hspace=0.05,
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax1_cbar = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_cbar = fig.add_subplot(gs[1, 3])
    ax1_bar = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2_bar = fig.add_subplot(gs[0, 2], sharex=ax2)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 3]).axis("off")

    mean_nn = np.nanmean(X_ord.to_numpy(), axis=0)
    clean_barplot(
        ax1_bar,
        mean_nn,
        obsm_key,
        y_max=None,
        y_label="Mean distance",
        y_ticks=None,
    )

    nn_cmap = plt.get_cmap("viridis").copy()
    nn_cmap.set_bad(nn_nan_color)
    sns.heatmap(
        X_ord,
        ax=ax1,
        cmap=nn_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax1_cbar,
    )
    label_source = subset.uns.get(f"{obsm_key}_centers")
    if label_source is None:
        label_source = subset.uns.get(f"{obsm_key}_starts")
    if label_source is not None:
        label_source = np.asarray(label_source)
        window_labels = _format_labels(label_source)
        try:
            col_idx = X_ord.columns.to_numpy()
            if np.issubdtype(col_idx.dtype, np.number):
                col_idx = col_idx.astype(int)
                if col_idx.size and col_idx.max() < len(label_source):
                    window_labels = _format_labels(label_source[col_idx])
        except Exception:
            window_labels = _format_labels(label_source)
        _apply_xticks(ax1, window_labels, xtick_step)

    methylation_fraction = _methylation_fraction_for_layer(L_ord.to_numpy(), layer_key)
    clean_barplot(
        ax2_bar,
        methylation_fraction,
        layer_key,
        y_max=1.0,
        y_label="Methylation fraction",
        y_ticks=[0.0, 0.5, 1.0],
    )

    layer_cmap = plt.get_cmap("coolwarm").copy()
    if read_span_mask is not None:
        layer_cmap.set_bad(outside_read_color)

    sns.heatmap(
        L_plot,
        ax=ax2,
        cmap=layer_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax2_cbar,
    )
    _apply_xticks(ax2, [str(x) for x in L_plot.columns], xtick_step)

    if title:
        fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved rolling NN/layer plot to %s.", fname)
    else:
        plt.show()

    return ordered_index


def plot_zero_hamming_span_and_layer(
    subset,
    span_layer_key: str,
    layer_key: str = "nan0_0minus1",
    meta_cols: tuple[str, ...] = ("Reference_strand", "Sample"),
    col_cluster: bool = False,
    fill_span_value: float = 0.0,
    fill_layer_value: float = 0.0,
    drop_all_nan_positions: bool = True,
    max_nan_fraction: float | None = None,
    var_valid_fraction_col: str | None = None,
    var_nan_fraction_col: str | None = None,
    read_span_layer: str | None = "read_span_mask",
    outside_read_color: str = "#bdbdbd",
    span_color: str = "#2ca25f",
    figsize: tuple[float, float] = (14, 10),
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name: str | None = None,
):
    """
    Plot zero-Hamming span clustermap alongside a layer clustermap.

    Args:
        subset: AnnData subset with zero-Hamming span annotations stored in ``layers``.
        span_layer_key: Layer name with the binary zero-Hamming span mask.
        layer_key: Layer name to plot alongside the span mask.
        meta_cols: Obs columns used for row color annotations.
        col_cluster: Whether to cluster columns in the span mask clustermap.
        fill_span_value: Value to fill NaNs in the span mask.
        fill_layer_value: Value to fill NaNs in the layer heatmap.
        drop_all_nan_positions: Drop positions that are all NaN in the span mask.
        max_nan_fraction: Maximum allowed NaN fraction per position (filtering columns).
        var_valid_fraction_col: ``subset.var`` column with valid fractions (1 - NaN fraction).
        var_nan_fraction_col: ``subset.var`` column with NaN fractions.
        read_span_layer: Layer name with read span mask; 0 values are treated as outside read.
        outside_read_color: Color used to show positions outside each read.
        span_color: Color for zero-Hamming span mask values.
        figsize: Figure size for the combined plot.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")

    logger.info(
        "Plotting zero-Hamming span mask '%s' with layer '%s'.",
        span_layer_key,
        layer_key,
    )

    def _apply_xticks(ax, labels, step):
        if labels is None or len(labels) == 0:
            ax.set_xticks([])
            return
        if step is None or step <= 0:
            step = max(1, len(labels) // 10)
        ticks = np.arange(0, len(labels), step)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(
            [labels[i] for i in ticks],
            rotation=xtick_rotation,
            fontsize=xtick_fontsize,
        )

    span = subset.layers[span_layer_key]
    span = span.toarray() if hasattr(span, "toarray") else np.asarray(span)
    span_df = pd.DataFrame(span, index=subset.obs_names, columns=subset.var_names)
    span_df.index = span_df.index.astype(str)

    if drop_all_nan_positions:
        span_df = span_df.loc[:, ~span_df.isna().all(axis=0)]

    nan_mask = None
    if max_nan_fraction is not None:
        nan_fraction = None
        if var_nan_fraction_col and var_nan_fraction_col in subset.var:
            nan_fraction = pd.to_numeric(
                subset.var[var_nan_fraction_col], errors="coerce"
            ).to_numpy()
        elif var_valid_fraction_col and var_valid_fraction_col in subset.var:
            valid_fraction = pd.to_numeric(
                subset.var[var_valid_fraction_col], errors="coerce"
            ).to_numpy()
            nan_fraction = 1 - valid_fraction
        if nan_fraction is not None:
            nan_mask = nan_fraction <= max_nan_fraction
            span_df = span_df.loc[:, nan_mask]

    span_df_filled = span_df.fillna(fill_span_value)
    span_df_filled.index = span_df_filled.index.astype(str)

    meta = subset.obs.loc[span_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    span_cmap = colors.ListedColormap(["white", span_color])
    span_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], span_cmap.N)

    g = sns.clustermap(
        span_df_filled,
        cmap=span_cmap,
        norm=span_norm,
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = span_df_filled.index[row_order]
    plt.close(g.fig)

    span_ord = span_df_filled.loc[ordered_index]

    layer = subset.layers[layer_key]
    layer = layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)
    layer_df = pd.DataFrame(layer, index=subset.obs_names, columns=subset.var_names)
    layer_df.index = layer_df.index.astype(str)

    if max_nan_fraction is not None and nan_mask is not None:
        layer_df = layer_df.loc[:, nan_mask]

    read_span_mask = None
    if read_span_layer and read_span_layer in subset.layers:
        span_mask = subset.layers[read_span_layer]
        span_mask = span_mask.toarray() if hasattr(span_mask, "toarray") else np.asarray(span_mask)
        span_mask_df = pd.DataFrame(span_mask, index=subset.obs_names, columns=subset.var_names)
        span_mask_df.index = span_mask_df.index.astype(str)
        if max_nan_fraction is not None and nan_mask is not None:
            span_mask_df = span_mask_df.loc[:, nan_mask]
        read_span_mask = span_mask_df.loc[ordered_index].to_numpy() == 0

    layer_ord = layer_df.loc[ordered_index]
    layer_plot = layer_ord.fillna(fill_layer_value)
    if read_span_mask is not None:
        layer_plot = layer_plot.mask(read_span_mask)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 6],
        wspace=0.2,
        hspace=0.05,
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax1_cbar = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_cbar = fig.add_subplot(gs[1, 3])
    ax1_bar = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2_bar = fig.add_subplot(gs[0, 2], sharex=ax2)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 3]).axis("off")

    mean_span = np.nanmean(span_ord.to_numpy(), axis=0)
    clean_barplot(
        ax1_bar,
        mean_span,
        span_layer_key,
        y_max=1.0,
        y_label="Span fraction",
        y_ticks=[0.0, 0.5, 1.0],
    )

    methylation_fraction = _methylation_fraction_for_layer(layer_ord.to_numpy(), layer_key)
    clean_barplot(
        ax2_bar,
        methylation_fraction,
        layer_key,
        y_max=1.0,
        y_label="Methylation fraction",
        y_ticks=[0.0, 0.5, 1.0],
    )

    sns.heatmap(
        span_ord,
        ax=ax1,
        cmap=span_cmap,
        norm=span_norm,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax1_cbar,
    )

    layer_cmap = plt.get_cmap("coolwarm").copy()
    if read_span_mask is not None:
        layer_cmap.set_bad(outside_read_color)

    sns.heatmap(
        layer_plot,
        ax=ax2,
        cmap=layer_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax2_cbar,
    )

    _apply_xticks(ax1, [str(x) for x in span_ord.columns], xtick_step)
    _apply_xticks(ax2, [str(x) for x in layer_plot.columns], xtick_step)

    if title:
        fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved zero-Hamming span/layer plot to %s.", fname)
    else:
        plt.show()

    return ordered_index


def _window_center_labels(var_names: Sequence, starts: np.ndarray, window: int) -> list[str]:
    coords = np.asarray(var_names)
    if coords.size == 0:
        return []
    try:
        coords_numeric = coords.astype(float)
        centers = np.array(
            [floor(np.nanmean(coords_numeric[s : s + window])) for s in starts], dtype=float
        )
        return [str(c) for c in centers]
    except Exception:
        mid = np.clip(starts + (window // 2), 0, coords.size - 1)
        return [str(coords[idx]) for idx in mid]


def plot_zero_hamming_pair_counts(
    subset,
    zero_pairs_uns_key: str,
    meta_cols: tuple[str, ...] = ("Reference_strand", "Sample"),
    col_cluster: bool = False,
    figsize: tuple[float, float] = (14, 10),
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name: str | None = None,
):
    """
    Plot a heatmap of zero-Hamming pair counts per read across rolling windows.

    Args:
        subset: AnnData subset containing zero-pair window data in ``.uns``.
        zero_pairs_uns_key: Key in ``subset.uns`` with zero-pair window data.
        meta_cols: Obs columns used for row color annotations.
        col_cluster: Whether to cluster columns in the heatmap.
        figsize: Figure size for the plot.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    if zero_pairs_uns_key not in subset.uns:
        raise KeyError(f"Missing zero-pair data in subset.uns[{zero_pairs_uns_key!r}].")

    zero_pairs_by_window = subset.uns[zero_pairs_uns_key]
    starts = np.asarray(subset.uns.get(f"{zero_pairs_uns_key}_starts", []))
    window = int(subset.uns.get(f"{zero_pairs_uns_key}_window", 0))

    n_windows = len(zero_pairs_by_window)
    counts = np.zeros((subset.n_obs, n_windows), dtype=int)

    for wi, pairs in enumerate(zero_pairs_by_window):
        if pairs is None or len(pairs) == 0:
            continue
        pair_arr = np.asarray(pairs, dtype=int)
        if pair_arr.size == 0:
            continue
        if pair_arr.ndim != 2 or pair_arr.shape[1] != 2:
            raise ValueError("Zero-pair entries must be arrays of shape (n, 2).")
        np.add.at(counts[:, wi], pair_arr[:, 0], 1)
        np.add.at(counts[:, wi], pair_arr[:, 1], 1)

    if starts.size == n_windows and window > 0:
        labels = _window_center_labels(subset.var_names, starts, window)
    else:
        labels = [str(i) for i in range(n_windows)]

    counts_df = pd.DataFrame(counts, index=subset.obs_names.astype(str), columns=labels)
    meta = subset.obs.loc[counts_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    def _apply_xticks(ax, labels, step):
        if labels is None or len(labels) == 0:
            ax.set_xticks([])
            return
        if step is None or step <= 0:
            step = max(1, len(labels) // 10)
        ticks = np.arange(0, len(labels), step)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(
            [labels[i] for i in ticks],
            rotation=xtick_rotation,
            fontsize=xtick_fontsize,
        )

    g = sns.clustermap(
        counts_df,
        cmap="viridis",
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        figsize=figsize,
        robust=robust,
    )
    _apply_xticks(g.ax_heatmap, labels, xtick_step)

    if title:
        g.fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        g.fig.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved zero-Hamming pair count plot to %s.", fname)
    else:
        plt.show()

    return g
