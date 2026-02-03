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

    methylation_fraction = _methylation_fraction_for_layer(L_plot.to_numpy(), layer_key)
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


def plot_rolling_nn_and_two_layers(
    subset,
    obsm_key: str = "rolling_nn_dist",
    layer_keys: Sequence[str] = ("nan0_0minus1", "nan0_0minus1"),
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
    figsize: tuple[float, float] = (20, 10),
    layer_var_mask=None,
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name: str | None = None,
):
    """
    Plot rolling NN distances alongside two layer clustermaps.

    Args:
        subset: AnnData subset with rolling NN distances stored in ``obsm``.
        obsm_key: Key in ``subset.obsm`` containing rolling NN distances.
        layer_keys: Two layer names to plot alongside rolling NN distances.
        meta_cols: Obs columns used for row color annotations.
        col_cluster: Whether to cluster columns in the rolling NN clustermap.
        fill_nn_with_colmax: Fill NaNs in rolling NN distances with per-column max values.
        fill_layer_value: Fill NaNs in the layer heatmaps with this value.
        drop_all_nan_windows: Drop rolling windows that are all NaN.
        max_nan_fraction: Maximum allowed NaN fraction per position (filtering columns).
        var_valid_fraction_col: ``subset.var`` column with valid fractions (1 - NaN fraction).
        var_nan_fraction_col: ``subset.var`` column with NaN fractions.
        read_span_layer: Layer name with read span mask; 0 values are treated as outside read.
        outside_read_color: Color used to show positions outside each read.
        nn_nan_color: Color used for NaNs in the rolling NN heatmap.
        figsize: Figure size for the combined plot.
        layer_var_mask: Optional boolean mask over ``subset.var`` for the layer panels.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    if len(layer_keys) != 2:
        raise ValueError("layer_keys must contain exactly two layer names.")
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")

    layer_key_one, layer_key_two = layer_keys
    logger.info(
        "Plotting rolling NN distances with layers '%s' and '%s'.",
        layer_key_one,
        layer_key_two,
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

    if layer_var_mask is not None:
        if hasattr(layer_var_mask, "values"):
            layer_var_mask = layer_var_mask.values
        layer_var_mask = np.asarray(layer_var_mask, dtype=bool)

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
            if layer_var_mask is None:
                layer_var_mask = nan_mask
            else:
                layer_var_mask = layer_var_mask & nan_mask

    if layer_var_mask is not None and layer_var_mask.size != subset.n_vars:
        raise ValueError("layer_var_mask must align with subset.var_names.")

    read_span_mask = None
    if read_span_layer and read_span_layer in subset.layers:
        span = subset.layers[read_span_layer]
        span = span.toarray() if hasattr(span, "toarray") else np.asarray(span)
        span_df = pd.DataFrame(span[valid], index=subset.obs_names[valid], columns=subset.var_names)
        span_df.index = span_df.index.astype(str)
        if layer_var_mask is not None:
            span_df = span_df.loc[:, layer_var_mask]
        read_span_mask = span_df.loc[ordered_index].to_numpy() == 0

    def _layer_df_for_key(layer_key: str) -> pd.DataFrame:
        layer = subset.layers[layer_key]
        layer = layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)
        layer_df = pd.DataFrame(
            layer[valid], index=subset.obs_names[valid], columns=subset.var_names
        )
        layer_df.index = layer_df.index.astype(str)
        if layer_var_mask is not None:
            layer_df = layer_df.loc[:, layer_var_mask]
        return layer_df.loc[ordered_index]

    layer_df_one = _layer_df_for_key(layer_key_one)
    layer_df_two = _layer_df_for_key(layer_key_two)

    layer_plot_one = layer_df_one.fillna(fill_layer_value)
    layer_plot_two = layer_df_two.fillna(fill_layer_value)
    if read_span_mask is not None:
        layer_plot_one = layer_plot_one.mask(read_span_mask)
        layer_plot_two = layer_plot_two.mask(read_span_mask)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05],
        height_ratios=[1, 6],
        wspace=0.2,
        hspace=0.05,
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax1_cbar = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_cbar = fig.add_subplot(gs[1, 3])
    ax3 = fig.add_subplot(gs[1, 4])
    ax3_cbar = fig.add_subplot(gs[1, 5])
    ax1_bar = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2_bar = fig.add_subplot(gs[0, 2], sharex=ax2)
    ax3_bar = fig.add_subplot(gs[0, 4], sharex=ax3)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 3]).axis("off")
    fig.add_subplot(gs[0, 5]).axis("off")

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

    for ax_bar, lp, layer_key in (
        (ax2_bar, layer_plot_one, layer_key_one),
        (ax3_bar, layer_plot_two, layer_key_two),
    ):
        methylation_fraction = _methylation_fraction_for_layer(lp.to_numpy(), layer_key)
        clean_barplot(
            ax_bar,
            methylation_fraction,
            layer_key,
            y_max=1.0,
            y_label="Methylation fraction",
            y_ticks=[0.0, 0.5, 1.0],
        )

    layer_cmap = plt.get_cmap("coolwarm").copy()
    if read_span_mask is not None:
        layer_cmap.set_bad(outside_read_color)

    layer2_cmap = plt.get_cmap("Greens").copy()
    if read_span_mask is not None:
        layer2_cmap.set_bad(outside_read_color)

    sns.heatmap(
        layer_plot_one,
        ax=ax2,
        cmap=layer_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax2_cbar,
    )
    sns.heatmap(
        layer_plot_two,
        ax=ax3,
        cmap=layer2_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax3_cbar,
    )
    _apply_xticks(ax2, [str(x) for x in layer_plot_one.columns], xtick_step)
    _apply_xticks(ax3, [str(x) for x in layer_plot_two.columns], xtick_step)

    if title:
        fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved rolling NN/layer pair plot to %s.", fname)
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
    variant_call_data: "pd.DataFrame | None" = None,
    seq1_label: str = "seq1",
    seq2_label: str = "seq2",
    ref1_marker_color: str = "white",
    ref2_marker_color: str = "black",
    variant_marker_size: float = 4.0,
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
        variant_call_data: Optional DataFrame (obs × full var_names) with variant calls
            (1=seq1, 2=seq2). When provided, circles are overlaid at positions that
            overlap with the plotted columns. Built from the full-width adata before
            column filtering so mismatch sites outside modification sites are mapped.
        seq1_label: Label for seq1 in the legend.
        seq2_label: Label for seq2 in the legend.
        ref1_marker_color: Circle color for seq1 variant calls.
        ref2_marker_color: Circle color for seq2 variant calls.
        variant_marker_size: Size of variant call overlay circles.
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

    # Apply read span mask to span layer for barplot and heatmap
    span_plot = span_ord.copy()
    if read_span_mask is not None:
        span_plot = span_plot.mask(read_span_mask)

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

    mean_span = np.nanmean(span_plot.to_numpy(), axis=0)
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

    span_cmap.set_bad(outside_read_color)
    sns.heatmap(
        span_plot,
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

    # Overlay variant call circles on both heatmaps if data is available
    if variant_call_data is not None:
        plotted_cols = span_ord.columns
        # Convert plotted column names to numeric positions for nearest-neighbour mapping
        try:
            plotted_positions = np.array([float(c) for c in plotted_cols])
        except (ValueError, TypeError):
            plotted_positions = None

        if plotted_positions is not None and len(plotted_positions) > 0:
            # Find variant call columns that have any calls
            call_cols_mask = variant_call_data.isin([1, 2]).any(axis=0)
            call_col_names = variant_call_data.columns[call_cols_mask]
            try:
                call_col_positions = np.array([float(c) for c in call_col_names])
            except (ValueError, TypeError):
                call_col_positions = None

            if call_col_positions is not None and len(call_col_positions) > 0:
                # Map each variant call position to the nearest plotted heatmap column
                insert_idx = np.searchsorted(plotted_positions, call_col_positions)
                insert_idx = np.clip(insert_idx, 0, len(plotted_positions) - 1)
                # Also check the index to the left in case it's closer
                left_idx = np.clip(insert_idx - 1, 0, len(plotted_positions) - 1)
                dist_right = np.abs(plotted_positions[insert_idx] - call_col_positions)
                dist_left = np.abs(plotted_positions[left_idx] - call_col_positions)
                nearest_heatmap_col = np.where(dist_left < dist_right, left_idx, insert_idx)

                call_sub = variant_call_data.loc[:, call_col_names]
                call_sub.index = call_sub.index.astype(str)
                common_rows = [r for r in ordered_index if r in call_sub.index]
                if common_rows:
                    call_ord = call_sub.loc[common_rows].to_numpy()
                    row_index_map = {r: i for i, r in enumerate(ordered_index)}
                    heatmap_row_indices = np.array([row_index_map[r] for r in common_rows])

                    for call_val, marker_color, label in [
                        (1, ref1_marker_color, f"{seq1_label} call"),
                        (2, ref2_marker_color, f"{seq2_label} call"),
                    ]:
                        local_rows, local_cols = np.where(call_ord == call_val)
                        if len(local_rows) == 0:
                            continue
                        plot_y = heatmap_row_indices[local_rows]
                        plot_x = nearest_heatmap_col[local_cols]
                        for ax in (ax1, ax2):
                            ax.scatter(
                                plot_x + 0.5, plot_y + 0.5,
                                c=marker_color, s=variant_marker_size,
                                marker="o", edgecolors="gray", linewidths=0.3, zorder=3,
                                label=label,
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


def plot_delta_hamming_summary(
    subset,
    self_obsm_key: str = "rolling_nn_dist",
    cross_obsm_key: str = "rolling_nn_dist",
    layer_key: str = "nan0_0minus1",
    self_span_layer_key: str = "zero_hamming_distance_spans",
    cross_span_layer_key: str = "cross_sample_zero_hamming_distance_spans",
    delta_span_layer_key: str = "delta_zero_hamming_distance_spans",
    meta_cols: tuple[str, ...] = ("Reference_strand", "Sample"),
    col_cluster: bool = False,
    fill_nn_with_colmax: bool = True,
    fill_layer_value: float = 0.0,
    fill_span_value: float = 0.0,
    drop_all_nan_windows: bool = True,
    max_nan_fraction: float | None = None,
    var_valid_fraction_col: str | None = None,
    var_nan_fraction_col: str | None = None,
    read_span_layer: str | None = "read_span_mask",
    outside_read_color: str = "#bdbdbd",
    nn_nan_color: str = "#bdbdbd",
    span_color: str = "#2ca25f",
    cross_span_color: str = "#e6550d",
    delta_span_color: str = "#756bb1",
    figsize: tuple[float, float] = (30, 24),
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name: str | None = None,
):
    """
    Plot a 2×3 summary: row 1 = self NN, cross NN, signal layer;
    row 2 = self hamming spans, cross hamming spans, delta hamming spans.

    Cluster order is determined by the delta hamming span layer.
    Barplots are drawn above each clustermap.

    Args:
        subset: AnnData subset with all required obsm/layers.
        self_obsm_key: obsm key for within-sample rolling NN distances.
        cross_obsm_key: obsm key for cross-sample rolling NN distances.
        layer_key: Signal layer to plot in top-right panel.
        self_span_layer_key: Layer with within-sample zero-Hamming spans.
        cross_span_layer_key: Layer with cross-sample zero-Hamming spans.
        delta_span_layer_key: Layer with delta (self - cross) zero-Hamming spans.
        meta_cols: Obs columns for row color annotations.
        col_cluster: Cluster columns.
        fill_nn_with_colmax: Fill NN NaNs with per-column max for display.
        fill_layer_value: Fill NaN in signal layer.
        fill_span_value: Fill NaN in span layers.
        drop_all_nan_windows: Drop all-NaN rolling NN windows.
        max_nan_fraction: Max NaN fraction filter for layer columns.
        var_valid_fraction_col: Var column with valid fraction.
        var_nan_fraction_col: Var column with NaN fraction.
        read_span_layer: Layer with read span mask.
        outside_read_color: Color for outside-read positions.
        nn_nan_color: Color for NaN in NN heatmaps.
        span_color: Color for self hamming span (1 values).
        cross_span_color: Color for cross hamming span (1 values).
        delta_span_color: Color for delta hamming span (1 values).
        figsize: Figure size.
        robust: Robust color scaling.
        title: Figure suptitle.
        xtick_step: Spacing between x-tick labels.
        xtick_rotation: X-tick label rotation.
        xtick_fontsize: X-tick label font size.
        save_name: Output path.
    """
    logger.info(
        "Plotting delta hamming summary: self_nn=%s cross_nn=%s delta_span=%s.",
        self_obsm_key,
        cross_obsm_key,
        delta_span_layer_key,
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

    def _format_labels(values):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.number):
            if np.all(np.isfinite(values)) and np.all(np.isclose(values, np.round(values))):
                values = np.round(values).astype(int)
        return [str(v) for v in values]

    # --- Determine row order from delta span layer ---
    delta_span = subset.layers[delta_span_layer_key]
    delta_span = delta_span.toarray() if hasattr(delta_span, "toarray") else np.asarray(delta_span)
    delta_span_df = pd.DataFrame(delta_span, index=subset.obs_names, columns=subset.var_names)
    delta_span_df.index = delta_span_df.index.astype(str)

    # NaN fraction filtering for layer columns
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
            delta_span_df = delta_span_df.loc[:, nan_mask]

    delta_span_filled = delta_span_df.fillna(fill_span_value)
    delta_span_filled.index = delta_span_filled.index.astype(str)

    meta = subset.obs.loc[delta_span_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    delta_cmap = colors.ListedColormap(["white", delta_span_color])
    delta_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], delta_cmap.N)

    g = sns.clustermap(
        delta_span_filled,
        cmap=delta_cmap,
        norm=delta_norm,
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = delta_span_filled.index[row_order]
    plt.close(g.fig)

    # --- Helper to extract + order a span layer ---
    def _span_df(layer_key):
        raw = subset.layers[layer_key]
        raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        df = pd.DataFrame(raw, index=subset.obs_names, columns=subset.var_names)
        df.index = df.index.astype(str)
        if nan_mask is not None:
            df = df.loc[:, nan_mask]
        return df.loc[ordered_index].fillna(fill_span_value)

    self_span_ord = _span_df(self_span_layer_key)
    cross_span_ord = _span_df(cross_span_layer_key)
    delta_span_ord = delta_span_filled.loc[ordered_index]

    # --- Read span mask for layer-resolution panels ---
    read_span_outside = None
    if read_span_layer and read_span_layer in subset.layers:
        rsm = subset.layers[read_span_layer]
        rsm = rsm.toarray() if hasattr(rsm, "toarray") else np.asarray(rsm)
        rsm_df = pd.DataFrame(rsm, index=subset.obs_names, columns=subset.var_names)
        rsm_df.index = rsm_df.index.astype(str)
        if nan_mask is not None:
            rsm_df = rsm_df.loc[:, nan_mask]
        read_span_outside = rsm_df.loc[ordered_index].to_numpy() == 0

    # Apply read span mask to span layers (NaN outside read → grey in heatmap, excluded from barplot)
    self_span_plot = self_span_ord.copy()
    cross_span_plot = cross_span_ord.copy()
    delta_span_plot = delta_span_ord.copy()
    if read_span_outside is not None:
        self_span_plot = self_span_plot.mask(read_span_outside)
        cross_span_plot = cross_span_plot.mask(read_span_outside)
        delta_span_plot = delta_span_plot.mask(read_span_outside)

    # --- NN data ---
    def _nn_df(obsm_key):
        X = subset.obsm[obsm_key]
        valid = ~np.all(np.isnan(X), axis=1)
        df = pd.DataFrame(X, index=subset.obs_names)
        df.index = df.index.astype(str)
        if drop_all_nan_windows:
            df = df.loc[:, ~df.isna().all(axis=0)]
        col_max = df.max(axis=0, skipna=True).fillna(0)
        df_cluster = df.fillna(col_max)
        if fill_nn_with_colmax:
            df_display = df_cluster
        else:
            df_display = df.copy()
        return df_display.loc[ordered_index]

    self_nn_ord = _nn_df(self_obsm_key)
    cross_nn_ord = _nn_df(cross_obsm_key)

    # --- Signal layer ---
    layer_raw = subset.layers[layer_key]
    layer_raw = layer_raw.toarray() if hasattr(layer_raw, "toarray") else np.asarray(layer_raw)
    layer_df = pd.DataFrame(layer_raw, index=subset.obs_names, columns=subset.var_names)
    layer_df.index = layer_df.index.astype(str)
    if nan_mask is not None:
        layer_df = layer_df.loc[:, nan_mask]

    layer_ord = layer_df.loc[ordered_index]
    layer_plot = layer_ord.fillna(fill_layer_value)
    if read_span_outside is not None:
        layer_plot = layer_plot.mask(read_span_outside)

    # --- Figure layout: 5 rows × 6 cols ---
    # row 0: barplots for top row
    # row 1: heatmaps (self NN, cross NN, signal)
    # row 2: spacer
    # row 3: barplots for bottom row
    # row 4: heatmaps (self span, cross span, delta span)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        5,
        6,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05],
        height_ratios=[1, 8, 0.8, 1, 8],
        wspace=0.2,
        hspace=0.05,
    )

    # Row 1 heatmaps + colorbars
    ax_self_nn = fig.add_subplot(gs[1, 0])
    ax_self_nn_cbar = fig.add_subplot(gs[1, 1])
    ax_cross_nn = fig.add_subplot(gs[1, 2])
    ax_cross_nn_cbar = fig.add_subplot(gs[1, 3])
    ax_signal = fig.add_subplot(gs[1, 4])
    ax_signal_cbar = fig.add_subplot(gs[1, 5])

    # Row 1 barplots
    ax_self_nn_bar = fig.add_subplot(gs[0, 0], sharex=ax_self_nn)
    ax_cross_nn_bar = fig.add_subplot(gs[0, 2], sharex=ax_cross_nn)
    ax_signal_bar = fig.add_subplot(gs[0, 4], sharex=ax_signal)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 3]).axis("off")
    fig.add_subplot(gs[0, 5]).axis("off")

    # Spacer row
    for col in range(6):
        fig.add_subplot(gs[2, col]).axis("off")

    # Row 2 heatmaps + colorbars
    ax_self_span = fig.add_subplot(gs[4, 0])
    ax_self_span_cbar = fig.add_subplot(gs[4, 1])
    ax_cross_span = fig.add_subplot(gs[4, 2])
    ax_cross_span_cbar = fig.add_subplot(gs[4, 3])
    ax_delta_span = fig.add_subplot(gs[4, 4])
    ax_delta_span_cbar = fig.add_subplot(gs[4, 5])

    # Row 2 barplots
    ax_self_span_bar = fig.add_subplot(gs[3, 0], sharex=ax_self_span)
    ax_cross_span_bar = fig.add_subplot(gs[3, 2], sharex=ax_cross_span)
    ax_delta_span_bar = fig.add_subplot(gs[3, 4], sharex=ax_delta_span)
    fig.add_subplot(gs[3, 1]).axis("off")
    fig.add_subplot(gs[3, 3]).axis("off")
    fig.add_subplot(gs[3, 5]).axis("off")

    # --- Row 1: NN + signal barplots ---
    mean_self_nn = np.nanmean(self_nn_ord.to_numpy(), axis=0)
    mean_cross_nn = np.nanmean(cross_nn_ord.to_numpy(), axis=0)
    nn_y_max = float(np.nanmax(np.concatenate([mean_self_nn, mean_cross_nn])))
    if not np.isfinite(nn_y_max) or nn_y_max <= 0:
        nn_y_max = None
    clean_barplot(
        ax_self_nn_bar,
        mean_self_nn,
        "Self NN",
        y_max=nn_y_max,
        y_label="Mean distance",
        y_ticks=None,
    )
    clean_barplot(
        ax_cross_nn_bar,
        mean_cross_nn,
        "Cross NN",
        y_max=nn_y_max,
        y_label="Mean distance",
        y_ticks=None,
    )
    methylation_fraction = _methylation_fraction_for_layer(layer_ord.to_numpy(), layer_key)
    clean_barplot(
        ax_signal_bar,
        methylation_fraction,
        layer_key,
        y_max=1.0,
        y_label="Methylation fraction",
        y_ticks=[0.0, 0.5, 1.0],
    )

    # --- Row 1: NN + signal heatmaps ---
    nn_cmap = plt.get_cmap("viridis").copy()
    nn_cmap.set_bad(nn_nan_color)

    sns.heatmap(
        self_nn_ord,
        ax=ax_self_nn,
        cmap=nn_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_self_nn_cbar,
    )
    sns.heatmap(
        cross_nn_ord,
        ax=ax_cross_nn,
        cmap=nn_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_cross_nn_cbar,
    )

    layer_cmap = plt.get_cmap("coolwarm").copy()
    if read_span_outside is not None:
        layer_cmap.set_bad(outside_read_color)
    sns.heatmap(
        layer_plot,
        ax=ax_signal,
        cmap=layer_cmap,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_signal_cbar,
    )

    # NN x-tick labels
    for ax_nn, obsm_key in ((ax_self_nn, self_obsm_key), (ax_cross_nn, cross_obsm_key)):
        label_source = subset.uns.get(f"{obsm_key}_centers")
        if label_source is None:
            label_source = subset.uns.get(f"{obsm_key}_starts")
        if label_source is not None:
            _apply_xticks(ax_nn, _format_labels(np.asarray(label_source)), xtick_step)

    _apply_xticks(ax_signal, [str(x) for x in layer_plot.columns], xtick_step)

    # --- Row 2: span barplots (matched y-scale, using read-span-masked data) ---
    mean_self_span = np.nanmean(self_span_plot.to_numpy(), axis=0)
    mean_cross_span = np.nanmean(cross_span_plot.to_numpy(), axis=0)
    mean_delta_span = np.nanmean(delta_span_plot.to_numpy(), axis=0)
    span_y_max = float(
        np.nanmax(np.concatenate([mean_self_span, mean_cross_span, mean_delta_span]))
    )
    if not np.isfinite(span_y_max) or span_y_max <= 0:
        span_y_max = 1.0
    # Round up to nearest 0.1 for clean ticks
    span_y_max = np.ceil(span_y_max * 10) / 10
    span_y_ticks = [0.0, span_y_max / 2, span_y_max]
    clean_barplot(
        ax_self_span_bar,
        mean_self_span,
        "Self spans",
        y_max=span_y_max,
        y_label="Span fraction",
        y_ticks=span_y_ticks,
    )
    clean_barplot(
        ax_cross_span_bar,
        mean_cross_span,
        "Cross spans",
        y_max=span_y_max,
        y_label="Span fraction",
        y_ticks=span_y_ticks,
    )
    clean_barplot(
        ax_delta_span_bar,
        mean_delta_span,
        "Delta spans",
        y_max=span_y_max,
        y_label="Span fraction",
        y_ticks=span_y_ticks,
    )

    # --- Row 2: span heatmaps (read-span-masked, outside-read = grey) ---
    self_span_cmap = colors.ListedColormap(["white", span_color])
    self_span_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], self_span_cmap.N)
    self_span_cmap.set_bad(outside_read_color)
    cross_span_cmap = colors.ListedColormap(["white", cross_span_color])
    cross_span_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cross_span_cmap.N)
    cross_span_cmap.set_bad(outside_read_color)
    delta_cmap.set_bad(outside_read_color)

    sns.heatmap(
        self_span_plot,
        ax=ax_self_span,
        cmap=self_span_cmap,
        norm=self_span_norm,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_self_span_cbar,
    )
    sns.heatmap(
        cross_span_plot,
        ax=ax_cross_span,
        cmap=cross_span_cmap,
        norm=cross_span_norm,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_cross_span_cbar,
    )
    sns.heatmap(
        delta_span_plot,
        ax=ax_delta_span,
        cmap=delta_cmap,
        norm=delta_norm,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax_delta_span_cbar,
    )

    col_labels = [str(x) for x in self_span_ord.columns]
    for ax in (ax_self_span, ax_cross_span, ax_delta_span):
        _apply_xticks(ax, col_labels, xtick_step)

    if title:
        fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved delta hamming summary plot to %s.", fname)
    else:
        plt.show()

    plt.close(fig)
    return ordered_index


def plot_span_length_distributions(
    subset,
    self_span_layer_key: str = "zero_hamming_distance_spans",
    cross_span_layer_key: str = "cross_sample_zero_hamming_distance_spans",
    delta_span_layer_key: str = "delta_zero_hamming_distance_spans",
    read_span_layer: str | None = "read_span_mask",
    bins: int = 30,
    self_color: str = "#2ca25f",
    cross_color: str = "#e6550d",
    delta_color: str = "#756bb1",
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Overlay probability histograms of contiguous span lengths from three layers.

    Span length is measured in base-pair coordinates using ``subset.var_names``.
    Positions outside the valid read span (where ``read_span_layer == 0``) are
    excluded before detecting contiguous runs.

    Args:
        subset: AnnData subset containing the span layers.
        self_span_layer_key: Layer with within-sample zero-Hamming spans.
        cross_span_layer_key: Layer with cross-sample zero-Hamming spans.
        delta_span_layer_key: Layer with delta (self - cross) spans.
        read_span_layer: Layer with read span mask; 0 = outside read.
        bins: Number of histogram bins.
        self_color: Histogram color for self spans.
        cross_color: Histogram color for cross spans.
        delta_color: Histogram color for delta spans.
        figsize: Figure size.
        title: Figure title.
        save_name: Output path.
    """

    def _extract_span_lengths(layer_arr, positions, read_mask):
        """Extract lengths (in bp) of contiguous runs of 1 in each row."""
        lengths = []
        for i in range(layer_arr.shape[0]):
            row = layer_arr[i].copy()
            if read_mask is not None:
                row[~read_mask[i]] = 0
            # Find contiguous runs of 1
            diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if e > s:
                    span_bp = float(positions[e - 1] - positions[s])
                    if span_bp > 0:
                        lengths.append(span_bp)
        return np.array(lengths, dtype=float)

    # Parse genomic positions from var_names
    try:
        positions = np.array(subset.var_names, dtype=float)
    except (ValueError, TypeError):
        positions = np.arange(subset.n_vars, dtype=float)

    # Read span mask
    read_mask = None
    if read_span_layer and read_span_layer in subset.layers:
        rsm = subset.layers[read_span_layer]
        rsm = rsm.toarray() if hasattr(rsm, "toarray") else np.asarray(rsm)
        read_mask = rsm.astype(bool)

    entries = []
    for layer_key, color, label in (
        (self_span_layer_key, self_color, "Self"),
        (cross_span_layer_key, cross_color, "Cross"),
        (delta_span_layer_key, delta_color, "Delta"),
    ):
        if layer_key not in subset.layers:
            continue
        arr = subset.layers[layer_key]
        arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
        span_lengths = _extract_span_lengths(arr, positions, read_mask)
        entries.append((label, color, span_lengths))

    if not entries:
        logger.warning("No span layers found for span length distribution plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for label, color, span_lengths in entries:
        if len(span_lengths) == 0:
            ax.axhline(0, color=color, label=f"{label} (n=0)")
            continue
        ax.hist(
            span_lengths,
            bins=bins,
            density=True,
            alpha=0.5,
            color=color,
            label=f"{label} (n={len(span_lengths)})",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Span length (bp)")
    ax.set_ylabel("Probability density")
    ax.legend()

    if title:
        ax.set_title(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved span length distribution plot to %s.", fname)
    else:
        plt.show()

    plt.close(fig)


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


def plot_segment_length_histogram(
    raw_lengths: np.ndarray,
    filtered_lengths: np.ndarray,
    bins: int = 30,
    title: str | None = None,
    raw_label: str = "All segments",
    filtered_label: str = "Filtered segments",
    figsize: tuple[float, float] = (8, 4),
    density: bool = True,
    save_name: str | None = None,
):
    """
    Plot an overlay histogram of segment lengths for raw vs filtered spans.

    Args:
        raw_lengths: Array of raw segment lengths.
        filtered_lengths: Array of filtered segment lengths.
        bins: Number of histogram bins.
        title: Optional plot title.
        raw_label: Label for raw segment histogram.
        filtered_label: Label for filtered segment histogram.
        figsize: Size of the matplotlib figure.
        density: If True, plot probabilities instead of counts.
        save_name: Optional output path for saving the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if raw_lengths.size:
        ax.hist(
            raw_lengths,
            bins=bins,
            alpha=0.6,
            label=raw_label,
            edgecolor="black",
            density=density,
        )
    if filtered_lengths.size:
        ax.hist(
            filtered_lengths,
            bins=bins,
            alpha=0.6,
            label=filtered_label,
            edgecolor="black",
            density=density,
        )
    ax.set_xlabel("Segment length")
    ax.set_ylabel("Probability" if density else "Count")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_name is not None:
        fname = os.path.join(save_name)
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved segment length histogram to %s.", fname)
    else:
        plt.show()

    plt.close(fig)
    return fig


def plot_hamming_span_trio(
    subset,
    self_span_layer_key: str = "zero_hamming_distance_spans",
    cross_span_layer_key: str = "cross_sample_zero_hamming_distance_spans",
    delta_span_layer_key: str = "delta_zero_hamming_distance_spans",
    read_span_layer: str | None = "read_span_mask",
    outside_read_color: str = "#bdbdbd",
    span_color: str = "#2ca25f",
    cross_span_color: str = "#e6550d",
    delta_span_color: str = "#756bb1",
    figsize: tuple[float, float] = (16, 8),
    robust: bool = True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    variant_call_data: "pd.DataFrame | None" = None,
    seq1_label: str = "seq1",
    seq2_label: str = "seq2",
    ref1_marker_color: str = "white",
    ref2_marker_color: str = "black",
    variant_marker_size: float = 4.0,
    classification_obs_col: str | None = "chimeric_by_mod_hamming_distance",
    classification_true_color: str = "#000000",
    classification_false_color: str = "#f0f0f0",
    classification_panel_title: str = "Mod-hamming chimera",
    save_name: str | None = None,
):
    """
    Plot a 1×3 trio of hamming span clustermaps (self, cross, delta) with no
    column subsetting, optionally overlaying variant call circles.

    Row order is determined by hierarchical clustering on the delta span layer.
    A barplot showing per-column mean span fraction is drawn above each panel.
    """
    logger.info(
        "Plotting hamming span trio: self=%s, cross=%s, delta=%s.",
        self_span_layer_key,
        cross_span_layer_key,
        delta_span_layer_key,
    )

    def _to_df(layer_key):
        arr = subset.layers[layer_key]
        arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
        df = pd.DataFrame(arr, index=subset.obs_names.astype(str), columns=subset.var_names)
        return df

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

    delta_df = _to_df(delta_span_layer_key)
    self_df = _to_df(self_span_layer_key)
    cross_df = _to_df(cross_span_layer_key)

    # Drop columns that are all-zero/NaN across all three layers
    has_data = (
        delta_df.fillna(0).any(axis=0)
        | self_df.fillna(0).any(axis=0)
        | cross_df.fillna(0).any(axis=0)
    )
    delta_df = delta_df.loc[:, has_data]
    self_df = self_df.loc[:, has_data]
    cross_df = cross_df.loc[:, has_data]

    # Hierarchical clustering on delta layer for row ordering
    delta_filled = delta_df.fillna(0)
    g = sns.clustermap(
        delta_filled,
        col_cluster=False,
        row_cluster=True,
        xticklabels=False,
        yticklabels=False,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = delta_filled.index[row_order]
    plt.close(g.fig)

    # Read span mask
    read_span_mask = None
    if read_span_layer and read_span_layer in subset.layers:
        rsm = subset.layers[read_span_layer]
        rsm = rsm.toarray() if hasattr(rsm, "toarray") else np.asarray(rsm)
        rsm_df = pd.DataFrame(rsm, index=subset.obs_names.astype(str), columns=subset.var_names)
        rsm_df = rsm_df.loc[:, has_data]
        read_span_mask = rsm_df.loc[ordered_index].to_numpy() == 0

    panels = [
        (self_df, span_color, self_span_layer_key, "Self spans"),
        (cross_df, cross_span_color, cross_span_layer_key, "Cross spans"),
        (delta_df, delta_span_color, delta_span_layer_key, "Delta spans"),
    ]

    has_classification = bool(classification_obs_col) and classification_obs_col in subset.obs

    fig = plt.figure(figsize=figsize)
    if has_classification:
        gs = fig.add_gridspec(
            2,
            4,
            height_ratios=[1, 6],
            width_ratios=[1, 1, 1, 0.12],
            wspace=0.08,
            hspace=0.05,
        )
    else:
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1, 6],
            wspace=0.08,
            hspace=0.05,
        )

    axes = []
    for col_idx, (df, color, layer_name, panel_title) in enumerate(panels):
        ax_bar = fig.add_subplot(gs[0, col_idx])
        ax_heat = fig.add_subplot(gs[1, col_idx])
        axes.append(ax_heat)

        ordered = df.loc[ordered_index].fillna(0)
        plot_data = ordered.copy()
        if read_span_mask is not None:
            plot_data = plot_data.mask(read_span_mask)

        cmap = colors.ListedColormap(["white", color])
        norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        cmap.set_bad(outside_read_color)

        mean_span = np.nanmean(plot_data.to_numpy(), axis=0)
        clean_barplot(ax_bar, mean_span, panel_title, y_max=1.0, y_label="Span frac", y_ticks=[0.0, 0.5, 1.0])

        sns.heatmap(
            plot_data,
            ax=ax_heat,
            cmap=cmap,
            norm=norm,
            xticklabels=False,
            yticklabels=False,
            robust=robust,
            cbar=False,
        )
        _apply_xticks(ax_heat, [str(x) for x in ordered.columns], xtick_step)

    if has_classification:
        class_values = subset.obs.loc[ordered_index, classification_obs_col].astype(bool).astype(int)
        class_df = pd.DataFrame(
            {classification_panel_title: class_values.to_numpy()},
            index=ordered_index,
        )
        class_cmap = colors.ListedColormap([classification_false_color, classification_true_color])
        class_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], class_cmap.N)

        ax_class_top = fig.add_subplot(gs[0, 3])
        ax_class_top.axis("off")
        ax_class = fig.add_subplot(gs[1, 3], sharey=axes[-1])
        sns.heatmap(
            class_df,
            ax=ax_class,
            cmap=class_cmap,
            norm=class_norm,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            robust=robust,
        )

    # Overlay variant call circles on all three panels
    if variant_call_data is not None:
        plotted_cols = list(self_df.loc[ordered_index].columns)
        plotted_col_set = set(plotted_cols)
        col_to_idx = {c: i for i, c in enumerate(plotted_cols)}

        call_cols_mask = variant_call_data.isin([1, 2]).any(axis=0)
        call_col_names = variant_call_data.columns[call_cols_mask]

        # Since no column subsetting, try exact match first; fall back to nearest
        try:
            plotted_positions = np.array([float(c) for c in plotted_cols])
            call_col_positions = np.array([float(c) for c in call_col_names])
            use_searchsorted = True
        except (ValueError, TypeError):
            use_searchsorted = False

        heatmap_col_indices = {}
        for cn in call_col_names:
            if cn in col_to_idx:
                heatmap_col_indices[cn] = col_to_idx[cn]
            elif use_searchsorted:
                pos = float(cn)
                idx = np.searchsorted(plotted_positions, pos)
                idx = np.clip(idx, 0, len(plotted_positions) - 1)
                left = max(0, idx - 1)
                if abs(plotted_positions[left] - pos) < abs(plotted_positions[idx] - pos):
                    idx = left
                heatmap_col_indices[cn] = idx

        if heatmap_col_indices:
            active_cols = [c for c in call_col_names if c in heatmap_col_indices]
            call_sub = variant_call_data.loc[:, active_cols]
            call_sub.index = call_sub.index.astype(str)
            common_rows = [r for r in ordered_index if r in call_sub.index]
            if common_rows:
                call_ord = call_sub.loc[common_rows].to_numpy()
                row_index_map = {r: i for i, r in enumerate(ordered_index)}
                heatmap_row_indices = np.array([row_index_map[r] for r in common_rows])
                col_idx_arr = np.array([heatmap_col_indices[c] for c in active_cols])

                for call_val, marker_color, label in [
                    (1, ref1_marker_color, f"{seq1_label} call"),
                    (2, ref2_marker_color, f"{seq2_label} call"),
                ]:
                    local_rows, local_cols = np.where(call_ord == call_val)
                    if len(local_rows) == 0:
                        continue
                    plot_y = heatmap_row_indices[local_rows]
                    plot_x = col_idx_arr[local_cols]
                    for ax in axes:
                        ax.scatter(
                            plot_x + 0.5, plot_y + 0.5,
                            c=marker_color, s=variant_marker_size,
                            marker="o", edgecolors="gray", linewidths=0.3, zorder=3,
                            label=label,
                        )

        # Add legend to rightmost axis
        handles, labels = axes[-1].get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                unique_handles.append(h)
                unique_labels.append(l)
        if unique_handles:
            legend_x_anchor = 1.3 if has_classification else 1.02
            axes[-1].legend(
                unique_handles, unique_labels,
                loc="upper left", bbox_to_anchor=(legend_x_anchor, 1.0),
                fontsize=8, framealpha=0.9,
            )

    if title:
        fig.suptitle(title, fontsize=12)

    if save_name is not None:
        fname = os.path.join(save_name)
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        logger.info("Saved hamming span trio to %s.", fname)
    else:
        plt.show()

    plt.close(fig)
    return ordered_index
