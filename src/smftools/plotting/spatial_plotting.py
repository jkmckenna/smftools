from __future__ import annotations

import os
from math import floor
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger
from smftools.optional_imports import require
from smftools.plotting.plotting_utils import (
    _fixed_tick_positions,
    _layer_to_numpy,
    _layer_to_numpy_np,
    _methylation_fraction_for_layer,
    _select_labels,
    clean_barplot,
    make_row_colors,
)

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
        figsize: Figure size for the combined plot.
        right_panel_var_mask: Optional boolean mask over ``subset.var`` for the right panel.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
    colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
    sns = require("seaborn", extra="plotting", purpose="plot styling")

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

    X_df_filled = X_df.copy()
    if fill_nn_with_colmax:
        col_max = X_df_filled.max(axis=0, skipna=True)
        X_df_filled = X_df_filled.fillna(col_max)

    X_df_filled.index = X_df_filled.index.astype(str)

    meta = subset.obs.loc[X_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

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

    X_ord = X_df_filled.loc[ordered_index]

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

    sns.heatmap(
        X_ord,
        ax=ax1,
        cmap="viridis",
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
    plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
    colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
    sns = require("seaborn", extra="plotting", purpose="plot styling")

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
    plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
    sns = require("seaborn", extra="plotting", purpose="plot styling")

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


from smftools.parallel_utils import resolve_n_jobs as _resolve_n_jobs


def _combined_raw_clustermap_one_group(args: dict):
    """Render one (ref, sample) combined raw clustermap. Module-level for pickling."""
    try:
        import matplotlib
        import numpy as np
        import scipy.cluster.hierarchy as sch
        from pathlib import Path

        from smftools.optional_imports import require
        from smftools.plotting.plotting_utils import (
            _fixed_tick_positions,
            _layer_to_numpy_np,
            _methylation_fraction_for_layer,
            clean_barplot,
        )

        # Use headless backend — no display connection in worker processes.
        matplotlib.use("Agg")
        _plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
        _grid_spec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
        _sns = require("seaborn", extra="plotting", purpose="plot styling")

        ref: str = args["ref"]
        sample: str = args["sample"]
        display_sample: str = args["display_sample"]
        layer_data: dict = args["layer_data"]  # {layer_name: np.ndarray (n_reads, n_positions)}
        any_c_sites: np.ndarray = args["any_c_sites"]
        gpc_sites: np.ndarray = args["gpc_sites"]
        cpg_sites: np.ndarray = args["cpg_sites"]
        any_a_sites: np.ndarray = args["any_a_sites"]
        any_c_labels: np.ndarray = args["any_c_labels"]
        gpc_labels: np.ndarray = args["gpc_labels"]
        cpg_labels: np.ndarray = args["cpg_labels"]
        any_a_labels: np.ndarray = args["any_a_labels"]
        bins_np: dict = args["bins_np"]  # {bin_label: bool ndarray over group rows}
        obs_sort_values: np.ndarray = args["obs_sort_values"]  # for sort_by="obs:<col>", else None
        total_reads: int = args["total_reads"]
        include_any_c: bool = args["include_any_c"]
        include_any_a: bool = args["include_any_a"]
        sort_by: str = args["sort_by"]
        fill_nan_strategy: str = args["fill_nan_strategy"]
        fill_nan_value: float = args["fill_nan_value"]
        layer_c: str = args["layer_c"]
        layer_gpc: str = args["layer_gpc"]
        layer_cpg: str = args["layer_cpg"]
        layer_a: str = args["layer_a"]
        cmap_c: str = args["cmap_c"]
        cmap_gpc: str = args["cmap_gpc"]
        cmap_cpg: str = args["cmap_cpg"]
        cmap_a: str = args["cmap_a"]
        n_xticks_any_c: int = args["n_xticks_any_c"]
        n_xticks_gpc: int = args["n_xticks_gpc"]
        n_xticks_cpg: int = args["n_xticks_cpg"]
        n_xticks_any_a: int = args["n_xticks_any_a"]
        xtick_rotation: int = args["xtick_rotation"]
        xtick_fontsize: int = args["xtick_fontsize"]
        save_path = args["save_path"]  # str or None

        num_any_c = len(any_c_sites)
        num_gpc = len(gpc_sites)
        num_cpg = len(cpg_sites)
        num_any_a = len(any_a_sites)

        stacked_any_c, stacked_gpc, stacked_cpg, stacked_any_a = [], [], [], []
        stacked_any_c_raw, stacked_gpc_raw, stacked_cpg_raw, stacked_any_a_raw = [], [], [], []
        row_labels, bin_labels_list, bin_boundaries = [], [], []
        percentages: dict = {}
        last_idx = 0

        for bin_label, bin_filter in bins_np.items():
            bin_row_idx = np.where(bin_filter)[0]
            num_reads = len(bin_row_idx)
            if num_reads == 0:
                percentages[bin_label] = 0.0
                continue

            percent_reads = (num_reads / total_reads) * 100
            percentages[bin_label] = percent_reads

            # Compute sort order for this bin
            if sort_by.startswith("obs:") and obs_sort_values is not None:
                order = np.argsort(obs_sort_values[bin_row_idx])
            elif sort_by == "gpc" and num_gpc > 0:
                m = _layer_to_numpy_np(layer_data[layer_gpc][bin_row_idx], gpc_sites, fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
                order = sch.leaves_list(sch.linkage(m, method="ward"))
            elif sort_by == "cpg" and num_cpg > 0:
                m = _layer_to_numpy_np(layer_data[layer_cpg][bin_row_idx], cpg_sites, fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
                order = sch.leaves_list(sch.linkage(m, method="ward"))
            elif sort_by == "c" and num_any_c > 0:
                m = _layer_to_numpy_np(layer_data[layer_c][bin_row_idx], any_c_sites, fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
                order = sch.leaves_list(sch.linkage(m, method="ward"))
            elif sort_by == "gpc_cpg":
                m = _layer_to_numpy_np(layer_data[layer_gpc][bin_row_idx], None, fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
                order = sch.leaves_list(sch.linkage(m, method="ward"))
            elif sort_by == "a" and num_any_a > 0:
                m = _layer_to_numpy_np(layer_data[layer_a][bin_row_idx], any_a_sites, fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
                order = sch.leaves_list(sch.linkage(m, method="ward"))
            else:
                order = np.arange(num_reads)

            ordered_idx = bin_row_idx[order]

            kw = dict(fill_nan_strategy=fill_nan_strategy, fill_nan_value=fill_nan_value)
            kw_raw = dict(fill_nan_strategy="none", fill_nan_value=fill_nan_value)

            if include_any_c and num_any_c > 0:
                stacked_any_c.append(_layer_to_numpy_np(layer_data[layer_c][ordered_idx], any_c_sites, **kw))
                stacked_any_c_raw.append(_layer_to_numpy_np(layer_data[layer_c][ordered_idx], any_c_sites, **kw_raw))
            if include_any_c and num_gpc > 0:
                stacked_gpc.append(_layer_to_numpy_np(layer_data[layer_gpc][ordered_idx], gpc_sites, **kw))
                stacked_gpc_raw.append(_layer_to_numpy_np(layer_data[layer_gpc][ordered_idx], gpc_sites, **kw_raw))
            if include_any_c and num_cpg > 0:
                stacked_cpg.append(_layer_to_numpy_np(layer_data[layer_cpg][ordered_idx], cpg_sites, **kw))
                stacked_cpg_raw.append(_layer_to_numpy_np(layer_data[layer_cpg][ordered_idx], cpg_sites, **kw_raw))
            if include_any_a and num_any_a > 0:
                stacked_any_a.append(_layer_to_numpy_np(layer_data[layer_a][ordered_idx], any_a_sites, **kw))
                stacked_any_a_raw.append(_layer_to_numpy_np(layer_data[layer_a][ordered_idx], any_a_sites, **kw_raw))

            row_labels.extend([bin_label] * num_reads)
            bin_labels_list.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
            last_idx += num_reads
            bin_boundaries.append(last_idx)

        blocks = []

        if include_any_c and stacked_any_c:
            any_c_matrix = np.vstack(stacked_any_c)
            any_c_matrix_raw = np.vstack(stacked_any_c_raw)
            gpc_matrix = np.vstack(stacked_gpc) if stacked_gpc else np.empty((0, 0))
            gpc_matrix_raw = np.vstack(stacked_gpc_raw) if stacked_gpc_raw else np.empty((0, 0))
            cpg_matrix = np.vstack(stacked_cpg) if stacked_cpg else np.empty((0, 0))
            cpg_matrix_raw = np.vstack(stacked_cpg_raw) if stacked_cpg_raw else np.empty((0, 0))
            mean_any_c = _methylation_fraction_for_layer(any_c_matrix_raw, layer_c) if any_c_matrix_raw.size else None
            mean_gpc = _methylation_fraction_for_layer(gpc_matrix_raw, layer_gpc) if gpc_matrix_raw.size else None
            mean_cpg = _methylation_fraction_for_layer(cpg_matrix_raw, layer_cpg) if cpg_matrix_raw.size else None
            if any_c_matrix.size:
                blocks.append(dict(name="c", matrix=any_c_matrix, mean=mean_any_c, labels=any_c_labels, cmap=cmap_c, n_xticks=n_xticks_any_c, title="any C site Modification Signal"))
            if gpc_matrix.size:
                blocks.append(dict(name="gpc", matrix=gpc_matrix, mean=mean_gpc, labels=gpc_labels, cmap=cmap_gpc, n_xticks=n_xticks_gpc, title="GpC Modification Signal"))
            if cpg_matrix.size:
                blocks.append(dict(name="cpg", matrix=cpg_matrix, mean=mean_cpg, labels=cpg_labels, cmap=cmap_cpg, n_xticks=n_xticks_cpg, title="CpG Modification Signal"))

        if include_any_a and stacked_any_a:
            any_a_matrix = np.vstack(stacked_any_a)
            any_a_matrix_raw = np.vstack(stacked_any_a_raw)
            mean_any_a = _methylation_fraction_for_layer(any_a_matrix_raw, layer_a) if any_a_matrix_raw.size else None
            if any_a_matrix.size:
                blocks.append(dict(name="a", matrix=any_a_matrix, mean=mean_any_a, labels=any_a_labels, cmap=cmap_a, n_xticks=n_xticks_any_a, title="any A site Modification Signal"))

        if not blocks:
            return None

        gs_dim = len(blocks)
        fig = _plt.figure(figsize=(5.5 * gs_dim, 11))
        gs = _grid_spec.GridSpec(2, gs_dim, height_ratios=[1, 6], hspace=0.02)
        fig.suptitle(f"{display_sample} - {ref} - {total_reads} reads", fontsize=14, y=0.97)

        axes_heat = [fig.add_subplot(gs[1, i]) for i in range(gs_dim)]
        axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(gs_dim)]

        for i, blk in enumerate(blocks):
            labels = np.asarray(blk["labels"], dtype=str)
            clean_barplot(axes_bar[i], blk["mean"], blk["title"])
            _sns.heatmap(blk["matrix"], cmap=blk["cmap"], ax=axes_heat[i], yticklabels=False, cbar=False)
            tick_pos = _fixed_tick_positions(len(labels), blk["n_xticks"])
            axes_heat[i].set_xticks(tick_pos)
            axes_heat[i].set_xticklabels(labels[tick_pos], rotation=xtick_rotation, fontsize=xtick_fontsize)
            for boundary in bin_boundaries[:-1]:
                axes_heat[i].axhline(y=boundary, color="black", linewidth=2)
            axes_heat[i].set_xlabel("Position", fontsize=9)

        _plt.tight_layout()

        out_file = None
        if save_path is not None:
            safe_name = (
                f"{ref}__{display_sample}"
                .replace("=", "")
                .replace("__", "_")
                .replace(",", "_")
                .replace(" ", "_")
            )
            out_file = Path(save_path) / f"{safe_name}.png"
            fig.savefig(out_file, dpi=300)
            _plt.close(fig)
        else:
            _plt.show()

        return {
            "sample": str(sample),
            "ref": str(ref),
            "row_labels": row_labels,
            "bin_labels": bin_labels_list,
            "bin_boundaries": bin_boundaries,
            "percentages": percentages,
            "output_path": str(out_file) if out_file is not None else None,
        }

    except Exception:
        import traceback

        traceback.print_exc()
        return None


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
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
    n_jobs: int = 1,
    omit_chimeric_reads: bool = False,
):
    """
    Plot stacked heatmaps + per-position mean barplots for C, GpC, CpG, and optional A.

    Key fixes vs old version:
      - order computed ONCE per bin, applied to all matrices
      - no hard-coded axes indices
      - NaNs excluded from methylation denominators
      - var_names not forced to int
      - fixed count of x tick labels per block (controllable)
      - optional NaN fill strategy for clustering/plotting (in-memory only)
      - adata.uns updated once at end

    Returns
    -------
    results : list[dict]
        One entry per (sample, ref) plot with output metadata.
    """
    logger.info("Plotting combined raw clustermaps.")
    if fill_nan_strategy not in {"none", "value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")

    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    base_set = set(mod_target_bases)
    include_any_c = any(b in {"C", "CpG", "GpC"} for b in base_set)
    include_any_a = "A" in base_set

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    # ------------------------------------------------------------------
    # Lazy group generator — yields one group's arrays at a time
    # ------------------------------------------------------------------
    def _iter_group_args():
        for ref in adata.obs[reference_col].cat.categories:
            for sample in adata.obs[sample_col].cat.categories:
                display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample

                qmask = _mask_or_true("read_quality", (lambda s: s >= float(min_quality)) if min_quality is not None else (lambda s: pd.Series(True, index=s.index)))
                lm_mask = _mask_or_true("mapped_length", (lambda s: s >= float(min_length)) if min_length is not None else (lambda s: pd.Series(True, index=s.index)))
                lrr_mask = _mask_or_true("mapped_length_to_reference_length_ratio", (lambda s: s >= float(min_mapped_length_to_reference_length_ratio)) if min_mapped_length_to_reference_length_ratio is not None else (lambda s: pd.Series(True, index=s.index)))
                demux_mask = _mask_or_true("demux_type", (lambda s: s.astype("string").isin(list(demux_types))) if demux_types is not None else (lambda s: pd.Series(True, index=s.index)))

                chimeric_mask = _mask_or_true("chimeric_variant_sites", lambda s: ~s.astype(bool)) if omit_chimeric_reads else pd.Series(True, index=adata.obs.index)
                row_mask = (adata.obs[reference_col] == ref) & (adata.obs[sample_col] == sample) & qmask & lm_mask & lrr_mask & demux_mask & chimeric_mask

                if not bool(row_mask.any()):
                    logger.warning("No reads for %s - %s after read quality and length filtering.", display_sample, ref)
                    continue

                try:
                    subset = adata[row_mask, :].copy()

                    if min_position_valid_fraction is not None:
                        valid_key = f"{ref}_valid_fraction"
                        if valid_key in subset.var:
                            v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                            col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                            if col_mask.any():
                                subset = subset[:, col_mask].copy()
                            else:
                                logger.warning("No positions left after valid_fraction filter for %s - %s.", display_sample, ref)
                                continue

                    if subset.shape[0] == 0:
                        logger.warning("No reads left after filtering for %s - %s.", display_sample, ref)
                        continue

                    # Site indices and labels
                    any_c_sites = gpc_sites = cpg_sites = np.array([], dtype=int)
                    any_a_sites = np.array([], dtype=int)
                    any_c_labels = gpc_labels = cpg_labels = any_a_labels = np.array([], dtype=str)

                    if include_any_c:
                        any_c_sites = np.where(subset.var.get(f"{ref}_C_site", False).values)[0]
                        gpc_sites = np.where(subset.var.get(f"{ref}_GpC_site", False).values)[0]
                        cpg_sites = np.where(subset.var.get(f"{ref}_CpG_site", False).values)[0]
                        any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                        gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                        cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)
                    if include_any_a:
                        any_a_sites = np.where(subset.var.get(f"{ref}_A_site", False).values)[0]
                        any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                    # Extract layer arrays (unique layers only to avoid duplicate copies)
                    unique_layer_names = {layer_c, layer_gpc, layer_cpg, layer_a}
                    layer_data = {}
                    for lname in unique_layer_names:
                        if lname in subset.layers:
                            arr = subset.layers[lname]
                            layer_data[lname] = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr, dtype=float)

                    # Bin filters as numpy bool arrays over subset rows
                    if bins is None:
                        bins_temp = {"All": np.ones(subset.shape[0], dtype=bool)}
                    else:
                        bins_temp = {bl: np.asarray(bf, dtype=bool) for bl, bf in bins.items()}

                    # obs sort column (for sort_by="obs:<col>")
                    obs_sort_values = None
                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        if colname in subset.obs:
                            obs_sort_values = subset.obs[colname].values

                    yield {
                        "ref": str(ref),
                        "sample": str(sample),
                        "display_sample": str(display_sample),
                        "layer_data": layer_data,
                        "any_c_sites": any_c_sites,
                        "gpc_sites": gpc_sites,
                        "cpg_sites": cpg_sites,
                        "any_a_sites": any_a_sites,
                        "any_c_labels": np.asarray(any_c_labels, dtype=str),
                        "gpc_labels": np.asarray(gpc_labels, dtype=str),
                        "cpg_labels": np.asarray(cpg_labels, dtype=str),
                        "any_a_labels": np.asarray(any_a_labels, dtype=str),
                        "bins_np": bins_temp,
                        "obs_sort_values": obs_sort_values,
                        "total_reads": subset.shape[0],
                        "include_any_c": include_any_c,
                        "include_any_a": include_any_a,
                        "sort_by": sort_by,
                        "fill_nan_strategy": fill_nan_strategy,
                        "fill_nan_value": fill_nan_value,
                        "layer_c": layer_c,
                        "layer_gpc": layer_gpc,
                        "layer_cpg": layer_cpg,
                        "layer_a": layer_a,
                        "cmap_c": cmap_c,
                        "cmap_gpc": cmap_gpc,
                        "cmap_cpg": cmap_cpg,
                        "cmap_a": cmap_a,
                        "n_xticks_any_c": n_xticks_any_c,
                        "n_xticks_gpc": n_xticks_gpc,
                        "n_xticks_cpg": n_xticks_cpg,
                        "n_xticks_any_a": n_xticks_any_a,
                        "xtick_rotation": xtick_rotation,
                        "xtick_fontsize": xtick_fontsize,
                        "save_path": str(save_path) if save_path is not None else None,
                    }

                except Exception:
                    import traceback
                    traceback.print_exc()
                    continue

    # ------------------------------------------------------------------
    # Dispatch — parallel only when saving to disk
    # ------------------------------------------------------------------
    n_workers = _resolve_n_jobs(n_jobs) if save_path is not None else 1

    if n_workers <= 1:
        raw_results = [_combined_raw_clustermap_one_group(a) for a in _iter_group_args()]
    else:
        from concurrent.futures import ProcessPoolExecutor

        from smftools.parallel_utils import configure_worker_threads

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=configure_worker_threads,
            initargs=(1,),
        ) as executor:
            raw_results = list(executor.map(_combined_raw_clustermap_one_group, _iter_group_args()))

    results = [r for r in raw_results if r is not None]

    for r in results:
        logger.info("Summary for %s - %s:", r["sample"], r["ref"])
        for bin_label, percent in r.get("percentages", {}).items():
            logger.info("  - %s: %.1f%%", bin_label, percent)

    return results
