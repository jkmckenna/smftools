"""
embeddings.py — Scatter and density plots for 2D embeddings (PCA/UMAP).

Functions
---------
plot_embedding_scatter             Scatter plot of an embedding, coloured by a categorical column.
plot_embedding_density_grid        Grid of per-category 2D KDE density panels over a shared extent.
plot_cluster_composition_barplot   Stacked barplot of per-sample cluster proportions.
plot_cluster_proportion_grouped_barplot   Grouped barplot (cluster on x-axis, one bar per sample per cluster) with cell-type fill colours, WT/enh-del hatching, and per-biorep points overlaid.
plot_explained_variance           Scree plot of per-PC explained variance ratio with cumulative explained variance overlaid.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_embedding_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_path: Path,
    color_map: dict | None = None,
    color_labels: dict | None = None,
    color_order: list | None = None,
    point_size: float = 5.0,
    alpha: float = 0.6,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (4.5, 4.0),
    dpi: int = 300,
    marginal_density: bool = False,
    marginal_grid_size: int = 200,
    marginal_bw_method: float | str | None = None,
) -> None:
    """
    Scatter plot of a 2D embedding (e.g. UMAP1 vs UMAP2), coloured by a categorical column.

    Parameters
    ----------
    df               : DataFrame containing x_col, y_col, color_col.
    color_map        : category -> hex colour; missing categories auto-assigned from tab10.
    color_labels     : category -> display label for the legend.
    color_order      : explicit category order for plotting/legend (default: first appearance).
    marginal_density : if True, add 1D KDE marginal panels per category alongside the scatter.
    """
    color_map = dict(color_map or {})
    color_labels = color_labels or {}
    cycle = itertools.cycle(_TAB10)

    if color_order is None:
        seen: set = set()
        color_order = [v for v in df[color_col] if v not in seen and not seen.add(v)]

    for label in color_order:
        if label not in color_map:
            color_map[label] = next(cycle)

    if not marginal_density:
        fig, ax = plt.subplots(figsize=figsize)
        for label in color_order:
            sub = df[df[color_col] == label]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=point_size,
                alpha=alpha,
                color=color_map.get(label, "#888888"),
                label=color_labels.get(label, label),
                edgecolors="none",
            )

        ax.set_xlabel(xlabel or x_col, fontsize=9)
        ax.set_ylabel(ylabel or y_col, fontsize=9)
        if title:
            ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(
            fontsize=7,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            markerscale=2,
        )
        fig.tight_layout(rect=(0, 0, 0.82, 1))
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    # ------------------------------------------------------------
    # marginal_density=True: scatter + 1D KDE marginals (top/right)
    # ------------------------------------------------------------
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(figsize[0] + 1.4, figsize[1] + 1.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    for label in color_order:
        sub = df[df[color_col] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=point_size,
            alpha=alpha,
            color=color_map.get(label, "#888888"),
            label=color_labels.get(label, label),
            edgecolors="none",
        )

    x_all = df[x_col].to_numpy(dtype=float)
    y_all = df[y_col].to_numpy(dtype=float)
    xpad = (x_all.max() - x_all.min()) * 0.05
    ypad = (y_all.max() - y_all.min()) * 0.05
    x_grid = np.linspace(x_all.min() - xpad, x_all.max() + xpad, marginal_grid_size)
    y_grid = np.linspace(y_all.min() - ypad, y_all.max() + ypad, marginal_grid_size)

    for label in color_order:
        sub = df[df[color_col] == label]
        if sub.empty:
            continue
        color = color_map.get(label, "#888888")

        x_vals = sub[x_col].to_numpy(dtype=float)
        if len(x_vals) >= 2 and np.ptp(x_vals) > 0:
            kde_x = gaussian_kde(x_vals, bw_method=marginal_bw_method)
            ax_top.plot(x_grid, kde_x(x_grid), color=color, lw=1.2)

        y_vals = sub[y_col].to_numpy(dtype=float)
        if len(y_vals) >= 2 and np.ptp(y_vals) > 0:
            kde_y = gaussian_kde(y_vals, bw_method=marginal_bw_method)
            ax_right.plot(kde_y(y_grid), y_grid, color=color, lw=1.2)

    ax_top.tick_params(labelbottom=False, labelleft=False, length=0)
    ax_right.tick_params(labelleft=False, labelbottom=False, length=0)
    ax_top.set_ylabel("density", fontsize=7)
    ax_right.set_xlabel("density", fontsize=7)
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    for spine in ("top", "right", "bottom"):
        ax_right.spines[spine].set_visible(False)

    ax.set_xlabel(xlabel or x_col, fontsize=9)
    ax.set_ylabel(ylabel or y_col, fontsize=9)
    ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=9)

    handles, labels_ = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels_, fontsize=7, frameon=False, loc="center", markerscale=2)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_density_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    facet_col: str,
    output_path: Path,
    facet_order: list | None = None,
    facet_labels: dict | None = None,
    grid_size: int = 100,
    bw_method: float | str | None = None,
    n_cols: int = 3,
    cmap: str = "viridis",
    shared_scale: bool = True,
    pad_frac: float = 0.05,
    figsize_per_panel: tuple[float, float] = (3.0, 2.6),
    dpi: int = 300,
) -> None:
    """
    Grid of 2D KDE density panels, one per category in ``facet_col``.

    Each panel is a Gaussian KDE of (x_col, y_col) for that category, evaluated on
    a shared grid spanning the full df extent and renormalised so the discretised
    grid integrates to 1 (a probability density, comparable across panels
    regardless of category size).

    Parameters
    ----------
    facet_order  : explicit category order (default: order of first appearance).
    facet_labels : category -> display label for panel titles.
    grid_size    : number of grid points per axis.
    bw_method    : passed to scipy.stats.gaussian_kde (None -> Scott's rule).
    shared_scale : use the same vmax (density colour scale) across all panels.
    """
    from scipy.stats import gaussian_kde

    if facet_order is None:
        seen: set = set()
        facet_order = [v for v in df[facet_col] if v not in seen and not seen.add(v)]
    facet_labels = facet_labels or {}

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    xpad = (x.max() - x.min()) * pad_frac
    ypad = (y.max() - y.min()) * pad_frac
    xmin, xmax = x.min() - xpad, x.max() + xpad
    ymin, ymax = y.min() - ypad, y.max() + ypad

    xx, yy = np.mgrid[xmin : xmax : complex(0, grid_size), ymin : ymax : complex(0, grid_size)]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    cell_area = ((xmax - xmin) / grid_size) * ((ymax - ymin) / grid_size)

    densities: dict = {}
    for label in facet_order:
        sub = df[df[facet_col] == label]
        if len(sub) < 3:
            densities[label] = None
            continue
        pts = np.vstack([sub[x_col].to_numpy(dtype=float), sub[y_col].to_numpy(dtype=float)])
        kde = gaussian_kde(pts, bw_method=bw_method)
        z = kde(grid_coords).reshape(xx.shape)
        z = z / (z.sum() * cell_area)  # renormalise to a probability density on this grid
        densities[label] = z

    vmax = None
    if shared_scale:
        valid = [z for z in densities.values() if z is not None]
        vmax = max(z.max() for z in valid) if valid else None

    n = len(facet_order)
    n_cols_eff = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols_eff))
    fig, axes = plt.subplots(
        n_rows,
        n_cols_eff,
        figsize=(figsize_per_panel[0] * n_cols_eff, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    im = None
    for i, label in enumerate(facet_order):
        ax = axes[i // n_cols_eff][i % n_cols_eff]
        z = densities[label]
        n_pts = int((df[facet_col] == label).sum())
        if z is None:
            ax.text(
                0.5,
                0.5,
                "n too small",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
        else:
            im = ax.imshow(
                z.T,
                origin="lower",
                extent=(xmin, xmax, ymin, ymax),
                aspect="auto",
                cmap=cmap,
                vmin=0,
                vmax=vmax if shared_scale else None,
            )
        ax.set_title(f"{facet_labels.get(label, label)} (n={n_pts})", fontsize=8)
        ax.tick_params(labelsize=7)
        if i % n_cols_eff == 0:
            ax.set_ylabel(y_col, fontsize=8)
        if i // n_cols_eff == n_rows - 1:
            ax.set_xlabel(x_col, fontsize=8)

    for i in range(n, n_rows * n_cols_eff):
        axes[i // n_cols_eff][i % n_cols_eff].axis("off")

    fig.suptitle(f"{x_col} / {y_col} density by {facet_col}", fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 0.9, 0.93))

    if im is not None:
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
        fig.colorbar(im, cax=cbar_ax, label="probability density")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_composition_barplot(
    df: pd.DataFrame,
    sample_col: str,
    cluster_col: str,
    output_path: Path,
    sample_order: list | None = None,
    sample_labels: dict | None = None,
    cluster_order: list | None = None,
    cluster_labels: dict | None = None,
    cluster_colors: dict | None = None,
    title: str = "",
    bar_width_in: float = 0.6,
    height_in: float = 4.0,
    dpi: int = 300,
) -> None:
    """
    Stacked barplot of per-sample cluster composition.

    For each category in ``sample_col`` (one bar), shows the proportion of
    rows falling into each category of ``cluster_col``, stacked to 1.0.

    Parameters
    ----------
    sample_order   : explicit bar order (default: order of first appearance).
    cluster_order  : explicit stacking order (default: sorted unique values).
    cluster_colors : cluster -> hex colour; unspecified clusters auto-assigned from tab10.
    """
    if sample_order is None:
        seen: set = set()
        sample_order = [v for v in df[sample_col] if v not in seen and not seen.add(v)]
    sample_labels = sample_labels or {}

    if cluster_order is None:
        cluster_order = sorted(df[cluster_col].unique())
    cluster_labels = cluster_labels or {}

    cycle = itertools.cycle(_TAB10)
    cluster_colors = dict(cluster_colors or {})
    for c in cluster_order:
        if c not in cluster_colors:
            cluster_colors[c] = next(cycle)

    n_samples = len(sample_order)
    props = np.zeros((n_samples, len(cluster_order)))
    counts = np.zeros(n_samples, dtype=int)
    for i, s in enumerate(sample_order):
        sub = df[df[sample_col] == s]
        counts[i] = len(sub)
        for j, c in enumerate(cluster_order):
            props[i, j] = (sub[cluster_col] == c).sum() / counts[i] if counts[i] else 0.0

    fig_width = max(3.0, bar_width_in * n_samples + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, height_in))

    x = np.arange(n_samples)
    bottom = np.zeros(n_samples)
    for j, c in enumerate(cluster_order):
        ax.bar(
            x,
            props[:, j],
            bottom=bottom,
            color=cluster_colors[c],
            label=cluster_labels.get(c, f"cluster {c}"),
            width=0.7,
        )
        bottom += props[:, j]

    ax.set_xticks(x)
    xticklabels = [
        f"{sample_labels.get(s, s)}\n(n={counts[i]})" for i, s in enumerate(sample_order)
    ]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("proportion of reads", fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        title="Leiden cluster",
        title_fontsize=7,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_proportion_grouped_barplot(
    df: pd.DataFrame,
    sample_col: str,
    cluster_col: str,
    biorep_col: str,
    cell_type_col: str,
    output_path: Path,
    sample_order: list | None = None,
    cluster_order: list | None = None,
    cluster_labels: dict | None = None,
    cell_type_colors: dict | None = None,
    cell_type_labels: dict | None = None,
    color_overrides: dict | None = None,
    color_override_labels: dict | None = None,
    enh_del_keyword: str = "enh-del",
    point_size: float = 14.0,
    jitter: float = 0.06,
    group_width: float = 0.8,
    figsize: tuple[float, float] = (9.5, 4.5),
    title: str = "",
    dpi: int = 300,
) -> None:
    """
    Single-axes grouped barplot of per-sample cluster proportions.

    ``cluster_col`` categories form the x-axis; within each cluster, one bar
    per category in ``sample_col`` is drawn side by side. Bar fill colour is
    keyed off ``cell_type_col``; bars whose ``sample_col`` value contains
    ``enh_del_keyword`` (case-insensitive) are drawn with a cross-hatch,
    all others solid. For each bar, the height is the mean across
    ``biorep_col`` of that biorep's proportion of reads in the cluster, with
    individual biorep proportions overlaid as jittered points. A legend maps
    cell-type colours and the solid/hatched (WT/enh-del) convention.

    Parameters
    ----------
    sample_order          : explicit bar order within each cluster group (default: first appearance).
    cluster_order         : explicit x-axis order (default: sorted unique values).
    cell_type_colors      : cell type -> hex colour; unspecified types auto-assigned from tab10.
    cell_type_labels      : cell type -> display label for the legend.
    color_overrides       : sample -> hex colour, overriding ``cell_type_colors`` for that bar.
    color_override_labels : sample -> legend label, overriding ``cell_type_labels`` when colour is overridden.
    """
    from matplotlib.patches import Patch

    if sample_order is None:
        seen: set = set()
        sample_order = [v for v in df[sample_col] if v not in seen and not seen.add(v)]

    if cluster_order is None:
        cluster_order = sorted(df[cluster_col].unique())
    cluster_labels = cluster_labels or {}
    cell_type_labels = cell_type_labels or {}
    color_overrides = color_overrides or {}
    color_override_labels = color_override_labels or {}

    sample_cell_type = {s: df.loc[df[sample_col] == s, cell_type_col].iloc[0] for s in sample_order}
    sample_is_enh_del = {s: enh_del_keyword.lower() in str(s).lower() for s in sample_order}

    cycle = itertools.cycle(_TAB10)
    cell_type_colors = dict(cell_type_colors or {})
    for ct in dict.fromkeys(sample_cell_type.values()):
        if ct not in cell_type_colors:
            cell_type_colors[ct] = next(cycle)

    totals = df.groupby([sample_col, biorep_col]).size()
    sample_bioreps = {s: sorted({b for (ss, b) in totals.index if ss == s}) for s in sample_order}

    n_samples = len(sample_order)
    bar_width = group_width / n_samples
    x = np.arange(len(cluster_order), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    ymax = 0.0

    for i, cluster in enumerate(cluster_order):
        cluster_counts = df[df[cluster_col] == cluster].groupby([sample_col, biorep_col]).size()
        for j, s in enumerate(sample_order):
            xpos = x[i] + (j - (n_samples - 1) / 2) * bar_width

            props = []
            for b in sample_bioreps[s]:
                total = totals.get((s, b), 0)
                if total == 0:
                    continue
                props.append(cluster_counts.get((s, b), 0) / total)
            props = np.array(props)
            mean_prop = props.mean() if len(props) else 0.0
            sem_prop = props.std(ddof=1) / np.sqrt(len(props)) if len(props) > 1 else 0.0
            ymax = max(ymax, mean_prop + sem_prop)

            color = color_overrides.get(s, cell_type_colors[sample_cell_type[s]])
            hatch = "////" if sample_is_enh_del[s] else None
            ax.bar(
                xpos,
                mean_prop,
                yerr=sem_prop,
                width=bar_width * 0.9,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
            )

            if len(props):
                jitter_x = xpos + rng.uniform(-jitter, jitter, size=len(props))
                ax.scatter(
                    jitter_x,
                    props,
                    color=color,
                    s=point_size,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                ymax = max(ymax, props.max())

    ax.set_xticks(x)
    ax.set_xticklabels([cluster_labels.get(c, f"cluster {c}") for c in cluster_order], fontsize=8)
    ax.set_ylabel("proportion of reads", fontsize=9)
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)

    legend_entries: dict[str, str] = {}
    for s in sample_order:
        ct = sample_cell_type[s]
        color = color_overrides.get(s, cell_type_colors[ct])
        label = color_override_labels.get(s, cell_type_labels.get(ct, ct))
        legend_entries.setdefault(color, label)

    cell_type_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in legend_entries.items()
    ]
    leg1 = ax.legend(
        handles=cell_type_handles,
        fontsize=7,
        frameon=False,
        title="cell type",
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    if any(sample_is_enh_del.values()):
        ax.add_artist(leg1)
        hatch_handles = [
            Patch(facecolor="white", edgecolor="black", label="WT"),
            Patch(facecolor="white", edgecolor="black", hatch="////", label="enh-del"),
        ]
        y_anchor = 1.0 - 0.12 * (len(cell_type_handles) + 1.5)
        ax.legend(
            handles=hatch_handles,
            fontsize=7,
            frameon=False,
            title="allele",
            title_fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, y_anchor),
            borderaxespad=0,
        )

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(
    explained_variance_ratio: np.ndarray,
    output_path: Path,
    n_pcs_show: int = 20,
    figsize: tuple[float, float] = (6.0, 4.0),
    title: str = "",
    dpi: int = 300,
) -> None:
    """
    Scree plot: bar chart of per-PC explained variance ratio, with cumulative
    explained variance overlaid on a secondary y-axis.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        1D array from sklearn ``PCA.explained_variance_ratio_``.
    output_path : Path
        PNG output path.
    n_pcs_show : int
        number of leading PCs to plot.
    """
    evr = np.asarray(explained_variance_ratio)
    n_show = min(n_pcs_show, len(evr))
    evr_show = evr[:n_show]
    cumvar = np.cumsum(evr)[:n_show]
    x = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, evr_show, color="#1f77b4", label="per-PC")
    ax.set_xlabel("principal component", fontsize=9)
    ax.set_ylabel("explained variance ratio", fontsize=9)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, cumvar, color="#d62728", marker="o", markersize=3, label="cumulative")
    ax2.set_ylabel("cumulative explained variance", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis="y", labelsize=8)

    if title:
        ax.set_title(title, fontsize=9)

    fig.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
