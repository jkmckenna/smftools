"""
autocorr.py — ACF overlay, LS periodogram, paired barplot, and metric histogram rendering.

All functions read summary CSVs (or DataFrames) produced by a compute pass and write
figures. Project-specific values (colors, labels, reference ordering) are passed as
parameters.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_smooth(arr: np.ndarray, win: int = 25) -> np.ndarray:
    """NaN-aware moving average."""
    if win <= 1:
        return arr.copy()
    valid = np.isfinite(arr).astype(float)
    arr0 = np.nan_to_num(arr, nan=0.0)
    k = np.ones(win, dtype=float)
    num = np.convolve(arr0, k, mode="same")
    den = np.convolve(valid, k, mode="same")
    out = np.full_like(arr, np.nan, dtype=float)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def plot_autocorr_overlay(
    summary_df: pd.DataFrame,
    ref_strand: str | None,
    output_path: Path,
    group_col: str = "cell_type",
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    ref_label: str = "",
    title_suffix: str = "",
    max_lag: int = 1000,
    smooth_win: int = 25,
    dpi: int = 300,
    figsize: tuple[float, float] = (5.5, 2.5),
) -> None:
    """
    Plot per-group mean ± SEM autocorrelation curves.

    Parameters
    ----------
    summary_df   : DataFrame with columns reference_strand, <group_col>, curve_csv.
    ref_strand   : filter summary_df to this reference_strand value.
        Pass ``None`` to include all rows (e.g. when groups already encode the
        reference distinction via ``group_col``).
    group_order  : order to plot groups; defaults to unique values in group_col.
    group_colors : {group_value: color}; defaults to matplotlib tab10.
    group_labels : {group_value: display_label}.
    """
    sub = summary_df if ref_strand is None else summary_df[summary_df["reference_strand"] == ref_strand]
    groups = group_order or sorted(sub[group_col].unique())
    colors = group_colors or {}
    labels = group_labels or {}
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps["tab10"]
    for i, grp in enumerate(groups):
        rows = sub[sub[group_col] == grp]
        if rows.empty:
            continue
        curves = [
            pd.read_csv(p)["autocorrelation"].to_numpy(dtype=float)
            for p in rows["curve_csv"]
            if Path(p).exists()
        ]
        if not curves:
            continue
        mat = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_c = np.nanmean(mat, axis=0)
            sem_c = (
                np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
                if mat.shape[0] > 1
                else np.zeros_like(mean_c)
            )
        color = colors.get(grp, cmap(i % 10))
        label = labels.get(grp, str(grp))
        ax.plot(lags, rolling_smooth(mean_c, smooth_win), color=color, linewidth=1.2, label=label)
        ax.fill_between(lags, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.2)

    ax.set_xlim(0, max_lag)
    ax.set_xlabel("Lag (bp)", fontsize=9)
    ax.set_ylabel("Autocorrelation", fontsize=9)
    title = f"{ref_label} autocorrelation"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15)
    ax.legend(
        fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.87, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ls_overlay(
    summary_df: pd.DataFrame,
    ref_strand: str | None,
    output_path: Path,
    group_col: str = "cell_type",
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    ref_label: str = "",
    title_suffix: str = "",
    nrl_range: tuple[float, float] = (120.0, 260.0),
    period_xlim: tuple[float, float] = (80, 400),
    dpi: int = 300,
    figsize: tuple[float, float] = (3.5, 2.5),
) -> None:
    """Plot mean ± SEM Lomb-Scargle spectra.

    Parameters
    ----------
    ref_strand : str or None
        Filter to this reference strand value, or ``None`` to include all rows.
    """
    base = summary_df if ref_strand is None else summary_df[summary_df["reference_strand"] == ref_strand]
    sub = base[base["ls_spectrum_csv"].notna() & (base["ls_spectrum_csv"] != "")]
    groups = group_order or sorted(sub[group_col].unique())
    colors = group_colors or {}
    labels = group_labels or {}

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps["tab10"]
    for i, grp in enumerate(groups):
        rows = sub[sub[group_col] == grp]
        if rows.empty:
            continue
        spectra, freqs = [], None
        for p in rows["ls_spectrum_csv"]:
            if not Path(p).exists():
                continue
            df = pd.read_csv(p)
            freqs = df["frequency"].to_numpy(dtype=float)
            spectra.append(df["power"].to_numpy(dtype=float))
        if not spectra or freqs is None:
            continue
        mat = np.vstack(spectra)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_p = np.nanmean(mat, axis=0)
            sem_p = (
                np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
                if mat.shape[0] > 1
                else np.zeros_like(mean_p)
            )
        periods = 1.0 / freqs
        color = colors.get(grp, cmap(i % 10))
        ax.plot(periods, mean_p, color=color, linewidth=1.2, label=labels.get(grp, str(grp)))
        ax.fill_between(periods, mean_p - sem_p, mean_p + sem_p, color=color, alpha=0.2)

    ax.axvline(nrl_range[0], color="#777777", linewidth=0.6, linestyle=":")
    ax.axvline(nrl_range[1], color="#777777", linewidth=0.6, linestyle=":")
    ax.set_xlim(*period_xlim)
    ax.set_xlabel("Period (bp)", fontsize=9)
    ax.set_ylabel("Normalised LS power", fontsize=9)
    title = f"{ref_label} LS spectrum"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15)
    ax.legend(
        fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_barplot_paired(
    summary_df: pd.DataFrame,
    metric_key: str,
    output_path: Path,
    group_col: str = "cell_type",
    group_order: list[str] | None = None,
    group_labels: dict[str, str] | None = None,
    group_colors: dict[str, str] | None = None,
    ref_col: str = "reference_strand",
    ref_order: list[str] | None = None,
    ref_labels: dict[str, str] | None = None,
    ylabel: str = "",
    title: str = "",
    dpi: int = 300,
    figsize: tuple[float, float] = (4.0, 2.7),
) -> None:
    """
    Paired barplot: x = group, paired bars = reference strands, dots = individual replicates.

    First ref in ref_order is solid; second uses hatch "//".
    """
    sub = summary_df.dropna(subset=[metric_key]).copy()
    if sub.empty:
        return

    groups = group_order or sorted(sub[group_col].unique())
    refs = ref_order or sorted(sub[ref_col].unique())
    g_labels = group_labels or {}
    g_colors = group_colors or {}
    r_labels = ref_labels or {}

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    width = 0.34
    offsets = [w - width / 2 for w in np.linspace(0, width * (len(refs) - 1), len(refs))]
    offsets = [o - np.mean(offsets) for o in offsets]
    cmap = plt.colormaps["tab10"]

    for ri, ref in enumerate(refs):
        for gi, grp in enumerate(groups):
            cell = sub[(sub[ref_col] == ref) & (sub[group_col] == grp)]
            vals = cell[metric_key].to_numpy(dtype=float)
            if not len(vals):
                continue
            mean = np.nanmean(vals)
            sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            xpos = gi + offsets[ri]
            color = g_colors.get(grp, cmap(gi % 10))
            hatch = None if ri == 0 else "//"
            alpha = 0.85 if ri == 0 else 0.45
            ax.bar(
                xpos,
                mean,
                yerr=sem,
                width=width * 0.92,
                color=color,
                alpha=alpha,
                edgecolor="#222222" if hatch else "none",
                linewidth=0.6 if hatch else 0.0,
                hatch=hatch,
                capsize=3,
            )
            jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(
                np.full(len(vals), xpos) + jitter, vals, color="#222222", s=16, alpha=0.8, zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels([g_labels.get(g, str(g)) for g in groups], fontsize=8)
    ax.set_ylabel(ylabel or metric_key, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.15)
    legend_handles = [
        mpatches.Patch(
            facecolor="black",
            edgecolor="black",
            label=r_labels.get(refs[0], refs[0]) if refs else "ref 1",
        ),
    ]
    if len(refs) > 1:
        legend_handles.append(
            mpatches.Patch(
                facecolor="white",
                edgecolor="black",
                hatch="//",
                label=r_labels.get(refs[1], refs[1]),
            )
        )
    ax.legend(
        handles=legend_handles,
        fontsize=7,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_histogram(
    df: pd.DataFrame,
    metric_key: str,
    output_path: Path,
    group_col: str = "cell_type",
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    ref_col: str | None = "reference_strand",
    ref_strand: str | None = None,
    ref_label: str = "",
    xlabel: str = "",
    title: str = "",
    title_suffix: str = "",
    bins: int | np.ndarray = 30,
    xlim: tuple[float, float] | None = None,
    vlines: list[float] | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (4.5, 2.7),
) -> None:
    """
    Overlaid per-group histograms (density-normalised) of a single-molecule metric.

    Parameters
    ----------
    df         : DataFrame with one row per read, columns ``<metric_key>``,
        ``<group_col>``, and optionally ``<ref_col>``.
    metric_key : column to histogram, e.g. ``"ls_nrl_bp"``, ``"ls_snr"``,
        ``"ls_peak_power"``.
    ref_strand : if given (with ``ref_col``), filter to this reference_strand value.
    vlines     : optional list of x positions to mark with dotted vertical lines
        (e.g. an NRL search-band boundary).
    """
    sub = df.dropna(subset=[metric_key]).copy()
    if ref_col is not None and ref_strand is not None and ref_col in sub.columns:
        sub = sub[sub[ref_col] == ref_strand]
    if sub.empty:
        return

    groups = group_order or sorted(sub[group_col].unique())
    colors = group_colors or {}
    labels = group_labels or {}

    vals_all = sub[metric_key].to_numpy(dtype=float)
    if isinstance(bins, int):
        lo, hi = (xlim if xlim is not None else (np.nanmin(vals_all), np.nanmax(vals_all)))
        bin_edges = np.linspace(lo, hi, bins + 1)
    else:
        bin_edges = bins

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps["tab10"]
    for i, grp in enumerate(groups):
        vals = sub.loc[sub[group_col] == grp, metric_key].to_numpy(dtype=float)
        if not vals.size:
            continue
        color = colors.get(grp, cmap(i % 10))
        label = f"{labels.get(grp, str(grp))} (n={vals.size})"
        ax.hist(
            vals, bins=bin_edges, density=True, histtype="stepfilled",
            color=color, alpha=0.15, zorder=1,
        )
        ax.hist(
            vals, bins=bin_edges, density=True, histtype="step",
            color=color, linewidth=1.4, label=label, zorder=2,
        )

    for x in vlines or []:
        ax.axvline(x, color="#777777", linewidth=0.6, linestyle=":")

    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel or metric_key, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    full_title = title or f"{ref_label} {metric_key}".strip()
    if title_suffix:
        full_title += f"\n{title_suffix}"
    if full_title:
        ax.set_title(full_title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15)
    ax.legend(
        fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
