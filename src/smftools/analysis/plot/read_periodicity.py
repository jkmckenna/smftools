"""Paired input / periodogram clustermaps of reads (`RPG-03`).

Left, the input layer over a region's positions; right, each read's
Lomb-Scargle periodogram (``magma``). Both panels share one row order -- by
peak period, optionally within bins (e.g. groups or Leiden clusters), or an
explicit order -- and carry a marginal track above: the mean signal per
position, and the mean power per period with the peak-search band shaded.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

matplotlib.use("Agg")

ZERO_COLOR = "#eee6d9"  # as the HMM clustermaps
NAN_COLOR = "#D0D0D0"
INPUT_COLOR = "#2E7D32"


def periodicity_row_order(
    peak_period: np.ndarray,
    *,
    bins: Sequence[str] | None = None,
    bin_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Rows with a peak, by bin (in ``bin_order``, else first appearance), then peak period.

    Rows without a finite peak period are left out.
    """
    peak_period = np.asarray(peak_period, dtype=float)
    keep = np.flatnonzero(np.isfinite(peak_period))
    if bins is None:
        return keep[np.argsort(peak_period[keep], kind="stable")]
    bins = np.asarray([str(b) for b in bins], dtype=object)
    present = list(dict.fromkeys(bins[keep]))
    ordered = [b for b in (bin_order or present) if b in present]
    ordered += [b for b in present if b not in ordered]
    rank = {b: i for i, b in enumerate(ordered)}
    keys = np.array([rank[b] for b in bins[keep]])
    return keep[np.lexsort((peak_period[keep], keys))]


def _subsample(order: np.ndarray, max_reads: int | None, seed: int) -> np.ndarray:
    """At most ``max_reads`` rows, chosen deterministically, order kept."""
    if max_reads is None or max_reads <= 0 or order.size <= max_reads:
        return order
    chosen = np.random.default_rng(seed).choice(order.size, size=max_reads, replace=False)
    return order[np.sort(chosen)]


def _input_cmap(binary: bool, color: str, zero_color: str, nan_color: str):
    if binary:
        cmap = ListedColormap([zero_color, color])
    else:
        cmap = LinearSegmentedColormap.from_list("input", [zero_color, color])
    cmap = cmap.copy()
    cmap.set_bad(nan_color)
    return cmap


def _ticks(axis_values: np.ndarray, n: int = 6) -> tuple[np.ndarray, list[str]]:
    index = np.unique(
        np.linspace(0, axis_values.size - 1, num=min(n, axis_values.size)).astype(int)
    )
    return index, [f"{axis_values[i]:g}" for i in index]


def plot_read_periodicity_clustermap(
    values: np.ndarray,
    positions: np.ndarray,
    power: np.ndarray,
    periods: np.ndarray,
    peak_period: np.ndarray,
    output_path: str | Path,
    *,
    bins: Sequence[str] | None = None,
    bin_order: Sequence[str] | None = None,
    order: Sequence[int] | None = None,
    peak_range: tuple[float, float] | None = None,
    max_reads: int | None = 2000,
    seed: int = 0,
    input_label: str = "input",
    input_color: str = INPUT_COLOR,
    zero_color: str = ZERO_COLOR,
    nan_color: str = NAN_COLOR,
    binary: bool | None = None,
    power_vmax: float | None = None,
    observed_columns_only: bool = True,
    title: str = "",
) -> None:
    """Input layer and periodogram of the same reads, in one row order.

    ``values`` (reads x positions, NaN = not observed) and ``power`` (reads x
    periods) share rows with ``peak_period``; reads without a finite peak are
    left out and counted in the title. Rows follow ``order`` (indices into the
    rows) when given, else `periodicity_row_order`; at most ``max_reads`` are
    drawn, chosen deterministically. ``binary`` (default: inferred) draws the
    input in two colours, otherwise a ramp from ``zero_color`` to
    ``input_color``. ``observed_columns_only`` drops positions no shown read
    observed -- for a site-restricted input, everything between its sites --
    while tick labels keep the true positions.
    """
    values = np.asarray(values, dtype=float)
    power = np.asarray(power, dtype=float)
    positions = np.asarray(positions)
    periods = np.asarray(periods, dtype=float)
    peak_period = np.asarray(peak_period, dtype=float)
    n = values.shape[0]
    if power.shape[0] != n or peak_period.shape[0] != n:
        raise ValueError("values, power and peak_period must have the same rows")
    if values.shape[1] != positions.size or power.shape[1] != periods.size:
        raise ValueError("values/positions or power/periods disagree in length")
    if bins is not None and len(bins) != n:
        raise ValueError("bins must give one label per row")

    if order is None:
        rows = periodicity_row_order(peak_period, bins=bins, bin_order=bin_order)
    else:
        rows = np.asarray(order, dtype=int)
        rows = rows[np.isfinite(peak_period[rows])]
    excluded = n - int(np.isfinite(peak_period).sum())
    rows = _subsample(rows, max_reads, seed)
    # Periods ascending, left to right.
    period_sort = np.argsort(periods)
    periods = periods[period_sort]
    power = power[:, period_sort]

    heading = title or ""
    counts = f"{rows.size} reads shown" + (
        f", {excluded} without a peak left out" if excluded else ""
    )
    if rows.size == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"no reads with a valid peak ({excluded} left out)", ha="center")
        if heading:
            ax.set_title(heading, fontsize=9)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    shown_values = values[rows]
    shown_power = power[rows]
    if observed_columns_only:
        kept = np.isfinite(shown_values).any(axis=0)
        if kept.any():
            shown_values, positions = shown_values[:, kept], positions[kept]
    if binary is None:
        finite = shown_values[np.isfinite(shown_values)]
        binary = bool(finite.size) and bool(np.isin(finite, (0.0, 1.0)).all())
    finite = shown_values[np.isfinite(shown_values)]
    vmax_input = 1.0 if binary or not finite.size else float(np.nanpercentile(finite, 99)) or 1.0
    if power_vmax is None:
        finite_power = shown_power[np.isfinite(shown_power)]
        power_vmax = float(np.percentile(finite_power, 99)) if finite_power.size else 1.0

    height = min(12.0, 3.0 + 0.004 * rows.size)
    fig = plt.figure(figsize=(13, height))
    grid = fig.add_gridspec(
        2, 4, width_ratios=[3.2, 0.06, 1.6, 0.06], height_ratios=[1, 5], wspace=0.08, hspace=0.05
    )
    top_input = fig.add_subplot(grid[0, 0])
    top_power = fig.add_subplot(grid[0, 2])
    input_ax = fig.add_subplot(grid[1, 0], sharex=top_input)
    input_bar = fig.add_subplot(grid[1, 1])
    power_ax = fig.add_subplot(grid[1, 2], sharex=top_power)
    power_bar = fig.add_subplot(grid[1, 3])

    columns = np.arange(positions.size)
    with warnings.catch_warnings():  # all-NaN columns: a NaN mean, not a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_signal = np.nanmean(shown_values, axis=0)
        mean_power = np.nanmean(shown_power, axis=0)
    finite_mean = np.isfinite(mean_signal)
    top_input.plot(columns[finite_mean], mean_signal[finite_mean], color=input_color, linewidth=0.8)
    top_input.set_ylabel(f"mean\n{input_label}", fontsize=7)
    top_input.set_xlim(-0.5, positions.size - 0.5)
    top_power.plot(np.arange(periods.size), mean_power, color="black", linewidth=0.9)
    top_power.set_ylabel("mean power", fontsize=7)
    top_power.set_xlim(-0.5, periods.size - 0.5)
    median_peak = float(np.median(peak_period[rows]))
    if peak_range is not None:
        low, high = np.searchsorted(periods, peak_range[0]), np.searchsorted(periods, peak_range[1])
        top_power.axvspan(low - 0.5, high - 0.5, color="#FFB74D", alpha=0.25, linewidth=0)
    top_power.axvline(
        np.searchsorted(periods, median_peak), color="#C62828", linewidth=0.8, linestyle="--"
    )
    top_power.set_title(f"median peak {median_peak:.0f} bp", fontsize=7, loc="right")
    for ax in (top_input, top_power):
        ax.tick_params(labelbottom=False, labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    image = input_ax.imshow(
        np.ma.masked_invalid(shown_values),
        aspect="auto",
        interpolation="nearest",
        cmap=_input_cmap(binary, input_color, zero_color, nan_color),
        vmin=0.0,
        vmax=vmax_input,
    )
    input_scale = fig.colorbar(image, cax=input_bar)
    input_scale.ax.tick_params(labelsize=6)
    if binary:
        input_scale.set_ticks([0.25, 0.75], labels=["0", "1"])
    ticks, labels = _ticks(positions)
    input_ax.set_xticks(ticks, labels, fontsize=6)
    input_ax.set_xlabel(
        "position" + (" (observed columns only)" if observed_columns_only else ""), fontsize=7
    )
    input_ax.set_yticks([])

    spectrum = power_ax.imshow(
        np.ma.masked_invalid(shown_power),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=power_vmax,
    )
    bar = fig.colorbar(spectrum, cax=power_bar)
    bar.ax.tick_params(labelsize=6)
    bar.set_label("Lomb-Scargle power", fontsize=7)
    ticks, labels = _ticks(periods)
    power_ax.set_xticks(ticks, labels, fontsize=6)
    power_ax.set_xlabel("period (bp)", fontsize=7)
    power_ax.set_yticks([])

    if bins is not None:
        labels_shown = np.asarray([str(b) for b in bins], dtype=object)[rows]
        edges = np.flatnonzero(labels_shown[1:] != labels_shown[:-1]) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, rows.size]
        for ax in (input_ax, power_ax):
            for edge in edges:
                ax.axhline(edge - 0.5, color="#212121", linewidth=0.7)
        for start, end in zip(starts, ends, strict=True):
            input_ax.annotate(
                labels_shown[start],
                xy=(-0.01, 1 - (start + end) / 2 / rows.size),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=7,
            )
    fig.suptitle("\n".join(part for part in (heading, counts) if part), fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
