"""
histograms.py — Interval distribution histograms with rolling mean, peak calling,
and optional Gaussian fit overlay.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

_CLIP_Q = 0.01


def plot_interval_histogram(
    values: np.ndarray,
    output_path: Path,
    title: str = "",
    xlabel: str = "Value",
    color: str = "#1f77b4",
    integer_bins: bool = False,
    hist_config: dict | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (3.5, 2.5),
) -> None:
    """
    Plot a histogram with optional rolling-mean overlay and peak annotations.

    Parameters
    ----------
    values       : 1-D float array of observations.
    output_path  : file to write.
    title        : axes title.
    xlabel       : x-axis label.
    color        : bar fill color.
    integer_bins : one bar per integer value (for count histograms).
    hist_config  : dict from tools.hmm_histogram_config.HISTOGRAM_CONFIGS[layer][hist_type].
                   Keys used: bin_size_bp, rolling_window_bp, peak_kwargs,
                              rolling_color, peak_color.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not values.size:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, transform=ax.transAxes)
    else:
        bin_size_bp = hist_config.get("bin_size_bp") if hist_config else None
        plot_values = values

        if integer_bins:
            max_val = int(np.nanmax(values))
            bins = np.arange(max_val + 2) - 0.5
        else:
            lo = np.nanquantile(values, _CLIP_Q)
            hi = np.nanquantile(values, 1.0 - _CLIP_Q)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                clipped = values[(values >= lo) & (values <= hi)]
                if clipped.size:
                    plot_values = clipped
            if bin_size_bp is not None:
                vmin, vmax = np.nanmin(plot_values), np.nanmax(plot_values)
                b0 = np.floor(vmin / bin_size_bp) * bin_size_bp
                b1 = np.ceil(vmax / bin_size_bp) * bin_size_bp + bin_size_bp
                bins = np.arange(b0, b1, bin_size_bp)
                if len(bins) < 2:
                    bins = np.array([b0, b0 + bin_size_bp])
            else:
                bins = min(40, max(10, int(np.sqrt(len(plot_values)))))

        counts, edges = np.histogram(plot_values, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        bin_width = edges[1] - edges[0]
        ax.bar(
            bin_centers,
            counts,
            width=bin_width * 0.9,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )

        if hist_config is not None:
            rolling_bp = hist_config.get("rolling_window_bp")
            peak_kwargs = hist_config.get("peak_kwargs")
            r_color = hist_config.get("rolling_color", "#333333")
            p_color = hist_config.get("peak_color", "#d62728")

            rolling_bins = (
                max(1, round(rolling_bp / bin_size_bp)) if rolling_bp and bin_size_bp else None
            )
            max_count = float(counts.max()) if counts.max() > 0 else 1.0
            norm = counts / max_count
            smooth = norm.copy()

            if rolling_bins and rolling_bins > 1:
                smooth = (
                    pd.Series(norm)
                    .rolling(rolling_bins, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                ax.plot(
                    bin_centers,
                    smooth * max_count,
                    color=r_color,
                    linewidth=1.0,
                    zorder=3,
                    label=f"rolling mean ({rolling_bp} bp)",
                )
                ax.legend(fontsize=7, loc="upper right", frameon=False)

            if peak_kwargs is not None:
                peaks, _ = find_peaks(smooth, **peak_kwargs)
                for pk in peaks:
                    xpos = bin_centers[pk]
                    ax.axvline(
                        xpos, color=p_color, linewidth=0.8, linestyle="--", alpha=0.85, zorder=4
                    )
                    ax.text(
                        xpos,
                        smooth[pk] * max_count,
                        f" {xpos:.0f}",
                        ha="left",
                        va="bottom",
                        fontsize=6,
                        color=p_color,
                        zorder=5,
                    )

    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _gaussian(x, amplitude, mean, std):
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)


def gaussian_fit_plot(
    values: np.ndarray,
    output_path: Path,
    title: str = "",
    xlabel: str = "Center-to-center distance (bp)",
    color: str = "#1f77b4",
    hist_config: dict | None = None,
    fit_config: dict | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (3.5, 2.5),
) -> None:
    """
    Histogram with least-squares Gaussian fit overlay.

    fit_config fields: fit_range_bp (tuple[float,float]), fit_color (str).
    """
    bin_size_bp = (hist_config or {}).get("bin_size_bp", 5)
    fit_lo, fit_hi = (fit_config or {}).get("fit_range_bp", (0, 300))
    fit_color = (fit_config or {}).get("fit_color", "#1f77b4")

    fig, ax = plt.subplots(figsize=figsize)
    if not values.size:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return

    lo = np.nanquantile(values, _CLIP_Q)
    hi = np.nanquantile(values, 1.0 - _CLIP_Q)
    plot_values = (
        values[(values >= lo) & (values <= hi)] if (np.isfinite(lo) and hi > lo) else values
    )

    vmin, vmax = np.nanmin(plot_values), np.nanmax(plot_values)
    b0 = np.floor(vmin / bin_size_bp) * bin_size_bp
    b1 = np.ceil(vmax / bin_size_bp) * bin_size_bp + bin_size_bp
    bins = np.arange(b0, b1, bin_size_bp)
    if len(bins) < 2:
        bins = np.array([b0, b0 + bin_size_bp])

    counts, edges = np.histogram(plot_values, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = edges[1] - edges[0]
    ax.bar(
        bin_centers,
        counts,
        width=bin_width * 0.9,
        color=color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )

    fit_mask = (bin_centers >= fit_lo) & (bin_centers <= fit_hi)
    fit_text = "Gaussian fit failed"
    if fit_mask.sum() >= 4 and counts[fit_mask].sum() > 0:
        fc, fx = counts[fit_mask].astype(float), bin_centers[fit_mask]
        mu0 = float(np.average(fx, weights=fc))
        var0 = float(np.average((fx - mu0) ** 2, weights=fc))
        try:
            popt, _ = curve_fit(
                _gaussian,
                fx,
                fc,
                p0=[fc.max(), mu0, max(np.sqrt(var0), bin_size_bp)],
                bounds=([0, fit_lo, 1], [np.inf, fit_hi, fit_hi - fit_lo]),
                maxfev=10_000,
            )
            amp, mu, sig = popt
            fitted = _gaussian(fx, *popt)
            ss_res = float(np.sum((fc - fitted) ** 2))
            ss_tot = float(np.sum((fc - fc.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            x_curve = np.linspace(bin_centers[0], bin_centers[-1], 500)
            ax.plot(
                x_curve,
                _gaussian(x_curve, *popt),
                color=fit_color,
                linewidth=1.2,
                zorder=3,
                label="Gaussian fit",
            )
            ax.axvline(mu, color=fit_color, linewidth=0.7, linestyle=":", alpha=0.8)
            fit_text = f"μ = {mu:.1f} bp\nσ = {sig:.1f} bp\nA = {amp:.1f}\nR² = {r2:.3f}"
        except (RuntimeError, ValueError):
            pass

    ax.text(
        0.97,
        0.97,
        fit_text,
        transform=ax.transAxes,
        fontsize=6,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )
    ax.axvspan(
        fit_lo, fit_hi, color="0.92", zorder=0, label=f"fit range {fit_lo:.0f}–{fit_hi:.0f} bp"
    )
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
