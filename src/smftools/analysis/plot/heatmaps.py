"""
heatmaps.py — Pearson correlation and covariance heatmap rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pearson_heatmap(
    mat: np.ndarray,
    coords: np.ndarray,
    output_path: Path,
    title: str = "",
    n_ticks: int = 10,
    dpi: int = 300,
    figsize: tuple[float, float] = (3.5, 3.0),
    cmap: str = "seismic",
) -> None:
    """
    Render a position × position Pearson correlation matrix and save to disk.

    Parameters
    ----------
    mat         : (n_pos × n_pos) Pearson matrix from smftools.analysis.compute.pearson.nan_pearson_matrix().
    coords      : 1-D int array of TSS-centred coordinates; length must equal mat.shape[0].
    output_path : file to write (PNG or PDF).
    title       : axes title string.
    n_ticks     : number of x/y axis ticks.
    dpi         : output resolution.
    figsize     : figure size in inches.
    cmap        : diverging colormap; "seismic" or "RdBu_r" recommended.
    """
    from smftools.analysis.compute.pearson import make_ticks

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=-1, vmax=1,
        origin="upper",
        interpolation="none",
    )
    tick_idx, tick_labels = make_ticks(coords, n_ticks=n_ticks)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("Position (bp, TSS = 0)", fontsize=9)
    ax.set_ylabel("Position (bp, TSS = 0)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Pearson r", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
