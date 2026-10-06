"""Figures for modification-site sequence-context bias (`SCB-02`).

Inputs are the tables of `smftools.analysis.compute.site_context_bias`
(``offset_enrichment``, ``kmer_rates``, ``group_differences``); each function
writes one figure with one panel per group.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

matplotlib.use("Agg")

BASE_COLORS = {"A": "#2E7D32", "C": "#1565C0", "G": "#F9A825", "T": "#C62828", "N": "#9E9E9E"}
_LETTER_FONT = FontProperties(family="DejaVu Sans", weight="bold")


def _groups(table: pd.DataFrame, groups: Sequence[str] | None) -> list[str]:
    present = list(dict.fromkeys(table["group"]))
    if groups is None:
        return present
    missing = [group for group in groups if group not in present]
    if missing:
        raise KeyError(f"groups not in table: {missing}")
    return list(groups)


def _grid(n: int, ncols: int, panel: tuple[float, float]):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel[0] * ncols, panel[1] * nrows),
        squeeze=False,
    )
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def _bases(table: pd.DataFrame, value: str) -> list[str]:
    """A/C/G/T, plus N only where it carries a value (reference-end padding)."""
    with_n = table.loc[(table["base"] == "N"), value].notna().any()
    return ["A", "C", "G", "T"] + (["N"] if with_n else [])


def _save(fig, output_path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _offset_heatmaps(
    table: pd.DataFrame,
    value: str,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None,
    ncols: int,
    title: str,
    colorbar_label: str,
    limit: float | None,
) -> None:
    groups = _groups(table, groups)
    bases = _bases(table, value)
    offsets = sorted(table["offset"].unique())
    values = table[value].to_numpy(dtype=float)
    if limit is None:
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    fig, axes = _grid(len(groups), ncols, (0.38 * len(offsets) + 1.2, 0.32 * len(bases) + 1.0))
    image = None
    for ax, group in zip(axes, groups, strict=False):
        matrix = (
            table.loc[table["group"] == group]
            .pivot(index="base", columns="offset", values=value)
            .reindex(index=bases, columns=offsets)
        )
        image = ax.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        image.cmap.set_bad("#E0E0E0")
        ax.set_xticks(range(len(offsets)), [f"{o:+d}" if o else "0" for o in offsets], fontsize=7)
        ax.set_yticks(range(len(bases)), bases, fontsize=7)
        if 0 in offsets:
            centre = offsets.index(0)
            ax.axvline(centre - 0.5, color="black", linewidth=0.6)
            ax.axvline(centre + 0.5, color="black", linewidth=0.6)
        ax.set_title(str(group), fontsize=8)
        ax.set_xlabel("offset from site (5'→3', modified strand)", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if image is not None:
        bar = fig.colorbar(image, ax=axes[: len(groups)], shrink=0.8, pad=0.02)
        bar.set_label(colorbar_label, fontsize=7)
        bar.ax.tick_params(labelsize=7)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_offset_enrichment_heatmap(
    enrichment: pd.DataFrame,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    ncols: int = 4,
    title: str = "",
    limit: float | None = None,
) -> None:
    """Offset x base ``log2_enrichment``, one panel per group, shared colour scale."""
    _offset_heatmaps(
        enrichment,
        "log2_enrichment",
        output_path,
        groups=groups,
        ncols=ncols,
        title=title,
        colorbar_label="log2 enrichment (modified / observed)",
        limit=limit,
    )


def plot_group_differences(
    differences: pd.DataFrame,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    ncols: int = 4,
    title: str = "",
    limit: float | None = None,
) -> None:
    """Offset x base enrichment difference from the reference group, one panel per group."""
    reference = differences["reference_group"].iloc[0] if len(differences) else ""
    _offset_heatmaps(
        differences,
        "difference",
        output_path,
        groups=groups,
        ncols=ncols,
        title=title or f"difference from {reference}",
        colorbar_label=f"log2 enrichment − {reference}",
        limit=limit,
    )


def _draw_letter(ax, letter: str, x: float, bottom: float, height: float) -> None:
    path = TextPath((0, 0), letter, size=1, prop=_LETTER_FONT)
    extent = path.get_extents()
    transform = (
        Affine2D()
        .translate(-extent.x0, -extent.y0)
        .scale(0.9 / extent.width, height / extent.height)
        .translate(x - 0.45, bottom)
    )
    ax.add_patch(
        PathPatch(transform.transform_path(path), facecolor=BASE_COLORS[letter], linewidth=0)
    )


def _draw_logo(ax, frame: pd.DataFrame, offsets: list, bound: float) -> None:
    """One group's logo: letter height = |log2 enrichment|, enriched above the axis."""
    for x, offset in enumerate(offsets):
        column = frame.loc[frame["offset"] == offset].dropna(subset=["log2_enrichment"])
        up = column[column["log2_enrichment"] > 0].sort_values("log2_enrichment")
        down = column[column["log2_enrichment"] < 0].sort_values("log2_enrichment", ascending=False)
        bottom = 0.0
        for base, height in zip(up["base"], up["log2_enrichment"], strict=True):
            _draw_letter(ax, base, x, bottom, height)
            bottom += height
        top = 0.0
        for base, height in zip(down["base"], down["log2_enrichment"], strict=True):
            _draw_letter(ax, base, x, top + height, -height)
            top += height
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.6, len(offsets) - 0.4)
    ax.set_ylim(-bound * 1.05, bound * 1.05)
    ax.set_xticks(range(len(offsets)), [f"{o:+d}" if o else "0" for o in offsets], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _logo_bound(enrichment: pd.DataFrame, groups: Sequence[str]) -> float:
    """Shared y-range: the tallest stack, enriched or depleted, of any shown panel."""
    shown = enrichment.loc[enrichment["group"].isin(groups)]
    values = shown["log2_enrichment"].where(np.isfinite(shown["log2_enrichment"]))
    keys = [shown["group"], shown["offset"]]
    stacks = pd.concat(
        [values.clip(lower=0).groupby(keys).sum(), (-values).clip(lower=0).groupby(keys).sum()]
    )
    return max(0.25, float(stacks.max()) if len(stacks) else 0.0)


def plot_enrichment_logo(
    enrichment: pd.DataFrame,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    ncols: int = 2,
    title: str = "",
    layout: Sequence[Sequence[str | None]] | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
) -> None:
    """Enrichment logo: letter height = |log2 enrichment|; enriched above, depleted below.

    Panels are ``groups`` in ``ncols`` columns, or -- with ``layout``, rows of
    group names with ``None`` for an empty cell -- a grid, labelled by
    ``row_labels`` (left) and ``col_labels`` (top) when given. The y-range is
    shared by every panel shown.
    """
    offsets = sorted(enrichment["offset"].unique())
    panel = (0.45 * len(offsets) + 1.2, 2.4)
    if layout is None:
        groups = _groups(enrichment, groups)
        fig, axes = _grid(len(groups), ncols, panel)
        cells = list(zip(axes, groups, strict=False))
        for ax, group in cells:
            ax.set_title(str(group), fontsize=8)
            ax.set_ylabel("log2 enrichment", fontsize=7)
    else:
        rows = [list(row) for row in layout]
        width = max((len(row) for row in rows), default=0)
        rows = [row + [None] * (width - len(row)) for row in rows]
        groups = _groups(enrichment, [g for row in rows for g in row if g is not None])
        if row_labels is not None and len(row_labels) != len(rows):
            raise ValueError("row_labels must match the layout's rows")
        if col_labels is not None and len(col_labels) != width:
            raise ValueError("col_labels must match the layout's columns")
        fig, grid = plt.subplots(
            len(rows),
            width,
            figsize=(panel[0] * width, panel[1] * len(rows)),
            squeeze=False,
        )
        cells = []
        for r, row in enumerate(rows):
            for c, group in enumerate(row):
                ax = grid[r][c]
                if r == 0 and col_labels is not None:
                    ax.set_title(str(col_labels[c]), fontsize=9, fontweight="bold")
                if c == 0 and row_labels is not None:
                    # Left of the row whether or not its first cell is empty.
                    ax.annotate(
                        str(row_labels[r]),
                        xy=(-0.18, 0.5),
                        xycoords="axes fraction",
                        ha="right",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )
                if group is None:
                    ax.axis("off")
                    continue
                if row_labels is None and col_labels is None:
                    ax.set_title(str(group), fontsize=7)
                cells.append((ax, group))
    bound = _logo_bound(enrichment, groups)
    for ax, group in cells:
        _draw_logo(ax, enrichment.loc[enrichment["group"] == group], offsets, bound)
    if layout is not None:
        fig.supylabel("log2 enrichment (modified / observed)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_kmer_rates(
    rates: pd.DataFrame,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    ncols: int = 4,
    max_kmers: int = 40,
    title: str = "",
) -> None:
    """Per-k-mer modification rate with Wilson interval, one panel per group.

    Rows are the ``max_kmers`` k-mers with the most observed calls overall,
    ordered by mean rate across groups and labelled with their distinct-site
    count (the most any group has), so a rate resting on one site shows.
    """
    groups = _groups(rates, groups)
    totals = rates.groupby("kmer")["observed"].sum().nlargest(max_kmers).index
    shown = rates.loc[rates["kmer"].isin(totals)]
    order = shown.groupby("kmer")["rate"].mean().sort_values().index.tolist()
    sites = shown.groupby("kmer")["n_sites"].max()
    labels = [f"{kmer} ({sites[kmer]})" for kmer in order]
    fig, axes = _grid(len(groups), ncols, (2.4, 0.16 * len(order) + 1.0))
    for ax, group in zip(axes, groups, strict=False):
        frame = shown.loc[shown["group"] == group].set_index("kmer").reindex(order)
        y = np.arange(len(order))
        low = (frame["rate"] - frame["rate_low"]).clip(lower=0)
        high = (frame["rate_high"] - frame["rate"]).clip(lower=0)
        ax.errorbar(
            frame["rate"],
            y,
            xerr=[low, high],
            fmt="o",
            markersize=2.5,
            color="#37474F",
            ecolor="#90A4AE",
            elinewidth=0.8,
            capsize=0,
        )
        ax.set_yticks(y, labels, fontsize=6, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_xlabel("modification rate", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(str(group), fontsize=8)
        ax.grid(axis="x", alpha=0.2)
    if title:
        fig.suptitle(title, fontsize=9)
    _save(fig, output_path)
