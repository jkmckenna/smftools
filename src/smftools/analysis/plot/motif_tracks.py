"""Feature-class tracks with motif lanes below (`MOT-03`)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from smftools.analysis.compute.motif_tracks import pack_lanes, track_matrix  # noqa: E402

TRACK_COLORS = {
    "tf": "#C62828",
    "small_bound": "#C62828",
    "medium": "#EF6C00",
    "medium_bound": "#EF6C00",
    "nucleosome": "#6A1B9A",
    "accessible": "#2E7D32",
}
FALLBACK_COLORS = ("#1565C0", "#00838F", "#AD1457", "#5D4037", "#455A64", "#9E9D24")


def track_colors(tracks: Sequence[str]) -> dict[str, str]:
    colors, spare = {}, iter(FALLBACK_COLORS * 4)
    for track in tracks:
        key = next((k for k in TRACK_COLORS if str(track).lower().startswith(k)), None)
        colors[track] = TRACK_COLORS[key] if key else next(spare)
    return colors


def family_colors(families: Sequence[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {family: cmap(i % 20) for i, family in enumerate(sorted(set(families)))}


def _x(positions: np.ndarray, origin: float | None, reverse: bool) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if origin is None:
        return -positions if reverse else positions
    return origin - positions if reverse else positions - origin


def _x_label(origin: float | None, reverse: bool) -> str:
    if origin is None:
        return "position" + (" (reversed)" if reverse else "")
    return f"position relative to {origin:g}" + (" (reversed)" if reverse else "")


def plot_motif_tracks(
    tracks: pd.DataFrame,
    hits: pd.DataFrame,
    output_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    track_order: Sequence[str] | None = None,
    window: tuple[int, int] | None = None,
    layout: str = "groups",
    coordinate_origin: float | None = None,
    coordinate_reverse: bool = False,
    highlight: Sequence[int] = (),
    label_top: int = 15,
    min_spanning: int = 10,
    title: str = "",
) -> None:
    """Class fractions along the reference, motif instances in lanes below.

    ``layout="groups"``: a panel per group, a line per track. ``"tracks"``: a
    panel per track, a line per group. ``window`` limits the x range
    (frame coordinates); positions spanned by fewer than ``min_spanning``
    reads are left blank. Instances in ``highlight`` (row positions of
    ``hits``) are outlined; the ``label_top`` lowest-p instances are labelled.
    """
    if layout not in ("groups", "tracks"):
        raise ValueError("layout must be 'groups' or 'tracks'")
    groups = list(groups) if groups is not None else list(dict.fromkeys(tracks["group"]))
    track_order = (
        list(track_order) if track_order is not None else list(dict.fromkeys(tracks["track"]))
    )
    positions = np.sort(tracks["position"].unique())
    if window is not None:
        positions = positions[(positions >= window[0]) & (positions < window[1])]
    xs = _x(positions, coordinate_origin, coordinate_reverse)
    sort = np.argsort(xs)
    positions, xs = positions[sort], xs[sort]
    panels = groups if layout == "groups" else track_order
    lines = track_order if layout == "groups" else groups
    colors = (
        track_colors(track_order)
        if layout == "groups"
        else {g: plt.get_cmap("tab10")(i % 10) for i, g in enumerate(groups)}
    )

    shown_hits = hits.reset_index(drop=True)
    lanes = (
        pack_lanes(shown_hits["start"], shown_hits["end"]) if len(shown_hits) else np.zeros(0, int)
    )
    n_lanes = int(lanes.max()) + 1 if lanes.size else 1
    heights = [1.6] * len(panels) + [max(0.8, 0.16 * n_lanes)]
    width = 13 if window is None else 10
    total = sum(heights) + 1.6
    figure, axes = plt.subplots(
        len(panels) + 1,
        1,
        figsize=(width, total),
        sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.12},
        squeeze=False,
    )
    # Margins in inches, not fractions: a tall stack would otherwise open a
    # band of blank space above the first panel.
    figure.subplots_adjust(top=1 - 0.9 / total, bottom=0.5 / total)
    axes = axes[:, 0]
    for axis, panel in zip(axes[:-1], panels, strict=True):
        for line in lines:
            group, track = (panel, line) if layout == "groups" else (line, panel)
            values = track_matrix(tracks, group, track, positions)
            frame = tracks[(tracks["group"] == group) & (tracks["track"] == track)]
            spanning = (
                pd.Series(frame["spanning"].to_numpy(), index=frame["position"].to_numpy())
                .reindex(positions)
                .fillna(0)
                .to_numpy()
            )
            values = np.where(spanning >= min_spanning, values, np.nan)
            axis.plot(xs, values, color=colors[line], linewidth=1.0, label=str(line))
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(str(panel), fontsize=7, rotation=0, ha="right", va="center")
        axis.tick_params(labelsize=6)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(
        fontsize=6,
        ncol=min(len(lines), 6),
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        title="fraction of spanning reads" if layout == "groups" else None,
        title_fontsize=6,
    )

    lane_axis = axes[-1]
    families = shown_hits["family"].astype(str).replace("", "other") if len(shown_hits) else []
    fam_colors = family_colors(list(families)) if len(shown_hits) else {}
    outlined = set(int(i) for i in highlight)
    label_rows = (
        set(shown_hits["pvalue"].nsmallest(label_top).index)
        if label_top and len(shown_hits)
        else set()
    )
    for row, hit in shown_hits.iterrows():
        x0, x1 = _x(np.array([hit["start"], hit["end"]]), coordinate_origin, coordinate_reverse)
        left, span = min(x0, x1), abs(x1 - x0)
        lane = lanes[row]
        lane_axis.add_patch(
            Rectangle(
                (left, -lane - 0.9),
                span,
                0.8,
                facecolor=fam_colors[families.iloc[row]],
                edgecolor="black" if row in outlined else "none",
                linewidth=1.2 if row in outlined else 0,
            )
        )
        if row in label_rows or row in outlined:
            lane_axis.text(
                left + span / 2,
                -lane - 0.5,
                str(hit["motif_name"]),
                fontsize=4.5,
                ha="center",
                va="center",
            )
    lane_axis.set_ylim(-n_lanes - 0.2, 0.2)
    lane_axis.set_yticks([])
    lane_axis.set_ylabel(
        f"motifs\n({len(shown_hits)})", fontsize=7, rotation=0, ha="right", va="center"
    )
    lane_axis.set_xlim(xs.min() if xs.size else 0, xs.max() if xs.size else 1)
    lane_axis.set_xlabel(_x_label(coordinate_origin, coordinate_reverse), fontsize=8)
    lane_axis.tick_params(labelsize=6)
    if fam_colors:
        handles = [
            Patch(facecolor=color, label=family) for family, color in sorted(fam_colors.items())
        ][:30]
        figure.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.25),
            fontsize=5,
            frameon=False,
            title="motif family",
            title_fontsize=6,
        )
    if title:
        figure.suptitle(title, fontsize=9, y=1 - 0.15 / total)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
