"""Bulk feature-class tracks and motif-instance contrast (`MOT-03`).

Per group and position: of the reads that span the position (the class
layer is observed there), how many are in the class. Per motif instance and
group: the class fraction inside the motif against flanks of the same width
on either side -- a footprint sitting on a motif, and not around it, gives a
high ``contrast``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["group", "track", "position", "spanning", "in_class", "fraction"]


class TrackCounts:
    """Running ``spanning`` / ``in_class`` counts: groups x positions x tracks."""

    def __init__(self, positions: Sequence[int], tracks: Sequence[str]):
        self.positions = np.asarray(positions, dtype=np.int64)
        self.tracks = list(tracks)
        self.groups: list[str] = []
        self._spanning: list[np.ndarray] = []
        self._in_class: list[np.ndarray] = []

    def _row(self, group: str) -> int:
        if group not in self.groups:
            shape = (self.positions.size, len(self.tracks))
            self.groups.append(group)
            self._spanning.append(np.zeros(shape, dtype=np.int64))
            self._in_class.append(np.zeros(shape, dtype=np.int64))
        return self.groups.index(group)

    def add(self, groups: Sequence[str], values: np.ndarray, observed: np.ndarray) -> None:
        """One batch: ``values`` / ``observed`` are reads x positions x tracks."""
        groups = np.asarray([str(group) for group in groups])
        observed = np.asarray(observed, dtype=bool)
        in_class = observed & (np.nan_to_num(np.asarray(values, dtype=float), nan=0.0) > 0)
        for group in np.unique(groups):
            rows = groups == group
            index = self._row(str(group))
            self._spanning[index] += observed[rows].sum(axis=0)
            self._in_class[index] += in_class[rows].sum(axis=0)

    def merge(self, other: "TrackCounts") -> None:
        for group, spanning, in_class in zip(
            other.groups, other._spanning, other._in_class, strict=True
        ):
            index = self._row(group)
            self._spanning[index] += spanning
            self._in_class[index] += in_class

    def table(self) -> pd.DataFrame:
        frames = []
        for group, spanning, in_class in zip(
            self.groups, self._spanning, self._in_class, strict=True
        ):
            for t_index, track in enumerate(self.tracks):
                span = spanning[:, t_index]
                hits = in_class[:, t_index]
                with np.errstate(invalid="ignore", divide="ignore"):
                    fraction = np.where(span > 0, hits / span, np.nan)
                frames.append(
                    pd.DataFrame(
                        {
                            "group": group,
                            "track": track,
                            "position": self.positions,
                            "spanning": span,
                            "in_class": hits,
                            "fraction": fraction,
                        }
                    )
                )
        if not frames:
            return pd.DataFrame(columns=TRACK_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(
            ["group", "track", "position"], kind="stable", ignore_index=True
        )


def instance_contrast(
    tracks: pd.DataFrame, hits: pd.DataFrame, *, flank: int | None = None
) -> pd.DataFrame:
    """Per motif instance, group and track: mean class fraction inside the motif
    vs in the flanks either side (``flank`` bp; default the motif's width).

    ``contrast = inside - flanks``; ``min_spanning`` is the fewest spanning
    reads at any position inside. Instances without positions inside or in
    the flanks get NaN.
    """
    columns = [
        "instance",
        "motif_id",
        "motif_name",
        "family",
        "start",
        "end",
        "motif_strand",
        "pvalue",
        "group",
        "track",
        "inside",
        "flanks",
        "contrast",
        "min_spanning",
    ]
    if tracks.empty or hits.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (group, track), frame in tracks.groupby(["group", "track"], sort=True):
        positions = frame["position"].to_numpy()
        order = np.argsort(positions)
        positions = positions[order]
        fraction = frame["fraction"].to_numpy()[order]
        spanning = frame["spanning"].to_numpy()[order]

        def window(lo, hi):
            a, b = np.searchsorted(positions, [lo, hi])
            return slice(a, b)

        for instance, hit in enumerate(hits.itertuples(index=False)):
            width = int(hit.end) - int(hit.start)
            pad = int(flank) if flank is not None else width
            inside = window(int(hit.start), int(hit.end))
            left = window(int(hit.start) - pad, int(hit.start))
            right = window(int(hit.end), int(hit.end) + pad)
            inner = fraction[inside]
            around = np.concatenate([fraction[left], fraction[right]])
            with np.errstate(invalid="ignore"):
                inside_mean = float(np.nanmean(inner)) if np.isfinite(inner).any() else np.nan
                flank_mean = float(np.nanmean(around)) if np.isfinite(around).any() else np.nan
            rows.append(
                (
                    instance,
                    hit.motif_id,
                    getattr(hit, "motif_name", hit.motif_id),
                    getattr(hit, "family", ""),
                    int(hit.start),
                    int(hit.end),
                    getattr(hit, "motif_strand", "+"),
                    float(getattr(hit, "pvalue", np.nan)),
                    group,
                    track,
                    inside_mean,
                    flank_mean,
                    inside_mean - flank_mean,
                    int(spanning[inside].min()) if inner.size else 0,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def pack_lanes(starts: Sequence[int], ends: Sequence[int], gap: int = 2) -> np.ndarray:
    """Greedy lane per interval (in start order) so intervals in a lane never
    overlap (and keep ``gap`` bp apart)."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    lanes = np.zeros(starts.size, dtype=int)
    lane_ends: list[int] = []
    for index in np.argsort(starts, kind="stable"):
        for lane, last in enumerate(lane_ends):
            if starts[index] >= last + gap:
                lanes[index] = lane
                lane_ends[lane] = int(ends[index])
                break
        else:
            lanes[index] = len(lane_ends)
            lane_ends.append(int(ends[index]))
    return lanes


def filter_hits(
    hits: pd.DataFrame,
    *,
    reference: str | None = None,
    max_pvalue: float | None = None,
    families: Sequence[str] | None = None,
    motifs: Sequence[str] | None = None,
    window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Motif instances on one reference, by p-value, family / motif name or ID,
    and inside a window."""
    selected = hits
    if reference is not None:
        selected = selected[selected["reference"].astype(str) == str(reference)]
    if max_pvalue is not None:
        selected = selected[selected["pvalue"] <= max_pvalue]
    if families:
        wanted = {str(f).lower() for f in families}
        family = selected["family"].astype(str).str.lower()
        selected = selected[
            family.isin(wanted) | family.str.split(",").map(lambda parts: bool(wanted & set(parts)))
        ]
    if motifs:
        wanted = {str(m) for m in motifs}
        selected = selected[
            selected["motif_id"].astype(str).isin(wanted)
            | selected["motif_name"].astype(str).isin(wanted)
        ]
    if window is not None:
        selected = selected[(selected["end"] > window[0]) & (selected["start"] < window[1])]
    return selected.reset_index(drop=True)


def track_matrix(
    tracks: pd.DataFrame, group: str, track: str, positions: Sequence[int]
) -> np.ndarray:
    """One group's class fraction for one track at ``positions`` (NaN where absent)."""
    frame = tracks[(tracks["group"] == group) & (tracks["track"] == track)]
    series = pd.Series(frame["fraction"].to_numpy(), index=frame["position"].to_numpy())
    return series.reindex(np.asarray(positions)).to_numpy(dtype=float)


def groups_in(tracks: pd.DataFrame, order: Mapping[str, int] | None = None) -> list[str]:
    groups = list(dict.fromkeys(tracks["group"].astype(str)))
    return sorted(groups, key=lambda g: (order or {}).get(g, len(groups))) if order else groups
