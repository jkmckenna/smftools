"""Bulk HMM feature-class tracks with motif lanes, for a plan dataset (`MOT-03`).

Every channel of the dataset is a track -- e.g. HMM class layers at every
position (``site_context: all``): ``small_bound_stretch`` (TF-sized),
``medium_bound_stretch``, ``putative_nucleosome``, any accessible feature.
Per grouping, group and position the reads spanning the position and those
in the class are counted over the dataset's molecules (streamed in blocks
over worker processes, as `RPG`). Counts are cached by plan, dataset,
referenced files and groupings; the motif-instance contrast and the figures
are recomputed from them, so motif filters and display options are free to
change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from smftools.analysis.compute.motif_tracks import (
    TrackCounts,
    filter_hits,
    instance_contrast,
)

TRACKS_FILE = "tracks.parquet"
CONTRAST_FILE = "motif_contrast.parquet"
KEY_FILE = "tracks_key.json"
ALL = "all"


def _groupings(group_by) -> list[str]:
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(dict.fromkeys(str(column) for column in group_by))


def _shard(document, dataset_name, scope, groupings, channels, worker, workers):
    """One worker's blocks; plain data back (bound datasets do not pickle)."""
    from smftools.machine_learning.orchestration import bind_ml_dataset
    from smftools.machine_learning.plan import parse_ml_plan

    bound = bind_ml_dataset(parse_ml_plan(document), dataset_name, group_by=groupings, **scope)
    plan = bound.dataset.plan
    declared = [item.name for item in plan.dataset.input_schema.channels]
    index = [declared.index(channel) for channel in channels]
    coordinates = np.asarray(plan.coordinates)
    identity = bound.identity.set_index("molecule_uid")
    counts = {grouping: TrackCounts(coordinates, channels) for grouping in groupings or [ALL]}
    batches = (
        bound.iter_batches(worker_id=worker, num_workers=workers)
        if workers > 1
        else bound.iter_batches()
    )
    molecules = 0
    for batch in batches:
        values = batch.values[:, :, index]
        observed = batch.observed_mask[:, :, index]
        molecules += len(batch.molecule_uids)
        for grouping, accumulator in counts.items():
            labels = (
                identity.loc[list(batch.molecule_uids), grouping].astype(str).to_list()
                if grouping != ALL
                else [ALL] * len(batch.molecule_uids)
            )
            accumulator.add(labels, values, observed)
    frame = str(plan.dataset.input_schema.reference)
    return counts, frame, molecules


def compute_tracks(
    plan,
    dataset_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, str, int]:
    """``(tracks, frame reference, molecules)``: per grouping, group, track and
    position the spanning reads, the reads in the class and their fraction."""
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channels = list(channels) if channels else declared
    unknown = sorted(set(channels) - set(declared))
    if unknown:
        raise KeyError(f"channels {unknown} not in dataset channels {declared}")
    groupings = _groupings(group_by)
    scope: dict[str, Any] = {"project_dir": project_dir, "experiment_dir": experiment_dir}
    document = plan.to_dict()
    jobs = [
        (document, dataset_name, scope, groupings, channels, worker, workers)
        for worker in range(workers)
    ]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        from smftools.parallel_utils import configure_worker_threads

        with ProcessPoolExecutor(
            max_workers=workers, initializer=configure_worker_threads, initargs=(1,)
        ) as pool:
            shards = list(pool.map(_shard, *zip(*jobs)))
    else:
        shards = [_shard(*jobs[0])]
    merged, frame, molecules = shards[0]
    for counts, _, n in shards[1:]:
        molecules += n
        for grouping, accumulator in counts.items():
            merged[grouping].merge(accumulator)
    tables = [
        accumulator.table().assign(grouping=grouping) for grouping, accumulator in merged.items()
    ]
    tracks = pd.concat(tables, ignore_index=True)
    leading = ["grouping", "group", "track", "position"]
    tracks = tracks[[*leading, *[c for c in tracks.columns if c not in leading]]]
    return tracks, frame, molecules


def read_hits(motif_hits: str | Path | pd.DataFrame) -> pd.DataFrame:
    """A motif-instance table (`MOT-01`): a DataFrame, a parquet file, or a
    ``motifs scan`` output directory."""
    if isinstance(motif_hits, pd.DataFrame):
        return motif_hits
    path = Path(motif_hits)
    if path.is_dir():
        path = path / "motif_hits.parquet"
    return pd.read_parquet(path)


def run_motif_tracks(
    plan,
    dataset_name: str,
    output_dir: str | Path,
    motif_hits: str | Path | pd.DataFrame,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    motif_reference: str | None = None,
    regions: Mapping[str, tuple[int, int]] | None = None,
    max_pvalue: float | None = None,
    families: Sequence[str] | None = None,
    motifs: Sequence[str] | None = None,
    flank: int | None = None,
    contrast_track: str | None = None,
    highlight_top: int = 10,
    label_top: int = 15,
    min_spanning: int = 10,
    layout: str = "groups",
    coordinate_origin: float | None = None,
    coordinate_reverse: bool = False,
    workers: int = 1,
    refresh: bool = False,
    figures: bool = True,
) -> dict:
    """Tracks (computed or reused), per-instance contrast, figures and ``run.json``.

    Motif instances are those of ``motif_reference`` (default: the dataset's
    frame reference), filtered by ``max_pvalue`` / ``families`` / ``motifs``.
    ``regions`` (name -> half-open frame interval) add zoomed figures to the
    whole-span one. The ``highlight_top`` instances with the highest
    ``contrast`` on ``contrast_track`` (default: the first track) in each group
    are outlined -- among instances whose every position is spanned by at
    least ``min_spanning`` reads, the same threshold the figures draw at.
    """
    from smftools import __version__
    from smftools.tools.analysis_cache import cache_key

    output_dir = Path(output_dir)
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channels = list(channels) if channels else declared
    groupings = _groupings(group_by)
    key = cache_key(
        plan,
        dataset_name,
        base_dir=project_dir or experiment_dir,
        parameters={"channels": channels, "group_by": groupings},
    )
    tracks_path = output_dir / TRACKS_FILE
    key_path = output_dir / KEY_FILE
    reused = (
        not refresh
        and tracks_path.is_file()
        and key_path.is_file()
        and json.loads(key_path.read_text()).get("key") == key
    )
    if reused:
        saved = json.loads(key_path.read_text())
        tracks, frame, molecules = (
            pd.read_parquet(tracks_path),
            saved["frame_reference"],
            saved["molecules"],
        )
    else:
        tracks, frame, molecules = compute_tracks(
            plan,
            dataset_name,
            project_dir=project_dir,
            experiment_dir=experiment_dir,
            group_by=groupings,
            channels=channels,
            workers=workers,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tracks.to_parquet(tracks_path, index=False)
        key_path.write_text(
            json.dumps({"key": key, "frame_reference": frame, "molecules": molecules}, default=str)
        )

    hits = read_hits(motif_hits)
    reference = motif_reference or frame
    if not (hits["reference"].astype(str) == str(reference)).any():
        available = sorted(hits["reference"].astype(str).unique())
        raise KeyError(
            f"no motif instances on reference {reference!r} (the dataset frame); "
            f"the table has {available} -- pass motif_reference"
        )
    selected = filter_hits(
        hits, reference=reference, max_pvalue=max_pvalue, families=families, motifs=motifs
    )
    contrast_track = contrast_track or channels[0]
    contrasts = []
    for grouping, frame_tracks in tracks.groupby("grouping", sort=False):
        contrasts.append(
            instance_contrast(frame_tracks.drop(columns="grouping"), selected, flank=flank).assign(
                grouping=grouping
            )
        )
    contrast = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    contrast.to_parquet(output_dir / CONTRAST_FILE, index=False)

    written = []
    if figures:
        written = _draw(
            tracks,
            selected,
            contrast,
            output_dir,
            regions=regions or {},
            contrast_track=contrast_track,
            highlight_top=highlight_top,
            label_top=label_top,
            min_spanning=min_spanning,
            layout=layout,
            coordinate_origin=coordinate_origin,
            coordinate_reverse=coordinate_reverse,
            title=dataset_name,
        )
    record = {
        "key": key,
        "tracks_reused": reused,
        "frame_reference": frame,
        "motif_reference": reference,
        "molecules": int(molecules),
        "tracks": channels,
        "groups": {
            grouping: sorted(frame_tracks["group"].astype(str).unique())
            for grouping, frame_tracks in tracks.groupby("grouping")
        },
        "motif_instances": int(len(selected)),
        "filters": {"max_pvalue": max_pvalue, "families": families, "motifs": motifs},
        "contrast_track": contrast_track,
        "figures": written,
        "smftools_version": __version__,
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def _draw(
    tracks,
    hits,
    contrast,
    output_dir: Path,
    *,
    regions,
    contrast_track,
    highlight_top,
    label_top,
    min_spanning,
    layout,
    coordinate_origin,
    coordinate_reverse,
    title,
) -> list[str]:
    from smftools.analysis.compute.motif_tracks import filter_hits as window_hits
    from smftools.analysis.plot.motif_tracks import plot_motif_tracks

    written = []
    windows = {"span": None, **{name: tuple(bounds) for name, bounds in regions.items()}}
    for grouping, frame_tracks in tracks.groupby("grouping", sort=False):
        frame_tracks = frame_tracks.drop(columns="grouping")
        scored = contrast[
            (contrast["grouping"] == grouping) & (contrast["track"] == contrast_track)
        ]
        top = (
            scored[scored["min_spanning"] >= min_spanning]
            .dropna(subset=["contrast"])
            .sort_values("contrast", ascending=False)
            .groupby("group")
            .head(highlight_top)
        )
        for name, window in windows.items():
            shown = window_hits(hits, window=window) if window else hits.reset_index(drop=True)
            if window:
                original = hits.reset_index(drop=True)
                keep = (original["end"] > window[0]) & (original["start"] < window[1])
                index_map = {old: new for new, old in enumerate(np.flatnonzero(keep.to_numpy()))}
            else:
                index_map = {i: i for i in range(len(shown))}
            highlight = sorted(
                {index_map[i] for i in top["instance"].astype(int) if i in index_map}
            )
            path = output_dir / "figures" / _safe(grouping) / f"{_safe(name)}_{layout}.png"
            plot_motif_tracks(
                frame_tracks,
                shown,
                path,
                window=window,
                layout=layout,
                coordinate_origin=coordinate_origin,
                coordinate_reverse=coordinate_reverse,
                highlight=highlight,
                label_top=label_top,
                min_spanning=min_spanning,
                title=f"{title} [{grouping}] {name}",
            )
            written.append(str(path.relative_to(output_dir)))
    return written
