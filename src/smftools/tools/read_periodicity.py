"""Per-read periodograms over regions for a plan dataset (`RPG-02`).

Molecules and the input signal come from a datasets-only ML plan through
`bind_ml_dataset` (`MLX-10`): the channel names the layer and the positions it
is read at (a site context, or ``all`` for every position, `RPG-01`), filters
the QC/dedup flags, a label table or identity column the groups. Batches
stream through `analysis.compute.read_periodicity` region by region, over
worker processes by whole blocks (`MLX-11`); results come back in the
dataset's molecule order whatever the worker count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.analysis.compute.ls_periodicity import LS_POLY_DEGREE, MIN_SITES_PER_READ
from smftools.analysis.compute.read_periodicity import (
    MIN_COVERAGE,
    MIN_CYCLES,
    PEAK_RANGE_BP,
    PERIOD_RANGE_BP,
    PeriodGrid,
    period_grid,
    read_periodograms,
)

ALL = "all"
# Statistics columns a grouping may not be named after.
RESERVED_COLUMNS = frozenset(
    {
        "molecule_uid",
        "region",
        "status",
        "n_sites",
        "coverage",
        "peak_period_bp",
        "snr",
        "peak_power",
        "peak_power_raw",
        "fwhm_bp",
        "peak_at_edge",
    }
)


@dataclass
class ReadPeriodicity:
    """Per-(read, region) statistics, and per region the power matrix and grid.

    ``power[region]`` rows follow ``stats`` rows of that region, in order.
    """

    stats: pd.DataFrame
    power: dict[str, np.ndarray]
    grids: dict[str, PeriodGrid]
    channel: str
    frame_reference: str
    parameters: dict = field(default_factory=dict)
    # Input values kept for plotting (`keep_per_group`): per region,
    # (molecule_uids, positions, values reads x positions, NaN unobserved).
    plot_values: dict[str, tuple[list, np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def regions(self) -> pd.DataFrame:
        return pd.DataFrame([grid.record() for grid in self.grids.values()])


def windows(coordinates: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous runs of coordinates as half-open ``(start, end)`` intervals."""
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if coordinates.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(coordinates) != 1)
    starts = np.r_[coordinates[0], coordinates[breaks + 1]]
    ends = np.r_[coordinates[breaks], coordinates[-1]] + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _groupings(group_by) -> list[str]:
    """``group_by`` as a list: one column, several, or none."""
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(dict.fromkeys(str(column) for column in group_by))


def plot_selection(identity: pd.DataFrame, group_by, per_group: int, seed: int) -> set:
    """Up to ``per_group`` molecules of each combination of groups, chosen deterministically.

    With several groupings every group of every grouping keeps molecules.
    Every worker computes the same set from the same identity table.
    """
    groupings = _groupings(group_by)
    if groupings:
        groups = identity[groupings].astype(str).agg("\x1f".join, axis=1)
    else:
        groups = pd.Series(ALL, index=identity.index)
    chosen = set()
    for _, uids in identity["molecule_uid"].astype(str).groupby(groups.to_numpy(), sort=True):
        uids = np.sort(uids.to_numpy())
        if uids.size > per_group:
            uids = np.sort(np.random.default_rng(seed).choice(uids, size=per_group, replace=False))
        chosen.update(uids.tolist())
    return chosen


def _bind(document, dataset_name, scope, group_by):
    from smftools.machine_learning.orchestration import bind_ml_dataset
    from smftools.machine_learning.plan import parse_ml_plan

    return bind_ml_dataset(
        parse_ml_plan(document),
        dataset_name,
        group_by=_groupings(group_by),
        **scope,
    )


def _design(batch, index: int) -> np.ndarray:
    """The channel's design mask per read; batches may hold one row for all reads."""
    design = batch.design_mask
    if design.ndim == 3:
        return design[:, :, index]
    return np.broadcast_to(design[:, index], batch.values.shape[:2])


def _shard(document, dataset_name, scope, group_by, channel, regions, options, worker, workers):
    """One worker's blocks; plain data back (bound datasets do not pickle)."""
    bound = _bind(document, dataset_name, scope, group_by)
    plan = bound.dataset.plan
    channels = [item.name for item in plan.dataset.input_schema.channels]
    index = channels.index(channel)
    coordinates = np.asarray(plan.coordinates)
    grids = {
        grid.name: grid
        for grid in (
            period_grid(
                start,
                end,
                period_range=options["period_range"],
                peak_range=options["peak_range"],
                min_cycles=options["min_cycles"],
            )
            for start, end in (regions or windows(coordinates))
        )
    }
    batches = (
        bound.iter_batches(worker_id=worker, num_workers=workers)
        if workers > 1
        else bound.iter_batches()
    )
    keep = (
        plot_selection(bound.identity, group_by, options["keep_per_group"], options["seed"])
        if options["keep_per_group"]
        else set()
    )
    inside = {
        name: (coordinates >= grid.start) & (coordinates < grid.end) for name, grid in grids.items()
    }
    uids, stats, power = [], [], {name: [] for name in grids}
    kept = {name: ([], []) for name in grids}
    for batch in batches:
        uids.extend(batch.molecule_uids)
        rows = [i for i, uid in enumerate(batch.molecule_uids) if uid in keep]
        if rows:
            signal = np.where(
                batch.observed_mask[rows, :, index], batch.values[rows, :, index], np.nan
            ).astype(np.float32)
            for name in grids:
                kept[name][0].extend(batch.molecule_uids[i] for i in rows)
                kept[name][1].append(signal[:, inside[name]])
        for name, grid in grids.items():
            block, table = read_periodograms(
                batch.coordinates,
                batch.values[:, :, index],
                batch.observed_mask[:, :, index],
                _design(batch, index),
                grid,
                poly_degree=options["poly_degree"],
                min_sites=options["min_sites"],
                min_coverage=options["min_coverage"],
            )
            stats.append(table.assign(molecule_uid=list(batch.molecule_uids)))
            power[name].append(block)
    frame = str(plan.dataset.input_schema.reference)
    positions = {name: coordinates[mask] for name, mask in inside.items()}
    return uids, stats, power, grids, frame, bound.identity, kept, positions


def compute_read_periodicity(
    plan,
    dataset_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    regions: Sequence[tuple[int, int]] | None = None,
    channel: str | None = None,
    group_by: str | Sequence[str] | None = None,
    period_range: tuple[float, float] = PERIOD_RANGE_BP,
    peak_range: tuple[float, float] = PEAK_RANGE_BP,
    min_cycles: float = MIN_CYCLES,
    poly_degree: int = LS_POLY_DEGREE,
    min_sites: int = MIN_SITES_PER_READ,
    min_coverage: float = MIN_COVERAGE,
    keep_per_group: int = 0,
    seed: int = 0,
    workers: int = 1,
) -> ReadPeriodicity:
    """Periodograms of every molecule of a plan dataset over each region.

    ``regions`` are half-open intervals in the dataset's frame coordinates;
    without them, one region per contiguous window of the plan's positions.
    ``channel`` defaults to the dataset's first; ``group_by`` adds an identity
    or label-table column to the statistics. Regions too short for the period
    range are narrowed, or skipped (`analysis.compute.read_periodicity`).
    ``keep_per_group`` > 0 also keeps the input values of up to that many
    molecules per group (chosen with ``seed``) for figures.
    """
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channel = channel or declared[0]
    if channel not in declared:
        raise KeyError(f"channel {channel!r} not in dataset channels {declared}")
    group_by = _groupings(group_by)
    reserved = sorted(set(group_by) & RESERVED_COLUMNS)
    if reserved:
        raise ValueError(f"grouping columns clash with statistics columns: {reserved}")
    regions = [(int(start), int(end)) for start, end in regions] if regions else None
    options = {
        "period_range": tuple(float(v) for v in period_range),
        "peak_range": tuple(float(v) for v in peak_range),
        "min_cycles": float(min_cycles),
        "poly_degree": int(poly_degree),
        "min_sites": int(min_sites),
        "min_coverage": float(min_coverage),
        "keep_per_group": int(keep_per_group),
        "seed": int(seed),
    }
    for start, end in regions or []:
        period_grid(
            start, end, **{k: options[k] for k in ("period_range", "peak_range", "min_cycles")}
        )
    scope: dict[str, Any] = {"project_dir": project_dir, "experiment_dir": experiment_dir}
    document = plan.to_dict()
    jobs = [
        (document, dataset_name, scope, group_by, channel, regions, options, worker, workers)
        for worker in range(workers)
    ]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            shards = list(pool.map(_shard, *zip(*jobs)))
    else:
        shards = [_shard(*jobs[0])]

    _, _, _, grids, frame, identity, _, positions = shards[0]
    # The dataset's molecule order, whatever the worker split.
    order = {uid: rank for rank, uid in enumerate(identity["molecule_uid"])}
    stats_parts, power, plot_values = [], {}, {}
    for name in grids:
        tables = [t for shard in shards for t in shard[1] if t["region"].iat[0] == name]
        blocks = [b for shard in shards for b in shard[2][name]]
        table = pd.concat(tables, ignore_index=True)
        matrix = (
            np.concatenate(blocks, axis=0) if blocks else np.empty((0, grids[name].periods.size))
        )
        ranks = table["molecule_uid"].map(order).to_numpy()
        sort = np.argsort(ranks, kind="stable")
        stats_parts.append(table.iloc[sort].reset_index(drop=True))
        power[name] = matrix[sort]
        kept_uids = [uid for shard in shards for uid in shard[6][name][0]]
        if kept_uids:
            kept_values = np.concatenate([v for shard in shards for v in shard[6][name][1]])
            ranks = np.argsort([order[uid] for uid in kept_uids], kind="stable")
            plot_values[name] = (
                [kept_uids[i] for i in ranks],
                positions[name],
                kept_values[ranks],
            )
    stats = pd.concat(stats_parts, ignore_index=True)
    columns = [c for c in ("read_id", "experiment_id", "physical_reference") if c in identity]
    extra = identity.set_index("molecule_uid")[columns]
    for column in group_by:
        extra = extra.assign(**{column: identity.set_index("molecule_uid")[column].astype(str)})
    stats = stats.join(extra, on="molecule_uid")
    # "group" names the first grouping (or "all") -- unless a grouping is
    # itself called "group" (a plan's label column), which it then is.
    if "group" not in group_by:
        stats["group"] = stats[group_by[0]] if group_by else ALL
    leading = ["molecule_uid", *columns, *group_by, "group", "region", "status"]
    leading = list(dict.fromkeys(leading))
    stats = stats[[*leading, *[c for c in stats.columns if c not in leading]]]
    return ReadPeriodicity(
        stats=stats,
        power=power,
        grids=grids,
        channel=channel,
        frame_reference=frame,
        parameters={**options, "regions": [g.name for g in grids.values()], "group_by": group_by},
        plot_values=plot_values,
    )


RESULT_KEY = "periodicity_key.json"
STATS_FILE = "read_periodicity.parquet"
REGIONS_FILE = "regions.parquet"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name)).strip("_") or "x"


def save_results(result: ReadPeriodicity, output_dir: Path, key: dict) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    result.stats.to_parquet(output_dir / STATS_FILE, index=False)
    result.regions.to_parquet(output_dir / REGIONS_FILE, index=False)
    for name, grid in result.grids.items():
        np.save(output_dir / f"power_{name}.npy", result.power[name])
        np.save(output_dir / f"periods_{name}.npy", grid.periods)
        if name in result.plot_values:
            uids, positions, values = result.plot_values[name]
            np.savez_compressed(
                output_dir / f"plot_values_{name}.npz",
                molecule_uids=np.asarray(uids, dtype=str),
                positions=positions,
                values=values,
            )
    (output_dir / RESULT_KEY).write_text(
        json.dumps(
            {
                "key": key,
                "channel": result.channel,
                "frame_reference": result.frame_reference,
                "parameters": result.parameters,
            },
            indent=2,
            default=str,
        )
    )


def load_results(output_dir: Path, key: dict) -> ReadPeriodicity | None:
    """Saved results made under ``key``, or None."""
    import json

    meta = output_dir / RESULT_KEY
    if not meta.is_file() or not (output_dir / STATS_FILE).is_file():
        return None
    saved = json.loads(meta.read_text())
    if saved.get("key") != key:
        return None
    parameters = saved["parameters"]
    stats = pd.read_parquet(output_dir / STATS_FILE)
    grids, power, plot_values = {}, {}, {}
    for record in pd.read_parquet(output_dir / REGIONS_FILE).to_dict("records"):
        grid = period_grid(
            int(record["start"]),
            int(record["end"]),
            period_range=tuple(parameters["period_range"]),
            peak_range=tuple(parameters["peak_range"]),
            min_cycles=parameters["min_cycles"],
        )
        grids[grid.name] = grid
        power[grid.name] = np.load(output_dir / f"power_{grid.name}.npy")
        values_path = output_dir / f"plot_values_{grid.name}.npz"
        if values_path.is_file():
            with np.load(values_path) as data:
                plot_values[grid.name] = (
                    data["molecule_uids"].astype(str).tolist(),
                    data["positions"],
                    data["values"],
                )
    return ReadPeriodicity(
        stats=stats,
        power=power,
        grids=grids,
        channel=saved["channel"],
        frame_reference=saved["frame_reference"],
        parameters=parameters,
        plot_values=plot_values,
    )


def draw_figures(
    result: ReadPeriodicity,
    output_dir: Path,
    *,
    max_reads: int,
    seed: int = 0,
    title: str = "",
    descending: bool = True,
    coordinate_origin: float | None = None,
    coordinate_reverse: bool = False,
) -> list[str]:
    """Per grouping and region: a paired clustermap per group, and all groups binned.

    Figures go to ``figures/<grouping>/<region>/``, or ``figures/<region>/``
    without a grouping.
    """
    from smftools.analysis.plot.read_periodicity import plot_read_periodicity_clustermap

    groupings = result.parameters.get("group_by") or [None]
    written = []
    for grouping, (name, grid) in (
        (grouping, item) for grouping in groupings for item in result.grids.items()
    ):
        if grid.status != "ok" or name not in result.plot_values:
            continue
        uids, positions, values = result.plot_values[name]
        region_stats = result.stats.loc[result.stats["region"] == name].reset_index(drop=True)
        row_of = {uid: i for i, uid in enumerate(region_stats["molecule_uid"])}
        rows = np.array([row_of[uid] for uid in uids])
        peaks = region_stats["peak_period_bp"].to_numpy()[rows]
        groups = region_stats[grouping or "group"].astype(str).to_numpy()[rows]
        power = result.power[name][rows]
        ranges = f"periods {grid.period_range[0]:g}-{grid.period_range[1]:g} bp" + (
            " (narrowed to the region)" if grid.narrowed else ""
        )
        heading = " | ".join(
            part for part in (title, f"{result.channel}", f"region {name}", ranges) if part
        )
        directory = output_dir / "figures" / (_safe(grouping) if grouping else "") / name
        directory.mkdir(parents=True, exist_ok=True)
        selections = [(group, groups == group) for group in dict.fromkeys(groups)]
        if len(selections) > 1:
            selections.append(("all_groups", np.ones(groups.size, dtype=bool)))
        for label, mask in selections:
            path = directory / f"{_safe(label)}.png"
            plot_read_periodicity_clustermap(
                values[mask],
                positions,
                power[mask],
                grid.periods,
                peaks[mask],
                path,
                bins=groups[mask] if label == "all_groups" else None,
                peak_range=grid.peak_range,
                max_reads=max_reads if label != "all_groups" else max_reads * 2,
                seed=seed,
                input_label=result.channel,
                descending=descending,
                coordinate_origin=coordinate_origin,
                coordinate_reverse=coordinate_reverse,
                title=f"{heading} | {grouping + ': ' if grouping else ''}{label}",
            )
            written.append(str(path.relative_to(output_dir)))
    return written


def run_periodicity(
    plan,
    dataset_name: str,
    output_dir: str | Path,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    regions: Sequence[tuple[int, int]] | None = None,
    channel: str | None = None,
    group_by: str | Sequence[str] | None = None,
    period_range: tuple[float, float] = PERIOD_RANGE_BP,
    peak_range: tuple[float, float] = PEAK_RANGE_BP,
    min_cycles: float = MIN_CYCLES,
    poly_degree: int = LS_POLY_DEGREE,
    min_sites: int = MIN_SITES_PER_READ,
    min_coverage: float = MIN_COVERAGE,
    max_reads_per_plot: int = 1000,
    seed: int = 0,
    workers: int = 1,
    refresh: bool = False,
    figures: bool = True,
    descending: bool = True,
    coordinate_origin: float | None = None,
    coordinate_reverse: bool = False,
) -> dict:
    """Compute (or reuse) per-read periodograms; write tables, figures and ``run.json``.

    Results are reused when their key -- plan, dataset, the files it references
    (`F72`), channel, grouping, regions and every parameter -- matches. Figure
    options (``descending``, ``coordinate_*``) are not part of it: a cached run
    redraws with them.
    """
    import json

    from smftools import __version__
    from smftools.tools.analysis_cache import cache_key

    output_dir = Path(output_dir)
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channel = channel or declared[0]
    regions = [(int(s), int(e)) for s, e in regions] if regions else None
    # Plot values are kept whether or not figures are drawn now, so the
    # cache key -- and a later run with figures -- does not depend on it.
    keep = int(np.ceil(max_reads_per_plot * 1.25))
    parameters = {
        "channel": channel,
        "group_by": _groupings(group_by),
        "regions": regions,
        "period_range": list(period_range),
        "peak_range": list(peak_range),
        "min_cycles": min_cycles,
        "poly_degree": poly_degree,
        "min_sites": min_sites,
        "min_coverage": min_coverage,
        "keep_per_group": keep,
        "seed": seed,
    }
    key = cache_key(
        plan, dataset_name, base_dir=project_dir or experiment_dir, parameters=parameters
    )
    result = None if refresh else load_results(output_dir, key)
    reused = result is not None
    if result is None:
        result = compute_read_periodicity(
            plan,
            dataset_name,
            project_dir=project_dir,
            experiment_dir=experiment_dir,
            regions=regions,
            channel=channel,
            group_by=group_by,
            period_range=period_range,
            peak_range=peak_range,
            min_cycles=min_cycles,
            poly_degree=poly_degree,
            min_sites=min_sites,
            min_coverage=min_coverage,
            keep_per_group=keep,
            seed=seed,
            workers=workers,
        )
        save_results(result, output_dir, key)
    written = (
        draw_figures(
            result,
            output_dir,
            max_reads=max_reads_per_plot,
            seed=seed,
            title=dataset_name,
            descending=descending,
            coordinate_origin=coordinate_origin,
            coordinate_reverse=coordinate_reverse,
        )
        if figures
        else []
    )
    status = result.stats.groupby(["region", "status"]).size()
    record = {
        "key": key,
        "results_reused": reused,
        "frame_reference": result.frame_reference,
        "regions": result.regions.to_dict("records"),
        "status_counts": {f"{r}|{s}": int(n) for (r, s), n in status.items()},
        "groups": {
            grouping: sorted(result.stats[grouping].astype(str).unique())
            for grouping in (result.parameters.get("group_by") or ["group"])
        },
        "molecules": int(result.stats["molecule_uid"].nunique()),
        "figures": written,
        "smftools_version": __version__,
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return record
