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


def _bind(document, dataset_name, scope, group_by):
    from smftools.machine_learning.orchestration import bind_ml_dataset
    from smftools.machine_learning.plan import parse_ml_plan

    return bind_ml_dataset(
        parse_ml_plan(document),
        dataset_name,
        group_by=[group_by] if group_by else [],
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
    uids, stats, power = [], [], {name: [] for name in grids}
    for batch in batches:
        uids.extend(batch.molecule_uids)
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
    return uids, stats, power, grids, frame, bound.identity


def compute_read_periodicity(
    plan,
    dataset_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    regions: Sequence[tuple[int, int]] | None = None,
    channel: str | None = None,
    group_by: str | None = None,
    period_range: tuple[float, float] = PERIOD_RANGE_BP,
    peak_range: tuple[float, float] = PEAK_RANGE_BP,
    min_cycles: float = MIN_CYCLES,
    poly_degree: int = LS_POLY_DEGREE,
    min_sites: int = MIN_SITES_PER_READ,
    min_coverage: float = MIN_COVERAGE,
    workers: int = 1,
) -> ReadPeriodicity:
    """Periodograms of every molecule of a plan dataset over each region.

    ``regions`` are half-open intervals in the dataset's frame coordinates;
    without them, one region per contiguous window of the plan's positions.
    ``channel`` defaults to the dataset's first; ``group_by`` adds an identity
    or label-table column to the statistics. Regions too short for the period
    range are narrowed, or skipped (`analysis.compute.read_periodicity`).
    """
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channel = channel or declared[0]
    if channel not in declared:
        raise KeyError(f"channel {channel!r} not in dataset channels {declared}")
    regions = [(int(start), int(end)) for start, end in regions] if regions else None
    options = {
        "period_range": tuple(float(v) for v in period_range),
        "peak_range": tuple(float(v) for v in peak_range),
        "min_cycles": float(min_cycles),
        "poly_degree": int(poly_degree),
        "min_sites": int(min_sites),
        "min_coverage": float(min_coverage),
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

    _, _, _, grids, frame, identity = shards[0]
    # The dataset's molecule order, whatever the worker split.
    order = {uid: rank for rank, uid in enumerate(identity["molecule_uid"])}
    stats_parts, power = [], {}
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
    stats = pd.concat(stats_parts, ignore_index=True)
    columns = [c for c in ("read_id", "experiment_id", "physical_reference") if c in identity]
    extra = identity.set_index("molecule_uid")[columns]
    if group_by:
        extra = extra.assign(group=identity.set_index("molecule_uid")[group_by].astype(str))
    stats = stats.join(extra, on="molecule_uid")
    if not group_by:
        stats["group"] = ALL
    leading = ["molecule_uid", *columns, "group", "region", "status"]
    stats = stats[[*leading, *[c for c in stats.columns if c not in leading]]]
    return ReadPeriodicity(
        stats=stats,
        power=power,
        grids=grids,
        channel=channel,
        frame_reference=frame,
        parameters={**options, "regions": [g.name for g in grids.values()]},
    )
