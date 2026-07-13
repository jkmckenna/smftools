"""Deterministic work planning for partitioned preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from smftools.constants import BARCODE, REFERENCE_STRAND, SAMPLE

BYTES_PER_WORKING_POSITION = 8


@dataclass(frozen=True)
class PreprocessTask:
    """One independently dispatchable reference/barcode/read-chunk work unit."""

    task_id: str
    reference: str
    barcode: str
    analysis_mode: str
    chunk_index: int
    core_start: int
    core_end: int
    load_start: int
    load_end: int
    n_reads: int
    estimated_memory_bytes: int
    read_ids: tuple[str, ...]

    def to_dict(self, *, include_read_ids: bool = True) -> dict[str, object]:
        """Return a serializable task record."""
        record = asdict(self)
        if not include_read_ids:
            record.pop("read_ids")
        return record


def _chunks(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for chunk_index, start in enumerate(range(0, len(values), size)):
        yield chunk_index, values[start : start + size]


def _plan_dict(spine, reference: str) -> dict[str, object]:
    plans = spine.uns.get("reference_plans", {})
    if not hasattr(plans, "get"):
        raise ValueError("spine lacks per-reference storage plans")
    plan = plans.get(reference)
    if not hasattr(plan, "items"):
        raise KeyError(f"spine lacks a storage plan for {reference!r}")
    return dict(plan)


def plan_preprocess_tasks(
    spine,
    *,
    target_task_memory_mb: int = 512,
    partition_by_barcode: bool = True,
) -> list[PreprocessTask]:
    """Plan bounded tasks by reference, genomic window, barcode, and read chunk.

    Genome tasks load a haloed interval but own only their non-overlapping core.
    Read chunks are sized from the loaded width so no task intentionally exceeds
    the configured dense working-memory target.
    """
    if target_task_memory_mb <= 0:
        raise ValueError("target_task_memory_mb must be positive")
    obs = spine.obs
    required = {REFERENCE_STRAND, "reference_start", "reference_end"}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"spine lacks preprocessing task columns: {sorted(missing)}")
    group_column = BARCODE if BARCODE in obs else SAMPLE if SAMPLE in obs else None
    memory_budget = int(target_task_memory_mb) * 1024**2
    tasks: list[PreprocessTask] = []

    for reference in sorted(obs[REFERENCE_STRAND].astype(str).unique()):
        plan = _plan_dict(spine, reference)
        reference_length = int(plan["reference_length"])
        analysis_mode = str(plan["analysis_mode"])
        tile_size = int(plan["tile_size"])
        tile_halo = int(plan["tile_halo"])
        reference_obs = obs.loc[obs[REFERENCE_STRAND].astype(str) == reference]
        windows = (
            [(0, reference_length, 0, reference_length)]
            if analysis_mode == "locus"
            else [
                (
                    core_start,
                    min(core_start + tile_size, reference_length),
                    max(0, core_start - tile_halo),
                    min(reference_length, core_start + tile_size + tile_halo),
                )
                for core_start in range(0, reference_length, tile_size)
            ]
        )
        for core_start, core_end, load_start, load_end in windows:
            # Core overlap assigns ownership. Halo-only reads must not create work
            # for a genomic span with no data of its own.
            overlapping = reference_obs.loc[
                (reference_obs["reference_start"].astype("int64") < core_end)
                & (reference_obs["reference_end"].astype("int64") > core_start)
            ]
            if overlapping.empty:
                continue
            if partition_by_barcode and group_column is not None:
                group_values = overlapping[group_column].astype(object)
                group_values = group_values.where(group_values.notna(), "unclassified").astype(str)
                groups = [
                    (barcode, overlapping.loc[index])
                    for barcode, index in group_values.groupby(
                        group_values, sort=True
                    ).groups.items()
                ]
            else:
                groups = [("*", overlapping)]
            loaded_width = load_end - load_start
            reads_per_chunk = max(
                1, memory_budget // max(1, loaded_width * BYTES_PER_WORKING_POSITION)
            )
            for barcode, barcode_obs in groups:
                read_ids = sorted(map(str, barcode_obs.index))
                for chunk_index, chunk_read_ids in _chunks(read_ids, reads_per_chunk):
                    task_id = f"{reference}|{barcode}|{core_start}-{core_end}|{chunk_index:05d}"
                    tasks.append(
                        PreprocessTask(
                            task_id=task_id,
                            reference=reference,
                            barcode=str(barcode),
                            analysis_mode=analysis_mode,
                            chunk_index=chunk_index,
                            core_start=core_start,
                            core_end=core_end,
                            load_start=load_start,
                            load_end=load_end,
                            n_reads=len(chunk_read_ids),
                            estimated_memory_bytes=(
                                len(chunk_read_ids) * loaded_width * BYTES_PER_WORKING_POSITION
                            ),
                            read_ids=tuple(chunk_read_ids),
                        )
                    )
    return tasks


def write_preprocess_task_catalog(tasks: Iterable[PreprocessTask], path: str | Path) -> Path:
    """Persist scheduler-facing task metadata without embedding read-id lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task.to_dict(include_read_ids=False) for task in tasks).to_parquet(
        path, index=False
    )
    return path
