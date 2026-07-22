"""Resolve presentation regions against completed tasks and stitch plot inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .partition_query import query_derived_index
from .partition_read import resolve_relative_path

PLOT_REGION_PLANNER_VERSION = 1
DEFAULT_PLOT_SELECTION_SEED = 0


@dataclass(frozen=True)
class PlotRegionPlan:
    """One stored-coordinate plot interval and its completed task provenance."""

    region_id: str
    name: str
    reference: str
    start: int
    end: int
    original_reference: str | None
    original_start: int | None
    original_end: int | None
    task_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    model_ids: tuple[str, ...]
    source: str = "task"
    gaps: tuple[tuple[int, int], ...] = ()
    planner_version: int = PLOT_REGION_PLANNER_VERSION

    def source_manifest(self) -> dict[str, object]:
        """Return stable interval and task provenance for a plot manifest."""
        return {
            "planner_version": self.planner_version,
            "region_id": self.region_id,
            "name": self.name,
            "reference": self.reference,
            "start": self.start,
            "end": self.end,
            "original_reference": self.original_reference,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "task_ids": list(self.task_ids),
            "artifact_paths": list(self.artifact_paths),
            "model_ids": list(self.model_ids),
            "source": self.source,
            "gaps": [list(gap) for gap in self.gaps],
        }


@dataclass(frozen=True)
class PlotReadSelection:
    """Deterministic molecule selection made before plot materialization."""

    read_ids: tuple[str, ...]
    molecule_uids: tuple[str, ...]
    seed: int
    selection_sha256: str


def _run_root(spine_path: str | Path | None) -> Path | None:
    return None if spine_path is None else Path(spine_path).parent.parent


def _inherited_path(spine, key: str, *, spine_path: str | Path | None) -> Path | None:
    value = spine.uns.get(key)
    resolved = resolve_relative_path(value, _run_root(spine_path))
    if value and resolved is None:
        raise ValueError(
            f"plot-region planning cannot resolve relative spine.uns[{key!r}] "
            "without the on-disk spine path"
        )
    return resolved


def plot_catalog_path(spine, *, spine_path: str | Path | None = None) -> Path | None:
    """Resolve the inherited normalized plot-region catalog, if configured."""
    catalogs = spine.uns.get("region_catalogs", {})
    if not isinstance(catalogs, Mapping):
        raise ValueError("spine.uns['region_catalogs'] must be a mapping")
    value = catalogs.get("plot")
    resolved = resolve_relative_path(value, _run_root(spine_path))
    if value and resolved is None:
        raise ValueError(
            "plot-region planning cannot resolve the relative plot catalog "
            "without the on-disk spine path"
        )
    return resolved


def _task_frame(task_catalog: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = (
        task_catalog.copy()
        if isinstance(task_catalog, pd.DataFrame)
        else pd.read_parquet(task_catalog)
    )
    required = {"task_id", "reference", "core_start", "core_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"task catalog lacks required plot-planning columns: {sorted(missing)}")
    if "status" in frame:
        frame = frame.loc[frame["status"].astype(str).isin({"complete", "completed"})]
    return frame


def _flatten_strings(values: Iterable[object]) -> tuple[str, ...]:
    flattened: set[str] = set()
    for value in values:
        if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
            continue
        if isinstance(value, str):
            if value:
                flattened.add(value)
            continue
        try:
            flattened.update(
                str(item)
                for item in value
                if item is not None
                and item is not pd.NA
                and not (np.isscalar(item) and pd.isna(item))
                and str(item)
            )
        except TypeError:
            flattened.add(str(value))
    return tuple(sorted(flattened))


def _merged(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _coverage_gaps(start: int, end: int, coverage: Iterable[tuple[int, int]]) -> tuple:
    cursor = start
    gaps: list[tuple[int, int]] = []
    for covered_start, covered_end in _merged(coverage):
        covered_start = max(start, covered_start)
        covered_end = min(end, covered_end)
        if covered_end <= covered_start:
            continue
        if covered_start > cursor:
            gaps.append((cursor, covered_start))
        cursor = max(cursor, covered_end)
    if cursor < end:
        gaps.append((cursor, end))
    return tuple(gaps)


def _fallback_region_id(reference: str, start: int, end: int) -> str:
    payload = f"{PLOT_REGION_PLANNER_VERSION}\0{reference}\0{start}\0{end}".encode()
    return f"plot_{hashlib.sha256(payload).hexdigest()[:24]}"


def _mapped_plot_requests(
    spine, *, spine_path: str | Path | None
) -> list[dict[str, object]] | None:
    catalog_path = plot_catalog_path(spine, spine_path=spine_path)
    if catalog_path is None:
        return None
    if not catalog_path.is_file():
        raise FileNotFoundError(f"plot region catalog not found: {catalog_path}")
    mapping_path = _inherited_path(spine, "reference_interval_map", spine_path=spine_path)
    if mapping_path is None or not mapping_path.is_file():
        raise FileNotFoundError(
            f"reference interval map required for plot planning was not found: {mapping_path}"
        )
    catalog = pd.read_parquet(catalog_path)
    mapping = pd.read_parquet(mapping_path)
    requests: list[dict[str, object]] = []
    for region in catalog.itertuples(index=False):
        candidates = mapping.loc[
            mapping["original_reference"].astype(str) == str(region.original_reference)
        ]
        for stored in candidates.itertuples(index=False):
            overlap_start = max(int(region.original_start), int(stored.original_start))
            overlap_end = min(int(region.original_end), int(stored.original_end))
            if overlap_end <= overlap_start:
                continue
            orientation = int(stored.coordinate_orientation)
            if orientation != 1:
                raise ValueError(
                    "plot-region planner currently requires forward stored-coordinate mappings"
                )
            requests.append(
                {
                    "region_id": str(region.region_id),
                    "name": (
                        str(region.name) if not pd.isna(region.name) else str(region.region_id)
                    ),
                    "reference": str(stored.stored_reference),
                    "start": int(stored.stored_start) + overlap_start - int(stored.original_start),
                    "end": int(stored.stored_start) + overlap_end - int(stored.original_start),
                    "original_reference": str(region.original_reference),
                    "original_start": overlap_start,
                    "original_end": overlap_end,
                    "source": "plot",
                }
            )
    if catalog.shape[0] and not requests:
        raise ValueError("plot region catalog has no intervals represented by stored references")
    return requests


def resolve_plot_region_plans(
    spine,
    task_catalog: str | Path | pd.DataFrame,
    *,
    spine_path: str | Path | None = None,
    allow_gaps: bool = False,
    fallback_regions: pd.DataFrame | None = None,
) -> list[PlotRegionPlan]:
    """Resolve requested plot intervals against completed authoritative cores.

    Explicit plot regions are mapped from original FASTA coordinates. Without
    one, callers may provide their legacy stored-coordinate regions; otherwise
    each distinct completed core remains one plot region for backward
    compatibility. Every contributing core is clipped to the requested window,
    so adjacent cores cover each output position exactly once.
    """
    tasks = _task_frame(task_catalog)
    mapped_requests = _mapped_plot_requests(spine, spine_path=spine_path)
    if mapped_requests is not None:
        requests = mapped_requests
        if not requests:
            return []
    elif fallback_regions is not None:
        required = {"reference", "start", "end"}
        missing = required.difference(fallback_regions.columns)
        if missing:
            raise ValueError(f"fallback plot regions lack required columns: {sorted(missing)}")
        requests = [
            {
                "region_id": _fallback_region_id(
                    str(region.reference), int(region.start), int(region.end)
                ),
                "name": str(getattr(region, "name", "") or ""),
                "reference": str(region.reference),
                "start": int(region.start),
                "end": int(region.end),
                "original_reference": None,
                "original_start": None,
                "original_end": None,
                "source": str(getattr(region, "source", "fallback")),
            }
            for region in fallback_regions.itertuples(index=False)
        ]
        if fallback_regions.empty:
            return []
    else:
        requests = []
        for row in tasks.drop_duplicates(
            ["reference", "core_start", "core_end"], keep="first"
        ).itertuples(index=False):
            analysis_core_id = getattr(row, "analysis_core_id", None)
            region_id = (
                str(analysis_core_id)
                if analysis_core_id is not None and not pd.isna(analysis_core_id)
                else _fallback_region_id(str(row.reference), int(row.core_start), int(row.core_end))
            )
            requests.append(
                {
                    "region_id": region_id,
                    "name": region_id,
                    "reference": str(row.reference),
                    "start": int(row.core_start),
                    "end": int(row.core_end),
                    "original_reference": None,
                    "original_start": None,
                    "original_end": None,
                    "source": "task",
                }
            )

    plans: list[PlotRegionPlan] = []
    for request in requests:
        reference = str(request["reference"])
        start, end = int(request["start"]), int(request["end"])
        contributing = tasks.loc[
            (tasks["reference"].astype(str) == reference)
            & (tasks["core_start"].astype("int64") < end)
            & (tasks["core_end"].astype("int64") > start)
        ]
        coverage = [
            (int(row.core_start), int(row.core_end)) for row in contributing.itertuples(index=False)
        ]
        gaps = _coverage_gaps(start, end, coverage)
        if gaps and not allow_gaps:
            formatted = ", ".join(f"{gap_start}-{gap_end}" for gap_start, gap_end in gaps)
            raise ValueError(
                f"plot region {reference}:{start}-{end} contains unanalyzed gaps: {formatted}"
            )
        artifact_paths = (
            _flatten_strings(contributing["group_path"]) if "group_path" in contributing else ()
        )
        model_column = next(
            (name for name in ("hmm_model_ids", "model_ids") if name in contributing),
            None,
        )
        model_ids = _flatten_strings(contributing[model_column]) if model_column else ()
        plans.append(
            PlotRegionPlan(
                region_id=str(request["region_id"]),
                name=str(request["name"]),
                reference=reference,
                start=start,
                end=end,
                original_reference=request["original_reference"],
                original_start=request["original_start"],
                original_end=request["original_end"],
                task_ids=_flatten_strings(contributing["task_id"]),
                artifact_paths=artifact_paths,
                model_ids=model_ids,
                source=str(request.get("source", "task")),
                gaps=gaps,
            )
        )
    return sorted(plans, key=lambda plan: (plan.reference, plan.start, plan.end, plan.region_id))


def select_plot_reads(
    read_index: str | Path,
    plan: PlotRegionPlan,
    *,
    max_reads_per_barcode: int | None,
    seed: int = DEFAULT_PLOT_SELECTION_SEED,
    eligible_read_ids: Iterable[str] | None = None,
) -> PlotReadSelection:
    """Select deterministic per-barcode molecules before array materialization."""
    indexed = query_derived_index(
        read_index,
        references=plan.reference,
        start=plan.start,
        end=plan.end,
    )
    if indexed.empty:
        return PlotReadSelection((), (), int(seed), hashlib.sha256(b"").hexdigest())
    if eligible_read_ids is not None:
        eligible = set(map(str, eligible_read_ids))
        indexed = indexed.loc[indexed["read_id"].astype(str).isin(eligible)]
        if indexed.empty:
            return PlotReadSelection((), (), int(seed), hashlib.sha256(b"").hexdigest())
    identity = "molecule_uid" if "molecule_uid" in indexed else "read_id"
    indexed = indexed.sort_values(["barcode", identity, "read_id"], kind="stable")
    indexed = indexed.drop_duplicates(identity, keep="first")
    selected = []
    for _barcode, group in indexed.groupby("barcode", sort=True, observed=True):
        if (
            max_reads_per_barcode is not None
            and max_reads_per_barcode > 0
            and len(group) > int(max_reads_per_barcode)
        ):
            rng = np.random.default_rng(int(seed))
            rows = np.sort(rng.choice(len(group), size=int(max_reads_per_barcode), replace=False))
            group = group.iloc[rows]
        selected.append(group)
    selection = pd.concat(selected, ignore_index=True).sort_values(identity, kind="stable")
    read_ids = tuple(selection["read_id"].astype(str))
    molecule_uids = tuple(selection[identity].astype(str))
    digest = hashlib.sha256()
    for molecule_uid in molecule_uids:
        payload = molecule_uid.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return PlotReadSelection(read_ids, molecule_uids, int(seed), digest.hexdigest())


def mask_unanalyzed_gaps(adata, gaps: Iterable[tuple[int, int]]):
    """Replace explicitly allowed unanalyzed columns with labeled ``NaN`` values."""
    gaps = tuple(gaps)
    positions = np.asarray(adata.var_names, dtype=np.int64)
    mask = np.zeros(adata.n_vars, dtype=bool)
    for start, end in gaps:
        mask |= (positions >= int(start)) & (positions < int(end))
    adata.var["plot_unanalyzed_gap"] = mask
    if not mask.any():
        return adata
    if adata.X is not None:
        values = np.asarray(adata.X, dtype=np.float32).copy()
        values[:, mask] = np.nan
        adata.X = values
    for layer in list(adata.layers):
        values = np.asarray(adata.layers[layer], dtype=np.float32).copy()
        values[:, mask] = np.nan
        adata.layers[layer] = values
    return adata
