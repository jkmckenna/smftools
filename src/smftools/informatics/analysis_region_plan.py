"""Shared authoritative-core planning for inherited analysis-region catalogs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from .partition_read import resolve_relative_path

ANALYSIS_REGION_PLANNER_VERSION = 1


@dataclass(frozen=True)
class AnalysisCore:
    """One non-overlapping authoritative analysis core in stored coordinates."""

    reference: str
    analysis_mode: str
    core_start: int
    core_end: int
    original_reference: str | None = None
    original_start: int | None = None
    original_end: int | None = None
    analysis_region_ids: tuple[str, ...] = ()
    analysis_core_id: str = ""
    planner_version: int = ANALYSIS_REGION_PLANNER_VERSION

    def load_bounds(self, reference_length: int, halo: int) -> tuple[int, int]:
        """Return stored-coordinate load bounds for a stage-specific halo."""
        return max(0, self.core_start - halo), min(reference_length, self.core_end + halo)


def _stable_core_id(
    reference: str,
    start: int,
    end: int,
    region_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    for value in (
        ANALYSIS_REGION_PLANNER_VERSION,
        reference,
        start,
        end,
        *region_ids,
    ):
        payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"acore_{digest.hexdigest()[:24]}"


def _run_root(spine_path: str | Path | None) -> Path | None:
    return None if spine_path is None else Path(spine_path).parent.parent


def _inherited_path(spine, key: str, *, spine_path: str | Path | None) -> Path | None:
    value = spine.uns.get(key)
    resolved = resolve_relative_path(value, _run_root(spine_path))
    if value and resolved is None:
        raise ValueError(
            f"analysis-region planning cannot resolve relative spine.uns[{key!r}] "
            "without the on-disk spine path"
        )
    return resolved


def analysis_catalog_path(spine, *, spine_path: str | Path | None = None) -> Path | None:
    """Resolve the inherited normalized analysis catalog, if one is configured."""
    catalogs = spine.uns.get("region_catalogs", {})
    if not isinstance(catalogs, Mapping):
        raise ValueError("spine.uns['region_catalogs'] must be a mapping")
    value = catalogs.get("analysis")
    resolved = resolve_relative_path(value, _run_root(spine_path))
    if value and resolved is None:
        raise ValueError(
            "analysis-region planning cannot resolve the relative analysis catalog "
            "without the on-disk spine path"
        )
    return resolved


def has_analysis_catalog(spine) -> bool:
    """Return whether the spine inherits an explicit analysis-region catalog."""
    catalogs = spine.uns.get("region_catalogs", {})
    return isinstance(catalogs, Mapping) and bool(catalogs.get("analysis"))


def _standard_cores(reference: str, plan: Mapping[str, object]) -> list[AnalysisCore]:
    length = int(plan["reference_length"])
    analysis_mode = str(plan["analysis_mode"])
    tile_size = length if analysis_mode == "locus" else int(plan["tile_size"])
    return [
        AnalysisCore(
            reference=reference,
            analysis_mode=analysis_mode,
            core_start=start,
            core_end=min(start + tile_size, length),
        )
        for start in range(0, length, tile_size)
    ]


def _merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _validate_analysis_coverage(catalog: pd.DataFrame, mapping: pd.DataFrame) -> None:
    for original_reference, rows in catalog.groupby("original_reference", sort=True):
        coverage = _merged_intervals(
            [
                (int(row.original_start), int(row.original_end))
                for row in mapping.loc[
                    mapping["original_reference"].astype(str) == str(original_reference)
                ].itertuples(index=False)
            ]
        )
        for region in rows.itertuples(index=False):
            cursor = int(region.original_start)
            end = int(region.original_end)
            for covered_start, covered_end in coverage:
                if covered_end <= cursor:
                    continue
                if covered_start > cursor:
                    break
                cursor = max(cursor, covered_end)
                if cursor >= end:
                    break
            if cursor < end:
                raise ValueError(
                    "analysis region is not fully represented by the alignment reference map: "
                    f"{original_reference}:{int(region.original_start)}-{end} "
                    f"({region.region_id})"
                )


def _mapping_by_stored_reference(mapping: pd.DataFrame) -> dict[str, dict[str, object]]:
    columns = [
        "stored_reference",
        "stored_start",
        "stored_end",
        "original_reference",
        "original_start",
        "original_end",
        "coordinate_orientation",
    ]
    unique = mapping[columns].drop_duplicates()
    result: dict[str, dict[str, object]] = {}
    for reference, rows in unique.groupby("stored_reference", sort=True):
        if len(rows) != 1:
            raise ValueError(
                f"reference_interval_map has conflicting mappings for stored reference {reference!r}"
            )
        record = rows.iloc[0].to_dict()
        if int(record["coordinate_orientation"]) != 1:
            raise ValueError(
                "analysis-region planner currently requires forward stored-coordinate mappings"
            )
        result[str(reference)] = record
    return result


def _catalog_cores(
    reference: str,
    plan: Mapping[str, object],
    catalog: pd.DataFrame,
    mapping: Mapping[str, object],
) -> list[AnalysisCore]:
    original_reference = str(mapping["original_reference"])
    original_start = int(mapping["original_start"])
    original_end = int(mapping["original_end"])
    stored_start = int(mapping["stored_start"])
    stored_end = int(mapping["stored_end"])
    reference_length = int(plan["reference_length"])
    if stored_start != 0 or stored_end != reference_length:
        raise ValueError(
            f"reference_interval_map bounds for {reference!r} do not match its storage plan"
        )

    mapped: list[tuple[int, int, str]] = []
    relevant = catalog.loc[catalog["original_reference"].astype(str) == original_reference]
    for region in relevant.itertuples(index=False):
        overlap_start = max(original_start, int(region.original_start))
        overlap_end = min(original_end, int(region.original_end))
        if overlap_end <= overlap_start:
            continue
        mapped.append(
            (
                stored_start + overlap_start - original_start,
                stored_start + overlap_end - original_start,
                str(region.region_id),
            )
        )

    tile_size = int(plan["tile_size"])
    cores: list[AnalysisCore] = []
    for tile_start in range(0, reference_length, tile_size):
        tile_end = min(tile_start + tile_size, reference_length)
        clipped = [
            (max(start, tile_start), min(end, tile_end))
            for start, end, _region_id in mapped
            if start < tile_end and end > tile_start
        ]
        for core_start, core_end in _merged_intervals(clipped):
            region_ids = tuple(
                sorted(
                    {
                        region_id
                        for start, end, region_id in mapped
                        if start < core_end and end > core_start
                    }
                )
            )
            cores.append(
                AnalysisCore(
                    reference=reference,
                    analysis_mode="genome",
                    core_start=core_start,
                    core_end=core_end,
                    original_reference=original_reference,
                    original_start=original_start + core_start - stored_start,
                    original_end=original_start + core_end - stored_start,
                    analysis_region_ids=region_ids,
                    analysis_core_id=_stable_core_id(reference, core_start, core_end, region_ids),
                )
            )
    return cores


def plan_analysis_cores(
    spine,
    *,
    spine_path: str | Path | None = None,
) -> list[AnalysisCore]:
    """Plan shared non-overlapping cores from inherited catalog provenance.

    Without an explicit analysis catalog this returns the historical full-locus
    and storage-tiled genome cores. When a catalog is present, only genome-mode
    references are restricted; locus references retain their full-reference
    behavior.
    """
    plans = {
        str(reference): dict(plan)
        for reference, plan in dict(spine.uns.get("reference_plans", {})).items()
    }
    if not plans:
        raise ValueError("analysis-region planning requires reference_plans")
    catalog_path = analysis_catalog_path(spine, spine_path=spine_path)
    if catalog_path is None:
        return [
            core
            for reference, plan in sorted(plans.items())
            for core in _standard_cores(reference, plan)
        ]
    if not catalog_path.is_file():
        raise FileNotFoundError(f"analysis region catalog not found: {catalog_path}")
    mapping_path = _inherited_path(spine, "reference_interval_map", spine_path=spine_path)
    if mapping_path is None or not mapping_path.is_file():
        raise FileNotFoundError(
            f"reference interval map required for analysis planning was not found: {mapping_path}"
        )
    catalog = pd.read_parquet(catalog_path)
    mapping = pd.read_parquet(mapping_path)
    _validate_analysis_coverage(catalog, mapping)
    stored_mappings = _mapping_by_stored_reference(mapping)

    cores: list[AnalysisCore] = []
    for reference, plan in sorted(plans.items()):
        if str(plan["analysis_mode"]) == "locus":
            cores.extend(_standard_cores(reference, plan))
            continue
        if reference not in stored_mappings:
            raise ValueError(
                f"reference_interval_map lacks stored reference required by the spine: {reference!r}"
            )
        cores.extend(_catalog_cores(reference, plan, catalog, stored_mappings[reference]))
    return cores
