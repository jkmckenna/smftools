from __future__ import annotations

from pathlib import Path

import pytest

from smftools.data.analysis_catalog import (
    AnalysisCatalogError,
    AnalysisLocation,
    add_location,
    load_catalog,
    locations_for,
    save_catalog,
)

pytestmark = pytest.mark.unit


def test_load_missing_catalog_returns_empty(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "nope.json") == {}


def test_add_location_creates_a_new_run_entry() -> None:
    catalog = add_location({}, "uid-a", volume_id="vol-1", path="runs/exp1")

    locations = locations_for(catalog, "uid-a")
    assert len(locations) == 1
    assert locations[0].volume_id == "vol-1"
    assert locations[0].path == "runs/exp1"


def test_add_location_appends_a_second_distinct_location() -> None:
    catalog = add_location({}, "uid-a", volume_id="vol-1", path="runs/exp1")
    catalog = add_location(catalog, "uid-a", volume_id="vol-2", path="backup/exp1")

    assert len(locations_for(catalog, "uid-a")) == 2


def test_add_location_refreshes_an_existing_location_in_place() -> None:
    catalog = add_location(
        {}, "uid-a", volume_id="vol-1", path="runs/exp1", scanned_at="2026-01-01T00:00:00+00:00"
    )

    catalog = add_location(
        catalog,
        "uid-a",
        volume_id="vol-1",
        path="runs/exp1",
        scanned_at="2026-06-01T00:00:00+00:00",
    )

    locations = locations_for(catalog, "uid-a")
    assert len(locations) == 1
    assert locations[0].scanned_at == "2026-06-01T00:00:00+00:00"


def test_add_location_does_not_mutate_the_input_catalog() -> None:
    original: dict[str, list[AnalysisLocation]] = {}
    add_location(original, "uid-a", volume_id="vol-1", path="runs/exp1")

    assert original == {}


def test_locations_for_unknown_run_is_empty() -> None:
    assert locations_for({}, "no-such-uid") == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    catalog_path = tmp_path / "analysis_catalog.json"
    catalog = add_location({}, "uid-a", volume_id="vol-1", path="runs/exp1")
    catalog = add_location(catalog, "uid-a", volume_id="vol-2", path="backup/exp1")

    save_catalog(catalog, path=catalog_path)
    reloaded = load_catalog(catalog_path)

    assert reloaded == catalog


def test_save_catalog_drops_runs_with_no_locations_left(tmp_path: Path) -> None:
    catalog_path = tmp_path / "analysis_catalog.json"
    save_catalog({"uid-a": []}, path=catalog_path)

    assert load_catalog(catalog_path) == {}


def test_load_rejects_corrupt_json(tmp_path: Path) -> None:
    catalog_path = tmp_path / "analysis_catalog.json"
    catalog_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(AnalysisCatalogError, match="not valid JSON"):
        load_catalog(catalog_path)


def test_load_rejects_wrong_schema_version(tmp_path: Path) -> None:
    catalog_path = tmp_path / "analysis_catalog.json"
    catalog_path.write_text('{"schema_version": 99, "runs": {}}', encoding="utf-8")

    with pytest.raises(AnalysisCatalogError, match="unsupported schema version"):
        load_catalog(catalog_path)


def test_load_rejects_malformed_run_entry(tmp_path: Path) -> None:
    catalog_path = tmp_path / "analysis_catalog.json"
    catalog_path.write_text(
        '{"schema_version": 1, "runs": {"uid-a": {"locations": "not-a-list"}}}',
        encoding="utf-8",
    )

    with pytest.raises(AnalysisCatalogError, match="no locations list"):
        load_catalog(catalog_path)
