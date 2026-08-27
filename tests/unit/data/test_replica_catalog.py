from __future__ import annotations

from pathlib import Path

import pytest

from smftools.data.replica_catalog import (
    Replica,
    ReplicaCatalogError,
    add_replica,
    load_catalog,
    replicas_for,
    resolve_replica,
    save_catalog,
)
from smftools.data.volume_discovery import DiscoveredVolume
from smftools.data.volume_stamp import VolumeStamp

pytestmark = pytest.mark.unit


def _stamp(volume_id: str, *, kind: str, label: str | None = None) -> VolumeStamp:
    return VolumeStamp(
        volume_id=volume_id,
        label=label or volume_id,
        kind=kind,
        created="2026-01-01T00:00:00+00:00",
    )


def test_load_missing_catalog_returns_empty(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "nope.json") == {}


def test_add_replica_creates_a_new_dataset_entry() -> None:
    catalog = add_replica({}, "digest-a", volume_id="vol-1", path="runs/exp1")

    replicas = replicas_for(catalog, "digest-a")
    assert len(replicas) == 1
    assert replicas[0].volume_id == "vol-1"
    assert replicas[0].path == "runs/exp1"
    assert replicas[0].digest == "digest-a"  # defaults to the dataset digest


def test_add_replica_appends_a_second_distinct_location() -> None:
    catalog = add_replica({}, "digest-a", volume_id="vol-1", path="runs/exp1")
    catalog = add_replica(catalog, "digest-a", volume_id="vol-2", path="backup/exp1")

    assert len(replicas_for(catalog, "digest-a")) == 2


def test_add_replica_refreshes_an_existing_location_in_place() -> None:
    catalog = add_replica(
        {}, "digest-a", volume_id="vol-1", path="runs/exp1", verified_at="2026-01-01T00:00:00+00:00"
    )

    catalog = add_replica(
        catalog,
        "digest-a",
        volume_id="vol-1",
        path="runs/exp1",
        verified_at="2026-06-01T00:00:00+00:00",
    )

    replicas = replicas_for(catalog, "digest-a")
    assert len(replicas) == 1
    assert replicas[0].verified_at == "2026-06-01T00:00:00+00:00"


def test_add_replica_does_not_mutate_the_input_catalog() -> None:
    original: dict[str, list[Replica]] = {}
    add_replica(original, "digest-a", volume_id="vol-1", path="runs/exp1")

    assert original == {}


def test_replicas_for_unknown_dataset_is_empty() -> None:
    assert replicas_for({}, "no-such-digest") == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    catalog_path = tmp_path / "replica_catalog.json"
    catalog = add_replica({}, "digest-a", volume_id="vol-1", path="runs/exp1")
    catalog = add_replica(catalog, "digest-a", volume_id="vol-2", path="backup/exp1")

    save_catalog(catalog, path=catalog_path)
    reloaded = load_catalog(catalog_path)

    assert reloaded == catalog


def test_save_catalog_drops_datasets_with_no_replicas_left(tmp_path: Path) -> None:
    catalog_path = tmp_path / "replica_catalog.json"
    save_catalog({"digest-a": []}, path=catalog_path)

    assert load_catalog(catalog_path) == {}


def test_load_rejects_corrupt_json(tmp_path: Path) -> None:
    catalog_path = tmp_path / "replica_catalog.json"
    catalog_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ReplicaCatalogError, match="not valid JSON"):
        load_catalog(catalog_path)


def test_load_rejects_wrong_schema_version(tmp_path: Path) -> None:
    catalog_path = tmp_path / "replica_catalog.json"
    catalog_path.write_text('{"schema_version": 99, "datasets": {}}', encoding="utf-8")

    with pytest.raises(ReplicaCatalogError, match="unsupported schema version"):
        load_catalog(catalog_path)


def test_load_rejects_malformed_dataset_entry(tmp_path: Path) -> None:
    catalog_path = tmp_path / "replica_catalog.json"
    catalog_path.write_text(
        '{"schema_version": 1, "datasets": {"digest-a": {"replicas": "not-a-list"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ReplicaCatalogError, match="no replica list"):
        load_catalog(catalog_path)


def test_resolve_replica_picks_the_first_attached_by_preference() -> None:
    catalog = add_replica({}, "digest-a", volume_id="archive-vol", path="a")
    catalog = add_replica(catalog, "digest-a", volume_id="working-vol", path="w")
    attached = [
        DiscoveredVolume(
            stamp=_stamp("archive-vol", kind="archive"), mount_path=Path("/mnt/archive")
        ),
        DiscoveredVolume(
            stamp=_stamp("working-vol", kind="working"), mount_path=Path("/mnt/working")
        ),
    ]

    resolved = resolve_replica(catalog, "digest-a", attached=attached)

    assert resolved is not None
    assert resolved.replica.volume_id == "working-vol"
    assert resolved.resolved_path == Path("/mnt/working/w")


def test_resolve_replica_skips_unattached_replicas() -> None:
    catalog = add_replica({}, "digest-a", volume_id="offline-vol", path="a")
    catalog = add_replica(catalog, "digest-a", volume_id="attached-vol", path="b")
    attached = [
        DiscoveredVolume(stamp=_stamp("attached-vol", kind="archive"), mount_path=Path("/mnt/x"))
    ]

    resolved = resolve_replica(catalog, "digest-a", attached=attached)

    assert resolved is not None
    assert resolved.replica.volume_id == "attached-vol"


def test_resolve_replica_returns_none_when_nothing_is_attached() -> None:
    catalog = add_replica({}, "digest-a", volume_id="offline-vol", path="a")

    assert resolve_replica(catalog, "digest-a", attached=[]) is None


def test_resolve_replica_unrecognized_kind_sorts_last_but_is_not_dropped() -> None:
    catalog = add_replica({}, "digest-a", volume_id="mystery-vol", path="a")
    attached = [
        DiscoveredVolume(stamp=_stamp("mystery-vol", kind="working"), mount_path=Path("/mnt/x"))
    ]

    resolved = resolve_replica(
        catalog, "digest-a", attached=attached, preference=("archive", "backup")
    )

    assert resolved is not None
    assert resolved.replica.volume_id == "mystery-vol"
