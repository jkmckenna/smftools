from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from smftools.constants import RAW_DIR
from smftools.data.run_locality import (
    STATE_AHEAD,
    STATE_BEHIND,
    STATE_DIVERGED,
    STATE_IDENTICAL,
    STATE_POINTER_CONFLICT,
    are_duplicates,
    compare_run_locations,
    run_identity,
)
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)

pytestmark = pytest.mark.unit


def _set_identity(run_root: Path, experiment_uid: str | None) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    payload: dict = {"schema_version": 1}
    if experiment_uid is not None:
        payload["experiment_uid"] = experiment_uid
    (run_root / "experiment_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _publish(run_root: Path, stage_dir: str, generation_id: str, *, current: bool = False) -> None:
    container = run_root / stage_dir
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / GENERATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "generation_id": generation_id,
                "config_hash": "hash0001",
                "artifacts": {"spine": "spine.h5ad"},
            }
        ),
        encoding="utf-8",
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "generation_path": f"{GENERATIONS_SUBDIR}/{generation_id}",
                }
            ),
            encoding="utf-8",
        )


# --- run_identity / are_duplicates ------------------------------------------


def test_run_identity_reads_the_recorded_uid(tmp_path: Path) -> None:
    uid = str(uuid.uuid4())
    _set_identity(tmp_path / "run", uid)

    assert run_identity(tmp_path / "run") == uid


def test_run_identity_none_without_a_manifest(tmp_path: Path) -> None:
    assert run_identity(tmp_path / "no-manifest") is None


def test_run_identity_none_for_a_legacy_manifest_with_no_uid(tmp_path: Path) -> None:
    _set_identity(tmp_path / "run", None)

    assert run_identity(tmp_path / "run") is None


def test_are_duplicates_true_for_matching_uids(tmp_path: Path) -> None:
    uid = str(uuid.uuid4())
    _set_identity(tmp_path / "a", uid)
    _set_identity(tmp_path / "b", uid)

    assert are_duplicates(tmp_path / "a", tmp_path / "b") is True


def test_are_duplicates_false_for_different_uids(tmp_path: Path) -> None:
    _set_identity(tmp_path / "a", str(uuid.uuid4()))
    _set_identity(tmp_path / "b", str(uuid.uuid4()))

    assert are_duplicates(tmp_path / "a", tmp_path / "b") is False


def test_are_duplicates_false_when_either_identity_is_unknown(tmp_path: Path) -> None:
    _set_identity(tmp_path / "a", str(uuid.uuid4()))
    _set_identity(tmp_path / "b", None)

    assert are_duplicates(tmp_path / "a", tmp_path / "b") is False
    assert are_duplicates(tmp_path / "b", tmp_path / "a") is False


# --- compare_run_locations ---------------------------------------------------


def test_identical_generation_set_and_pointer(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    result = compare_run_locations(a, b)

    assert len(result.stages) == 1
    assert result.stages[0].kind == "raw"
    assert result.stages[0].state == STATE_IDENTICAL
    assert result.identical is True


def test_a_ahead_of_b(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    result = compare_run_locations(a, b)

    assert result.stages[0].state == STATE_AHEAD
    assert result.stages[0].a_only == ("gen-2",)
    assert result.stages[0].b_only == ()


def test_a_behind_b(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-2", current=True)

    result = compare_run_locations(a, b)

    assert result.stages[0].state == STATE_BEHIND
    assert result.stages[0].b_only == ("gen-2",)


def test_diverged_when_each_side_holds_a_generation_the_other_lacks(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-a-only", current=True)
    _publish(b, RAW_DIR, "gen-1", current=False)
    _publish(b, RAW_DIR, "gen-b-only", current=True)

    result = compare_run_locations(a, b)

    assert result.stages[0].state == STATE_DIVERGED
    assert result.stages[0].a_only == ("gen-a-only",)
    assert result.stages[0].b_only == ("gen-b-only",)
    assert result.diverged_stages == ("raw",)
    assert result.identical is False


def test_pointer_conflict_when_generation_sets_match_but_current_differs(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=False)
    _publish(b, RAW_DIR, "gen-1", current=False)
    _publish(b, RAW_DIR, "gen-2", current=True)

    result = compare_run_locations(a, b)

    assert result.stages[0].state == STATE_POINTER_CONFLICT
    assert result.stages[0].a_current == "gen-1"
    assert result.stages[0].b_current == "gen-2"
    assert result.pointer_conflicts == ("raw",)


def test_stages_absent_from_both_locations_are_skipped(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    result = compare_run_locations(a, b)

    assert result.stages == ()
    assert result.identical is True  # vacuously -- nothing to disagree about


def test_a_stage_published_only_at_one_location_is_ahead(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    b.mkdir()

    result = compare_run_locations(a, b)

    assert result.stages[0].state == STATE_AHEAD
    assert result.stages[0].b_current is None
