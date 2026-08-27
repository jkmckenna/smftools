from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.constants import RAW_DIR
from smftools.data.run_locality import (
    STATE_AHEAD,
    STATE_BEHIND,
    STATE_DIVERGED,
    STATE_IDENTICAL,
    STATE_POINTER_CONFLICT,
)
from smftools.data.run_sync import sync_run_locations
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)

pytestmark = pytest.mark.unit


def _publish(run_root: Path, stage_dir: str, generation_id: str, *, current: bool = False) -> None:
    container = run_root / stage_dir
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / "payload.txt").write_text(f"data for {generation_id}", encoding="utf-8")
    (generation_dir / GENERATION_MANIFEST).write_text(
        json.dumps({"schema_version": 2, "status": "complete", "generation_id": generation_id}),
        encoding="utf-8",
    )
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps({"schema_version": 1, "generation_id": generation_id}), encoding="utf-8"
        )


def test_ahead_copies_the_missing_generation_from_a_to_b(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    result = sync_run_locations(a, b)

    assert result.stages[0].state == STATE_AHEAD
    assert result.stages[0].copied_a_to_b == ("gen-2",)
    assert result.stages[0].copied_b_to_a == ()
    copied_dir = b / RAW_DIR / GENERATIONS_SUBDIR / "gen-2"
    assert copied_dir.is_dir()
    assert (copied_dir / "payload.txt").read_text() == "data for gen-2"


def test_behind_copies_the_missing_generation_from_b_to_a(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-2", current=True)

    result = sync_run_locations(a, b)

    assert result.stages[0].state == STATE_BEHIND
    assert result.stages[0].copied_b_to_a == ("gen-2",)
    assert (a / RAW_DIR / GENERATIONS_SUBDIR / "gen-2").is_dir()


def test_sync_never_moves_current_json(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    sync_run_locations(a, b)

    b_current = json.loads((b / RAW_DIR / CURRENT_FILENAME).read_text())
    assert b_current["generation_id"] == "gen-1"  # unchanged, despite gen-2 now present


def test_identical_stage_copies_nothing(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    result = sync_run_locations(a, b)

    assert result.stages[0].state == STATE_IDENTICAL
    assert result.any_copied is False
    assert result.unresolved_stages == ()


def test_diverged_stage_copies_nothing_and_reports_a_reason(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-a-only", current=True)
    _publish(b, RAW_DIR, "gen-1", current=False)
    _publish(b, RAW_DIR, "gen-b-only", current=True)

    result = sync_run_locations(a, b)

    assert result.stages[0].state == STATE_DIVERGED
    assert result.stages[0].copied_a_to_b == ()
    assert result.stages[0].copied_b_to_a == ()
    assert result.stages[0].skipped_reason is not None
    assert result.any_copied is False
    assert result.unresolved_stages == ("raw",)
    # Neither side gained the other's generation.
    assert not (a / RAW_DIR / GENERATIONS_SUBDIR / "gen-b-only").exists()
    assert not (b / RAW_DIR / GENERATIONS_SUBDIR / "gen-a-only").exists()


def test_pointer_conflict_stage_copies_nothing(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=False)
    _publish(b, RAW_DIR, "gen-1", current=False)
    _publish(b, RAW_DIR, "gen-2", current=True)

    result = sync_run_locations(a, b)

    assert result.stages[0].state == STATE_POINTER_CONFLICT
    assert result.any_copied is False
    assert result.unresolved_stages == ("raw",)


def test_dry_run_reports_without_copying(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)

    result = sync_run_locations(a, b, dry_run=True)

    assert result.stages[0].copied_a_to_b == ("gen-2",)  # reported...
    assert not (b / RAW_DIR / GENERATIONS_SUBDIR / "gen-2").exists()  # ...but not copied


def test_rerun_after_a_sync_is_a_no_op(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(a, RAW_DIR, "gen-2", current=False)
    _publish(b, RAW_DIR, "gen-1", current=True)

    sync_run_locations(a, b)
    second = sync_run_locations(a, b)

    assert second.stages[0].state == STATE_IDENTICAL
    assert second.any_copied is False


def test_sync_across_multiple_stages_independently(tmp_path: Path) -> None:
    from smftools.constants import PREPROCESS_DIR

    a, b = tmp_path / "a", tmp_path / "b"
    _publish(a, RAW_DIR, "gen-1", current=True)
    _publish(b, RAW_DIR, "gen-1", current=True)
    _publish(a, PREPROCESS_DIR, "pp-1", current=True)
    # preprocess absent at b entirely -- b is "behind" with an empty starting set.

    result = sync_run_locations(a, b)

    stages_by_kind = {stage.kind: stage for stage in result.stages}
    assert stages_by_kind["raw"].state == STATE_IDENTICAL
    assert stages_by_kind["preprocess"].state == STATE_AHEAD
    assert stages_by_kind["preprocess"].copied_a_to_b == ("pp-1",)
    assert (b / PREPROCESS_DIR / GENERATIONS_SUBDIR / "pp-1").is_dir()
