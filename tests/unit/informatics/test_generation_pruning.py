from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.constants import RAW_DIR
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)
from smftools.informatics.generation_pruning import (
    BLOCKED_REPRODUCIBILITY,
    KEEP_CURRENT,
    KEEP_LAST,
    KEEP_PINNED,
    KEEP_RECENT,
    KEEP_UNREADABLE,
    GenerationPruneError,
    parse_older_than,
    plan_experiment_generation_prune,
)
from smftools.informatics.generation_retention import pin_generation

pytestmark = pytest.mark.unit


def _publish(
    container: Path,
    generation_id: str,
    created_at: str,
    *,
    current: bool = False,
) -> Path:
    generation = container / GENERATIONS_SUBDIR / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "payload.bin").write_bytes(generation_id.encode())
    (generation / GENERATION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "generation_id": generation_id,
                "created_at": created_at,
                "artifacts": {"payload": "payload.bin"},
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
    return generation


def test_plan_protects_current_pinned_and_keep_last_then_blocks_candidates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-new", "2026-03-01T00:00:00Z")
    _publish(container, "raw-pinned", "2026-02-01T00:00:00Z")
    _publish(container, "raw-old", "2026-01-01T00:00:00Z")
    pin_generation(container, "raw-pinned", reason="publication source")
    paths_before = sorted(path.relative_to(run_root) for path in run_root.rglob("*"))

    plan = plan_experiment_generation_prune(
        run_root,
        keep_last=2,
        older_than="2026-03-15T00:00:00Z",
        stages=("raw",),
    )

    dispositions = {decision.generation_id: decision.disposition for decision in plan.decisions}
    assert dispositions == {
        "raw-current": KEEP_CURRENT,
        "raw-new": KEEP_LAST,
        "raw-pinned": KEEP_PINNED,
        "raw-old": BLOCKED_REPRODUCIBILITY,
    }
    old = next(d for d in plan.decisions if d.generation_id == "raw-old")
    assert old.policy_candidate is True
    assert old.deletion_allowed is False
    assert plan.candidate_bytes > 0
    assert plan.reclaimable_bytes == 0
    assert plan.to_dict()["deletion_supported"] is False
    assert sorted(path.relative_to(run_root) for path in run_root.rglob("*")) == paths_before


def test_plan_protects_generations_at_or_after_cutoff(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-recent", "2026-03-15T00:00:00Z")
    _publish(container, "raw-old", "2026-03-14T23:59:59Z")

    plan = plan_experiment_generation_prune(
        run_root,
        older_than="2026-03-15T00:00:00Z",
        stages=("raw",),
    )

    dispositions = {decision.generation_id: decision.disposition for decision in plan.decisions}
    assert dispositions["raw-recent"] == KEEP_RECENT
    assert dispositions["raw-old"] == BLOCKED_REPRODUCIBILITY


def test_plan_protects_every_noncurrent_generation_when_retention_is_corrupt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-current", "2026-04-01T00:00:00Z", current=True)
    _publish(container, "raw-old", "2026-01-01T00:00:00Z")
    (container / "retention.json").write_text("{", encoding="utf-8")

    plan = plan_experiment_generation_prune(
        run_root,
        older_than="2100-01-01T00:00:00Z",
        stages=("raw",),
    )

    old = next(decision for decision in plan.decisions if decision.generation_id == "raw-old")
    assert old.disposition == KEEP_UNREADABLE
    assert old.policy_candidate is False
    assert old.deletion_allowed is False
    assert plan.candidate_bytes == 0


@pytest.mark.parametrize("value", ["", "not-a-date"])
def test_parse_older_than_rejects_invalid_values(value: str) -> None:
    with pytest.raises(GenerationPruneError, match="older-than"):
        parse_older_than(value)


def test_plan_requires_policy_and_known_stages(tmp_path: Path) -> None:
    with pytest.raises(GenerationPruneError, match="requires"):
        plan_experiment_generation_prune(tmp_path)
    with pytest.raises(GenerationPruneError, match="unknown generation stage"):
        plan_experiment_generation_prune(
            tmp_path,
            keep_last=1,
            stages=("unknown",),
        )
