"""`SLC-08`: applying a generation pruning plan -- archive, delete, restore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from smftools.cli_entry import cli
from smftools.constants import PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR
from smftools.informatics.generation_listing import (
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    PRUNED_RECORD,
    STATE_PRUNED,
    list_experiment_generations,
)
from smftools.informatics.generation_prune_apply import (
    PruneApplyError,
    apply_prune,
    restore_generation,
)
from smftools.informatics.generation_pruning import (
    PRUNE_CANDIDATE,
    PRUNED,
    plan_experiment_generation_prune,
)

from .test_generation_pruning import _refreshed_experiment

pytestmark = pytest.mark.unit

CANDIDATES = {"raw-stale", "pre-stale", "spatial-stale"}


def _gen(root: Path, directory: str, gid: str) -> Path:
    return root / directory / GENERATIONS_SUBDIR / gid


def _dispositions(root: Path) -> dict[str, str]:
    plan = plan_experiment_generation_prune(root, keep_last=0)
    return {d.generation_id: d.disposition for d in plan.decisions}


def test_archive_moves_candidates_and_leaves_stubs(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    archive = tmp_path / "archive"
    actions = apply_prune(root, keep_last=0, archive_dir=archive, allow_same_volume=True)

    assert {a.generation_id for a in actions} == CANDIDATES
    stub = _gen(root, RAW_DIR, "raw-stale")
    assert sorted(p.name for p in stub.iterdir()) == [GENERATION_MANIFEST, PRUNED_RECORD]
    record = json.loads((stub / PRUNED_RECORD).read_text())
    assert record["action"] == "archived" and record["files"] == 2
    copy = Path(record["archive_path"])
    assert copy == archive.resolve() / root.name / RAW_DIR / GENERATIONS_SUBDIR / "raw-stale"
    assert (copy / "payload.bin").read_bytes() == b"raw-stale"
    # kept generations untouched; pruned ones listed as pruned, never offered again
    assert (_gen(root, RAW_DIR, "raw-read") / "payload.bin").exists()
    states = {r.generation_id: r.state for r in list_experiment_generations(root)}
    assert {gid for gid, state in states.items() if state == STATE_PRUNED} == CANDIDATES
    assert {gid for gid, d in _dispositions(root).items() if d == PRUNED} == CANDIDATES
    assert PRUNE_CANDIDATE not in _dispositions(root).values()
    assert not (root / ".smftools-workflow" / "run.lock").exists()


def test_archive_on_the_same_volume_is_refused(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    with pytest.raises(PruneApplyError, match="same volume"):
        apply_prune(root, keep_last=0, archive_dir=tmp_path / "archive")
    assert set(_dispositions(root).values()) >= {PRUNE_CANDIDATE}


def test_delete_needs_every_id_confirmed(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    with pytest.raises(PruneApplyError, match="unconfirmed"):
        apply_prune(root, keep_last=0, delete=True, confirm=["raw-stale"])
    assert (_gen(root, PREPROCESS_DIR, "pre-stale") / "payload.bin").exists()

    actions = apply_prune(
        root, keep_last=0, delete=True, generation_ids=["raw-stale"], confirm=["raw-stale"]
    )
    assert [(a.generation_id, a.action) for a in actions] == [("raw-stale", "deleted")]
    assert not (_gen(root, RAW_DIR, "raw-stale") / "payload.bin").exists()
    with pytest.raises(PruneApplyError, match="deleted, not archived"):
        restore_generation(root, "raw", "raw-stale")


def test_restore_brings_an_archived_generation_back(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    archive = tmp_path / "archive"
    (action,) = apply_prune(
        root,
        keep_last=0,
        archive_dir=archive,
        allow_same_volume=True,
        generation_ids=["spatial-stale"],
    )
    restored = restore_generation(root, "spatial", "spatial-stale")
    assert (restored / "payload.bin").read_bytes() == b"spatial-stale"
    assert not (restored / PRUNED_RECORD).exists()
    assert not Path(action.archive_path).exists()
    assert _dispositions(root)["spatial-stale"] == PRUNE_CANDIDATE


def test_only_fresh_candidates_are_touched(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    with pytest.raises(PruneApplyError, match="not prune candidates"):
        apply_prune(
            root,
            keep_last=0,
            delete=True,
            generation_ids=["pre-parent"],  # read by the current preprocess generation
            confirm=["pre-parent"],
        )
    assert (_gen(root, PREPROCESS_DIR, "pre-parent") / "payload.bin").exists()


@pytest.mark.parametrize("busy", ["staging", "lock"])
def test_a_running_stage_or_workflow_blocks_pruning(tmp_path: Path, busy: str) -> None:
    root = _refreshed_experiment(tmp_path)
    if busy == "staging":
        staging = root / SPATIAL_DIR / ".staging" / "next"
        staging.mkdir(parents=True)
    else:
        lock = root / ".smftools-workflow" / "run.lock"
        lock.parent.mkdir()
        lock.write_text("123\n")
    with pytest.raises(PruneApplyError, match="refusing to prune"):
        apply_prune(root, keep_last=0, delete=True, confirm=sorted(CANDIDATES))
    assert (_gen(root, RAW_DIR, "raw-stale") / "payload.bin").exists()


def test_cli_archive_and_restore(tmp_path: Path) -> None:
    root = _refreshed_experiment(tmp_path)
    archive = tmp_path / "archive"
    result = CliRunner().invoke(
        cli,
        [
            "experiment",
            "generations",
            str(root),
            "prune",
            "--keep-last",
            "0",
            "--archive",
            str(archive),
            "--allow-same-volume",
            "--generation",
            "raw-stale",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [a["generation_id"] for a in json.loads(result.output)] == ["raw-stale"]
    restored = CliRunner().invoke(
        cli, ["experiment", "generations", str(root), "restore", "raw", "raw-stale"]
    )
    assert restored.exit_code == 0, restored.output
    refused = CliRunner().invoke(
        cli,
        ["experiment", "generations", str(root), "prune", "--keep-last", "0", "--delete"],
    )
    assert refused.exit_code != 0 and "unconfirmed" in refused.output
