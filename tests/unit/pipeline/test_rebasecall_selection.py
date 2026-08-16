from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from tests.unit.pipeline.test_rebasecall_plan import (
    _install_parent_fixtures,
    _pod5_index,
    _request,
)

from smftools.pipeline import rebasecall_plan
from smftools.pipeline import rebasecall_selection as selection_module
from smftools.pipeline.rebasecall_selection import (
    RebasecallSelectionError,
    freeze_rebasecall_selection,
    prepare_rebasecall_selection,
    read_frozen_rebasecall_selection,
)

pytestmark = pytest.mark.unit


def _build_ready_plan(tmp_path, monkeypatch, mode="all-parent-molecules"):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    request = _request(mode)
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )
    assert plan.status == "ready"
    return cfg, request, plan


def _skip_parent_validation(_plan):
    return None


def test_freeze_writes_exact_rows_and_reuses_content_addressed_result(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)
    root = tmp_path / "selection-results"

    frozen = freeze_rebasecall_selection(
        plan,
        root,
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    reused = freeze_rebasecall_selection(
        plan,
        root,
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    rows = pd.read_parquet(frozen.rows_path)

    assert reused.directory == frozen.directory
    assert frozen.directory.name == frozen.selection_id
    assert rows["observation_id"].tolist() == ["r1", "r2", "r3"]
    assert rows["pod5_read_id"].tolist() == ["r1", "r2", "r3"]
    assert frozen.manifest["accepted_plan_id"] == plan.plan_id
    assert frozen.manifest["counts"] == {
        "record_count": 3,
        "molecule_count": 3,
        "unique_pod5_read_count": 3,
        "duplicate_parent_reference_count": 0,
    }
    assert not any(path.name.endswith(".tmp") for path in root.iterdir())


def test_selection_identity_is_independent_of_requested_model(tmp_path, monkeypatch):
    cfg, request, first_plan = _build_ready_plan(tmp_path, monkeypatch)
    second_request = replace(request, basecall=replace(request.basecall, model="sup@latest"))
    second_plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        second_request,
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    first = freeze_rebasecall_selection(
        first_plan,
        tmp_path / "first",
        accepted_plan_id=first_plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    second = freeze_rebasecall_selection(
        second_plan,
        tmp_path / "second",
        accepted_plan_id=second_plan.plan_id,
        parent_validator=_skip_parent_validation,
    )

    assert first_plan.plan_id != second_plan.plan_id
    assert first.selection_id == second.selection_id


def test_all_signal_freezes_complete_pod5_inventory(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch, mode="all-signal")

    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    rows = pd.read_parquet(frozen.rows_path)

    assert len(rows) == 5
    assert rows["observation_id"].isna().all()
    assert frozen.manifest["counts"]["molecule_count"] == 0
    assert frozen.manifest["counts"]["duplicate_parent_reference_count"] == 0


def test_split_children_freeze_as_two_rows_with_one_signal_parent(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    pd.DataFrame(
        {
            "read_id": ["child-1", "child-2"],
            "molecule_uid": ["m1", "m2"],
            "pod5_read_id": ["parent", "parent"],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)
    request = _request("ids", id_kind="pod5_read_id", ids=["parent"])
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=lambda sources: _pod5_index("parent"),
    )

    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )

    assert frozen.manifest["counts"]["record_count"] == 2
    assert frozen.manifest["counts"]["unique_pod5_read_count"] == 1
    assert frozen.manifest["counts"]["duplicate_parent_reference_count"] == 1


def test_stale_acceptance_and_blocked_plan_write_nothing(tmp_path, monkeypatch):
    cfg, _, ready = _build_ready_plan(tmp_path, monkeypatch)
    stale_root = tmp_path / "stale"
    with pytest.raises(RebasecallSelectionError, match="does not match") as stale:
        freeze_rebasecall_selection(
            ready,
            stale_root,
            accepted_plan_id="not-the-plan",
            parent_validator=_skip_parent_validation,
        )
    assert stale.value.code == "accepted_plan_mismatch"
    assert not stale_root.exists()

    blocked = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("ids"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )
    blocked_root = tmp_path / "blocked"
    with pytest.raises(RebasecallSelectionError, match="blocked") as blocked_error:
        freeze_rebasecall_selection(
            blocked,
            blocked_root,
            accepted_plan_id=blocked.plan_id,
            parent_validator=_skip_parent_validation,
        )
    assert blocked_error.value.code == "accepted_plan_blocked"
    assert not blocked_root.exists()


def test_parent_drift_blocks_before_any_artifact_is_written(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)
    root = tmp_path / "selection-results"

    def changed_parent(_plan):
        raise RebasecallSelectionError("selection_parent_changed", "parent changed")

    with pytest.raises(RebasecallSelectionError, match="parent changed") as error:
        freeze_rebasecall_selection(
            plan,
            root,
            accepted_plan_id=plan.plan_id,
            parent_validator=changed_parent,
        )

    assert error.value.code == "selection_parent_changed"
    assert not root.exists()


def test_default_parent_validator_rechecks_manifest_identity(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        selection_module,
        "validate_raw_generation",
        lambda *_args, **_kwargs: dict(plan.raw_parent.manifest),
    )

    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
    )

    assert frozen.selection_id

    monkeypatch.setattr(
        selection_module,
        "validate_raw_generation",
        lambda *_args, **_kwargs: {"generation_id": "changed"},
    )
    with pytest.raises(RebasecallSelectionError, match="identity changed") as error:
        freeze_rebasecall_selection(
            plan,
            tmp_path / "changed-parent",
            accepted_plan_id=plan.plan_id,
        )
    assert error.value.code == "selection_parent_changed"


def test_default_parent_validator_rechecks_preprocess_manifest(tmp_path, monkeypatch):
    from smftools.preprocessing import preprocess_generation

    _, _, plan = _build_ready_plan(tmp_path, monkeypatch, mode="qc")
    monkeypatch.setattr(
        selection_module,
        "validate_raw_generation",
        lambda *_args, **_kwargs: dict(plan.raw_parent.manifest),
    )
    monkeypatch.setattr(
        preprocess_generation,
        "validate_preprocess_generation",
        lambda *_args, **_kwargs: dict(plan.preprocess_parent.manifest),
    )

    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
    )

    assert frozen.manifest["identity"]["parents"]["preprocess"]["generation_id"] == "pre-a"


def test_atomic_failure_leaves_no_partial_selection(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)
    root = tmp_path / "selection-results"

    def fail_manifest(*_args, **_kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(selection_module, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected write failure"):
        freeze_rebasecall_selection(
            plan,
            root,
            accepted_plan_id=plan.plan_id,
            parent_validator=_skip_parent_validation,
        )

    assert root.is_dir()
    assert not list(root.iterdir())


def test_tampered_rows_fail_validation(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    rows = pd.read_parquet(frozen.rows_path)
    rows.loc[0, "pod5_read_id"] = "changed"
    rows.to_parquet(frozen.rows_path, index=False)

    with pytest.raises(RebasecallSelectionError, match="checksum") as error:
        read_frozen_rebasecall_selection(frozen.directory)

    assert error.value.code == "selection_artifact_invalid"


def test_prepare_rebuilds_the_accepted_plan_before_freezing(tmp_path, monkeypatch):
    cfg, request, plan = _build_ready_plan(tmp_path, monkeypatch)

    frozen = prepare_rebasecall_selection(
        cfg,
        request,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
        parent_validator=_skip_parent_validation,
    )

    assert frozen.manifest["accepted_plan_id"] == plan.plan_id


def test_plan_fingerprints_complete_consumed_columns(tmp_path, monkeypatch):
    cfg, _, first = _build_ready_plan(tmp_path, monkeypatch, mode="qc")
    stage_obs = cfg.preprocess_parent.generation_dir / "stage_obs.parquet"
    changed = pd.read_parquet(stage_obs)
    changed.loc[0, "passes_read_qc"] = False
    changed.to_parquet(stage_obs, index=False)
    second = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("qc"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert set(first.selection.source_column_fingerprints) == {
        "passes_dedup",
        "passes_read_qc",
    }
    assert (
        first.selection.source_column_fingerprints["passes_read_qc"]
        != second.selection.source_column_fingerprints["passes_read_qc"]
    )
    assert first.plan_id != second.plan_id


def test_selection_freezing_is_no_longer_reported_as_deferred(tmp_path, monkeypatch):
    _, _, plan = _build_ready_plan(tmp_path, monkeypatch)

    assert "selection_freezing:srb-01b" not in plan.to_dict()["deferred_capabilities"]
    assert plan.to_dict()["plan_id"] == plan.plan_id
