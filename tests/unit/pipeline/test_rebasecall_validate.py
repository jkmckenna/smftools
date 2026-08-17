from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.pipeline.test_rebasecall_basecall import _case, _execute, _FakeDorado
from tests.unit.pipeline.test_rebasecall_lineage import _publish_lineage

from smftools.pipeline.rebasecall_lineage import (
    LINEAGE_MANIFEST_FILENAME,
    LINEAGE_VALIDATION_FILENAME,
    RebasecallLineageError,
)
from smftools.pipeline.rebasecall_transition import build_qc_transition, write_qc_transition
from smftools.pipeline.rebasecall_validate import (
    promote_rebasecall_lineage,
    validate_rebasecall_lineage,
)
from smftools.project import registry as reg

pytestmark = pytest.mark.unit


def _stage_generation(run_root: Path, stage_dir: str, generation_id: str) -> Path:
    """Publish a minimal generation the validator can resolve."""
    from smftools.informatics.generation import staged_generation

    with staged_generation(run_root / stage_dir, generation_id=generation_id) as staged:
        (staged.staging_dir / "spine.h5ad").touch()
        staged.record_manifest({"status": "complete", "stage": stage_dir})
    return run_root / stage_dir / "generations" / generation_id


def _validated_case(tmp_path, monkeypatch, *, with_transition=True):
    plan, frozen, basecall = None, None, None
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    basecall = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    run_root = tmp_path / "run"
    _stage_generation(run_root, "raw_outputs", "desc-a-0")
    lineage = _publish_lineage(plan, frozen, basecall, tmp_path / "rebasecall_outputs")
    if with_transition:
        selected = sorted(pd.read_parquet(frozen.rows_path)["pod5_read_id"].astype(str))
        raw_dir = run_root / "raw_outputs" / "generations" / "desc-a-0"
        pd.DataFrame(
            {"read_id": selected, "molecule_uid": [f"m-{value}" for value in selected]}
        ).to_parquet(raw_dir / "obs.parquet", index=False)
        frame, summary = build_qc_transition(frozen, basecall, raw_dir)
        write_qc_transition(lineage, frame, summary)
    return plan, frozen, basecall, lineage, run_root


def test_a_complete_lineage_validates_and_records_its_report(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)

    report = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )

    assert report.complete is True
    assert report.failures == ()
    assert {check.name for check in report.checks} >= {
        "manifest",
        "stage:raw",
        "qc_transition",
        "basecall",
        "replayable",
    }
    recorded = json.loads(
        (lineage.directory / LINEAGE_VALIDATION_FILENAME).read_text(encoding="utf-8")
    )
    assert recorded["complete"] is True
    assert recorded["lineage_id"] == lineage.lineage_id


def test_replayability_is_reported_separately_from_completeness(tmp_path, monkeypatch):
    """A lineage can be complete and still not replayable."""
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)

    without = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )
    required = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
        require_replayable=True,
        write_report=False,
    )

    # No filtered signal was materialized for this request.
    assert without.replayable is False
    assert without.complete is True
    # Asking for replayability folds it into completeness rather than changing
    # what "complete" means for everyone else.
    assert required.complete is False
    assert [check.name for check in required.failures] == ["replayable"]


def test_a_missing_stage_generation_fails_validation(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    import shutil

    shutil.rmtree(run_root / "raw_outputs" / "generations" / "desc-a-0")

    report = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )

    assert report.complete is False
    assert "stage:raw" in {check.name for check in report.failures}


def test_a_transition_that_does_not_reconcile_fails_validation(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    summary_path = lineage.directory / "qc_transition_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["selected_molecule_count"] += 7
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )

    assert report.complete is False
    assert "qc_transition" in {check.name for check in report.failures}


def test_a_missing_transition_report_fails_validation(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch, with_transition=False)

    report = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )

    assert report.complete is False
    assert "qc_transition" in {check.name for check in report.failures}


def test_a_tampered_basecall_fails_validation(tmp_path, monkeypatch):
    _, _, basecall, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    (basecall.directory / "calls.bam").write_text("tampered", encoding="utf-8")

    report = validate_rebasecall_lineage(
        lineage.directory,
        run_root,
        basecall_root=tmp_path / "basecalls",
    )

    assert report.complete is False
    assert "basecall" in {check.name for check in report.failures}


def test_an_unreadable_lineage_reports_rather_than_raises(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    (lineage.directory / LINEAGE_MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    report = validate_rebasecall_lineage(lineage.directory, run_root, write_report=False)

    assert report.complete is False
    assert [check.name for check in report.failures] == ["manifest"]


# --- promotion ---------------------------------------------------------------


def _project_with_lineage(tmp_path, lineage_id):
    project_dir = tmp_path / "project"
    reg.init_project(project_dir)
    run_root = tmp_path / "exp"
    run_root.mkdir(parents=True, exist_ok=True)
    spine = run_root / "spine.h5ad"
    spine.touch()
    registry = reg.load_registry(project_dir)
    registry["experiments"]["exp-a"] = {
        "path": str(run_root),
        "name": "exp-a",
        "experiment_uid": "uid-a",
        "status": "active",
        "spines": {"raw": str(spine)},
        "catalogs": {},
    }
    reg.save_registry(project_dir, registry)
    descendant = run_root / "descendant_spine.h5ad"
    descendant.touch()
    reg.register_experiment_lineage(
        project_dir,
        "exp-a",
        lineage_id,
        spines={"raw": descendant},
    )
    return project_dir


def test_an_incomplete_lineage_cannot_be_promoted(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch, with_transition=False)
    project_dir = _project_with_lineage(tmp_path, lineage.lineage_id)

    with pytest.raises(RebasecallLineageError) as error:
        promote_rebasecall_lineage(
            project_dir,
            "exp-a",
            lineage.lineage_id,
            lineage_dir=lineage.directory,
            run_root=run_root,
            basecall_root=tmp_path / "basecalls",
        )

    assert error.value.code == "lineage_incomplete"
    # The project still answers with the original.
    assert reg.list_experiments(project_dir)[0]["lineage"] == "original"


def test_promotion_requires_enough_context_to_verify(tmp_path, monkeypatch):
    _, _, _, lineage, _ = _validated_case(tmp_path, monkeypatch)
    project_dir = _project_with_lineage(tmp_path, lineage.lineage_id)

    with pytest.raises(RebasecallLineageError) as error:
        promote_rebasecall_lineage(project_dir, "exp-a", lineage.lineage_id)

    assert error.value.code == "promotion_unverifiable"
    assert reg.list_experiments(project_dir)[0]["lineage"] == "original"


def test_a_validated_lineage_is_promoted_and_rollback_is_the_same_operation(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    project_dir = _project_with_lineage(tmp_path, lineage.lineage_id)

    promoted = promote_rebasecall_lineage(
        project_dir,
        "exp-a",
        lineage.lineage_id,
        lineage_dir=lineage.directory,
        run_root=run_root,
        basecall_root=tmp_path / "basecalls",
    )
    assert promoted["active_lineage"] == lineage.lineage_id
    assert promoted["validation"]["complete"] is True
    assert reg.list_experiments(project_dir)[0]["lineage"] == lineage.lineage_id

    # Rollback needs no separate machinery.
    rolled_back = promote_rebasecall_lineage(project_dir, "exp-a", "original")

    assert rolled_back["active_lineage"] == "original"
    assert rolled_back["validation"] is None
    assert reg.list_experiments(project_dir)[0]["lineage"] == "original"
    # The descendant is still registered and still queryable by name.
    assert lineage.lineage_id in reg.list_experiments(project_dir)[0]["available_lineages"]


def test_promoting_an_unregistered_lineage_is_refused(tmp_path, monkeypatch):
    _, _, _, lineage, run_root = _validated_case(tmp_path, monkeypatch)
    project_dir = _project_with_lineage(tmp_path, "some-other-lineage")

    with pytest.raises(RebasecallLineageError) as error:
        promote_rebasecall_lineage(
            project_dir,
            "exp-a",
            lineage.lineage_id,
            lineage_dir=lineage.directory,
            run_root=run_root,
            basecall_root=tmp_path / "basecalls",
        )

    assert error.value.code == "promotion_refused"
