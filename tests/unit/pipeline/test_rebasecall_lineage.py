from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.informatics.test_raw_generation import _publication_sources, _publish
from tests.unit.pipeline.test_rebasecall_basecall import _case, _execute, _FakeDorado

from smftools.informatics.raw_append import plan_raw_append
from smftools.informatics.raw_generation import (
    RawGenerationError,
    publish_raw_generation,
    resolve_current_raw_generation,
    validate_raw_generation,
)
from smftools.pipeline import rebasecall_plan
from smftools.pipeline.rebasecall_lineage import (
    LINEAGE_MANIFEST_FILENAME,
    LINEAGE_STAGE_GENERATIONS_FILENAME,
    LINEAGE_STAGING_SUBDIR,
    LINEAGES_SUBDIR,
    RebasecallLineageError,
    build_lineage_identity,
    descendant_raw_provenance,
    list_published_rebasecall_lineages,
    read_published_rebasecall_lineage,
    staged_lineage,
    write_lineage_validation,
)

pytestmark = pytest.mark.unit


def _lineage_case(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    basecall = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    return plan, frozen, basecall


def _publish_lineage(plan, frozen, basecall, root, *, stages=("raw",), generation_id="desc-a"):
    with staged_lineage(plan, frozen, basecall, root, accepted_plan_id=plan.plan_id) as staged:
        for ordinal, stage in enumerate(stages):
            staged.record_stage_generation(stage, f"{generation_id}-{ordinal}")
    return read_published_rebasecall_lineage(
        Path(root) / LINEAGES_SUBDIR / _lineage_id(plan, frozen, basecall)
    )


def _lineage_id(plan, frozen, basecall):
    from smftools.pipeline.rebasecall_lineage import _sha256_payload

    return _sha256_payload(build_lineage_identity(plan, frozen, basecall))


def test_publishing_a_lineage_records_its_stage_generation_set(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"

    lineage = _publish_lineage(plan, frozen, basecall, root, stages=("raw", "preprocess"))

    assert lineage.directory.name == lineage.lineage_id
    assert lineage.stage_generations == {"raw": "desc-a-0", "preprocess": "desc-a-1"}
    assert lineage.basecall_id == basecall.basecall_id
    assert lineage.manifest["identity"]["selection_id"] == frozen.selection_id
    # Nothing is left staged once publication succeeds.
    staging = root / LINEAGE_STAGING_SUBDIR
    assert not staging.exists() or not any(staging.iterdir())


def test_lineage_identity_tracks_the_basecall_it_was_built_from(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"
    first = _publish_lineage(plan, frozen, basecall, root)

    upgraded = replace(
        plan._model_resolution,
        model_bundle_digest="d" * 64,
        simplex_model=replace(plan._model_resolution.simplex_model, name="chem_hac@v2.0.0"),
    )
    upgraded_plan = replace(plan, _model_resolution=upgraded)
    second_basecall = _execute(
        upgraded_plan,
        frozen,
        tmp_path / "basecalls",
        _FakeDorado(model="chem_hac@v2.0.0"),
    )
    second = _publish_lineage(upgraded_plan, frozen, second_basecall, root)

    assert first.lineage_id != second.lineage_id
    assert {item.lineage_id for item in list_published_rebasecall_lineages(root)} == {
        first.lineage_id,
        second.lineage_id,
    }


def test_a_killed_stage_leaves_no_lineage_and_no_staging(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"
    prior = _publish_lineage(plan, frozen, basecall, root)

    upgraded_plan = replace(
        plan,
        _model_resolution=replace(
            plan._model_resolution,
            model_bundle_digest="e" * 64,
            simplex_model=replace(plan._model_resolution.simplex_model, name="chem_hac@v3.0.0"),
        ),
    )
    second_basecall = _execute(
        upgraded_plan,
        frozen,
        tmp_path / "basecalls",
        _FakeDorado(model="chem_hac@v3.0.0"),
    )
    with pytest.raises(RuntimeError, match="stage was killed"):
        with staged_lineage(
            upgraded_plan,
            frozen,
            second_basecall,
            root,
            accepted_plan_id=upgraded_plan.plan_id,
        ) as staged:
            staged.record_stage_generation("raw", "desc-b")
            raise RuntimeError("stage was killed")

    # The parent's prior lineage is unchanged and still discoverable.
    assert [item.lineage_id for item in list_published_rebasecall_lineages(root)] == [
        prior.lineage_id
    ]
    read_published_rebasecall_lineage(prior.directory, expected_lineage_id=prior.lineage_id)
    staging = root / LINEAGE_STAGING_SUBDIR
    assert not any(staging.iterdir())


def test_a_lineage_without_a_descendant_raw_generation_is_refused(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"

    with pytest.raises(RebasecallLineageError) as error:
        with staged_lineage(plan, frozen, basecall, root, accepted_plan_id=plan.plan_id):
            pass

    assert error.value.code == "lineage_incomplete"
    assert not (root / LINEAGES_SUBDIR).exists() or not any((root / LINEAGES_SUBDIR).iterdir())


def test_republishing_one_lineage_identity_is_refused(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"
    _publish_lineage(plan, frozen, basecall, root)

    with pytest.raises(RebasecallLineageError) as error:
        _publish_lineage(plan, frozen, basecall, root)

    assert error.value.code == "lineage_already_published"


def test_a_basecall_from_another_selection_cannot_form_a_lineage(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    mismatched = replace(basecall, manifest={**basecall.manifest, "selection_id": "other"})

    with pytest.raises(RebasecallLineageError) as error:
        with staged_lineage(
            plan,
            frozen,
            mismatched,
            tmp_path / "rebasecall_outputs",
            accepted_plan_id=plan.plan_id,
        ):
            pass

    assert error.value.code == "lineage_basecall_mismatch"


def test_blocked_or_mismatched_plans_publish_nothing(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"
    blocked = replace(
        plan,
        blockers=(rebasecall_plan.RebasecallPlanReason("blocked", "blocked for test"),),
    )

    with pytest.raises(RebasecallLineageError) as mismatch:
        with staged_lineage(plan, frozen, basecall, root, accepted_plan_id="not-the-plan"):
            pass
    with pytest.raises(RebasecallLineageError) as blocked_error:
        with staged_lineage(blocked, frozen, basecall, root, accepted_plan_id=blocked.plan_id):
            pass

    assert mismatch.value.code == "accepted_plan_mismatch"
    assert blocked_error.value.code == "accepted_plan_blocked"
    assert not root.exists() or not any((root / LINEAGES_SUBDIR).glob("*"))


def test_a_tampered_lineage_manifest_is_rejected_on_read(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    lineage = _publish_lineage(plan, frozen, basecall, tmp_path / "rebasecall_outputs")
    manifest_path = lineage.directory / LINEAGE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage_generations"]["raw"] = "someone-elses-generation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RebasecallLineageError) as error:
        read_published_rebasecall_lineage(lineage.directory)

    assert error.value.code == "lineage_artifact_invalid"


def test_a_stage_map_that_disagrees_with_the_manifest_is_rejected(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    lineage = _publish_lineage(plan, frozen, basecall, tmp_path / "rebasecall_outputs")
    (lineage.directory / LINEAGE_STAGE_GENERATIONS_FILENAME).write_text(
        json.dumps({"raw": "different"}), encoding="utf-8"
    )

    with pytest.raises(RebasecallLineageError) as error:
        read_published_rebasecall_lineage(lineage.directory)

    assert error.value.code == "lineage_artifact_invalid"


def test_validation_reports_sit_outside_lineage_identity(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    lineage = _publish_lineage(plan, frozen, basecall, tmp_path / "rebasecall_outputs")

    write_lineage_validation(lineage, {"status": "ok", "checked": 3})
    revalidated = read_published_rebasecall_lineage(
        lineage.directory,
        expected_lineage_id=lineage.lineage_id,
    )

    assert revalidated.lineage_id == lineage.lineage_id


def test_conflicting_or_unknown_stage_entries_are_refused(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    root = tmp_path / "rebasecall_outputs"

    with pytest.raises(RebasecallLineageError) as error:
        with staged_lineage(plan, frozen, basecall, root, accepted_plan_id=plan.plan_id) as staged:
            staged.record_stage_generation("raw", "desc-a")
            staged.record_stage_generation("raw", "desc-b")

    assert error.value.code == "lineage_stage_generation_conflict"


# --- descendant raw generation provenance -----------------------------------


def test_descendant_provenance_derives_its_kind_from_the_basecall(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    identity = build_lineage_identity(plan, frozen, basecall)

    provenance = descendant_raw_provenance("lineage-a", identity, basecall)

    assert provenance["generation_kind"] == basecall.generation_kind == "parent_universe"
    assert provenance["basecall_id"] == basecall.basecall_id
    assert provenance["selection_id"] == frozen.selection_id


def test_a_descendant_generation_publishes_without_taking_current(tmp_path):
    _publish(tmp_path, generation_id="parent-a")
    parent_dir, parent_manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    sources, dependencies, regions = _publication_sources(tmp_path)

    outputs = publish_raw_generation(
        tmp_path,
        sources,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
        dependencies=dependencies,
        region_artifacts=regions,
        generation_id="descendant-a",
        select_current=False,
        lineage_provenance={
            "lineage_id": "lineage-a",
            "origin_experiment_uid": "uid-a",
            "parent_raw_generation_id": "parent-a",
            "parent_preprocess_generation_id": None,
            "selection_id": "selection-a",
            "source_resolution_digest": None,
            "basecall_id": "basecall-a",
            "generation_kind": "selected_cohort",
            "identity_map": None,
        },
    )

    still_current, still_manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    descendant = validate_raw_generation(outputs["generation"], run_root=tmp_path)

    # The descendant is published and addressable, but the parent still resolves.
    assert still_current == parent_dir
    assert still_manifest["generation_id"] == parent_manifest["generation_id"] == "parent-a"
    assert descendant["generation_id"] == "descendant-a"
    assert descendant["lineage"]["generation_kind"] == "selected_cohort"
    assert descendant["schema_version"] == 3


def test_malformed_descendant_provenance_publishes_nothing(tmp_path):
    _publish(tmp_path, generation_id="parent-a")
    sources, dependencies, regions = _publication_sources(tmp_path)
    generations = tmp_path / "raw_outputs" / "generations"
    before = sorted(path.name for path in generations.iterdir())

    with pytest.raises(RawGenerationError, match="lineage provenance"):
        publish_raw_generation(
            tmp_path,
            sources,
            config_hash="config-a",
            input_artifact_ids=["input-manifest:abc"],
            dependencies=dependencies,
            region_artifacts=regions,
            generation_id="descendant-a",
            select_current=False,
            lineage_provenance={"lineage_id": "lineage-a"},
        )

    assert sorted(path.name for path in generations.iterdir()) == before


def test_an_ordinary_generation_records_no_lineage(tmp_path):
    outputs = _publish(tmp_path, generation_id="parent-a")

    manifest = validate_raw_generation(outputs["generation"], run_root=tmp_path)

    assert manifest["lineage"] is None


def test_append_refuses_a_lineage_descendant_as_its_base(tmp_path, monkeypatch):
    """A selected descendant is a parallel lineage, never a later parent state."""
    from smftools.informatics import raw_append as append_module

    _publish(tmp_path, generation_id="parent-a")
    sources, dependencies, regions = _publication_sources(tmp_path)
    outputs = publish_raw_generation(
        tmp_path,
        sources,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
        dependencies=dependencies,
        region_artifacts=regions,
        generation_id="descendant-a",
        select_current=False,
        lineage_provenance={
            "lineage_id": "lineage-a",
            "origin_experiment_uid": "uid-a",
            "parent_raw_generation_id": "parent-a",
            "parent_preprocess_generation_id": None,
            "selection_id": "selection-a",
            "source_resolution_digest": None,
            "basecall_id": "basecall-a",
            "generation_kind": "selected_cohort",
            "identity_map": None,
        },
    )

    # Isolate the lineage gate: manifest reading and transition classification
    # have their own coverage, and a pure source addition is the case that would
    # otherwise be permitted.
    monkeypatch.setattr(append_module, "read_resolved_input_manifest", lambda _path: None)
    monkeypatch.setattr(
        append_module,
        "classify_input_manifest_transition",
        lambda *_args: SimpleNamespace(permits_incremental_append=True, kind=None),
    )

    descendant_plan = plan_raw_append(
        Path(outputs["generation"]),
        None,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
    )
    parent_plan = plan_raw_append(
        tmp_path / "raw_outputs" / "generations" / "parent-a",
        None,
        run_root=tmp_path,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
    )

    assert descendant_plan.eligible is False
    assert "lineage descendant" in descendant_plan.reason
    # The same call against the ordinary parent is still permitted, so the gate
    # is the lineage block and not something incidental to the fixture.
    assert parent_plan.eligible is True
