"""Retiring the model files of superseded runs (`MLR-16`)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from smftools.machine_learning.artifacts import (
    ArtifactReference,
    EnvironmentRecord,
    FailureRecord,
    MLRetentionError,
    ModelLineage,
    ModelManifest,
    ResolvedDefinition,
    RunManifest,
    SerializationPolicy,
    apply_model_retention,
    file_sha256,
    plan_model_retention,
    publish_bundle,
    retired_models,
    set_model_alias,
)
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

DATASET_ID = "3" * 64
SPLIT_ID = "4" * 64


def _workspace(tmp_path: Path) -> MLWorkspace:
    root = tmp_path / "project"
    return MLWorkspace(
        scope_kind="experiment", scope_id="experiment-1", owner_root=root, root=root / "ml_outputs"
    )


def _environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        smftools_version="2.19.0.dev0",
        python_version="3.12.4",
        platform="test",
        code_revision="abc123",
        dirty_tree=False,
        dependencies={"numpy": "2.1.0"},
    )


def _source(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "sources" / uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _reference(role: str, relative_path: str, source: Path) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=relative_path,
        sha256=file_sha256(source),
        size_bytes=source.stat().st_size,
        media_type="application/json",
    )


class Store:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.workspace = _workspace(tmp_path)

    def run(
        self,
        *,
        hour: int,
        plan: str = "7",
        tags: dict | None = None,
        model_keys=("nb",),
        state: str = "completed",
        source_run_ids=(),
        source_model_ids=(),
    ) -> str:
        run_id = str(uuid.uuid4())
        created = f"2026-07-30T{hour:02d}:00:00+00:00"
        plan_file = _source(self.tmp_path, b'{"plan": 1}\n')
        config_file = _source(self.tmp_path, b'{"config": 1}\n')
        tags_file = _source(self.tmp_path, json.dumps(tags or {}).encode())
        planned = RunManifest.create(
            run_id=run_id,
            workspace_id=self.workspace.workspace_id,
            action="train",
            job_name="train",
            plan_hash=plan * 64,
            resolved_plan=_reference("resolved_plan", "resolved_plan.json", plan_file),
            resolved_config=_reference("resolved_config", "resolved_config.json", config_file),
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
            model_keys=tuple(model_keys),
            source_model_ids=tuple(source_model_ids),
            source_run_ids=tuple(source_run_ids),
            environment=_environment(),
            seeds={"model": 1},
            device="cpu",
            created_at=created,
        )
        tags_reference = _reference("tags", "tags.json", tags_file)
        running = planned.transition("running", at=created)
        finished = (
            running.transition(state, at=created, artifacts=(tags_reference,))
            if state == "completed"
            else running.transition(
                "failed",
                at=created,
                failure=FailureRecord(error_type="RuntimeError", message="boom", phase="fit"),
                artifacts=(tags_reference,),
            )
        )
        publish_bundle(
            self.workspace,
            finished,
            sources={
                "resolved_plan.json": plan_file,
                "resolved_config.json": config_file,
                "tags.json": tags_file,
            },
        )
        return run_id

    def model(self, run_id: str, *, parents=()) -> str:
        source = _source(self.tmp_path, uuid.uuid4().bytes * 64)
        reference = _reference("model", "payload/model.bin", source)
        manifest = ModelManifest.create(
            model_key="nb",
            backend="sklearn",
            family="bernoulli_nb",
            task_type="binary_classification",
            originating_run_id=run_id,
            workspace_id=self.workspace.workspace_id,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
            input_schema_hash="5" * 64,
            label_schema_hash="6" * 64,
            architecture=ResolvedDefinition.create(name="nb", version="1", parameters={}),
            lineage=ModelLineage(
                kind="fine_tuned" if parents else "from_scratch",
                parent_model_ids=tuple(parents),
                parent_roles=tuple("base" for _ in parents),
            ),
            artifact=reference,
            serialization=SerializationPolicy(
                format="skops",
                loader="skops.io.load",
                requires_unsafe_load=False,
                allowed_types=("sklearn.naive_bayes.BernoulliNB",),
                package_versions={"scikit-learn": "1.6.1"},
            ),
            environment=_environment(),
            created_at="2026-07-30T12:00:00+00:00",
        )
        publish_bundle(self.workspace, manifest, sources={"payload/model.bin": source})
        return manifest.model_id


def test_a_rerun_with_the_same_identity_retires_the_older_models(tmp_path):
    store = Store(tmp_path)
    old = store.run(hour=1, tags={"task": "a"})
    old_model = store.model(old)
    new = store.run(hour=2, tags={"task": "a"})
    new_model = store.model(new)
    other = store.model(store.run(hour=0, tags={"task": "b"}))

    plan = plan_model_retention(store.workspace)
    assert [m.model_id for m in plan.retire] == [old_model]
    assert plan.retire[0].superseded_by == new
    assert plan.retire_bytes > 0
    assert {new_model, other} <= set(plan.protected)


def test_planning_writes_nothing(tmp_path):
    store = Store(tmp_path)
    old_model = store.model(store.run(hour=1))
    store.model(store.run(hour=2))
    plan_model_retention(store.workspace)
    assert store.workspace.model_dir(old_model).is_dir()
    assert retired_models(store.workspace) == {}


def test_a_changed_plan_is_a_different_job_unless_matched_by_tags(tmp_path):
    store = Store(tmp_path)
    old_model = store.model(store.run(hour=1, plan="7", tags={"task": "a"}))
    store.model(store.run(hour=2, plan="8", tags={"task": "a"}))
    assert plan_model_retention(store.workspace).retire == []
    by_tags = plan_model_retention(store.workspace, match_plan_hash=False)
    assert [m.model_id for m in by_tags.retire] == [old_model]


def test_different_model_keys_or_untagged_runs_never_supersede_each_other(tmp_path):
    store = Store(tmp_path)
    store.model(store.run(hour=1, plan="7", tags={"task": "a"}, model_keys=("nb", "rf")))
    store.model(store.run(hour=2, plan="8", tags={"task": "a"}, model_keys=("nb_cal",)))
    store.model(store.run(hour=1, plan="1"))
    store.model(store.run(hour=2, plan="2"))
    assert plan_model_retention(store.workspace, match_plan_hash=False).retire == []


def test_models_a_kept_run_uses_or_an_alias_names_are_kept(tmp_path):
    store = Store(tmp_path)
    base_run = store.run(hour=1, tags={"task": "a"})
    used = store.model(base_run)
    aliased_run = store.run(hour=1, tags={"task": "b"})
    aliased = store.model(aliased_run)
    parent_run = store.run(hour=1, tags={"task": "c"})
    parent = store.model(parent_run)
    for task in ("a", "b", "c"):
        store.model(store.run(hour=3, tags={"task": task}))
    store.run(hour=4, tags={"task": "evaluate"}, source_model_ids=(used,))
    set_model_alias(store.workspace, alias="published", model_id=aliased)
    child_run = store.run(hour=4, tags={"task": "tuned"})
    store.model(child_run, parents=(parent,))

    plan = plan_model_retention(store.workspace)
    assert plan.retire == []
    assert "used by kept run" in plan.protected[used]
    assert plan.protected[aliased] == "alias published"
    assert "parent of kept model" in plan.protected[parent]


def test_runs_a_kept_run_read_from_keep_their_models(tmp_path):
    store = Store(tmp_path)
    source_run = store.run(hour=1, tags={"task": "a"})
    model = store.model(source_run)
    store.model(store.run(hour=2, tags={"task": "a"}))
    store.run(hour=3, tags={"task": "report"}, source_run_ids=(source_run,))
    assert plan_model_retention(store.workspace).retire == []
    assert model in plan_model_retention(store.workspace).protected


def test_failed_reruns_do_not_supersede(tmp_path):
    store = Store(tmp_path)
    store.model(store.run(hour=1, tags={"task": "a"}))
    store.run(hour=2, tags={"task": "a"}, state="failed")
    assert plan_model_retention(store.workspace).retire == []


def test_apply_removes_records_and_reindexes(tmp_path):
    store = Store(tmp_path)
    old = store.run(hour=1, tags={"task": "a"})
    old_model = store.model(old)
    new = store.run(hour=2, tags={"task": "a"})
    new_model = store.model(new)
    plan = plan_model_retention(store.workspace)

    record = apply_model_retention(store.workspace, plan)

    assert not store.workspace.model_dir(old_model).exists()
    assert store.workspace.model_dir(new_model).is_dir()
    assert (store.workspace.runs_root / old).is_dir()  # the run record stays
    retired = retired_models(store.workspace)[old_model]
    assert retired["superseded_by"] == new and record.name == "retired_models.json"
    index = json.loads((store.workspace.index_root / "models.json").read_text())
    assert [entry["model_id"] for entry in index["records"]] == [new_model]


def test_apply_refuses_a_stale_plan(tmp_path):
    store = Store(tmp_path)
    store.model(store.run(hour=1, tags={"task": "a"}))
    store.model(store.run(hour=2, tags={"task": "a"}))
    plan = plan_model_retention(store.workspace, keep_last=1)
    set_model_alias(store.workspace, alias="keep", model_id=plan.retire[0].model_id)
    with pytest.raises(MLRetentionError, match="replan"):
        apply_model_retention(store.workspace, plan)


def test_loading_a_retired_model_names_the_replacement(tmp_path):
    from smftools.machine_learning.orchestration.runs import MLJobServiceError, _load_model

    store = Store(tmp_path)
    old_model = store.model(store.run(hour=1, tags={"task": "a"}))
    new = store.run(hour=2, tags={"task": "a"})
    store.model(new)
    apply_model_retention(store.workspace, plan_model_retention(store.workspace))
    with pytest.raises(MLJobServiceError, match=f"superseded by {new}"):
        _load_model(store.workspace, old_model, "sklearn")
