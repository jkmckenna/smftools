from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import smftools.machine_learning.artifacts.publication as publication
from smftools.machine_learning.artifacts import (
    ArtifactReference,
    EnvironmentRecord,
    MLArtifactConflictError,
    MLArtifactPublicationError,
    ModelLineage,
    ModelManifest,
    ResolvedDefinition,
    RunManifest,
    SerializationPolicy,
    cleanup_abandoned_staging,
    file_sha256,
    publish_bundle,
    rebuild_workspace_indexes,
    resolve_model_alias,
    set_model_alias,
    validate_published_bundle,
)
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

RUN_ID = "12345678-1234-5678-1234-567812345678"
NOW = "2026-07-30T12:00:00+00:00"
STARTED = "2026-07-30T12:01:00+00:00"
DONE = "2026-07-30T12:02:00+00:00"
DATASET_ID = "3" * 64
SPLIT_ID = "4" * 64
INPUT_HASH = "5" * 64
LABEL_HASH = "6" * 64
PLAN_HASH = "7" * 64


def _workspace(tmp_path: Path, *, owner: str = "experiment") -> MLWorkspace:
    root = tmp_path / owner
    return MLWorkspace(
        scope_kind="experiment",
        scope_id="experiment-1",
        owner_root=root,
        root=root / "ml_outputs",
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


def _source(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / "sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _reference(role: str, relative_path: str, source: Path) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=relative_path,
        sha256=file_sha256(source),
        size_bytes=source.stat().st_size,
        media_type="application/octet-stream",
    )


def _run_bundle(tmp_path: Path, workspace: MLWorkspace) -> tuple[RunManifest, dict[str, Path]]:
    plan = _source(tmp_path, "resolved_plan.json", b'{"plan": 1}\n')
    config = _source(tmp_path, "resolved_config.json", b'{"config": 1}\n')
    plan_reference = _reference("resolved_plan", "resolved_plan.json", plan)
    config_reference = _reference("resolved_config", "resolved_config.json", config)
    planned = RunManifest.create(
        run_id=RUN_ID,
        workspace_id=workspace.workspace_id,
        action="train",
        job_name="train-model",
        plan_hash=PLAN_HASH,
        resolved_plan=plan_reference,
        resolved_config=config_reference,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        model_keys=("classifier",),
        environment=_environment(),
        seeds={"model": 1},
        device="cpu",
        created_at=NOW,
    )
    completed = planned.transition("running", at=STARTED).transition("completed", at=DONE)
    return completed, {
        plan_reference.relative_path: plan,
        config_reference.relative_path: config,
    }


def _model_bundle(
    tmp_path: Path,
    workspace: MLWorkspace,
    *,
    name: str = "model.bin",
    content: bytes = b"trained-model",
) -> tuple[ModelManifest, dict[str, Path]]:
    source = _source(tmp_path, name, content)
    reference = _reference("model", f"payload/{name}", source)
    manifest = ModelManifest.create(
        model_key="classifier",
        backend="sklearn",
        family="random_forest",
        task_type="binary_classification",
        originating_run_id=RUN_ID,
        workspace_id=workspace.workspace_id,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        input_schema_hash=INPUT_HASH,
        label_schema_hash=LABEL_HASH,
        architecture=ResolvedDefinition.create(
            name="random_forest",
            version="1",
            parameters={"n_estimators": 100},
        ),
        lineage=ModelLineage(
            kind="from_scratch",
            parent_model_ids=(),
            parent_roles=(),
        ),
        artifact=reference,
        serialization=SerializationPolicy(
            format="skops",
            loader="skops.io.load",
            requires_unsafe_load=False,
            allowed_types=("sklearn.ensemble.RandomForestClassifier",),
            package_versions={"scikit-learn": "1.6.1", "skops": "0.12.0"},
        ),
        environment=_environment(),
        created_at=DONE,
    )
    return manifest, {reference.relative_path: source}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_publish_model_is_atomic_validated_and_retry_safe(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _model_bundle(tmp_path, workspace)

    first = publish_bundle(workspace, manifest, sources=sources)
    second = publish_bundle(workspace, manifest, sources=sources)

    assert first.created is True
    assert second.created is False
    assert first.path == workspace.model_dir(manifest.model_id)
    assert (
        validate_published_bundle(
            workspace,
            first.path,
            kind="model",
            expected_id=manifest.model_id,
        ).manifest_sha256
        == first.manifest_sha256
    )
    assert not (workspace.models_root / ".staging").exists()


def test_publish_run_accepts_workspace_prefixed_references(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _run_bundle(tmp_path, workspace)
    raw = manifest.to_dict()
    prefix = f"runs/{RUN_ID}/"
    raw["resolved_plan"]["relative_path"] = prefix + "resolved_plan.json"
    raw["resolved_config"]["relative_path"] = prefix + "resolved_config.json"
    manifest = RunManifest.from_dict(raw)
    sources = {
        prefix + "resolved_plan.json": sources["resolved_plan.json"],
        prefix + "resolved_config.json": sources["resolved_config.json"],
    }

    published = publish_bundle(workspace, manifest, sources=sources)

    assert published.path == workspace.run_paths(RUN_ID).root
    assert validate_published_bundle(workspace, published.path, kind="run").artifact_id == RUN_ID


def test_publish_rejects_checksum_mismatch_without_visible_artifact(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _model_bundle(tmp_path, workspace)
    next(iter(sources.values())).write_bytes(b"tampered")

    with pytest.raises(MLArtifactPublicationError, match="size mismatch|checksum mismatch"):
        publish_bundle(workspace, manifest, sources=sources)

    assert not workspace.model_dir(manifest.model_id).exists()


def test_publish_rejects_incomplete_or_extra_source_inventory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, _ = _model_bundle(tmp_path, workspace)

    with pytest.raises(MLArtifactPublicationError, match="exactly match"):
        publish_bundle(workspace, manifest, sources={})


def test_existing_tampered_identity_is_an_explicit_conflict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _model_bundle(tmp_path, workspace)
    published = publish_bundle(workspace, manifest, sources=sources)
    (published.path / "payload" / "model.bin").write_bytes(b"different")

    with pytest.raises(MLArtifactConflictError, match="already bound"):
        publish_bundle(workspace, manifest, sources=sources)


def test_same_run_id_cannot_be_rebound_to_a_different_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _run_bundle(tmp_path, workspace)
    publish_bundle(workspace, manifest, sources=sources)
    raw = manifest.to_dict()
    raw["job_name"] = "different-job"
    conflicting = RunManifest.from_dict(raw)

    with pytest.raises(MLArtifactConflictError, match="different manifest"):
        publish_bundle(workspace, conflicting, sources=sources)


def test_interrupted_publication_leaves_no_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _model_bundle(tmp_path, workspace)
    original_replace = publication.os.replace

    def fail_final_replace(source, destination):
        if Path(source).is_dir() and Path(destination) == workspace.model_dir(manifest.model_id):
            raise OSError("simulated interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(publication.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        publish_bundle(workspace, manifest, sources=sources)

    assert not workspace.model_dir(manifest.model_id).exists()
    staging = workspace.models_root / ".staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_concurrent_identical_publication_creates_exactly_once(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest, sources = _model_bundle(tmp_path, workspace)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_bundle, workspace, manifest, sources=sources) for _ in range(2)
        ]
    published = [future.result() for future in futures]

    assert sorted(item.created for item in published) == [False, True]
    assert {item.path for item in published} == {workspace.model_dir(manifest.model_id)}


def test_bundle_remains_valid_after_complete_workspace_relocation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, owner="first")
    manifest, sources = _model_bundle(tmp_path, workspace)
    publish_bundle(workspace, manifest, sources=sources)
    relocated = _workspace(tmp_path, owner="second")
    shutil.copytree(workspace.root, relocated.root)

    restored = validate_published_bundle(
        relocated,
        relocated.model_dir(manifest.model_id),
        kind="model",
        expected_id=manifest.model_id,
    )

    assert restored.artifact_id == manifest.model_id


def test_rebuild_indexes_is_deterministic_and_manifest_authoritative(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run, run_sources = _run_bundle(tmp_path, workspace)
    model, model_sources = _model_bundle(tmp_path, workspace)
    publish_bundle(workspace, run, sources=run_sources)
    publish_bundle(workspace, model, sources=model_sources)

    run_index, model_index = rebuild_workspace_indexes(workspace)
    expected_run = run_index.read_bytes()
    expected_model = model_index.read_bytes()
    run_index.unlink()
    model_index.unlink()
    rebuilt = rebuild_workspace_indexes(workspace)

    assert rebuilt == (run_index, model_index)
    assert run_index.read_bytes() == expected_run
    assert model_index.read_bytes() == expected_model
    assert _read_json(run_index)["records"][0]["run_id"] == RUN_ID
    assert _read_json(model_index)["records"][0]["model_id"] == model.model_id


def test_index_rebuild_rejects_tampered_authoritative_bundle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    model, sources = _model_bundle(tmp_path, workspace)
    published = publish_bundle(workspace, model, sources=sources)
    (published.path / "unexpected.txt").write_text("not checksummed", encoding="utf-8")

    with pytest.raises(MLArtifactPublicationError, match="inventory mismatch"):
        rebuild_workspace_indexes(workspace)


def test_aliases_repoint_without_mutating_immutable_models(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first, first_sources = _model_bundle(tmp_path, workspace)
    second, second_sources = _model_bundle(
        tmp_path,
        workspace,
        name="second.bin",
        content=b"second-trained-model",
    )
    first_bundle = publish_bundle(workspace, first, sources=first_sources)
    second_bundle = publish_bundle(workspace, second, sources=second_sources)
    first_manifest_bytes = (first_bundle.path / "model_manifest.json").read_bytes()

    set_model_alias(workspace, alias="promoted", model_id=first.model_id)
    assert resolve_model_alias(workspace, "promoted") == first
    set_model_alias(workspace, alias="promoted", model_id=second.model_id)
    assert resolve_model_alias(workspace, "promoted") == second
    rebuild_workspace_indexes(workspace)

    assert (first_bundle.path / "model_manifest.json").read_bytes() == first_manifest_bytes
    assert _read_json(workspace.index_root / "models.json")["aliases"] == {
        "promoted": second.model_id
    }
    assert second_bundle.path.is_dir()


def test_cleanup_only_removes_old_known_staging_transactions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    old = workspace.runs_root / ".staging" / "old"
    fresh = workspace.models_root / ".staging" / "fresh"
    old_lock = workspace.runs_root / ".locks" / "old.lock"
    fresh_lock = workspace.models_root / ".locks" / "fresh.lock"
    old.mkdir(parents=True)
    fresh.mkdir(parents=True)
    old_lock.parent.mkdir(parents=True)
    fresh_lock.parent.mkdir(parents=True)
    (old / "partial").write_text("old", encoding="utf-8")
    (fresh / "partial").write_text("fresh", encoding="utf-8")
    old_lock.write_text("", encoding="utf-8")
    fresh_lock.write_text("", encoding="utf-8")
    os.utime(old, (100.0, 100.0))
    os.utime(fresh, (190.0, 190.0))
    os.utime(old_lock, (100.0, 100.0))
    os.utime(fresh_lock, (190.0, 190.0))

    removed = cleanup_abandoned_staging(
        workspace,
        older_than_seconds=50.0,
        now=200.0,
    )

    assert set(removed) == {old, old_lock}
    assert not old.exists()
    assert not old_lock.exists()
    assert fresh.exists()
    assert fresh_lock.exists()


def test_publication_and_cleanup_reject_redirected_category_root(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace.root.mkdir(parents=True)
    workspace.models_root.symlink_to(outside, target_is_directory=True)
    manifest, sources = _model_bundle(tmp_path, workspace)

    with pytest.raises(MLArtifactPublicationError, match="symbolic link"):
        publish_bundle(workspace, manifest, sources=sources)
    with pytest.raises(MLArtifactPublicationError, match="symbolic link"):
        cleanup_abandoned_staging(workspace, older_than_seconds=0)
