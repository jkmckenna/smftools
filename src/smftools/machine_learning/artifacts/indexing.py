"""Rebuildable indexes and mutable aliases for immutable ML artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from smftools.readwrite import atomic_write_json

from ..workspace import MLWorkspace
from .model import ModelManifest
from .publication import (
    LOCKS_DIRNAME,
    MODEL_MANIFEST_FILENAME,
    RUN_MANIFEST_FILENAME,
    STAGING_DIRNAME,
    MLArtifactPublicationError,
    PublishedBundle,
    _assert_workspace_directory,
    validate_alias_name,
    validate_published_bundle,
)
from .run import RunManifest

ML_ARTIFACT_INDEX_VERSION = 1
ML_MODEL_ALIAS_VERSION = 1
RUN_INDEX_FILENAME = "runs.json"
MODEL_INDEX_FILENAME = "models.json"
MODEL_ALIASES_DIRNAME = "model_aliases"


def _manifest(path: Path, kind: str) -> RunManifest | ModelManifest:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise MLArtifactPublicationError(f"{kind} manifest root must be an object")
    return RunManifest.from_dict(raw) if kind == "run" else ModelManifest.from_dict(raw)


def _published_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name not in {STAGING_DIRNAME, LOCKS_DIRNAME}
    )


def _run_record(bundle: PublishedBundle, manifest: RunManifest, workspace: MLWorkspace) -> dict:
    return {
        "run_id": manifest.run_id,
        "state": manifest.state,
        "action": manifest.action,
        "job_name": manifest.job_name,
        "dataset_snapshot_id": manifest.dataset_snapshot_id,
        "split_id": manifest.split_id,
        "model_keys": list(manifest.model_keys),
        "source_model_ids": list(manifest.source_model_ids),
        "created_at": manifest.created_at,
        "finished_at": manifest.finished_at,
        "manifest_path": workspace.portable_reference(bundle.path / RUN_MANIFEST_FILENAME),
        "manifest_sha256": bundle.manifest_sha256,
    }


def _model_record(bundle: PublishedBundle, manifest: ModelManifest, workspace: MLWorkspace) -> dict:
    return {
        "model_id": manifest.model_id,
        "model_key": manifest.model_key,
        "backend": manifest.backend,
        "family": manifest.family,
        "task_type": manifest.task_type,
        "originating_run_id": manifest.originating_run_id,
        "dataset_snapshot_id": manifest.dataset_snapshot_id,
        "split_id": manifest.split_id,
        "lineage_kind": manifest.lineage.kind,
        "parent_model_ids": list(manifest.lineage.parent_model_ids),
        "created_at": manifest.created_at,
        "manifest_path": workspace.portable_reference(bundle.path / MODEL_MANIFEST_FILENAME),
        "manifest_sha256": bundle.manifest_sha256,
    }


def _alias_records(workspace: MLWorkspace) -> dict[str, str]:
    root = workspace.index_root / MODEL_ALIASES_DIRNAME
    _assert_workspace_directory(workspace, root, "model aliases root")
    if not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in sorted(root.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        expected = {"schema_version", "alias", "model_id", "manifest_sha256"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise MLArtifactPublicationError(f"invalid model alias record: {path}")
        alias = validate_alias_name(raw["alias"])
        if path.name != f"{alias}.json" or raw["schema_version"] != ML_MODEL_ALIAS_VERSION:
            raise MLArtifactPublicationError(f"invalid model alias identity: {path}")
        model_id = str(raw["model_id"])
        bundle = validate_published_bundle(
            workspace,
            workspace.model_dir(model_id),
            kind="model",
            expected_id=model_id,
        )
        if raw["manifest_sha256"] != bundle.manifest_sha256:
            raise MLArtifactPublicationError(f"stale model alias checksum: {path}")
        result[alias] = model_id
    return result


def rebuild_workspace_indexes(workspace: MLWorkspace) -> tuple[Path, Path]:
    """Rebuild disposable run/model indexes from authoritative manifests."""
    _assert_workspace_directory(workspace, workspace.runs_root, "runs root")
    _assert_workspace_directory(workspace, workspace.models_root, "models root")
    _assert_workspace_directory(workspace, workspace.index_root, "index root")
    run_records = []
    for path in _published_directories(workspace.runs_root):
        bundle = validate_published_bundle(workspace, path, kind="run", expected_id=path.name)
        manifest = _manifest(path / RUN_MANIFEST_FILENAME, "run")
        if not isinstance(manifest, RunManifest):
            raise AssertionError
        run_records.append(_run_record(bundle, manifest, workspace))
    model_records = []
    for path in _published_directories(workspace.models_root):
        bundle = validate_published_bundle(workspace, path, kind="model", expected_id=path.name)
        manifest = _manifest(path / MODEL_MANIFEST_FILENAME, "model")
        if not isinstance(manifest, ModelManifest):
            raise AssertionError
        model_records.append(_model_record(bundle, manifest, workspace))
    aliases = _alias_records(workspace)
    workspace.index_root.mkdir(parents=True, exist_ok=True)
    run_index = workspace.index_root / RUN_INDEX_FILENAME
    model_index = workspace.index_root / MODEL_INDEX_FILENAME
    atomic_write_json(
        run_index,
        {
            "schema_version": ML_ARTIFACT_INDEX_VERSION,
            "workspace_id": workspace.workspace_id,
            "records": sorted(run_records, key=lambda item: item["run_id"]),
        },
    )
    atomic_write_json(
        model_index,
        {
            "schema_version": ML_ARTIFACT_INDEX_VERSION,
            "workspace_id": workspace.workspace_id,
            "aliases": dict(sorted(aliases.items())),
            "records": sorted(model_records, key=lambda item: item["model_id"]),
        },
    )
    return run_index, model_index


def set_model_alias(workspace: MLWorkspace, *, alias: str, model_id: str) -> Path:
    """Atomically point a mutable alias at an existing immutable model."""
    alias = validate_alias_name(alias)
    bundle = validate_published_bundle(
        workspace,
        workspace.model_dir(model_id),
        kind="model",
        expected_id=model_id,
    )
    _assert_workspace_directory(workspace, workspace.index_root, "index root")
    aliases_root = workspace.index_root / MODEL_ALIASES_DIRNAME
    _assert_workspace_directory(workspace, aliases_root, "model aliases root")
    path = aliases_root / f"{alias}.json"
    return atomic_write_json(
        path,
        {
            "schema_version": ML_MODEL_ALIAS_VERSION,
            "alias": alias,
            "model_id": model_id,
            "manifest_sha256": bundle.manifest_sha256,
        },
    )


def resolve_model_alias(workspace: MLWorkspace, alias: str) -> ModelManifest:
    """Resolve and validate a mutable alias without loading model state."""
    alias = validate_alias_name(alias)
    _assert_workspace_directory(workspace, workspace.index_root, "index root")
    path = workspace.index_root / MODEL_ALIASES_DIRNAME / f"{alias}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown ML model alias: {alias}")
    aliases = _alias_records(workspace)
    model_id = aliases.get(alias)
    if model_id is None:
        raise MLArtifactPublicationError(f"model alias {alias!r} is invalid")
    manifest = _manifest(workspace.model_dir(model_id) / MODEL_MANIFEST_FILENAME, "model")
    if not isinstance(manifest, ModelManifest):
        raise AssertionError
    return manifest
