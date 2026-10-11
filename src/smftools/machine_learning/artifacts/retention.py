"""Retire the model files of superseded runs (`MLR-16`).

Every training run publishes its fitted models as immutable bundles under
``models/``. Rerunning a job -- after a code fix, a data refresh -- publishes
new bundles and leaves the old ones, so the store grows with every pass: on
one project 176 GB, a third of it from runs that newer runs had replaced.

A run is **superseded** when a newer completed run in the same workspace has
the same action, job name, model keys, plan hash and caller tags (the
identity a caller reuses runs by); optionally, without the plan hash, when the caller's tags
name a run's purpose and a changed plan is a deliberate replacement. Retention keeps the newest ``keep_last`` completed runs of
each such group and every run they depend on (``source_run_ids``); the model
bundles of the other runs are **retired** -- removed -- unless a kept run uses
them (``source_model_ids``), a kept model descends from them (lineage), or an
alias names them.

Run bundles are kept: they are small and hold the record (metrics,
predictions, which models were fit). A retired model is listed in
``index/retired_models.json`` with the run that superseded its own, so a
later load fails with that reason rather than "not found". Planning writes
nothing; :func:`apply_model_retention` deletes.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from smftools.readwrite import atomic_write_json

from ..workspace import MLWorkspace
from .indexing import _alias_records, _published_directories, rebuild_workspace_indexes
from .model import ModelManifest
from .publication import MODEL_MANIFEST_FILENAME, RUN_MANIFEST_FILENAME

RETIRED_MODELS_FILENAME = "retired_models.json"
RETENTION_SCHEMA_VERSION = 1
_TAGS_FILENAME = "tags.json"
_COMPLETED = "completed"


class MLRetentionError(RuntimeError):
    """A retention request cannot be planned or applied safely."""


@dataclass(frozen=True)
class RetiredModel:
    """A model bundle retention removes, and why."""

    model_id: str
    originating_run_id: str
    superseded_by: str
    size_bytes: int


@dataclass
class RetentionPlan:
    """What :func:`apply_model_retention` would remove; nothing is written."""

    workspace_root: str
    keep_last: int
    match_plan_hash: bool = True
    runs: int = 0
    kept_runs: int = 0
    superseded_runs: int = 0
    models: int = 0
    retire: list[RetiredModel] = field(default_factory=list)
    protected: dict[str, str] = field(default_factory=dict)
    orphan_models: list[str] = field(default_factory=list)

    @property
    def retire_bytes(self) -> int:
        return sum(model.size_bytes for model in self.retire)

    def to_dict(self) -> dict:
        return {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "workspace_root": self.workspace_root,
            "keep_last": self.keep_last,
            "match_plan_hash": self.match_plan_hash,
            "runs": self.runs,
            "kept_runs": self.kept_runs,
            "superseded_runs": self.superseded_runs,
            "models": self.models,
            "retire_models": len(self.retire),
            "retire_bytes": self.retire_bytes,
            "protected_models": len(self.protected),
            "orphan_models": len(self.orphan_models),
            "retire": [model.__dict__ for model in self.retire],
            "protected": self.protected,
        }


def _size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_records(workspace: MLWorkspace, *, match_plan_hash: bool) -> dict[str, dict]:
    records = {}
    for path in _published_directories(workspace.runs_root):
        manifest_path = path / RUN_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        tags_path = path / _TAGS_FILENAME
        tags = _read_json(tags_path) if tags_path.is_file() else {}
        # Without tags nothing names the run's purpose: only the plan does.
        plan_hash = manifest["plan_hash"] if match_plan_hash or not tags else None
        records[manifest["run_id"]] = {
            "identity": json.dumps(
                [
                    manifest["action"],
                    manifest["job_name"],
                    sorted(manifest.get("model_keys", ())),
                    plan_hash,
                    tags,
                ],
                sort_keys=True,
            ),
            "state": manifest["state"],
            "created_at": manifest["created_at"],
            "source_run_ids": list(manifest.get("source_run_ids", ())),
            "source_model_ids": list(manifest.get("source_model_ids", ())),
        }
    return records


def _model_records(workspace: MLWorkspace) -> dict[str, ModelManifest]:
    models = {}
    for path in _published_directories(workspace.models_root):
        manifest_path = path / MODEL_MANIFEST_FILENAME
        if manifest_path.is_file():
            manifest = ModelManifest.from_dict(_read_json(manifest_path))
            models[manifest.model_id] = manifest
    return models


def plan_model_retention(
    workspace: MLWorkspace,
    *,
    keep_last: int = 1,
    keep_run_ids: Iterable[str] = (),
    match_plan_hash: bool = True,
) -> RetentionPlan:
    """Plan which model bundles to retire. Writes nothing.

    Args:
        workspace: The ML workspace.
        keep_last: Completed runs kept per identity (newest first); >= 1.
        keep_run_ids: Runs to keep regardless (and what they depend on).
        match_plan_hash: Group runs by plan hash too (default). ``False``
            treats a newer run with the same action, job, model keys and tags as
            replacing an older one even when the plan changed (new labels,
            masks, data) -- for callers whose tags name a run's purpose.
            Untagged runs are always grouped by plan hash.
    """
    if keep_last < 1:
        raise MLRetentionError("keep_last must be at least 1")
    runs = _run_records(workspace, match_plan_hash=match_plan_hash)
    models = _model_records(workspace)
    plan = RetentionPlan(
        workspace_root=str(workspace.root), keep_last=keep_last, match_plan_hash=match_plan_hash
    )
    plan.runs, plan.models = len(runs), len(models)

    groups: dict[str, list[str]] = {}
    for run_id, record in runs.items():
        groups.setdefault(record["identity"], []).append(run_id)
    kept: set[str] = set(keep_run_ids) & set(runs)
    superseded_by: dict[str, str] = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda run_id: runs[run_id]["created_at"], reverse=True)
        completed = [run_id for run_id in ordered if runs[run_id]["state"] == _COMPLETED]
        if not completed:
            kept.update(ordered)  # nothing newer replaces them
            continue
        keepers = completed[:keep_last]
        kept.update(keepers)
        for run_id in ordered:
            if run_id not in keepers:
                superseded_by.setdefault(run_id, keepers[0])
    # A kept run keeps the runs it read from.
    pending = list(kept)
    while pending:
        for source in runs.get(pending.pop(), {}).get("source_run_ids", ()):
            if source in runs and source not in kept:
                kept.add(source)
                pending.append(source)
    plan.kept_runs = len(kept)
    plan.superseded_runs = len(set(runs) - kept)

    protected: dict[str, str] = {}
    for alias, model_id in _alias_records(workspace).items():
        protected.setdefault(model_id, f"alias {alias}")
    for run_id in kept:
        for model_id in runs[run_id]["source_model_ids"]:
            protected.setdefault(model_id, f"used by kept run {run_id}")
    for model_id, manifest in models.items():
        if manifest.originating_run_id in kept:
            protected.setdefault(model_id, f"made by kept run {manifest.originating_run_id}")
    pending = list(protected)
    while pending:
        manifest = models.get(pending.pop())
        if manifest is None:
            continue
        for parent in manifest.lineage.parent_model_ids:
            if parent not in protected:
                protected[parent] = f"parent of kept model {manifest.model_id}"
                pending.append(parent)
    plan.protected = protected

    for model_id, manifest in sorted(models.items()):
        if model_id in protected:
            continue
        origin = manifest.originating_run_id
        if origin not in runs:
            plan.orphan_models.append(model_id)  # no run record: left alone
            continue
        plan.retire.append(
            RetiredModel(
                model_id=model_id,
                originating_run_id=origin,
                superseded_by=superseded_by.get(origin, ""),
                size_bytes=_size(workspace.model_dir(model_id)),
            )
        )
    return plan


def retired_models(workspace: MLWorkspace) -> dict[str, dict]:
    """Model id -> retirement record (empty when nothing was retired)."""
    path = workspace.index_root / RETIRED_MODELS_FILENAME
    if not path.is_file():
        return {}
    return {record["model_id"]: record for record in _read_json(path).get("records", [])}


def apply_model_retention(workspace: MLWorkspace, plan: RetentionPlan) -> Path:
    """Remove the plan's model bundles, record them, rebuild the indexes.

    Each bundle is recorded before it is removed, so an interruption leaves a
    record of every model that may be gone, never a missing model without one.
    """
    if Path(plan.workspace_root) != workspace.root:
        raise MLRetentionError("plan was made for a different workspace")
    current = plan_model_retention(
        workspace, keep_last=plan.keep_last, match_plan_hash=plan.match_plan_hash
    )
    allowed = {model.model_id for model in current.retire}
    stale = [model.model_id for model in plan.retire if model.model_id not in allowed]
    if stale:
        raise MLRetentionError(
            f"{len(stale)} planned model(s) are no longer retirable (the store changed); replan"
        )
    path = workspace.index_root / RETIRED_MODELS_FILENAME
    workspace.index_root.mkdir(parents=True, exist_ok=True)
    records = retired_models(workspace)
    retired_at = datetime.now(timezone.utc).isoformat()
    for model in plan.retire:
        records[model.model_id] = {**model.__dict__, "retired_at": retired_at}
    atomic_write_json(
        path,
        {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "workspace_id": workspace.workspace_id,
            "records": sorted(records.values(), key=lambda record: record["model_id"]),
        },
    )
    for model in plan.retire:
        directory = workspace.model_dir(model.model_id)
        if directory.is_dir():
            shutil.rmtree(directory)
    rebuild_workspace_indexes(workspace)
    return path
