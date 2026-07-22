"""Consolidated experiment-level provenance manifest.

One JSON file per experiment (at the run root, alongside ``molecules.parquet``),
appended to -- never replaced -- as each pipeline stage runs. Closes two gaps
``spine.uns`` never covered: config parameters and the raw instrument input path were
previously nowhere in the store at all (only a ``config_hash`` for staleness
detection, not the values); "which stages have run" was previously inferred only from
directory presence, with no single readable index.

Phase 2 of ``dev/experiment_storage_schema.md`` (not tracked in git). Currently wired
into the raw stage only (``cli/load_adata.py``) -- preprocess/spatial/hmm can adopt
``record_stage_completion`` the same way once their executors are touched for the
schema's later phases; nothing here is raw-specific.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..readwrite import atomic_write_json

MANIFEST_FILENAME = "experiment_manifest.json"
MANIFEST_SCHEMA_VERSION = 2
STAGE_STATES = frozenset({"planned", "running", "complete", "failed"})
_ALLOWED_STAGE_TRANSITIONS = {
    None: frozenset({"planned", "complete"}),
    "planned": frozenset({"running", "failed"}),
    "running": frozenset({"complete", "failed"}),
    "complete": frozenset({"planned", "complete"}),
    "failed": frozenset({"planned", "failed"}),
}


def experiment_manifest_path(run_root: str | Path) -> Path:
    return Path(run_root) / MANIFEST_FILENAME


def config_hash(config: dict[str, Any]) -> str:
    """Stable, short hash of a resolved config dict (e.g. ``ExperimentConfig.to_dict()``)."""
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _save(path: Path, manifest: dict[str, Any]) -> None:
    manifest = dict(manifest)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    atomic_write_json(path, manifest)


def _artifact_path(run_root: Path, artifact: dict[str, Any]) -> Path | None:
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str):
        return None
    path = Path(raw_path)
    path_kind = artifact.get("path_kind")
    if path_kind == "relative":
        if artifact.get("anchor") != "run_root":
            return None
        return run_root / path
    if path_kind == "absolute" or path.is_absolute():
        return path
    # Version-1 records did not declare path semantics. Relative values in the
    # experiment manifest have historically been anchored at the run root.
    return run_root / path


def artifact_record(path: str | Path, run_root: str | Path, **metadata: Any) -> dict[str, Any]:
    """Describe an artifact with explicit, relocatable path semantics."""
    import os

    path = Path(path)
    run_root = Path(run_root)
    try:
        serialized_path = Path(os.path.relpath(path.resolve(), run_root.resolve())).as_posix()
        payload: dict[str, Any] = {
            "path": serialized_path,
            "path_kind": "relative",
            "anchor": "run_root",
        }
    except ValueError:
        # Windows raises for paths on different drives. Preserve access without
        # pretending that such a pointer is relocatable.
        payload = {
            "path": str(path.resolve()),
            "path_kind": "absolute",
            "anchor": None,
        }
    payload["kind"] = "directory" if path.is_dir() else "file"
    if path.is_file():
        payload["size_bytes"] = path.stat().st_size
    payload.update(metadata)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_stage_state(
    run_root: str | Path,
    stage: str,
    state: str,
    *,
    config_hash: str | None = None,
    input_artifact_ids: list[str] | None = None,
    artifacts: dict[str, dict[str, Any]] | None = None,
    expected_tasks: int | None = None,
    successful_tasks: int | None = None,
    schema_versions: dict[str, int] | None = None,
    timings: dict[str, float] | None = None,
    outcome: str | None = None,
    **extra: Any,
) -> Path:
    """Record one state transition for a versioned pipeline stage."""
    state = str(state).lower()
    if state not in STAGE_STATES:
        raise ValueError(f"stage state must be one of {sorted(STAGE_STATES)}")
    path = experiment_manifest_path(run_root)
    manifest = _load(path)
    stages = manifest.setdefault("stages", {})
    previous = stages.get(stage, {}) if isinstance(stages.get(stage), dict) else {}
    previous_state = previous.get("state")
    if previous_state is None and "completed_at" in previous:
        previous_state = "complete"
    allowed = _ALLOWED_STAGE_TRANSITIONS.get(previous_state, frozenset())
    if state not in allowed:
        raise ValueError(f"invalid stage transition for {stage!r}: {previous_state!r} -> {state!r}")
    entry = {} if state == "planned" else dict(previous)
    entry["state"] = state
    entry["updated_at"] = _now()
    timestamp_key = {
        "planned": "planned_at",
        "running": "started_at",
        "complete": "completed_at",
        "failed": "failed_at",
    }[state]
    entry[timestamp_key] = entry["updated_at"]
    optional_fields = {
        "config_hash": config_hash,
        "input_artifact_ids": input_artifact_ids,
        "artifacts": artifacts,
        "expected_tasks": expected_tasks,
        "successful_tasks": successful_tasks,
        "schema_versions": schema_versions,
        "timings": timings,
        "outcome": outcome,
    }
    entry.update({key: value for key, value in optional_fields.items() if value is not None})
    entry.update(extra)
    stages[stage] = entry
    _save(path, manifest)
    return path


def stage_is_complete(
    run_root: str | Path,
    stage: str,
    *,
    config_hash: str | None = None,
    required_artifacts: tuple[str, ...] = (),
) -> bool:
    """Return whether a compatible stage record and its required artifacts are valid."""
    run_root = Path(run_root)
    entry = read_experiment_manifest(run_root).get("stages", {}).get(stage)
    if not isinstance(entry, dict):
        return False
    state = entry.get("state")
    if state is None and "completed_at" in entry:
        state = "complete"
    if state != "complete":
        return False
    if config_hash is not None and entry.get("config_hash") != config_hash:
        return False
    expected_tasks = entry.get("expected_tasks")
    successful_tasks = entry.get("successful_tasks")
    if (
        expected_tasks is not None
        and successful_tasks is not None
        and int(expected_tasks) != int(successful_tasks)
    ):
        return False
    artifacts = entry.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return not required_artifacts
    for key in required_artifacts:
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict):
            return False
        path = _artifact_path(run_root, artifact)
        if path is None or not path.exists():
            return False
        expected_kind = artifact.get("kind")
        if expected_kind == "file" and not path.is_file():
            return False
        if expected_kind == "directory" and not path.is_dir():
            return False
        expected_size = artifact.get("size_bytes")
        if (
            expected_size is not None
            and path.is_file()
            and path.stat().st_size != int(expected_size)
        ):
            return False
        expected_sha256 = artifact.get("sha256")
        if expected_sha256 is not None and path.is_file() and _sha256(path) != expected_sha256:
            return False
    return True


@dataclass
class StageLifecycle(AbstractContextManager["StageLifecycle"]):
    """Context manager that guarantees a terminal stage state."""

    run_root: Path
    stage: str
    config_hash: str | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    _complete: bool = field(default=False, init=False)

    def __enter__(self) -> "StageLifecycle":
        record_stage_state(
            self.run_root,
            self.stage,
            "planned",
            config_hash=self.config_hash,
            input_artifact_ids=self.input_artifact_ids,
        )
        record_stage_state(self.run_root, self.stage, "running")
        return self

    def complete(self, **fields: Any) -> Path:
        """Publish the terminal complete state after all artifacts validate."""
        path = record_stage_state(self.run_root, self.stage, "complete", **fields)
        self._complete = True
        return path

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is not None:
            record_stage_state(
                self.run_root,
                self.stage,
                "failed",
                outcome=f"{type(exc_value).__name__}: {exc_value}",
            )
        elif not self._complete:
            record_stage_state(
                self.run_root,
                self.stage,
                "failed",
                outcome="stage exited without publishing completion",
            )
        return False


def update_experiment_manifest(run_root: str | Path, **fields: Any) -> Path:
    """Merge top-level fields into the manifest (``None`` values are skipped, so
    callers can pass everything they have without clobbering an already-recorded
    field with an absent one). Never touches the ``"stages"`` sub-dict -- see
    :func:`record_stage_completion` for that.
    """
    path = experiment_manifest_path(run_root)
    manifest = _load(path)
    manifest.update({key: value for key, value in fields.items() if value is not None})
    _save(path, manifest)
    return path


def record_stage_completion(
    run_root: str | Path,
    stage: str,
    *,
    config_hash: str | None = None,
    n_molecules: int | None = None,
    **extra: Any,
) -> Path:
    """Record (or overwrite, on a re-run) one stage's entry in the manifest's
    ``"stages"`` index -- a readable completion/provenance log, in addition to (not
    instead of) the existing structural signal of whether ``<stage>_outputs/`` exists.
    """
    if n_molecules is not None:
        extra["n_molecules"] = int(n_molecules)
    return record_stage_state(
        run_root,
        stage,
        "complete",
        config_hash=config_hash,
        **extra,
    )


def read_experiment_manifest(run_root: str | Path) -> dict[str, Any]:
    return _load(experiment_manifest_path(run_root))
