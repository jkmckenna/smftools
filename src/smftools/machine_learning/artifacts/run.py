"""Immutable run lifecycle manifests independent of execution backend."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from ..plan import SUPPORTED_JOB_ACTIONS
from ._validation import (
    MLArtifactManifestError,
    canonical_json,
    digest,
    fail,
    integer,
    keys,
    mapping,
    optional_string,
    optional_timestamp,
    sequence,
    string,
    strings,
    timestamp,
    version,
)
from .common import (
    ArtifactReference,
    EnvironmentRecord,
    FailureRecord,
    unique_artifact_roles,
)

ML_RUN_MANIFEST_VERSION = 1
RUN_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled"})
_RUN_TRANSITIONS = {
    "planned": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def new_run_id() -> str:
    """Return a unique execution-attempt identity."""
    return str(uuid.uuid4())


def _run_id(value: Any, path: str) -> str:
    result = string(value, path)
    try:
        return str(uuid.UUID(result))
    except ValueError as exc:
        fail(path, "must be a UUID")
        raise AssertionError from exc


def _seed_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, int]:
    result: dict[str, int] = {}
    for name, seed in value.items():
        result[string(name, f"{path}.key")] = integer(seed, f"{path}.{name}")
    return MappingProxyType(dict(sorted(result.items())))


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class RunManifest:
    """Tracker-neutral provenance and lifecycle state for one execution attempt."""

    schema_version: int
    run_id: str
    workspace_id: str
    action: str
    job_name: str
    state: str
    plan_hash: str
    resolved_plan: ArtifactReference
    resolved_config: ArtifactReference
    dataset_snapshot_id: str | None
    split_id: str | None
    model_keys: tuple[str, ...]
    source_model_ids: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    environment: EnvironmentRecord
    seeds: Mapping[str, int]
    device: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    failure: FailureRecord | None
    artifacts: tuple[ArtifactReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _run_id(self.run_id, "run.run_id"))
        object.__setattr__(self, "model_keys", tuple(sorted(self.model_keys)))
        object.__setattr__(self, "source_model_ids", tuple(sorted(self.source_model_ids)))
        object.__setattr__(self, "source_run_ids", tuple(sorted(self.source_run_ids)))
        object.__setattr__(self, "seeds", _seed_mapping(self.seeds, "run.seeds"))
        object.__setattr__(
            self,
            "artifacts",
            unique_artifact_roles(tuple(self.artifacts), "run.artifacts"),
        )
        _validate_run(self)

    @classmethod
    def create(
        cls,
        *,
        workspace_id: str,
        action: str,
        job_name: str,
        plan_hash: str,
        resolved_plan: ArtifactReference,
        resolved_config: ArtifactReference,
        environment: EnvironmentRecord,
        seeds: Mapping[str, int],
        device: str,
        created_at: str,
        dataset_snapshot_id: str | None = None,
        split_id: str | None = None,
        model_keys: tuple[str, ...] = (),
        source_model_ids: tuple[str, ...] = (),
        source_run_ids: tuple[str, ...] = (),
        run_id: str | None = None,
    ) -> RunManifest:
        """Create a distinct planned run for one resolved attempt."""
        return cls(
            schema_version=ML_RUN_MANIFEST_VERSION,
            run_id=run_id or new_run_id(),
            workspace_id=workspace_id,
            action=action,
            job_name=job_name,
            state="planned",
            plan_hash=plan_hash,
            resolved_plan=resolved_plan,
            resolved_config=resolved_config,
            dataset_snapshot_id=dataset_snapshot_id,
            split_id=split_id,
            model_keys=model_keys,
            source_model_ids=source_model_ids,
            source_run_ids=source_run_ids,
            environment=environment,
            seeds=seeds,
            device=device,
            created_at=created_at,
            started_at=None,
            finished_at=None,
            failure=None,
            artifacts=(),
        )

    def transition(
        self,
        state: str,
        *,
        at: str,
        failure: FailureRecord | None = None,
        artifacts: tuple[ArtifactReference, ...] | None = None,
    ) -> RunManifest:
        """Return the next valid lifecycle record without mutating this one."""
        state = string(state, "run.state").lower()
        if state not in _RUN_TRANSITIONS[self.state]:
            raise MLArtifactManifestError(
                f"run.state: cannot transition from {self.state!r} to {state!r}"
            )
        event_at = timestamp(at, "run.transition_at")
        return replace(
            self,
            state=state,
            started_at=event_at if state == "running" else self.started_at,
            finished_at=event_at if state in TERMINAL_RUN_STATES else None,
            failure=failure,
            artifacts=self.artifacts if artifacts is None else artifacts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable run manifest."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "action": self.action,
            "job_name": self.job_name,
            "state": self.state,
            "plan_hash": self.plan_hash,
            "resolved_plan": self.resolved_plan.to_dict(),
            "resolved_config": self.resolved_config.to_dict(),
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
            "model_keys": list(self.model_keys),
            "source_model_ids": list(self.source_model_ids),
            "source_run_ids": list(self.source_run_ids),
            "environment": self.environment.to_dict(),
            "seeds": dict(self.seeds),
            "device": self.device,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunManifest:
        """Validate and restore a version-1 run manifest."""
        value = mapping(raw, "run")
        fields = {
            "schema_version",
            "run_id",
            "workspace_id",
            "action",
            "job_name",
            "state",
            "plan_hash",
            "resolved_plan",
            "resolved_config",
            "dataset_snapshot_id",
            "split_id",
            "model_keys",
            "source_model_ids",
            "source_run_ids",
            "environment",
            "seeds",
            "device",
            "created_at",
            "started_at",
            "finished_at",
            "failure",
            "artifacts",
        }
        keys(value, path="run", fields=fields)
        failure_raw = value["failure"]
        return cls(
            schema_version=version(
                value["schema_version"], ML_RUN_MANIFEST_VERSION, "run.schema_version"
            ),
            run_id=_run_id(value["run_id"], "run.run_id"),
            workspace_id=digest(value["workspace_id"], "run.workspace_id"),
            action=string(value["action"], "run.action"),
            job_name=string(value["job_name"], "run.job_name"),
            state=string(value["state"], "run.state"),
            plan_hash=digest(value["plan_hash"], "run.plan_hash"),
            resolved_plan=ArtifactReference.from_dict(
                mapping(value["resolved_plan"], "run.resolved_plan")
            ),
            resolved_config=ArtifactReference.from_dict(
                mapping(value["resolved_config"], "run.resolved_config")
            ),
            dataset_snapshot_id=optional_string(
                value["dataset_snapshot_id"], "run.dataset_snapshot_id"
            ),
            split_id=optional_string(value["split_id"], "run.split_id"),
            model_keys=strings(value["model_keys"], "run.model_keys"),
            source_model_ids=strings(value["source_model_ids"], "run.source_model_ids"),
            source_run_ids=strings(value["source_run_ids"], "run.source_run_ids"),
            environment=EnvironmentRecord.from_dict(
                mapping(value["environment"], "run.environment")
            ),
            seeds=_seed_mapping(mapping(value["seeds"], "run.seeds"), "run.seeds"),
            device=string(value["device"], "run.device"),
            created_at=timestamp(value["created_at"], "run.created_at"),
            started_at=optional_timestamp(value["started_at"], "run.started_at"),
            finished_at=optional_timestamp(value["finished_at"], "run.finished_at"),
            failure=(
                None
                if failure_raw is None
                else FailureRecord.from_dict(mapping(failure_raw, "run.failure"))
            ),
            artifacts=tuple(
                ArtifactReference.from_dict(item)
                for item in sequence(value["artifacts"], "run.artifacts")
            ),
        )

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return canonical_json(self.to_dict())


def _validate_run(run: RunManifest) -> None:
    version(run.schema_version, ML_RUN_MANIFEST_VERSION, "run.schema_version")
    digest(run.workspace_id, "run.workspace_id")
    digest(run.plan_hash, "run.plan_hash")
    if run.action not in SUPPORTED_JOB_ACTIONS:
        fail("run.action", f"must be one of {sorted(SUPPORTED_JOB_ACTIONS)}")
    string(run.job_name, "run.job_name")
    if run.state not in RUN_STATES:
        fail("run.state", f"must be one of {sorted(RUN_STATES)}")
    if run.resolved_plan.role != "resolved_plan":
        fail("run.resolved_plan.role", "must be 'resolved_plan'")
    if run.resolved_config.role != "resolved_config":
        fail("run.resolved_config.role", "must be 'resolved_config'")
    dataset_required = run.action in {"train", "apply", "evaluate", "explain"}
    if dataset_required and run.dataset_snapshot_id is None:
        fail("run.dataset_snapshot_id", f"is required for {run.action!r}")
    if run.dataset_snapshot_id is not None:
        digest(run.dataset_snapshot_id, "run.dataset_snapshot_id")
    if run.split_id is not None:
        digest(run.split_id, "run.split_id")
    if run.action == "train":
        if run.split_id is None:
            fail("run.split_id", "is required for training")
        if not run.model_keys:
            fail("run.model_keys", "must identify at least one requested model")
    if run.action in {"apply", "evaluate", "explain"} and not run.source_model_ids:
        fail("run.source_model_ids", f"must identify a model for {run.action!r}")
    if run.action == "plot" and not run.source_run_ids:
        fail("run.source_run_ids", "must identify at least one source run for plotting")
    for index, model_id in enumerate(run.source_model_ids):
        digest(model_id, f"run.source_model_ids[{index}]")
    for index, source_run_id in enumerate(run.source_run_ids):
        _run_id(source_run_id, f"run.source_run_ids[{index}]")
    string(run.device, "run.device")
    timestamp(run.created_at, "run.created_at")
    optional_timestamp(run.started_at, "run.started_at")
    optional_timestamp(run.finished_at, "run.finished_at")
    if run.state == "planned":
        if run.started_at is not None or run.finished_at is not None:
            fail("run", "planned runs cannot have start or finish timestamps")
    elif run.state == "running":
        if run.started_at is None or run.finished_at is not None:
            fail("run", "running runs require started_at and cannot have finished_at")
    elif run.state in TERMINAL_RUN_STATES:
        if run.finished_at is None:
            fail("run.finished_at", "is required for a terminal run")
        if run.state == "completed" and run.started_at is None:
            fail("run.started_at", "is required for a completed run")
    created_at = _datetime(run.created_at)
    if run.started_at is not None and _datetime(run.started_at) < created_at:
        fail("run.started_at", "cannot precede created_at")
    if run.finished_at is not None:
        lower_bound = _datetime(run.started_at) if run.started_at is not None else created_at
        if _datetime(run.finished_at) < lower_bound:
            fail("run.finished_at", "cannot precede the run start")
    if run.state == "failed" and run.failure is None:
        fail("run.failure", "is required for a failed run")
    if run.state not in {"failed", "cancelled"} and run.failure is not None:
        fail("run.failure", "is only valid for a failed or cancelled run")
