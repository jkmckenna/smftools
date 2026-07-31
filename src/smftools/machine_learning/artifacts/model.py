"""Reusable model and resumable checkpoint artifact manifests."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._validation import (
    MLArtifactManifestError,
    canonical_json,
    digest,
    fail,
    integer,
    keys,
    mapping,
    optional_string,
    sequence,
    sha256,
    string,
    strings,
    timestamp,
    version,
)
from .common import (
    ArtifactReference,
    EnvironmentRecord,
    ResolvedDefinition,
    SerializationPolicy,
)

ML_CHECKPOINT_MANIFEST_VERSION = 1
ML_MODEL_MANIFEST_VERSION = 1
LINEAGE_KINDS = frozenset(
    {"from_scratch", "pretrained", "fine_tuned", "promoted", "continued_training"}
)
CHECKPOINT_KINDS = frozenset({"best", "last", "periodic", "manual"})


def _run_id(value: Any, path: str) -> str:
    result = string(value, path)
    try:
        return str(uuid.UUID(result))
    except ValueError:
        fail(path, "must be a UUID")


@dataclass(frozen=True)
class ModelLineage:
    """Explicit parent-child relationship for a reusable model artifact."""

    kind: str
    parent_model_ids: tuple[str, ...]
    parent_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        kind = string(self.kind, "lineage.kind").lower()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parent_model_ids", tuple(self.parent_model_ids))
        object.__setattr__(self, "parent_roles", tuple(self.parent_roles))
        if kind not in LINEAGE_KINDS:
            fail("lineage.kind", f"must be one of {sorted(LINEAGE_KINDS)}")
        if len(self.parent_model_ids) != len(set(self.parent_model_ids)):
            fail("lineage.parent_model_ids", "cannot contain duplicates")
        for index, model_id in enumerate(self.parent_model_ids):
            digest(model_id, f"lineage.parent_model_ids[{index}]")
        for index, role in enumerate(self.parent_roles):
            string(role, f"lineage.parent_roles[{index}]")
        if len(self.parent_model_ids) != len(self.parent_roles):
            fail("lineage", "parent model IDs and roles must have equal length")
        if kind == "from_scratch" and self.parent_model_ids:
            fail("lineage.parent_model_ids", "from-scratch models cannot have parents")
        if kind != "from_scratch" and not self.parent_model_ids:
            fail("lineage.parent_model_ids", f"{kind!r} models require a parent")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lineage record."""
        return {
            "kind": self.kind,
            "parent_model_ids": list(self.parent_model_ids),
            "parent_roles": list(self.parent_roles),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelLineage:
        """Validate and restore a lineage record."""
        value = mapping(raw, "lineage")
        fields = {"kind", "parent_model_ids", "parent_roles"}
        keys(value, path="lineage", fields=fields)
        return cls(
            kind=string(value["kind"], "lineage.kind"),
            parent_model_ids=strings(value["parent_model_ids"], "lineage.parent_model_ids"),
            parent_roles=strings(value["parent_roles"], "lineage.parent_roles"),
        )


def _model_identity_payload(
    *,
    backend: str,
    family: str,
    task_type: str,
    dataset_snapshot_id: str,
    split_id: str,
    input_schema_hash: str,
    label_schema_hash: str | None,
    architecture: ResolvedDefinition,
    lineage: ModelLineage,
    artifact: ArtifactReference,
    serialization: SerializationPolicy,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "family": family,
        "task_type": task_type,
        "dataset_snapshot_id": dataset_snapshot_id,
        "split_id": split_id,
        "input_schema_hash": input_schema_hash,
        "label_schema_hash": label_schema_hash,
        "architecture": architecture.to_dict(),
        "lineage": lineage.to_dict(),
        "artifact": artifact.to_dict(),
        "serialization": serialization.to_dict(),
    }


@dataclass(frozen=True)
class ModelManifest:
    """Immutable, locally understandable reusable model artifact."""

    schema_version: int
    model_id: str
    model_key: str
    backend: str
    family: str
    task_type: str
    originating_run_id: str
    workspace_id: str
    dataset_snapshot_id: str
    split_id: str
    input_schema_hash: str
    label_schema_hash: str | None
    architecture: ResolvedDefinition
    lineage: ModelLineage
    artifact: ArtifactReference
    serialization: SerializationPolicy
    environment: EnvironmentRecord
    created_at: str

    def __post_init__(self) -> None:
        _validate_model(self)

    @classmethod
    def create(
        cls,
        *,
        model_key: str,
        backend: str,
        family: str,
        task_type: str,
        originating_run_id: str,
        workspace_id: str,
        dataset_snapshot_id: str,
        split_id: str,
        input_schema_hash: str,
        label_schema_hash: str | None,
        architecture: ResolvedDefinition,
        lineage: ModelLineage,
        artifact: ArtifactReference,
        serialization: SerializationPolicy,
        environment: EnvironmentRecord,
        created_at: str,
    ) -> ModelManifest:
        """Create a content- and provenance-addressed model manifest."""
        payload = _model_identity_payload(
            backend=backend,
            family=family,
            task_type=task_type,
            dataset_snapshot_id=dataset_snapshot_id,
            split_id=split_id,
            input_schema_hash=input_schema_hash,
            label_schema_hash=label_schema_hash,
            architecture=architecture,
            lineage=lineage,
            artifact=artifact,
            serialization=serialization,
        )
        return cls(
            schema_version=ML_MODEL_MANIFEST_VERSION,
            model_id=sha256(payload),
            model_key=model_key,
            backend=backend,
            family=family,
            task_type=task_type,
            originating_run_id=originating_run_id,
            workspace_id=workspace_id,
            dataset_snapshot_id=dataset_snapshot_id,
            split_id=split_id,
            input_schema_hash=input_schema_hash,
            label_schema_hash=label_schema_hash,
            architecture=architecture,
            lineage=lineage,
            artifact=artifact,
            serialization=serialization,
            environment=environment,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable model manifest."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_key": self.model_key,
            "backend": self.backend,
            "family": self.family,
            "task_type": self.task_type,
            "originating_run_id": self.originating_run_id,
            "workspace_id": self.workspace_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
            "input_schema_hash": self.input_schema_hash,
            "label_schema_hash": self.label_schema_hash,
            "architecture": self.architecture.to_dict(),
            "lineage": self.lineage.to_dict(),
            "artifact": self.artifact.to_dict(),
            "serialization": self.serialization.to_dict(),
            "environment": self.environment.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelManifest:
        """Validate and restore a version-1 model manifest."""
        value = mapping(raw, "model")
        fields = {
            "schema_version",
            "model_id",
            "model_key",
            "backend",
            "family",
            "task_type",
            "originating_run_id",
            "workspace_id",
            "dataset_snapshot_id",
            "split_id",
            "input_schema_hash",
            "label_schema_hash",
            "architecture",
            "lineage",
            "artifact",
            "serialization",
            "environment",
            "created_at",
        }
        keys(value, path="model", fields=fields)
        return cls(
            schema_version=version(
                value["schema_version"],
                ML_MODEL_MANIFEST_VERSION,
                "model.schema_version",
            ),
            model_id=digest(value["model_id"], "model.model_id"),
            model_key=string(value["model_key"], "model.model_key"),
            backend=string(value["backend"], "model.backend"),
            family=string(value["family"], "model.family"),
            task_type=string(value["task_type"], "model.task_type"),
            originating_run_id=_run_id(value["originating_run_id"], "model.originating_run_id"),
            workspace_id=digest(value["workspace_id"], "model.workspace_id"),
            dataset_snapshot_id=digest(value["dataset_snapshot_id"], "model.dataset_snapshot_id"),
            split_id=digest(value["split_id"], "model.split_id"),
            input_schema_hash=digest(value["input_schema_hash"], "model.input_schema_hash"),
            label_schema_hash=(
                None
                if value["label_schema_hash"] is None
                else digest(value["label_schema_hash"], "model.label_schema_hash")
            ),
            architecture=ResolvedDefinition.from_dict(
                mapping(value["architecture"], "model.architecture")
            ),
            lineage=ModelLineage.from_dict(mapping(value["lineage"], "model.lineage")),
            artifact=ArtifactReference.from_dict(mapping(value["artifact"], "model.artifact")),
            serialization=SerializationPolicy.from_dict(
                mapping(value["serialization"], "model.serialization")
            ),
            environment=EnvironmentRecord.from_dict(
                mapping(value["environment"], "model.environment")
            ),
            created_at=timestamp(value["created_at"], "model.created_at"),
        )

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return canonical_json(self.to_dict())


def _validate_model(model: ModelManifest) -> None:
    version(model.schema_version, ML_MODEL_MANIFEST_VERSION, "model.schema_version")
    digest(model.model_id, "model.model_id")
    string(model.model_key, "model.model_key")
    string(model.backend, "model.backend")
    string(model.family, "model.family")
    string(model.task_type, "model.task_type")
    _run_id(model.originating_run_id, "model.originating_run_id")
    digest(model.workspace_id, "model.workspace_id")
    digest(model.dataset_snapshot_id, "model.dataset_snapshot_id")
    digest(model.split_id, "model.split_id")
    digest(model.input_schema_hash, "model.input_schema_hash")
    if model.label_schema_hash is not None:
        digest(model.label_schema_hash, "model.label_schema_hash")
    if model.artifact.role != "model":
        fail("model.artifact.role", "must be 'model'")
    if model.artifact.size_bytes == 0:
        fail("model.artifact.size_bytes", "model payload cannot be empty")
    timestamp(model.created_at, "model.created_at")
    expected = sha256(
        _model_identity_payload(
            backend=model.backend,
            family=model.family,
            task_type=model.task_type,
            dataset_snapshot_id=model.dataset_snapshot_id,
            split_id=model.split_id,
            input_schema_hash=model.input_schema_hash,
            label_schema_hash=model.label_schema_hash,
            architecture=model.architecture,
            lineage=model.lineage,
            artifact=model.artifact,
            serialization=model.serialization,
        )
    )
    if model.model_id != expected:
        fail("model.model_id", "does not match model content and provenance")


@dataclass(frozen=True)
class CheckpointManifest:
    """Identity of resumable backend state at one point in a run."""

    schema_version: int
    checkpoint_id: str
    run_id: str
    model_key: str
    backend: str
    kind: str
    epoch: int
    step: int
    input_schema_hash: str
    architecture_hash: str
    artifact: ArtifactReference
    created_at: str

    def __post_init__(self) -> None:
        _validate_checkpoint(self)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        model_key: str,
        backend: str,
        kind: str,
        epoch: int,
        step: int,
        input_schema_hash: str,
        architecture_hash: str,
        artifact: ArtifactReference,
        created_at: str,
    ) -> CheckpointManifest:
        """Create a checkpoint identity that includes the payload byte digest."""
        payload = {
            "run_id": run_id,
            "model_key": model_key,
            "backend": backend,
            "kind": kind,
            "epoch": epoch,
            "step": step,
            "input_schema_hash": input_schema_hash,
            "architecture_hash": architecture_hash,
            "artifact": artifact.to_dict(),
        }
        return cls(
            schema_version=ML_CHECKPOINT_MANIFEST_VERSION,
            checkpoint_id=sha256(payload),
            run_id=run_id,
            model_key=model_key,
            backend=backend,
            kind=kind,
            epoch=epoch,
            step=step,
            input_schema_hash=input_schema_hash,
            architecture_hash=architecture_hash,
            artifact=artifact,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable checkpoint manifest."""
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "model_key": self.model_key,
            "backend": self.backend,
            "kind": self.kind,
            "epoch": self.epoch,
            "step": self.step,
            "input_schema_hash": self.input_schema_hash,
            "architecture_hash": self.architecture_hash,
            "artifact": self.artifact.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CheckpointManifest:
        """Validate and restore a version-1 checkpoint manifest."""
        value = mapping(raw, "checkpoint")
        fields = {
            "schema_version",
            "checkpoint_id",
            "run_id",
            "model_key",
            "backend",
            "kind",
            "epoch",
            "step",
            "input_schema_hash",
            "architecture_hash",
            "artifact",
            "created_at",
        }
        keys(value, path="checkpoint", fields=fields)
        return cls(
            schema_version=version(
                value["schema_version"],
                ML_CHECKPOINT_MANIFEST_VERSION,
                "checkpoint.schema_version",
            ),
            checkpoint_id=digest(value["checkpoint_id"], "checkpoint.checkpoint_id"),
            run_id=_run_id(value["run_id"], "checkpoint.run_id"),
            model_key=string(value["model_key"], "checkpoint.model_key"),
            backend=string(value["backend"], "checkpoint.backend"),
            kind=string(value["kind"], "checkpoint.kind"),
            epoch=integer(value["epoch"], "checkpoint.epoch"),
            step=integer(value["step"], "checkpoint.step"),
            input_schema_hash=digest(value["input_schema_hash"], "checkpoint.input_schema_hash"),
            architecture_hash=digest(value["architecture_hash"], "checkpoint.architecture_hash"),
            artifact=ArtifactReference.from_dict(mapping(value["artifact"], "checkpoint.artifact")),
            created_at=timestamp(value["created_at"], "checkpoint.created_at"),
        )

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return canonical_json(self.to_dict())


def _validate_checkpoint(checkpoint: CheckpointManifest) -> None:
    version(
        checkpoint.schema_version,
        ML_CHECKPOINT_MANIFEST_VERSION,
        "checkpoint.schema_version",
    )
    digest(checkpoint.checkpoint_id, "checkpoint.checkpoint_id")
    _run_id(checkpoint.run_id, "checkpoint.run_id")
    string(checkpoint.model_key, "checkpoint.model_key")
    string(checkpoint.backend, "checkpoint.backend")
    if checkpoint.kind not in CHECKPOINT_KINDS:
        fail("checkpoint.kind", f"must be one of {sorted(CHECKPOINT_KINDS)}")
    integer(checkpoint.epoch, "checkpoint.epoch")
    integer(checkpoint.step, "checkpoint.step")
    digest(checkpoint.input_schema_hash, "checkpoint.input_schema_hash")
    digest(checkpoint.architecture_hash, "checkpoint.architecture_hash")
    if checkpoint.artifact.role != "checkpoint":
        fail("checkpoint.artifact.role", "must be 'checkpoint'")
    if checkpoint.artifact.size_bytes == 0:
        fail("checkpoint.artifact.size_bytes", "checkpoint payload cannot be empty")
    timestamp(checkpoint.created_at, "checkpoint.created_at")
    expected = sha256(
        {
            "run_id": checkpoint.run_id,
            "model_key": checkpoint.model_key,
            "backend": checkpoint.backend,
            "kind": checkpoint.kind,
            "epoch": checkpoint.epoch,
            "step": checkpoint.step,
            "input_schema_hash": checkpoint.input_schema_hash,
            "architecture_hash": checkpoint.architecture_hash,
            "artifact": checkpoint.artifact.to_dict(),
        }
    )
    if checkpoint.checkpoint_id != expected:
        fail("checkpoint.checkpoint_id", "does not match checkpoint payload and state")
