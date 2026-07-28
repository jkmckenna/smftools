"""Typed records for engine-neutral semantic analysis graphs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..constants import SEMANTIC_NODE_RESULT_SCHEMA_VERSION, SEMANTIC_PLAN_SCHEMA_VERSION

_ANALYSIS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_RESULT_STATES = frozenset({"planned", "running", "complete", "failed"})


def _require_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value)
    if not normalized or not _ANALYSIS_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, '.', '_', or '-'"
        )
    return normalized


def _require_nonempty(value: str, *, field_name: str) -> str:
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=str))
    return value


class AnalysisScope(str, Enum):
    """Semantic graph scope for one analysis node."""

    EXPERIMENT_STAGE = "experiment_stage"
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    PROJECT_ANALYSIS = "project_analysis"


class PlanState(str, Enum):
    """Compatibility classification for a requested semantic node."""

    COMPATIBLE = "compatible"
    MISSING = "missing"
    STALE_CONFIG = "stale_config"
    STALE_ALGORITHM = "stale_algorithm"
    STALE_INPUT = "stale_input"
    INVALID_ARTIFACT = "invalid_artifact"
    DEPENDENT_RECOMPUTE = "dependent_recompute"
    BLOCKED_MISSING_INPUT = "blocked_missing_input"


@dataclass(frozen=True)
class ChannelSpec:
    """One versioned output channel produced by a semantic node."""

    channel_id: str
    schema_version: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            _require_identifier(self.channel_id, field_name="channel_id"),
        )
        if int(self.schema_version) < 1:
            raise ValueError("channel schema_version must be positive")
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, object]:
        return {"channel_id": self.channel_id, "schema_version": self.schema_version}


@dataclass(frozen=True)
class ChannelDependency:
    """One channel consumed from a declared upstream dependency."""

    analysis_id: str
    channel_id: str
    schema_version: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_id",
            _require_identifier(self.analysis_id, field_name="analysis_id"),
        )
        object.__setattr__(
            self,
            "channel_id",
            _require_identifier(self.channel_id, field_name="channel_id"),
        )
        if int(self.schema_version) < 1:
            raise ValueError("consumed channel schema_version must be positive")
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "channel_id": self.channel_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class SemanticNodeSpec:
    """Immutable scientific contract for one semantic analysis node."""

    analysis_id: str
    scope: AnalysisScope
    dependencies: tuple[str, ...] = ()
    consumed_channels: tuple[ChannelDependency, ...] = ()
    produced_channels: tuple[ChannelSpec, ...] = ()
    semantic_config_keys: tuple[str, ...] = ()
    algorithm_version: str = "1"
    output_schema_version: int = 1
    task_scope: str = "experiment"
    validator_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_id",
            _require_identifier(self.analysis_id, field_name="analysis_id"),
        )
        object.__setattr__(self, "scope", AnalysisScope(self.scope))
        dependencies = tuple(sorted(map(str, self.dependencies)))
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"node {self.analysis_id!r} declares duplicate dependencies")
        for dependency in dependencies:
            _require_identifier(dependency, field_name="dependency analysis_id")
        object.__setattr__(self, "dependencies", dependencies)
        consumed = tuple(
            sorted(
                self.consumed_channels,
                key=lambda item: (item.analysis_id, item.channel_id),
            )
        )
        consumed_keys = [(item.analysis_id, item.channel_id) for item in consumed]
        if len(set(consumed_keys)) != len(consumed_keys):
            raise ValueError(f"node {self.analysis_id!r} declares duplicate consumed channels")
        object.__setattr__(self, "consumed_channels", consumed)
        produced = tuple(sorted(self.produced_channels, key=lambda item: item.channel_id))
        produced_ids = [item.channel_id for item in produced]
        if len(set(produced_ids)) != len(produced_ids):
            raise ValueError(f"node {self.analysis_id!r} declares duplicate produced channels")
        object.__setattr__(self, "produced_channels", produced)
        config_keys = tuple(sorted(map(str, self.semantic_config_keys)))
        if any(not key for key in config_keys):
            raise ValueError("semantic_config_keys must not contain empty values")
        if len(set(config_keys)) != len(config_keys):
            raise ValueError(f"node {self.analysis_id!r} declares duplicate semantic config keys")
        object.__setattr__(self, "semantic_config_keys", config_keys)
        object.__setattr__(
            self,
            "algorithm_version",
            _require_nonempty(self.algorithm_version, field_name="algorithm_version"),
        )
        if int(self.output_schema_version) < 1:
            raise ValueError("output_schema_version must be positive")
        object.__setattr__(self, "output_schema_version", int(self.output_schema_version))
        object.__setattr__(
            self,
            "task_scope",
            _require_nonempty(self.task_scope, field_name="task_scope"),
        )
        object.__setattr__(
            self,
            "validator_id",
            _require_identifier(self.validator_id, field_name="validator_id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "scope": self.scope.value,
            "dependencies": list(self.dependencies),
            "consumed_channels": [item.to_dict() for item in self.consumed_channels],
            "produced_channels": [item.to_dict() for item in self.produced_channels],
            "semantic_config_keys": list(self.semantic_config_keys),
            "algorithm_version": self.algorithm_version,
            "output_schema_version": self.output_schema_version,
            "task_scope": self.task_scope,
            "validator_id": self.validator_id,
        }


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity and checksum for one scientific input artifact."""

    artifact_id: str
    checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _require_nonempty(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "checksum",
            _require_nonempty(self.checksum, field_name="artifact checksum"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "checksum": self.checksum}


@dataclass(frozen=True)
class ChannelFingerprint:
    """Identity of one versioned channel in a completed node result."""

    channel_id: str
    schema_version: int
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            _require_identifier(self.channel_id, field_name="channel_id"),
        )
        if int(self.schema_version) < 1:
            raise ValueError("channel fingerprint schema_version must be positive")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(
            self,
            "fingerprint",
            _require_nonempty(self.fingerprint, field_name="channel fingerprint"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class DependencyResultIdentity:
    """Upstream result ID and the exact channels consumed from it."""

    analysis_id: str
    result_id: str
    channel_fingerprints: tuple[ChannelFingerprint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_id",
            _require_identifier(self.analysis_id, field_name="analysis_id"),
        )
        object.__setattr__(
            self,
            "result_id",
            _require_nonempty(self.result_id, field_name="result_id"),
        )
        channels = tuple(sorted(self.channel_fingerprints, key=lambda item: item.channel_id))
        channel_ids = [item.channel_id for item in channels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError(
                f"dependency result {self.analysis_id!r} declares duplicate channel fingerprints"
            )
        object.__setattr__(self, "channel_fingerprints", channels)

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "result_id": self.result_id,
            "channel_fingerprints": [channel.to_dict() for channel in self.channel_fingerprints],
        }


@dataclass(frozen=True)
class NodeInputs:
    """Scientific inputs used to evaluate one node's compatibility."""

    semantic_config: Mapping[str, Any] = field(default_factory=dict)
    input_artifacts: tuple[ArtifactIdentity, ...] = ()
    dependency_results: tuple[DependencyResultIdentity, ...] = ()
    logical_scope_identity: str = ""
    logical_task_plan_digest: str = ""
    unavailable_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(key, str) for key in self.semantic_config):
            raise ValueError("semantic_config keys must be strings")
        object.__setattr__(self, "semantic_config", _freeze(dict(self.semantic_config)))
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        dependencies = tuple(sorted(self.dependency_results, key=lambda item: item.analysis_id))
        dependency_ids = [item.analysis_id for item in dependencies]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError("node inputs declare duplicate dependency result identities")
        object.__setattr__(self, "dependency_results", dependencies)
        object.__setattr__(
            self,
            "logical_scope_identity",
            _require_nonempty(
                self.logical_scope_identity,
                field_name="logical_scope_identity",
            ),
        )
        object.__setattr__(
            self,
            "logical_task_plan_digest",
            _require_nonempty(
                self.logical_task_plan_digest,
                field_name="logical_task_plan_digest",
            ),
        )
        object.__setattr__(
            self,
            "unavailable_inputs",
            tuple(sorted({str(value) for value in self.unavailable_inputs})),
        )


@dataclass(frozen=True)
class CompatibilityFingerprint:
    """Canonical semantic identity expected for one node execution."""

    compatibility_key: str
    semantic_config_hash: str
    input_artifacts: tuple[ArtifactIdentity, ...]
    dependency_results: tuple[DependencyResultIdentity, ...]
    logical_scope_identity: str
    logical_task_plan_digest: str


@dataclass(frozen=True)
class ArtifactRecord:
    """One immutable output artifact published by a node result."""

    artifact_id: str
    relative_path: str
    checksum: str
    schema_version: int
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _require_nonempty(self.artifact_id, field_name="artifact_id"),
        )
        relative_path = _require_nonempty(self.relative_path, field_name="relative_path")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact relative_path must remain within its generation root")
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(
            self,
            "checksum",
            _require_nonempty(self.checksum, field_name="artifact checksum"),
        )
        if int(self.schema_version) < 1:
            raise ValueError("artifact schema_version must be positive")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "kind", _require_nonempty(self.kind, field_name="kind"))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "checksum": self.checksum,
            "schema_version": self.schema_version,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class NodeResult:
    """Immutable compatibility and artifact record for one completed node."""

    analysis_id: str
    result_id: str
    algorithm_version: str
    output_schema_version: int
    compatibility_key: str
    semantic_config_hash: str
    input_artifacts: tuple[ArtifactIdentity, ...]
    dependency_results: tuple[DependencyResultIdentity, ...]
    logical_scope_identity: str
    logical_task_plan_digest: str
    produced_channels: tuple[ChannelFingerprint, ...]
    artifacts: tuple[ArtifactRecord, ...] = ()
    state: str = "complete"
    reused_from_generation_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    execution_provenance: tuple[tuple[str, str], ...] = ()
    schema_version: int = SEMANTIC_NODE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_id",
            _require_identifier(self.analysis_id, field_name="analysis_id"),
        )
        object.__setattr__(
            self,
            "result_id",
            _require_nonempty(self.result_id, field_name="result_id"),
        )
        object.__setattr__(
            self,
            "algorithm_version",
            _require_nonempty(self.algorithm_version, field_name="algorithm_version"),
        )
        object.__setattr__(
            self,
            "compatibility_key",
            _require_nonempty(self.compatibility_key, field_name="compatibility_key"),
        )
        object.__setattr__(
            self,
            "semantic_config_hash",
            _require_nonempty(self.semantic_config_hash, field_name="semantic_config_hash"),
        )
        object.__setattr__(
            self,
            "logical_scope_identity",
            _require_nonempty(
                self.logical_scope_identity,
                field_name="logical_scope_identity",
            ),
        )
        object.__setattr__(
            self,
            "logical_task_plan_digest",
            _require_nonempty(
                self.logical_task_plan_digest,
                field_name="logical_task_plan_digest",
            ),
        )
        state = str(self.state)
        if state not in _RESULT_STATES:
            raise ValueError(f"node result state must be one of {sorted(_RESULT_STATES)}")
        object.__setattr__(self, "state", state)
        if int(self.output_schema_version) < 1:
            raise ValueError("output_schema_version must be positive")
        object.__setattr__(self, "output_schema_version", int(self.output_schema_version))
        channels = tuple(sorted(self.produced_channels, key=lambda item: item.channel_id))
        channel_ids = [item.channel_id for item in channels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError(f"node result {self.analysis_id!r} has duplicate channel fingerprints")
        object.__setattr__(self, "produced_channels", channels)
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        object.__setattr__(
            self,
            "dependency_results",
            tuple(sorted(self.dependency_results, key=lambda item: item.analysis_id)),
        )
        dependency_ids = [item.analysis_id for item in self.dependency_results]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError(f"node result {self.analysis_id!r} has duplicate dependencies")
        artifacts = tuple(sorted(self.artifacts, key=lambda item: item.artifact_id))
        artifact_ids = [item.artifact_id for item in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError(f"node result {self.analysis_id!r} has duplicate artifacts")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "execution_provenance",
            tuple(sorted((str(key), str(value)) for key, value in self.execution_provenance)),
        )
        if int(self.schema_version) != SEMANTIC_NODE_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "node result schema_version is incompatible with this smftools version"
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def channel(self, channel_id: str) -> ChannelFingerprint | None:
        """Return a produced channel fingerprint by ID."""
        return next(
            (channel for channel in self.produced_channels if channel.channel_id == channel_id),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "state": self.state,
            "result_id": self.result_id,
            "algorithm_version": self.algorithm_version,
            "output_schema_version": self.output_schema_version,
            "compatibility_key": self.compatibility_key,
            "semantic_config_hash": self.semantic_config_hash,
            "input_artifacts": [item.to_dict() for item in self.input_artifacts],
            "dependency_results": [item.to_dict() for item in self.dependency_results],
            "logical_scope_identity": self.logical_scope_identity,
            "logical_task_plan_digest": self.logical_task_plan_digest,
            "produced_channels": [item.to_dict() for item in self.produced_channels],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "execution_provenance": dict(self.execution_provenance),
        }
        optional = {
            "reused_from_generation_id": self.reused_from_generation_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass(frozen=True)
class ArtifactValidation:
    """Read-only artifact validation outcome returned by a registered validator."""

    valid: bool
    reason_code: str = "artifact_validation_failed"
    reason: str = "published artifacts failed validation"


@dataclass(frozen=True)
class PlanDecision:
    """One explainable node classification in a semantic plan."""

    analysis_id: str
    state: PlanState
    reason_code: str
    reason: str
    expected_outputs: tuple[ChannelSpec, ...]
    compatibility_key: str | None = None
    selected_result_id: str | None = None
    rejected_result_id: str | None = None
    invalidated_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "analysis_id": self.analysis_id,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "expected_outputs": [item.to_dict() for item in self.expected_outputs],
            "invalidated_by": list(self.invalidated_by),
        }
        optional = {
            "compatibility_key": self.compatibility_key,
            "selected_result_id": self.selected_result_id,
            "rejected_result_id": self.rejected_result_id,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass(frozen=True)
class SemanticPlan:
    """Deterministic read-only plan for one requested target."""

    requested_target: str
    topological_order: tuple[str, ...]
    decisions: tuple[PlanDecision, ...]
    graph_definition_version: int
    current_generation_id: str | None = None
    schema_version: int = SEMANTIC_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "graph_definition_version": self.graph_definition_version,
            "requested_target": self.requested_target,
            "topological_order": list(self.topological_order),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
        if self.current_generation_id is not None:
            payload["current_generation_id"] = self.current_generation_id
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return stable JSON for machine-readable plan consumers."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), indent=indent)
