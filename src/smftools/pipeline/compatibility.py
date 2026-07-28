"""Deterministic compatibility keys and read-only semantic planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ..constants import SEMANTIC_GRAPH_DEFINITION_VERSION
from .analysis_registry import AnalysisRegistry
from .semantic_graph import (
    ArtifactRecord,
    ArtifactValidation,
    ChannelFingerprint,
    CompatibilityFingerprint,
    DependencyResultIdentity,
    NodeInputs,
    NodeResult,
    PlanDecision,
    PlanState,
    SemanticNodeSpec,
    SemanticPlan,
)


class MissingCompatibilityInput(ValueError):
    """Raised when a compatibility key cannot be constructed without guessing."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(item) for item in value), key=str)
    return value


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_inputs(
    spec: SemanticNodeSpec,
    inputs: NodeInputs,
) -> tuple[DependencyResultIdentity, ...]:
    by_id = {dependency.analysis_id: dependency for dependency in inputs.dependency_results}
    missing = [dependency for dependency in spec.dependencies if dependency not in by_id]
    if missing:
        raise MissingCompatibilityInput(
            f"node {spec.analysis_id!r} lacks dependency result identities for {missing}"
        )
    unexpected = sorted(set(by_id).difference(spec.dependencies))
    if unexpected:
        raise ValueError(
            f"node {spec.analysis_id!r} received undeclared dependency results: {unexpected}"
        )
    consumed_by_dependency: dict[str, dict[str, int]] = {}
    for channel in spec.consumed_channels:
        consumed_by_dependency.setdefault(channel.analysis_id, {})[channel.channel_id] = (
            channel.schema_version
        )
    normalized: list[DependencyResultIdentity] = []
    for dependency_id in spec.dependencies:
        dependency = by_id[dependency_id]
        channel_by_id = {channel.channel_id: channel for channel in dependency.channel_fingerprints}
        expected_channels = consumed_by_dependency.get(dependency_id, {})
        missing_channels = sorted(set(expected_channels).difference(channel_by_id))
        if missing_channels:
            raise MissingCompatibilityInput(
                f"node {spec.analysis_id!r} lacks consumed channel fingerprints from "
                f"{dependency_id!r}: {missing_channels}"
            )
        unexpected_channels = sorted(set(channel_by_id).difference(expected_channels))
        if unexpected_channels:
            raise ValueError(
                f"node {spec.analysis_id!r} received undeclared channel fingerprints from "
                f"{dependency_id!r}: {unexpected_channels}"
            )
        for channel_id, expected_schema in expected_channels.items():
            actual_schema = channel_by_id[channel_id].schema_version
            if actual_schema != expected_schema:
                raise ValueError(
                    f"node {spec.analysis_id!r} expected channel schema {expected_schema} for "
                    f"{dependency_id!r}/{channel_id!r}, received {actual_schema}"
                )
        normalized.append(dependency)
    return tuple(normalized)


def compatibility_fingerprint(
    spec: SemanticNodeSpec,
    inputs: NodeInputs,
) -> CompatibilityFingerprint:
    """Build a scientific compatibility identity without execution provenance."""
    missing_config = [key for key in spec.semantic_config_keys if key not in inputs.semantic_config]
    if missing_config:
        raise MissingCompatibilityInput(
            f"node {spec.analysis_id!r} lacks semantic config keys {missing_config}"
        )
    selected_config = {key: inputs.semantic_config[key] for key in spec.semantic_config_keys}
    semantic_config_hash = _stable_hash(selected_config)
    dependencies = _dependency_inputs(spec, inputs)
    payload = {
        "analysis_id": spec.analysis_id,
        "algorithm_version": spec.algorithm_version,
        "output_schema_version": spec.output_schema_version,
        "produced_channels": [channel.to_dict() for channel in spec.produced_channels],
        "semantic_config_hash": semantic_config_hash,
        "input_artifacts": [artifact.to_dict() for artifact in inputs.input_artifacts],
        "dependency_results": [dependency.to_dict() for dependency in dependencies],
        "task_scope": spec.task_scope,
        "logical_scope_identity": inputs.logical_scope_identity,
        "logical_task_plan_digest": inputs.logical_task_plan_digest,
    }
    return CompatibilityFingerprint(
        compatibility_key=_stable_hash(payload),
        semantic_config_hash=semantic_config_hash,
        input_artifacts=inputs.input_artifacts,
        dependency_results=dependencies,
        logical_scope_identity=inputs.logical_scope_identity,
        logical_task_plan_digest=inputs.logical_task_plan_digest,
    )


def node_result_from_inputs(
    spec: SemanticNodeSpec,
    inputs: NodeInputs,
    *,
    result_id: str,
    produced_channels: tuple[ChannelFingerprint, ...],
    artifacts: tuple[ArtifactRecord, ...] = (),
    state: str = "complete",
    reused_from_generation_id: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    execution_provenance: tuple[tuple[str, str], ...] = (),
) -> NodeResult:
    """Construct a node result from the exact semantic inputs it consumed."""
    expected_channels = {
        channel.channel_id: channel.schema_version for channel in spec.produced_channels
    }
    actual_channels = {channel.channel_id: channel.schema_version for channel in produced_channels}
    if actual_channels != expected_channels:
        raise ValueError(
            f"node {spec.analysis_id!r} produced channel schemas {actual_channels}, "
            f"expected {expected_channels}"
        )
    fingerprint = compatibility_fingerprint(spec, inputs)
    return NodeResult(
        analysis_id=spec.analysis_id,
        result_id=result_id,
        algorithm_version=spec.algorithm_version,
        output_schema_version=spec.output_schema_version,
        compatibility_key=fingerprint.compatibility_key,
        semantic_config_hash=fingerprint.semantic_config_hash,
        input_artifacts=fingerprint.input_artifacts,
        dependency_results=fingerprint.dependency_results,
        logical_scope_identity=fingerprint.logical_scope_identity,
        logical_task_plan_digest=fingerprint.logical_task_plan_digest,
        produced_channels=produced_channels,
        artifacts=artifacts,
        state=state,
        reused_from_generation_id=reused_from_generation_id,
        started_at=started_at,
        completed_at=completed_at,
        execution_provenance=execution_provenance,
    )


def _dependency_identity(
    consumer: SemanticNodeSpec,
    dependency: NodeResult,
) -> DependencyResultIdentity:
    expected = {
        channel.channel_id: channel.schema_version
        for channel in consumer.consumed_channels
        if channel.analysis_id == dependency.analysis_id
    }
    channels: list[ChannelFingerprint] = []
    for channel_id, schema_version in sorted(expected.items()):
        channel = dependency.channel(channel_id)
        if channel is None or channel.schema_version != schema_version:
            raise MissingCompatibilityInput(
                f"result {dependency.result_id!r} lacks required channel "
                f"{dependency.analysis_id!r}/{channel_id!r} schema {schema_version}"
            )
        channels.append(channel)
    return DependencyResultIdentity(
        analysis_id=dependency.analysis_id,
        result_id=dependency.result_id,
        channel_fingerprints=tuple(channels),
    )


class SemanticPlanner:
    """Classify compatible reuse and dependency-driven recomputation without writes."""

    def __init__(
        self,
        registry: AnalysisRegistry,
        *,
        graph_definition_version: int = SEMANTIC_GRAPH_DEFINITION_VERSION,
    ) -> None:
        if int(graph_definition_version) < 1:
            raise ValueError("graph_definition_version must be positive")
        self.registry = registry
        self.graph_definition_version = int(graph_definition_version)

    def plan(
        self,
        target: str,
        *,
        inputs_by_node: Mapping[str, NodeInputs],
        current_results: Mapping[str, NodeResult] | None = None,
        current_generation_id: str | None = None,
    ) -> SemanticPlan:
        """Return a deterministic plan and perform no artifact publication."""
        current_results = dict(current_results or {})
        order = self.registry.topological_order((target,))
        unexpected_inputs = sorted(set(inputs_by_node).difference(self.registry.nodes))
        if unexpected_inputs:
            raise ValueError(f"planning inputs contain unknown nodes: {unexpected_inputs}")
        unexpected_results = sorted(set(current_results).difference(self.registry.nodes))
        if unexpected_results:
            raise ValueError(f"current results contain unknown nodes: {unexpected_results}")
        decisions: dict[str, PlanDecision] = {}
        for analysis_id in order:
            spec = self.registry.node(analysis_id)
            inputs = inputs_by_node.get(analysis_id)
            current = current_results.get(analysis_id)
            decision = self._direct_decision(spec, inputs, current, current_results)
            dependency_decisions = [decisions[item] for item in spec.dependencies]
            blocked = [
                item.analysis_id
                for item in dependency_decisions
                if item.state is PlanState.BLOCKED_MISSING_INPUT
            ]
            if blocked:
                decision = replace(
                    decision,
                    state=PlanState.BLOCKED_MISSING_INPUT,
                    reason_code="dependency_blocked",
                    reason=f"required dependencies are blocked: {blocked}",
                    selected_result_id=None,
                    rejected_result_id=current.result_id if current is not None else None,
                    invalidated_by=tuple(blocked),
                )
            elif (
                decision.state is PlanState.BLOCKED_MISSING_INPUT
                and decision.reason_code == "compatibility_input_unavailable"
            ):
                pending = [
                    item.analysis_id
                    for item in dependency_decisions
                    if item.state is not PlanState.COMPATIBLE
                ]
                if pending:
                    decision = replace(
                        decision,
                        state=PlanState.DEPENDENT_RECOMPUTE,
                        reason_code="dependency_result_pending",
                        reason=f"dependency results will be replaced: {pending}",
                        invalidated_by=tuple(pending),
                    )
            elif decision.state is PlanState.COMPATIBLE:
                changed = [
                    item.analysis_id
                    for item in dependency_decisions
                    if item.state is not PlanState.COMPATIBLE
                ]
                if changed:
                    decision = replace(
                        decision,
                        state=PlanState.DEPENDENT_RECOMPUTE,
                        reason_code="dependency_recompute",
                        reason=f"dependencies require recomputation: {changed}",
                        selected_result_id=None,
                        rejected_result_id=current.result_id if current is not None else None,
                        invalidated_by=tuple(changed),
                    )
            decisions[analysis_id] = decision
        return SemanticPlan(
            requested_target=target,
            topological_order=order,
            decisions=tuple(decisions[analysis_id] for analysis_id in order),
            graph_definition_version=self.graph_definition_version,
            current_generation_id=current_generation_id,
        )

    def _direct_decision(
        self,
        spec: SemanticNodeSpec,
        inputs: NodeInputs | None,
        current: NodeResult | None,
        current_results: Mapping[str, NodeResult],
    ) -> PlanDecision:
        if inputs is None:
            return self._decision(
                spec,
                PlanState.BLOCKED_MISSING_INPUT,
                "missing_node_inputs",
                "planning inputs are unavailable",
                current=current,
            )
        if inputs.unavailable_inputs:
            return self._decision(
                spec,
                PlanState.BLOCKED_MISSING_INPUT,
                "required_input_unavailable",
                f"required scientific inputs are unavailable: {list(inputs.unavailable_inputs)}",
                current=current,
            )
        missing_config = [
            key for key in spec.semantic_config_keys if key not in inputs.semantic_config
        ]
        if missing_config:
            return self._decision(
                spec,
                PlanState.BLOCKED_MISSING_INPUT,
                "missing_semantic_config",
                f"required semantic config keys are unavailable: {missing_config}",
                current=current,
            )
        try:
            effective_inputs = self._with_current_dependencies(
                spec,
                inputs,
                current_results,
            )
            fingerprint = compatibility_fingerprint(spec, effective_inputs)
        except MissingCompatibilityInput as exc:
            if current is None:
                return self._decision(
                    spec,
                    PlanState.MISSING,
                    "missing_result",
                    "no current result exists; compatibility will be finalized after dependencies",
                )
            return self._decision(
                spec,
                PlanState.BLOCKED_MISSING_INPUT,
                "compatibility_input_unavailable",
                str(exc),
                current=current,
            )
        if current is None:
            return self._decision(
                spec,
                PlanState.MISSING,
                "missing_result",
                "no current result exists",
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.analysis_id != spec.analysis_id:
            return self._decision(
                spec,
                PlanState.INVALID_ARTIFACT,
                "result_analysis_mismatch",
                f"stored result belongs to {current.analysis_id!r}",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.state != "complete":
            return self._decision(
                spec,
                PlanState.INVALID_ARTIFACT,
                "result_not_complete",
                f"stored result state is {current.state!r}",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.algorithm_version != spec.algorithm_version:
            return self._decision(
                spec,
                PlanState.STALE_ALGORITHM,
                "algorithm_version_changed",
                f"algorithm version changed from {current.algorithm_version!r} "
                f"to {spec.algorithm_version!r}",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.output_schema_version != spec.output_schema_version:
            return self._decision(
                spec,
                PlanState.STALE_ALGORITHM,
                "output_schema_version_changed",
                f"output schema version changed from {current.output_schema_version} "
                f"to {spec.output_schema_version}",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.semantic_config_hash != fingerprint.semantic_config_hash:
            return self._decision(
                spec,
                PlanState.STALE_CONFIG,
                "semantic_config_changed",
                "declared semantic configuration changed",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        input_reason = self._input_change_reason(current, fingerprint)
        if input_reason is not None:
            reason_code, reason = input_reason
            return self._decision(
                spec,
                PlanState.STALE_INPUT,
                reason_code,
                reason,
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        if current.compatibility_key != fingerprint.compatibility_key:
            return self._decision(
                spec,
                PlanState.STALE_INPUT,
                "compatibility_key_changed",
                "semantic compatibility identity changed",
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        validation = self.registry.validator_for(spec)(current)
        if isinstance(validation, bool):
            validation = ArtifactValidation(valid=validation)
        if not isinstance(validation, ArtifactValidation):
            raise TypeError(
                f"validator {spec.validator_id!r} must return bool or ArtifactValidation"
            )
        if not validation.valid:
            return self._decision(
                spec,
                PlanState.INVALID_ARTIFACT,
                validation.reason_code,
                validation.reason,
                current=current,
                compatibility_key=fingerprint.compatibility_key,
            )
        return self._decision(
            spec,
            PlanState.COMPATIBLE,
            "compatible_result",
            "stored result and required artifacts are compatible",
            current=current,
            compatibility_key=fingerprint.compatibility_key,
            selected=True,
        )

    @staticmethod
    def _with_current_dependencies(
        spec: SemanticNodeSpec,
        inputs: NodeInputs,
        current_results: Mapping[str, NodeResult],
    ) -> NodeInputs:
        supplied = {dependency.analysis_id: dependency for dependency in inputs.dependency_results}
        for dependency_id in spec.dependencies:
            if dependency_id not in supplied and dependency_id in current_results:
                supplied[dependency_id] = _dependency_identity(
                    spec,
                    current_results[dependency_id],
                )
        return replace(
            inputs,
            dependency_results=tuple(supplied.values()),
        )

    @staticmethod
    def _input_change_reason(
        current: NodeResult,
        expected: CompatibilityFingerprint,
    ) -> tuple[str, str] | None:
        if current.input_artifacts != expected.input_artifacts:
            return "input_artifacts_changed", "ordered input artifact identities changed"
        if current.dependency_results != expected.dependency_results:
            return "dependency_results_changed", "consumed dependency results changed"
        if current.logical_scope_identity != expected.logical_scope_identity:
            return "logical_scope_changed", "logical scientific scope changed"
        if current.logical_task_plan_digest != expected.logical_task_plan_digest:
            return "logical_task_plan_changed", "logical task plan changed"
        return None

    @staticmethod
    def _decision(
        spec: SemanticNodeSpec,
        state: PlanState,
        reason_code: str,
        reason: str,
        *,
        current: NodeResult | None = None,
        compatibility_key: str | None = None,
        selected: bool = False,
    ) -> PlanDecision:
        return PlanDecision(
            analysis_id=spec.analysis_id,
            state=state,
            reason_code=reason_code,
            reason=reason,
            expected_outputs=spec.produced_channels,
            compatibility_key=compatibility_key,
            selected_result_id=current.result_id if selected and current is not None else None,
            rejected_result_id=(
                current.result_id if not selected and current is not None else None
            ),
        )
