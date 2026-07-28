"""Explicit semantic-node registry and deterministic dependency validation."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

from .semantic_graph import (
    AnalysisScope,
    ArtifactValidation,
    NodeResult,
    SemanticNodeSpec,
)

NodeValidator = Callable[[NodeResult], ArtifactValidation | bool]
NodeExecutor = Callable[..., Any]


class RegistryError(ValueError):
    """Raised when a semantic graph definition is internally inconsistent."""


class AnalysisRegistry:
    """Validated, side-effect-free registry of semantic node contracts and callables."""

    def __init__(
        self,
        specs: Iterable[SemanticNodeSpec],
        *,
        validators: Mapping[str, NodeValidator],
        executors: Mapping[str, NodeExecutor] | None = None,
    ) -> None:
        nodes: dict[str, SemanticNodeSpec] = {}
        for spec in specs:
            if spec.analysis_id in nodes:
                raise RegistryError(f"duplicate semantic analysis_id {spec.analysis_id!r}")
            nodes[spec.analysis_id] = spec
        self._nodes = MappingProxyType(nodes)
        self._validators = MappingProxyType(dict(validators))
        self._executors = MappingProxyType(dict(executors or {}))
        self._validate()

    @property
    def nodes(self) -> Mapping[str, SemanticNodeSpec]:
        """Return the immutable node mapping."""
        return self._nodes

    def node(self, analysis_id: str) -> SemanticNodeSpec:
        """Return a registered node or raise an actionable error."""
        try:
            return self._nodes[analysis_id]
        except KeyError as exc:
            raise RegistryError(f"unknown semantic analysis_id {analysis_id!r}") from exc

    def validator_for(self, spec: SemanticNodeSpec) -> NodeValidator:
        """Return the registered validator for a node specification."""
        return self._validators[spec.validator_id]

    def executor_for(self, analysis_id: str) -> NodeExecutor | None:
        """Return an optional registered executor without invoking it."""
        self.node(analysis_id)
        return self._executors.get(analysis_id)

    def dependency_closure(self, targets: Iterable[str]) -> frozenset[str]:
        """Return targets and all transitive dependencies."""
        closure: set[str] = set()

        def visit(analysis_id: str) -> None:
            spec = self.node(analysis_id)
            if analysis_id in closure:
                return
            closure.add(analysis_id)
            for dependency in spec.dependencies:
                visit(dependency)

        for target in targets:
            visit(str(target))
        return frozenset(closure)

    def topological_order(self, targets: Iterable[str]) -> tuple[str, ...]:
        """Return a stable dependency-first order independent of registration order."""
        closure = self.dependency_closure(targets)
        indegree = {
            analysis_id: sum(
                dependency in closure for dependency in self._nodes[analysis_id].dependencies
            )
            for analysis_id in closure
        }
        dependents: dict[str, list[str]] = {analysis_id: [] for analysis_id in closure}
        for analysis_id in closure:
            for dependency in self._nodes[analysis_id].dependencies:
                if dependency in closure:
                    dependents[dependency].append(analysis_id)
        ready = [analysis_id for analysis_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            analysis_id = heapq.heappop(ready)
            ordered.append(analysis_id)
            for dependent in sorted(dependents[analysis_id]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, dependent)
        if len(ordered) != len(closure):
            raise RegistryError("semantic graph contains a dependency cycle")
        return tuple(ordered)

    def _validate(self) -> None:
        invalid_validators = sorted(
            validator_id
            for validator_id, validator in self._validators.items()
            if not callable(validator)
        )
        if invalid_validators:
            raise RegistryError(f"registered validators are not callable: {invalid_validators}")
        unknown_executors = sorted(set(self._executors).difference(self._nodes))
        if unknown_executors:
            raise RegistryError(f"executors registered for unknown nodes: {unknown_executors}")
        for analysis_id, spec in sorted(self._nodes.items()):
            unknown = sorted(set(spec.dependencies).difference(self._nodes))
            if unknown:
                raise RegistryError(
                    f"node {analysis_id!r} declares unknown dependencies: {unknown}"
                )
            if spec.validator_id not in self._validators:
                raise RegistryError(
                    f"node {analysis_id!r} declares unknown validator_id {spec.validator_id!r}"
                )
            if analysis_id in self._executors and not callable(self._executors[analysis_id]):
                raise RegistryError(f"executor for node {analysis_id!r} is not callable")
            for dependency_id in spec.dependencies:
                dependency = self._nodes[dependency_id]
                if (
                    spec.scope is not AnalysisScope.PROJECT_ANALYSIS
                    and dependency.scope is AnalysisScope.PROJECT_ANALYSIS
                ):
                    raise RegistryError(
                        f"illegal scope dependency: {analysis_id!r} ({spec.scope.value}) "
                        f"cannot depend on project node {dependency_id!r}"
                    )
            for consumed in spec.consumed_channels:
                if consumed.analysis_id not in spec.dependencies:
                    raise RegistryError(
                        f"node {analysis_id!r} consumes {consumed.analysis_id!r}/"
                        f"{consumed.channel_id!r} without declaring that node as a dependency"
                    )
                producer = self._nodes[consumed.analysis_id]
                produced = {
                    channel.channel_id: channel.schema_version
                    for channel in producer.produced_channels
                }
                if consumed.channel_id not in produced:
                    raise RegistryError(
                        f"node {analysis_id!r} consumes unknown channel "
                        f"{consumed.analysis_id!r}/{consumed.channel_id!r}"
                    )
                actual_schema = produced[consumed.channel_id]
                if actual_schema != consumed.schema_version:
                    raise RegistryError(
                        f"node {analysis_id!r} requires schema {consumed.schema_version} for "
                        f"{consumed.analysis_id!r}/{consumed.channel_id!r}, but the producer "
                        f"declares schema {actual_schema}"
                    )
        self._validate_cycles()

    def _validate_cycles(self) -> None:
        visited: set[str] = set()
        active: list[str] = []
        active_set: set[str] = set()

        def visit(analysis_id: str) -> None:
            if analysis_id in active_set:
                start = active.index(analysis_id)
                cycle = [*active[start:], analysis_id]
                raise RegistryError(
                    "semantic graph contains a dependency cycle: " + " -> ".join(cycle)
                )
            if analysis_id in visited:
                return
            active.append(analysis_id)
            active_set.add(analysis_id)
            for dependency in self._nodes[analysis_id].dependencies:
                visit(dependency)
            active.pop()
            active_set.remove(analysis_id)
            visited.add(analysis_id)

        for analysis_id in sorted(self._nodes):
            visit(analysis_id)
