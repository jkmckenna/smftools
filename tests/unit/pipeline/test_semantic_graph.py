from __future__ import annotations

from dataclasses import replace

import pytest

from smftools.pipeline import (
    AnalysisRegistry,
    AnalysisScope,
    ArtifactIdentity,
    ArtifactValidation,
    ChannelDependency,
    ChannelFingerprint,
    ChannelSpec,
    DependencyResultIdentity,
    NodeInputs,
    PlanState,
    RegistryError,
    SemanticNodeSpec,
    SemanticPlanner,
    node_result_from_inputs,
)

pytestmark = pytest.mark.unit


def _valid(_result):
    return ArtifactValidation(valid=True)


def _specs() -> tuple[SemanticNodeSpec, ...]:
    raw = SemanticNodeSpec(
        analysis_id="experiment.raw.store",
        scope=AnalysisScope.EXPERIMENT_STAGE,
        produced_channels=(ChannelSpec("raw.records", 1),),
        semantic_config_keys=("raw_mode",),
        task_scope="experiment",
        validator_id="valid",
    )
    metrics = SemanticNodeSpec(
        analysis_id="preprocess.metrics",
        scope=AnalysisScope.EXPERIMENT_ANALYSIS,
        dependencies=(raw.analysis_id,),
        consumed_channels=(ChannelDependency(raw.analysis_id, "raw.records", 1),),
        produced_channels=(ChannelSpec("preprocess.metrics", 1),),
        semantic_config_keys=("minimum_quality",),
        task_scope="experiment_reference",
        validator_id="valid",
    )
    complete = SemanticNodeSpec(
        analysis_id="experiment.preprocess.complete",
        scope=AnalysisScope.EXPERIMENT_STAGE,
        dependencies=(metrics.analysis_id,),
        consumed_channels=(ChannelDependency(metrics.analysis_id, "preprocess.metrics", 1),),
        produced_channels=(ChannelSpec("preprocess.complete", 1),),
        task_scope="experiment",
        validator_id="valid",
    )
    return raw, metrics, complete


def _registry(
    specs: tuple[SemanticNodeSpec, ...] | None = None,
    *,
    validators=None,
    executors=None,
) -> AnalysisRegistry:
    return AnalysisRegistry(
        specs or _specs(),
        validators=validators or {"valid": _valid},
        executors=executors,
    )


def _channel(spec: SemanticNodeSpec, fingerprint: str) -> tuple[ChannelFingerprint, ...]:
    return tuple(
        ChannelFingerprint(channel.channel_id, channel.schema_version, fingerprint)
        for channel in spec.produced_channels
    )


def _dependency(
    result,
    consumer: SemanticNodeSpec,
) -> DependencyResultIdentity:
    expected = {
        channel.channel_id
        for channel in consumer.consumed_channels
        if channel.analysis_id == result.analysis_id
    }
    return DependencyResultIdentity(
        analysis_id=result.analysis_id,
        result_id=result.result_id,
        channel_fingerprints=tuple(
            channel for channel in result.produced_channels if channel.channel_id in expected
        ),
    )


def _complete_fixture(
    specs: tuple[SemanticNodeSpec, ...] | None = None,
):
    raw, metrics, complete = specs or _specs()
    raw_inputs = NodeInputs(
        semantic_config={"raw_mode": "aligned_bam"},
        input_artifacts=(ArtifactIdentity("source-bam", "sha256:raw"),),
        logical_scope_identity="experiment:exp-1",
        logical_task_plan_digest="logical:raw-v1",
    )
    raw_result = node_result_from_inputs(
        raw,
        raw_inputs,
        result_id="raw-result-1",
        produced_channels=_channel(raw, "raw-channel-1"),
    )
    metrics_inputs = NodeInputs(
        semantic_config={"minimum_quality": 0.8},
        dependency_results=(_dependency(raw_result, metrics),),
        logical_scope_identity="experiment:exp-1/reference:ref-1",
        logical_task_plan_digest="logical:metrics-v1",
    )
    metrics_result = node_result_from_inputs(
        metrics,
        metrics_inputs,
        result_id="metrics-result-1",
        produced_channels=_channel(metrics, "metrics-channel-1"),
    )
    complete_inputs = NodeInputs(
        semantic_config={},
        dependency_results=(_dependency(metrics_result, complete),),
        logical_scope_identity="experiment:exp-1",
        logical_task_plan_digest="logical:preprocess-complete-v1",
    )
    complete_result = node_result_from_inputs(
        complete,
        complete_inputs,
        result_id="preprocess-result-1",
        produced_channels=_channel(complete, "preprocess-channel-1"),
    )
    return (
        {
            raw.analysis_id: raw_inputs,
            metrics.analysis_id: metrics_inputs,
            complete.analysis_id: complete_inputs,
        },
        {
            raw.analysis_id: raw_result,
            metrics.analysis_id: metrics_result,
            complete.analysis_id: complete_result,
        },
    )


def _states(plan) -> dict[str, PlanState]:
    return {decision.analysis_id: decision.state for decision in plan.decisions}


def test_registration_order_does_not_change_topological_order():
    raw, metrics, complete = _specs()
    forward = _registry((raw, metrics, complete))
    reverse = _registry((complete, metrics, raw))

    expected = (raw.analysis_id, metrics.analysis_id, complete.analysis_id)
    assert forward.topological_order((complete.analysis_id,)) == expected
    assert reverse.topological_order((complete.analysis_id,)) == expected


def test_topological_order_uses_node_ids_to_break_independent_branch_ties():
    raw, metrics, complete = _specs()
    diagnostics = SemanticNodeSpec(
        analysis_id="preprocess.diagnostics",
        scope=AnalysisScope.EXPERIMENT_ANALYSIS,
        dependencies=(raw.analysis_id,),
        produced_channels=(ChannelSpec("preprocess.diagnostics", 1),),
        validator_id="valid",
    )
    joined = replace(
        complete,
        dependencies=(metrics.analysis_id, diagnostics.analysis_id),
        consumed_channels=(),
    )

    order = _registry((joined, metrics, raw, diagnostics)).topological_order((joined.analysis_id,))

    assert order == (
        raw.analysis_id,
        diagnostics.analysis_id,
        metrics.analysis_id,
        joined.analysis_id,
    )


def test_registry_rejects_duplicate_unknown_cycle_and_illegal_scope_dependencies():
    raw, metrics, complete = _specs()
    with pytest.raises(RegistryError, match="duplicate semantic analysis_id"):
        _registry((raw, raw))

    unknown = replace(metrics, dependencies=("missing.node",), consumed_channels=())
    with pytest.raises(RegistryError, match="unknown dependencies.*missing.node"):
        _registry((raw, unknown))

    cyclic_raw = replace(raw, dependencies=(complete.analysis_id,))
    with pytest.raises(RegistryError, match="dependency cycle.*experiment.raw.store"):
        _registry((cyclic_raw, metrics, complete))

    project = SemanticNodeSpec(
        analysis_id="project.embedding",
        scope=AnalysisScope.PROJECT_ANALYSIS,
        produced_channels=(ChannelSpec("project.embedding", 1),),
        validator_id="valid",
    )
    illegal = replace(raw, dependencies=(project.analysis_id,))
    with pytest.raises(RegistryError, match="illegal scope dependency"):
        _registry((project, illegal))


def test_registry_validates_consumed_channels_and_registered_validators():
    raw, metrics, _complete = _specs()
    missing_channel = replace(
        metrics,
        consumed_channels=(ChannelDependency(raw.analysis_id, "raw.missing", 1),),
    )
    with pytest.raises(RegistryError, match="consumes unknown channel"):
        _registry((raw, missing_channel))

    with pytest.raises(RegistryError, match="unknown validator_id"):
        AnalysisRegistry((raw,), validators={})


def test_project_nodes_may_depend_on_experiment_nodes():
    raw = _specs()[0]
    project = SemanticNodeSpec(
        analysis_id="project.materialization",
        scope=AnalysisScope.PROJECT_ANALYSIS,
        dependencies=(raw.analysis_id,),
        consumed_channels=(ChannelDependency(raw.analysis_id, "raw.records", 1),),
        produced_channels=(ChannelSpec("project.materialized", 1),),
        validator_id="valid",
    )

    registry = _registry((project, raw))

    assert registry.topological_order((project.analysis_id,)) == (
        raw.analysis_id,
        project.analysis_id,
    )


def test_unrelated_config_and_execution_provenance_do_not_invalidate_results():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    metrics = specs[1]
    inputs[metrics.analysis_id] = replace(
        inputs[metrics.analysis_id],
        semantic_config={
            "minimum_quality": 0.8,
            "worker_count": 64,
            "task_order": ["last", "first"],
            "machine": "different-host",
        },
    )
    results[metrics.analysis_id] = replace(
        results[metrics.analysis_id],
        execution_provenance=(
            ("worker_count", "1"),
            ("machine", "original-host"),
            ("elapsed_seconds", "9.1"),
        ),
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    assert set(_states(plan).values()) == {PlanState.COMPATIBLE}


def test_semantic_config_change_invalidates_node_and_declared_dependent():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    metrics = specs[1]
    inputs[metrics.analysis_id] = replace(
        inputs[metrics.analysis_id],
        semantic_config={"minimum_quality": 0.9},
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    states = _states(plan)
    assert states[specs[0].analysis_id] is PlanState.COMPATIBLE
    assert states[metrics.analysis_id] is PlanState.STALE_CONFIG
    assert states[specs[-1].analysis_id] is PlanState.DEPENDENT_RECOMPUTE


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("algorithm_version", "2", "algorithm_version_changed"),
        ("output_schema_version", 2, "output_schema_version_changed"),
    ],
)
def test_algorithm_and_output_schema_changes_invalidate_correct_dependents(
    field,
    value,
    reason_code,
):
    old_specs = _specs()
    inputs, results = _complete_fixture(old_specs)
    new_metrics = replace(old_specs[1], **{field: value})
    new_specs = (old_specs[0], new_metrics, old_specs[2])

    plan = SemanticPlanner(_registry(new_specs)).plan(
        new_specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    decision_by_id = {decision.analysis_id: decision for decision in plan.decisions}
    assert decision_by_id[new_metrics.analysis_id].state is PlanState.STALE_ALGORITHM
    assert decision_by_id[new_metrics.analysis_id].reason_code == reason_code
    assert decision_by_id[new_specs[-1].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE


def test_input_artifact_change_invalidates_node_and_transitive_dependents():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    raw = specs[0]
    inputs[raw.analysis_id] = replace(
        inputs[raw.analysis_id],
        input_artifacts=(ArtifactIdentity("source-bam", "sha256:changed"),),
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    decision_by_id = {decision.analysis_id: decision for decision in plan.decisions}
    assert decision_by_id[raw.analysis_id].state is PlanState.STALE_INPUT
    assert decision_by_id[raw.analysis_id].reason_code == "input_artifacts_changed"
    assert decision_by_id[specs[1].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE
    assert decision_by_id[specs[2].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE


def test_changed_dependency_result_invalidates_direct_consumer():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    metrics = specs[1]
    changed_dependency = DependencyResultIdentity(
        analysis_id=specs[0].analysis_id,
        result_id="raw-result-2",
        channel_fingerprints=(ChannelFingerprint("raw.records", 1, "raw-channel-2"),),
    )
    inputs[metrics.analysis_id] = replace(
        inputs[metrics.analysis_id],
        dependency_results=(changed_dependency,),
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    decision_by_id = {decision.analysis_id: decision for decision in plan.decisions}
    assert decision_by_id[metrics.analysis_id].state is PlanState.STALE_INPUT
    assert decision_by_id[metrics.analysis_id].reason_code == "dependency_results_changed"
    assert decision_by_id[specs[-1].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE


def test_invalid_artifact_and_missing_inputs_have_distinct_states():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    invalid_registry = _registry(
        specs,
        validators={
            "valid": lambda _result: ArtifactValidation(
                valid=False,
                reason_code="checksum_mismatch",
                reason="manifest checksum does not match",
            )
        },
    )
    invalid_plan = SemanticPlanner(invalid_registry).plan(
        specs[0].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )
    assert invalid_plan.decisions[0].state is PlanState.INVALID_ARTIFACT
    assert invalid_plan.decisions[0].reason_code == "checksum_mismatch"

    inputs[specs[0].analysis_id] = replace(
        inputs[specs[0].analysis_id],
        unavailable_inputs=("raw.sequence",),
    )
    blocked_plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )
    assert set(_states(blocked_plan).values()) == {PlanState.BLOCKED_MISSING_INPUT}


def test_missing_declared_semantic_config_is_blocked():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    inputs[specs[0].analysis_id] = replace(
        inputs[specs[0].analysis_id],
        semantic_config={},
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[0].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    assert plan.decisions[0].state is PlanState.BLOCKED_MISSING_INPUT
    assert plan.decisions[0].reason_code == "missing_semantic_config"


def test_fresh_plan_classifies_missing_results_without_requiring_future_result_ids():
    specs = _specs()
    inputs, _results = _complete_fixture(specs)
    for spec in specs[1:]:
        inputs[spec.analysis_id] = replace(
            inputs[spec.analysis_id],
            dependency_results=(),
        )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
    )

    assert set(_states(plan).values()) == {PlanState.MISSING}


def test_existing_downstream_result_recomputes_when_upstream_result_is_missing():
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    results.pop(specs[0].analysis_id)
    inputs[specs[1].analysis_id] = replace(
        inputs[specs[1].analysis_id],
        dependency_results=(),
    )

    plan = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )

    decisions = {decision.analysis_id: decision for decision in plan.decisions}
    assert decisions[specs[0].analysis_id].state is PlanState.MISSING
    assert decisions[specs[1].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE
    assert decisions[specs[1].analysis_id].reason_code == "dependency_result_pending"
    assert decisions[specs[2].analysis_id].state is PlanState.DEPENDENT_RECOMPUTE


def test_plan_json_is_deterministic_and_planning_does_not_write(tmp_path):
    specs = _specs()
    inputs, results = _complete_fixture(specs)
    marker = tmp_path / "existing.txt"
    marker.write_text("unchanged", encoding="utf-8")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    first = SemanticPlanner(_registry(specs)).plan(
        specs[-1].analysis_id,
        inputs_by_node=dict(reversed(tuple(inputs.items()))),
        current_results=dict(reversed(tuple(results.items()))),
        current_generation_id="preprocess-generation-1",
    )
    second = SemanticPlanner(_registry(tuple(reversed(specs)))).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
        current_generation_id="preprocess-generation-1",
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert first.to_json() == second.to_json()
    assert before == after
    assert first.to_dict()["current_generation_id"] == "preprocess-generation-1"


def test_registered_executor_is_discoverable_but_never_run_by_planner():
    specs = _specs()
    calls = []

    def executor():
        calls.append("executed")

    registry = _registry(specs, executors={specs[0].analysis_id: executor})
    inputs, results = _complete_fixture(specs)

    assert registry.executor_for(specs[0].analysis_id) is executor
    SemanticPlanner(registry).plan(
        specs[-1].analysis_id,
        inputs_by_node=inputs,
        current_results=results,
    )
    assert calls == []
