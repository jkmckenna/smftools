import json

from smftools.pipeline.semantic_graph import PlanDecision, PlanState, SemanticPlan
from smftools.pipeline.upgrade_impact import build_upgrade_impact, format_upgrade_impact


def _decision(
    analysis_id: str,
    state: PlanState,
    *,
    invalidated_by: tuple[str, ...] = (),
) -> PlanDecision:
    return PlanDecision(
        analysis_id=analysis_id,
        state=state,
        reason_code=f"reason_{state.value}",
        reason=f"because {state.value}",
        expected_outputs=(),
        invalidated_by=invalidated_by,
    )


def test_upgrade_impact_groups_states_triggers_and_partial_historical_cost():
    plan = SemanticPlan(
        requested_target="node.blocked",
        topological_order=("node.ok", "node.root", "node.child", "node.blocked"),
        decisions=(
            _decision("node.ok", PlanState.COMPATIBLE),
            _decision("node.root", PlanState.STALE_ALGORITHM),
            _decision(
                "node.child",
                PlanState.DEPENDENT_RECOMPUTE,
                invalidated_by=("node.root",),
            ),
            _decision("node.blocked", PlanState.BLOCKED_MISSING_INPUT),
        ),
        graph_definition_version=7,
    )

    report = build_upgrade_impact(
        plan,
        scope="experiment",
        historical_seconds={
            "node.root": 2.5,
            "node.child": "not-a-number",
            "not-in-plan": 100.0,
        },
    )
    payload = report.to_dict()

    assert payload["has_impact"] is True
    assert payload["impact_node_count"] == 3
    assert payload["trigger_node_count"] == 2
    assert payload["dependent_node_count"] == 1
    assert payload["blocked_node_count"] == 1
    assert [trigger["analysis_id"] for trigger in payload["triggers"]] == [
        "node.root",
        "node.blocked",
    ]
    assert [group["state"] for group in payload["plan_state_groups"]] == [
        "compatible",
        "stale_algorithm",
        "dependent_recompute",
        "blocked_missing_input",
    ]
    assert payload["recompute_cost"] == {
        "basis": "historical_elapsed_seconds",
        "estimated_seconds": 2.5,
        "complete": False,
        "recompute_node_count": 3,
        "known_node_count": 1,
        "known_nodes": ["node.root"],
        "unknown_nodes": ["node.child", "node.blocked"],
    }
    assert report.to_json() == report.to_json()
    assert json.loads(report.to_json()) == payload
    rendered = format_upgrade_impact(report)
    assert "stale_algorithm (1):" in rendered
    assert "invalidated by node.root" in rendered
    assert "partial historical coverage" in rendered


def test_compatible_upgrade_impact_has_zero_complete_recompute_cost():
    plan = SemanticPlan(
        requested_target="node.ok",
        topological_order=("node.ok",),
        decisions=(_decision("node.ok", PlanState.COMPATIBLE),),
        graph_definition_version=1,
    )

    payload = build_upgrade_impact(plan, scope="project").to_dict()

    assert payload["has_impact"] is False
    assert payload["triggers"] == []
    assert payload["recompute_cost"]["estimated_seconds"] == 0.0
    assert payload["recompute_cost"]["complete"] is True
    assert payload["recompute_cost"]["basis"] == "no_recompute"
