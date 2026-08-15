"""Read-only upgrade-impact projections over semantic compatibility plans."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping

from .semantic_graph import PlanDecision, PlanState, SemanticPlan

UPGRADE_IMPACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UpgradeImpactReport:
    """A stable grouped view of one existing semantic plan.

    ``historical_seconds`` contains observed elapsed times from prior executions,
    never synthetic estimates. Missing observations remain explicit in the cost
    summary instead of being replaced by guessed throughput.
    """

    scope: str
    plan: SemanticPlan
    historical_seconds: tuple[tuple[str, float], ...] = ()
    schema_version: int = UPGRADE_IMPACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.scope not in {"experiment", "project"}:
            raise ValueError("upgrade-impact scope must be 'experiment' or 'project'")

    @property
    def timing_by_node(self) -> dict[str, float]:
        return dict(self.historical_seconds)

    @property
    def triggers(self) -> tuple[PlanDecision, ...]:
        return tuple(
            decision
            for decision in self.plan.decisions
            if decision.state not in {PlanState.COMPATIBLE, PlanState.DEPENDENT_RECOMPUTE}
        )

    @property
    def recompute_decisions(self) -> tuple[PlanDecision, ...]:
        return tuple(
            decision
            for decision in self.plan.decisions
            if decision.state is not PlanState.COMPATIBLE
        )

    def _node_payload(self, decision: PlanDecision) -> dict[str, object]:
        payload = decision.to_dict()
        if decision.state is PlanState.COMPATIBLE:
            role = "compatible"
        elif decision.state is PlanState.BLOCKED_MISSING_INPUT:
            role = "blocked"
        elif decision.state is PlanState.DEPENDENT_RECOMPUTE:
            role = "dependent"
        else:
            role = "trigger"
        payload["role"] = role
        if decision.analysis_id in self.timing_by_node and decision in self.recompute_decisions:
            payload["historical_elapsed_seconds"] = self.timing_by_node[decision.analysis_id]
        return payload

    def _cost_payload(self) -> dict[str, object]:
        recompute_ids = [decision.analysis_id for decision in self.recompute_decisions]
        timings = self.timing_by_node
        known = [analysis_id for analysis_id in recompute_ids if analysis_id in timings]
        unknown = [analysis_id for analysis_id in recompute_ids if analysis_id not in timings]
        if not recompute_ids:
            basis = "no_recompute"
            estimated_seconds: float | None = 0.0
        elif not known:
            basis = "unavailable"
            estimated_seconds = None
        else:
            basis = "historical_elapsed_seconds"
            estimated_seconds = sum(timings[analysis_id] for analysis_id in known)
        return {
            "basis": basis,
            "estimated_seconds": estimated_seconds,
            "complete": not unknown,
            "recompute_node_count": len(recompute_ids),
            "known_node_count": len(known),
            "known_nodes": known,
            "unknown_nodes": unknown,
        }

    def to_dict(self) -> dict[str, object]:
        groups = []
        for state in PlanState:
            decisions = tuple(
                decision for decision in self.plan.decisions if decision.state is state
            )
            if decisions:
                groups.append(
                    {
                        "state": state.value,
                        "count": len(decisions),
                        "nodes": [self._node_payload(decision) for decision in decisions],
                    }
                )
        noncompatible = tuple(
            decision
            for decision in self.plan.decisions
            if decision.state is not PlanState.COMPATIBLE
        )
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "requested_target": self.plan.requested_target,
            "source_plan_schema_version": self.plan.schema_version,
            "graph_definition_version": self.plan.graph_definition_version,
            "topological_order": list(self.plan.topological_order),
            "has_impact": bool(noncompatible),
            "impact_node_count": len(noncompatible),
            "trigger_node_count": len(self.triggers),
            "dependent_node_count": sum(
                decision.state is PlanState.DEPENDENT_RECOMPUTE for decision in self.plan.decisions
            ),
            "blocked_node_count": sum(
                decision.state is PlanState.BLOCKED_MISSING_INPUT
                for decision in self.plan.decisions
            ),
            "triggers": [self._node_payload(decision) for decision in self.triggers],
            "plan_state_groups": groups,
            "recompute_cost": self._cost_payload(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return stable JSON for automation and diffing between installations."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), indent=indent)


def build_upgrade_impact(
    plan: SemanticPlan,
    *,
    scope: str,
    historical_seconds: Mapping[str, object] | None = None,
) -> UpgradeImpactReport:
    """Project one semantic plan into grouped upgrade-impact reporting."""
    valid_ids = set(plan.topological_order)
    normalized: list[tuple[str, float]] = []
    for analysis_id, raw_seconds in (historical_seconds or {}).items():
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            continue
        if analysis_id in valid_ids and math.isfinite(seconds) and seconds >= 0:
            normalized.append((str(analysis_id), seconds))
    return UpgradeImpactReport(
        scope=scope,
        plan=plan,
        historical_seconds=tuple(sorted(normalized)),
    )


def format_upgrade_impact(report: UpgradeImpactReport) -> str:
    """Render a deterministic grouped upgrade-impact report."""
    payload = report.to_dict()
    cost = payload["recompute_cost"]
    assert isinstance(cost, dict)
    lines = [
        f"{report.scope.title()} target: {report.plan.requested_target}",
        f"Graph definition version: {report.plan.graph_definition_version}",
        f"Impact: {payload['impact_node_count']} node(s) require attention; "
        f"{payload['dependent_node_count']} downstream recompute(s); "
        f"{payload['blocked_node_count']} blocked",
    ]
    estimate = cost["estimated_seconds"]
    if cost["basis"] == "no_recompute":
        lines.append("Estimated recompute cost: 0 seconds (no recomputation)")
    elif estimate is None:
        lines.append("Estimated recompute cost: unknown (no historical timings)")
    else:
        coverage = f"{cost['known_node_count']}/{cost['recompute_node_count']} nodes"
        qualifier = "complete" if cost["complete"] else "partial"
        lines.append(
            f"Estimated recompute cost: {float(estimate):.3f} seconds "
            f"({qualifier} historical coverage, {coverage})"
        )

    lines.extend(("", "Triggers:"))
    if report.triggers:
        lines.extend(
            f"- {decision.state.value} {decision.analysis_id} — {decision.reason}"
            for decision in report.triggers
        )
    else:
        lines.append("- none")

    lines.extend(("", "Plan states:"))
    timings = report.timing_by_node
    for group in payload["plan_state_groups"]:
        assert isinstance(group, dict)
        lines.append(f"{group['state']} ({group['count']}):")
        nodes = group["nodes"]
        assert isinstance(nodes, list)
        for node in nodes:
            assert isinstance(node, dict)
            suffix = ""
            analysis_id = str(node["analysis_id"])
            if analysis_id in timings and node["role"] != "compatible":
                suffix = f" [historical {timings[analysis_id]:.3f}s]"
            invalidated_by = node.get("invalidated_by", [])
            cause = f"; invalidated by {', '.join(invalidated_by)}" if invalidated_by else ""
            lines.append(f"  - {analysis_id}: {node['reason']}{cause}{suffix}")
    return "\n".join(lines)
