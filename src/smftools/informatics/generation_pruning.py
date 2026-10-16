"""Conservative, read-only pruning plans for immutable generations.

The planner never deletes. It keeps current and pinned generations, every
generation they read -- transitively, through spine pointers and manifest
lineage (`SLC-06`) -- and every generation an outside file (a project
registry) names; then applies the age / count policy to what is left. A
generation that nothing live reads is a prune candidate (`SLC-07`): no
current result depends on it, so its byte reproducibility (the former
`EGL-03a` blocker) is not needed to remove it safely. When any generation's
references cannot be read, every non-current generation is kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .generation_listing import (
    STAGE_GENERATION_DIRS,
    STATE_OK,
    STATE_PRUNED,
    GenerationRecord,
    list_experiment_generations,
)
from .generation_references import (
    GenerationReferenceError,
    build_reference_graph,
    external_references,
    protected_generations,
)

PRUNE_PLAN_SCHEMA_VERSION = 2

KEEP_CURRENT = "keep_current"
KEEP_PINNED = "keep_pinned"
KEEP_UNREADABLE = "keep_unreadable"
KEEP_LAST = "keep_last"
KEEP_RECENT = "keep_recent"
KEEP_REFERENCED = "keep_referenced"
KEEP_REFERENCES_UNKNOWN = "keep_references_unknown"
PRUNE_CANDIDATE = "prune_candidate"
PRUNED = "pruned"


class GenerationPruneError(ValueError):
    """Raised when a pruning request cannot be planned safely."""


@dataclass(frozen=True)
class PruneDecision:
    """One generation's disposition in a read-only pruning plan."""

    kind: str
    generation_id: str
    path: str
    disposition: str
    policy_candidate: bool
    deletion_allowed: bool
    is_current: bool
    pinned: bool
    timestamp: str
    size_bytes: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation."""
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class PrunePlan:
    """A non-destructive retention-policy evaluation for one experiment."""

    output_root: str
    keep_last: int | None
    older_than: str | None
    stages: tuple[str, ...]
    decisions: tuple[PruneDecision, ...]

    @property
    def candidate_bytes(self) -> int:
        """Bytes of the generations the policy selects for pruning."""
        return sum(
            decision.size_bytes or 0 for decision in self.decisions if decision.policy_candidate
        )

    @property
    def reclaimable_bytes(self) -> int:
        """Bytes of candidates nothing live reads (equal to ``candidate_bytes``)."""
        return sum(
            decision.size_bytes or 0 for decision in self.decisions if decision.deletion_allowed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned machine-readable plan."""
        return {
            "schema_version": PRUNE_PLAN_SCHEMA_VERSION,
            "dry_run": True,
            "deletion_supported": False,
            "output_root": self.output_root,
            "criteria": {
                "keep_last": self.keep_last,
                "older_than": self.older_than,
                "stages": list(self.stages),
            },
            "candidate_bytes": self.candidate_bytes,
            "reclaimable_bytes": self.reclaimable_bytes,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def parse_older_than(value: str | datetime | None) -> datetime | None:
    """Parse an ISO-8601 pruning cutoff, treating naive values as UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = str(value).strip()
        if not normalized:
            raise GenerationPruneError("older-than timestamp must not be empty")
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise GenerationPruneError("older-than must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_timestamp(record: GenerationRecord) -> tuple[datetime | None, str]:
    raw = record.created_at or record.modified_at
    if not raw:
        return None, ""
    try:
        parsed = parse_older_than(raw)
    except GenerationPruneError:
        return None, raw
    return parsed, raw


def _selected_stages(stages: Iterable[str] | None) -> tuple[str, ...]:
    if stages is None:
        return tuple(STAGE_GENERATION_DIRS)
    selected = tuple(dict.fromkeys(str(stage).strip().lower() for stage in stages))
    unknown = sorted(set(selected).difference(STAGE_GENERATION_DIRS))
    if unknown:
        raise GenerationPruneError(f"unknown generation stage(s): {unknown}")
    if not selected:
        raise GenerationPruneError("at least one generation stage is required")
    return selected


def plan_experiment_generation_prune(
    output_root: str | Path,
    *,
    keep_last: int | None = None,
    older_than: str | datetime | None = None,
    stages: Iterable[str] | None = None,
    protect_from: Iterable[str | Path] = (),
) -> PrunePlan:
    """Evaluate retention policy without deleting or modifying any artifact.

    ``protect_from``: files (or directories, read as their ``registry.json``)
    whose ``<stage dir>/generations/<id>`` paths name generations that must
    stay, with everything they read.
    """
    output_root = Path(output_root)
    if keep_last is None and older_than is None:
        raise GenerationPruneError("prune planning requires --keep-last or --older-than")
    if keep_last is not None and keep_last < 0:
        raise GenerationPruneError("keep-last must be zero or greater")
    cutoff = parse_older_than(older_than)
    selected_stages = _selected_stages(stages)
    records = [
        record
        for record in list_experiment_generations(output_root, include_size=True)
        if record.kind in selected_stages
    ]

    graph = build_reference_graph(output_root)
    try:
        named = external_references(protect_from, graph)
    except GenerationReferenceError as exc:
        raise GenerationPruneError(str(exc)) from exc
    all_records = list_experiment_generations(output_root, include_size=False)
    roots = {
        (record.kind, record.generation_id): "current"
        for record in all_records
        if record.is_current
    }
    roots.update(
        {
            (record.kind, record.generation_id): "pinned"
            for record in all_records
            if record.pinned and (record.kind, record.generation_id) not in roots
        }
    )
    for key, why in named.items():
        roots.setdefault(key, why)
    referenced = protected_generations(graph, roots)
    references_unknown = sorted(
        f"{kind} {gid[:8]}: {error}" for (kind, gid), error in graph.unreadable.items()
    )

    newest_ids: dict[str, set[str]] = {}
    if keep_last is not None:
        for kind in selected_stages:
            kind_records = [record for record in records if record.kind == kind]
            ordered = sorted(
                kind_records,
                key=lambda record: (
                    _record_timestamp(record)[0] or datetime.min.replace(tzinfo=timezone.utc),
                    record.generation_id,
                ),
                reverse=True,
            )
            newest_ids[kind] = {record.generation_id for record in ordered[:keep_last]}

    decisions: list[PruneDecision] = []
    for record in records:
        timestamp, raw_timestamp = _record_timestamp(record)
        if record.state == STATE_PRUNED:
            disposition = PRUNED
            reasons = (f"already {record.status or 'pruned'}",)
        elif record.is_current:
            disposition = KEEP_CURRENT
            reasons = ("selected by current.json",)
        elif record.pinned:
            disposition = KEEP_PINNED
            reasons = tuple(f"retention: {reason}" for reason in record.retention_reasons) or (
                "pinned by retention.json",
            )
        elif (record.kind, record.generation_id) in named:
            disposition = KEEP_REFERENCED
            reasons = (named[(record.kind, record.generation_id)],)
        elif (record.kind, record.generation_id) in referenced:
            disposition = KEEP_REFERENCED
            reasons = (referenced[(record.kind, record.generation_id)],)
        elif references_unknown:
            disposition = KEEP_REFERENCES_UNKNOWN
            reasons = ("references could not be read: " + "; ".join(references_unknown),)
        elif record.state != STATE_OK or record.issues or timestamp is None:
            disposition = KEEP_UNREADABLE
            reasons = tuple(record.issues) or ("generation metadata is incomplete",)
        elif keep_last is not None and record.generation_id in newest_ids.get(record.kind, set()):
            disposition = KEEP_LAST
            reasons = (f"among the newest {keep_last} {record.kind} generation(s)",)
        elif cutoff is not None and timestamp >= cutoff:
            disposition = KEEP_RECENT
            reasons = (f"not older than {cutoff.isoformat()}",)
        else:
            disposition = PRUNE_CANDIDATE
            reasons = ("matches retention policy; no current, pinned or named generation reads it",)

        policy_candidate = disposition == PRUNE_CANDIDATE
        decisions.append(
            PruneDecision(
                kind=record.kind,
                generation_id=record.generation_id,
                path=record.path,
                disposition=disposition,
                policy_candidate=policy_candidate,
                deletion_allowed=policy_candidate,
                is_current=record.is_current,
                pinned=record.pinned,
                timestamp=raw_timestamp,
                size_bytes=record.size_bytes,
                reasons=reasons,
            )
        )

    return PrunePlan(
        output_root=output_root.as_posix(),
        keep_last=keep_last,
        older_than=cutoff.isoformat() if cutoff is not None else None,
        stages=selected_stages,
        decisions=tuple(decisions),
    )
