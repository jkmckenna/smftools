"""Deterministic fit ownership and bounded read selection for partitioned HMMs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .model_artifacts import hmm_fit_config_hash, training_selection_metadata

HMM_FIT_BYTES_PER_SIGNAL_VALUE = 16
SUPPORTED_HMM_GROUP_FIELDS = frozenset({"reference", "methbase", "sample", "barcode"})
REQUIRED_HMM_GROUP_FIELDS = frozenset({"reference", "methbase"})


@dataclass(frozen=True)
class HMMModelSpec:
    """One configured signal/model definition expanded from HMM task config."""

    name: str
    label: str
    signals: tuple[str, ...]
    feature_groups: tuple[str, ...]
    architecture: str

    @property
    def n_channels(self) -> int:
        """Return the number of signal channels materialized for fitting."""
        return max(1, len(self.signals))


@dataclass(frozen=True)
class HMMFitPlan:
    """One immutable model fit or adaptation completed before apply workers start."""

    fit_id: str
    fit_kind: str
    strategy: str
    reference: str
    barcode: str
    core_start: int
    core_end: int
    load_start: int
    load_end: int
    model_spec: HMMModelSpec
    candidate_n_reads: int
    selected_read_ids: tuple[str, ...]
    configured_max_fit_reads: int
    memory_max_fit_reads: int
    selection_seed: int
    selection_sha256: str
    estimated_memory_bytes: int
    apply_task_ids: tuple[str, ...]
    parent_fit_id: str | None = None

    def to_dict(self, *, include_read_ids: bool = False) -> dict[str, Any]:
        """Serialize a fit plan, excluding selected molecule IDs by default."""
        record = asdict(self)
        record["model_spec"] = asdict(self.model_spec)
        record["selected_n_reads"] = len(self.selected_read_ids)
        if not include_read_ids:
            record.pop("selected_read_ids")
        return record

    def training_metadata(self) -> dict[str, Any]:
        """Return persisted checkpoint provenance for the selected fit population."""
        return {
            **training_selection_metadata(
                self.selected_read_ids,
                n_reads=len(self.selected_read_ids),
            ),
            "fit_id": self.fit_id,
            "fit_kind": self.fit_kind,
            "strategy": self.strategy,
            "candidate_n_reads": self.candidate_n_reads,
            "configured_max_fit_reads": self.configured_max_fit_reads,
            "memory_max_fit_reads": self.memory_max_fit_reads,
            "selection_seed": self.selection_seed,
            "reference": self.reference,
            "barcode": self.barcode,
            "core_start": self.core_start,
            "core_end": self.core_end,
            "parent_fit_id": self.parent_fit_id,
        }


@dataclass(frozen=True)
class HMMFitPlanningResult:
    """Ordered fits plus each apply task's final model assignment."""

    base_plans: tuple[HMMFitPlan, ...]
    adaptation_plans: tuple[HMMFitPlan, ...]
    apply_assignments: dict[str, dict[str, str]]

    @property
    def all_plans(self) -> tuple[HMMFitPlan, ...]:
        """Return base fits followed by dependent adaptations."""
        return (*self.base_plans, *self.adaptation_plans)


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _normalize_group_fields(cfg: Any, name: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = _cfg_value(cfg, name, default)
    fields = tuple(str(value).strip().lower() for value in (raw or []))
    aliases = tuple("sample" if value == "barcode" else value for value in fields)
    unknown = sorted(set(aliases).difference(SUPPORTED_HMM_GROUP_FIELDS))
    if unknown:
        raise ValueError(f"{name} contains unsupported partitioned HMM fields: {unknown}")
    missing = sorted(REQUIRED_HMM_GROUP_FIELDS.difference(aliases))
    if missing:
        raise ValueError(f"{name} must include {missing} for bounded partitioned HMM fitting")
    return tuple(dict.fromkeys(aliases))


def _stable_fit_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"hmm-fit-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _hash_rank(read_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{read_id}".encode("utf-8")).hexdigest()
    return digest, read_id


def _bounded_selection(
    read_ids: Iterable[str],
    *,
    seed: int,
    configured_max: int,
    memory_max: int,
) -> tuple[str, ...]:
    candidates = sorted(set(map(str, read_ids)))
    limit = min(len(candidates), configured_max, memory_max)
    if limit >= len(candidates):
        return tuple(candidates)
    ranked = sorted((_hash_rank(read_id, seed) for read_id in candidates))
    return tuple(read_id for _, read_id in ranked[:limit])


def _group_barcode(fields: Sequence[str], tasks: Sequence[Any]) -> str:
    if "sample" not in fields:
        return "ALL"
    barcodes = sorted({str(task.barcode) for task in tasks})
    if len(barcodes) != 1:
        raise ValueError(f"HMM sample group unexpectedly contains barcodes {barcodes}")
    return barcodes[0]


def _group_tasks(tasks: Sequence[Any], *, include_sample: bool) -> list[list[Any]]:
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for task in tasks:
        key = (
            str(task.reference),
            int(task.core_start),
            int(task.core_end),
            int(task.load_start),
            int(task.load_end),
            str(task.barcode) if include_sample else "ALL",
        )
        grouped.setdefault(key, []).append(task)
    return [grouped[key] for key in sorted(grouped)]


def _make_plan(
    tasks: Sequence[Any],
    spec: HMMModelSpec,
    cfg: Any,
    *,
    fit_kind: str,
    strategy: str,
    fields: Sequence[str],
    parent_fit_id: str | None = None,
) -> HMMFitPlan:
    first = tasks[0]
    candidate_ids = sorted({str(read_id) for task in tasks for read_id in task.read_ids})
    configured_max = int(_cfg_value(cfg, "hmm_max_fit_reads", 1000))
    selection_seed = int(_cfg_value(cfg, "hmm_fit_selection_seed", 0))
    loaded_width = int(first.load_end) - int(first.load_start)
    bytes_per_read = max(
        1,
        loaded_width * spec.n_channels * HMM_FIT_BYTES_PER_SIGNAL_VALUE,
    )
    memory_bytes = int(_cfg_value(cfg, "target_task_memory_mb", 512)) * 1024**2
    memory_max = max(1, memory_bytes // bytes_per_read)
    selected = _bounded_selection(
        candidate_ids,
        seed=selection_seed,
        configured_max=configured_max,
        memory_max=memory_max,
    )
    selection = training_selection_metadata(selected)
    barcode = _group_barcode(fields, tasks)
    identity = {
        "fit_kind": fit_kind,
        "strategy": strategy,
        "reference": str(first.reference),
        "barcode": barcode,
        "core_start": int(first.core_start),
        "core_end": int(first.core_end),
        "label": spec.label,
        "architecture": spec.architecture,
        "fit_config_hash": hmm_fit_config_hash(cfg),
        "parent_fit_id": parent_fit_id,
    }
    return HMMFitPlan(
        fit_id=_stable_fit_id(identity),
        fit_kind=fit_kind,
        strategy=strategy,
        reference=str(first.reference),
        barcode=barcode,
        core_start=int(first.core_start),
        core_end=int(first.core_end),
        load_start=int(first.load_start),
        load_end=int(first.load_end),
        model_spec=spec,
        candidate_n_reads=len(candidate_ids),
        selected_read_ids=selected,
        configured_max_fit_reads=configured_max,
        memory_max_fit_reads=memory_max,
        selection_seed=selection_seed,
        selection_sha256=str(selection["selection_sha256"]),
        estimated_memory_bytes=len(selected) * bytes_per_read,
        apply_task_ids=tuple(sorted(str(task.task_id) for task in tasks)),
        parent_fit_id=parent_fit_id,
    )


def build_hmm_fit_plan(
    tasks: Sequence[Any], model_specs: Sequence[HMMModelSpec], cfg: Any
) -> HMMFitPlanningResult:
    """Plan deterministic fits and final artifact assignments for apply tasks."""
    strategy = str(_cfg_value(cfg, "hmm_fit_strategy", "per_group")).strip().lower()
    if strategy not in {"per_group", "shared_transitions"}:
        raise ValueError(
            f"hmm_fit_strategy must be 'per_group' or 'shared_transitions'; got {strategy!r}"
        )
    group_fields = _normalize_group_fields(
        cfg,
        "hmm_groupby",
        ("sample", "reference", "methbase"),
    )
    shared_fields = _normalize_group_fields(
        cfg,
        "hmm_shared_scope",
        ("reference", "methbase"),
    )
    base_plans: list[HMMFitPlan] = []
    adaptations: list[HMMFitPlan] = []
    assignments: dict[str, dict[str, str]] = {str(task.task_id): {} for task in tasks}

    for spec in model_specs:
        if strategy == "per_group":
            groups = _group_tasks(tasks, include_sample="sample" in group_fields)
            for group in groups:
                plan = _make_plan(
                    group,
                    spec,
                    cfg,
                    fit_kind="PER",
                    strategy=strategy,
                    fields=group_fields,
                )
                base_plans.append(plan)
                for task_id in plan.apply_task_ids:
                    assignments[task_id][spec.label] = plan.fit_id
            continue

        shared_groups = _group_tasks(tasks, include_sample="sample" in shared_fields)
        adapt_enabled = bool(_cfg_value(cfg, "hmm_adapt_emissions", True)) or bool(
            _cfg_value(cfg, "hmm_adapt_startprobs", True)
        )
        for shared_group in shared_groups:
            base = _make_plan(
                shared_group,
                spec,
                cfg,
                fit_kind="GLOBAL",
                strategy=strategy,
                fields=shared_fields,
            )
            base_plans.append(base)
            if not adapt_enabled:
                for task_id in base.apply_task_ids:
                    assignments[task_id][spec.label] = base.fit_id
                continue
            for adapt_group in _group_tasks(shared_group, include_sample="sample" in group_fields):
                adapted = _make_plan(
                    adapt_group,
                    spec,
                    cfg,
                    fit_kind="ADAPT",
                    strategy=strategy,
                    fields=group_fields,
                    parent_fit_id=base.fit_id,
                )
                adaptations.append(adapted)
                for task_id in adapted.apply_task_ids:
                    assignments[task_id][spec.label] = adapted.fit_id

    return HMMFitPlanningResult(
        base_plans=tuple(sorted(base_plans, key=lambda plan: plan.fit_id)),
        adaptation_plans=tuple(sorted(adaptations, key=lambda plan: plan.fit_id)),
        apply_assignments=assignments,
    )
