"""Deterministic fit ownership and bounded read selection for partitioned HMMs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .model_artifacts import hmm_fit_config_hash, training_selection_metadata

HMM_FIT_BYTES_PER_SIGNAL_VALUE = 16
SUPPORTED_HMM_GROUP_FIELDS = frozenset({"reference", "methbase", "sample", "barcode"})
REQUIRED_HMM_GROUP_FIELDS = frozenset({"reference", "methbase"})


@dataclass(frozen=True)
class HMMModelSpec:
    """One configured signal/model definition expanded from HMM task config.

    ``variant`` names an emission variant (`HCE-06`): "" is the default, whose
    ``label`` -- the prefix of every layer it writes -- is unchanged; another
    variant's label is ``<base_label>_<variant>``. ``overrides`` are the
    variant's config settings, as ``(key, value)`` pairs.
    """

    name: str
    label: str
    signals: tuple[str, ...]
    feature_groups: tuple[str, ...]
    architecture: str
    variant: str = ""
    base_label: str = ""
    overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def n_channels(self) -> int:
        """Return the number of signal channels materialized for fitting."""
        return max(1, len(self.signals))

    def config(self, cfg: Any) -> Any:
        """``cfg`` with this variant's settings applied (``cfg`` itself for the default)."""
        if not self.overrides:
            return cfg
        variant = copy.copy(cfg)
        for key, value in self.overrides:
            setattr(variant, key, value)
        return variant


@dataclass(frozen=True)
class HMMFitMember:
    """One reference's share of a fit pooled across references (`HXR-02`)."""

    reference: str
    core_start: int
    core_end: int
    load_start: int
    load_end: int
    candidate_n_reads: int
    selected_read_ids: tuple[str, ...]

    def to_dict(self, *, include_read_ids: bool = False) -> dict[str, Any]:
        record = asdict(self)
        record["selected_n_reads"] = len(self.selected_read_ids)
        if not include_read_ids:
            record.pop("selected_read_ids")
        return record


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
    # A fit pooled across references (`HXR`): one entry per reference, its
    # window and selected reads; empty for a single-reference fit, whose
    # ``reference`` is that reference. A pooled fit's ``reference`` is its
    # members joined with "+".
    members: tuple[HMMFitMember, ...] = ()
    # Emission scopes (`HPS-02`): fits with the same emission group share one
    # emission model, fitted jointly (tied EM); the efficiency key names the
    # offset group (empty: no offset). Both empty under the default scope.
    emission_group: str = ""
    efficiency_key: str = ""
    efficiency_covariates: tuple[float, ...] = ()  # log values, hmm_efficiency_covariates order

    @property
    def references(self) -> tuple[str, ...]:
        """The references the fit is trained on and applied to."""
        return tuple(m.reference for m in self.members) or (self.reference,)

    def to_dict(self, *, include_read_ids: bool = False) -> dict[str, Any]:
        """Serialize a fit plan, excluding selected molecule IDs by default."""
        record = asdict(self)
        record["model_spec"] = asdict(self.model_spec)
        record["selected_n_reads"] = len(self.selected_read_ids)
        if not include_read_ids:
            record.pop("selected_read_ids")
        record["references"] = list(self.references)
        record["members"] = json.dumps(
            [m.to_dict(include_read_ids=include_read_ids) for m in self.members],
            separators=(",", ":"),
        )
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
            **(
                {
                    "references": list(self.references),
                    "reference_n_reads": {
                        m.reference: len(m.selected_read_ids) for m in self.members
                    },
                }
                if self.members
                else {}
            ),
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


_STRAND_SUFFIXES = ("_top", "_bottom")


def _strand_of(reference: str) -> tuple[str, str]:
    for suffix in _STRAND_SUFFIXES:
        if reference.endswith(suffix):
            return reference[: -len(suffix)], suffix[1:]
    return reference, ""


def reference_groups(cfg: Any) -> dict[str, tuple[str, ...]]:
    """``hmm_reference_groups`` (`HXR-01`): group name -> member references.

    A member is a reference name or a base name standing for every strand of
    it; a reference may belong to one group only.
    """
    raw = _cfg_value(cfg, "hmm_reference_groups", None) or {}
    if not isinstance(raw, dict):
        raise ValueError("hmm_reference_groups must map a group name to references")
    groups: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for name, members in raw.items():
        members = [members] if isinstance(members, str) else list(members or [])
        groups[str(name)] = tuple(str(member) for member in members)
        for member in groups[str(name)]:
            if member in owner:
                raise ValueError(
                    f"hmm_reference_groups lists {member!r} in both {owner[member]!r} and {name!r}"
                )
            owner[member] = str(name)
    return groups


def _reference_group(reference: str, groups: dict[str, tuple[str, ...]]) -> str:
    """The grouping key of ``reference``: its group and strand, or itself."""
    base, strand = _strand_of(reference)
    for name, members in groups.items():
        if reference in members or base in members:
            return f"\0{name}\0{strand}"
    return reference


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


def _group_tasks(
    tasks: Sequence[Any],
    *,
    include_sample: bool,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> list[list[Any]]:
    """Tasks sharing one fit: same reference and window (and sample), or --
    for references in one ``hmm_reference_groups`` group -- every task of the
    group's references (`HXR-02`), each reference holding one analysis core."""
    groups = groups or {}
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for task in tasks:
        group = _reference_group(str(task.reference), groups)
        window = (
            (int(task.core_start), int(task.core_end), int(task.load_start), int(task.load_end))
            if group == str(task.reference)
            else (0, 0, 0, 0)
        )
        key = (group, *window, str(task.barcode) if include_sample else "ALL")
        grouped.setdefault(key, []).append(task)
    for key, members in grouped.items():
        windows: dict[str, set[tuple[int, int]]] = {}
        for task in members:
            windows.setdefault(str(task.reference), set()).add(
                (int(task.core_start), int(task.core_end))
            )
        split = sorted(ref for ref, cores in windows.items() if len(cores) > 1)
        if len(windows) > 1 and split:
            raise ValueError(
                "hmm_reference_groups pools whole references; "
                f"{split} have several analysis cores (genome-scale references cannot be pooled)"
            )
    return [grouped[key] for key in sorted(grouped)]


def emission_scope(cfg: Any) -> str:
    """``hmm_emission_scope`` (`HPS-01`)."""
    scope = str(_cfg_value(cfg, "hmm_emission_scope", "fit_group") or "fit_group").lower()
    if scope not in ("fit_group", "shared", "shared_efficiency"):
        raise ValueError(f"unknown hmm_emission_scope {scope!r}")
    return scope


def _emission_keys(
    plan_barcode: str,
    label: str,
    cfg: Any,
    sample_attributes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """The emission group and efficiency key of one fit group's sample."""
    scope = emission_scope(cfg)
    emission_columns = [str(c) for c in (_cfg_value(cfg, "hmm_emission_groupby", []) or [])]
    efficiency_columns, covariate_columns = efficiency_columns_of(cfg)
    attributes = sample_attributes.get(str(plan_barcode))
    needed = emission_columns + (
        efficiency_columns + covariate_columns if scope == "shared_efficiency" else []
    )
    if needed and attributes is None:
        raise ValueError(f"no sample-sheet row for barcode {plan_barcode!r}")
    missing = [c for c in needed if c not in (attributes or {})]
    if missing:
        raise ValueError(f"sample sheet lacks {missing} for barcode {plan_barcode!r}")

    def key(columns: Sequence[str]) -> str:
        return "|".join(f"{c}={attributes[c]}" for c in columns) if columns else "all"

    group = f"{label}|{key(emission_columns)}"
    efficiency = (
        key(efficiency_columns) if scope == "shared_efficiency" and efficiency_columns else ""
    )
    covariates: tuple[float, ...] = ()
    if scope == "shared_efficiency":
        values = []
        for column in covariate_columns:
            try:
                value = float(attributes[column])
            except (TypeError, ValueError):
                value = float("nan")
            if not value > 0:
                raise ValueError(
                    f"efficiency covariate {column!r} must be a positive number for barcode "
                    f"{plan_barcode!r} (it is used on the log scale), not {attributes[column]!r}"
                )
            values.append(float(np.log(value)))
        covariates = tuple(values)
    return group, efficiency, covariates


def efficiency_columns_of(cfg: Any) -> tuple[list[str], list[str]]:
    """(categorical, numeric) technical columns an efficiency offset may follow."""
    return (
        [str(c) for c in (_cfg_value(cfg, "hmm_efficiency_groupby", []) or [])],
        [str(c) for c in (_cfg_value(cfg, "hmm_efficiency_covariates", []) or [])],
    )


def _make_plan(
    tasks: Sequence[Any],
    spec: HMMModelSpec,
    cfg: Any,
    *,
    fit_kind: str,
    strategy: str,
    fields: Sequence[str],
    parent_fit_id: str | None = None,
    sample_attributes: Mapping[str, Mapping[str, Any]] | None = None,
) -> HMMFitPlan:
    by_reference: dict[str, list[Any]] = {}
    for task in tasks:
        by_reference.setdefault(str(task.reference), []).append(task)
    first = tasks[0] if len(by_reference) == 1 else by_reference[min(by_reference)][0]
    candidate_ids = sorted({str(read_id) for task in tasks for read_id in task.read_ids})
    configured_max = int(_cfg_value(cfg, "hmm_max_fit_reads", 1000))
    selection_seed = int(_cfg_value(cfg, "hmm_fit_selection_seed", 0))
    loaded_width = max(int(task.load_end) - int(task.load_start) for task in tasks)
    bytes_per_read = max(
        1,
        loaded_width * spec.n_channels * HMM_FIT_BYTES_PER_SIGNAL_VALUE,
    )
    memory_bytes = int(_cfg_value(cfg, "target_task_memory_mb", 512)) * 1024**2
    memory_max = max(1, memory_bytes // bytes_per_read)
    members: tuple[HMMFitMember, ...] = ()
    if len(by_reference) == 1:
        selected = _bounded_selection(
            candidate_ids,
            seed=selection_seed,
            configured_max=configured_max,
            memory_max=memory_max,
        )
    else:
        # Equal shares per reference (`HXR-02`), so the larger allele does
        # not dominate the shared fit; a reference with fewer reads than its
        # share gives all of them and the rest is not redistributed.
        n_references = len(by_reference)
        member_list = []
        for reference in sorted(by_reference):
            ref_tasks = by_reference[reference]
            ref_ids = sorted({str(read_id) for task in ref_tasks for read_id in task.read_ids})
            ref_selected = _bounded_selection(
                ref_ids,
                seed=selection_seed,
                configured_max=max(1, configured_max // n_references),
                memory_max=max(1, memory_max // n_references),
            )
            member_list.append(
                HMMFitMember(
                    reference=reference,
                    core_start=int(ref_tasks[0].core_start),
                    core_end=int(ref_tasks[0].core_end),
                    load_start=int(ref_tasks[0].load_start),
                    load_end=int(ref_tasks[0].load_end),
                    candidate_n_reads=len(ref_ids),
                    selected_read_ids=ref_selected,
                )
            )
        members = tuple(member_list)
        selected = tuple(read_id for member in members for read_id in member.selected_read_ids)
    selection = training_selection_metadata(selected)
    barcode = _group_barcode(fields, tasks)
    reference = "+".join(sorted(by_reference))
    identity = {
        "fit_kind": fit_kind,
        "strategy": strategy,
        "reference": reference,
        "barcode": barcode,
        "core_start": int(first.core_start),
        "core_end": int(first.core_end),
        "label": spec.label,
        "architecture": spec.architecture,
        "fit_config_hash": hmm_fit_config_hash(spec.config(cfg)),
        "parent_fit_id": parent_fit_id,
    }
    if members:
        identity["members"] = [
            [m.reference, m.core_start, m.core_end, m.load_start, m.load_end] for m in members
        ]
    emission_group = efficiency_key = ""
    efficiency_covariates: tuple[float, ...] = ()
    if sample_attributes is not None:
        emission_group, efficiency_key, efficiency_covariates = _emission_keys(
            barcode, spec.label, cfg, sample_attributes
        )
        identity["emission_group"] = emission_group
        identity["efficiency_key"] = efficiency_key
        identity["efficiency_covariates"] = list(efficiency_covariates)
    return HMMFitPlan(
        fit_id=_stable_fit_id(identity),
        fit_kind=fit_kind,
        strategy=strategy,
        reference=reference,
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
        members=members,
        emission_group=emission_group,
        efficiency_key=efficiency_key,
        efficiency_covariates=efficiency_covariates,
    )


def build_hmm_fit_plan(
    tasks: Sequence[Any],
    model_specs: Sequence[HMMModelSpec],
    cfg: Any,
    *,
    sample_attributes: Mapping[str, Mapping[str, Any]] | None = None,
) -> HMMFitPlanningResult:
    """Plan deterministic fits and final artifact assignments for apply tasks.

    Every model spec is planned under its own settings (``spec.config(cfg)``):
    a variant is a named fitting plan (`HPS-05`) that may change grouping,
    strategy, reference groups, emission scope or axis, not only emissions.
    Under a shared emission scope (`HPS-02`) every fit is per sample and
    carries its emission group and efficiency key, from ``sample_attributes``
    (barcode -> sample-sheet columns).
    """
    base_plans: list[HMMFitPlan] = []
    adaptations: list[HMMFitPlan] = []
    assignments: dict[str, dict[str, str]] = {str(task.task_id): {} for task in tasks}
    for spec in model_specs:
        _plan_spec(
            tasks,
            spec,
            spec.config(cfg),
            sample_attributes,
            base_plans,
            adaptations,
            assignments,
        )
    return HMMFitPlanningResult(
        base_plans=tuple(sorted(base_plans, key=lambda plan: plan.fit_id)),
        adaptation_plans=tuple(sorted(adaptations, key=lambda plan: plan.fit_id)),
        apply_assignments=assignments,
    )


def _plan_spec(
    tasks: Sequence[Any],
    spec: HMMModelSpec,
    cfg: Any,
    sample_attributes: Mapping[str, Mapping[str, Any]] | None,
    base_plans: list[HMMFitPlan],
    adaptations: list[HMMFitPlan],
    assignments: dict[str, dict[str, str]],
) -> None:
    """Plan one model spec's fits under its own settings ``cfg``."""
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
    groups = reference_groups(cfg)
    tied = emission_scope(cfg) != "fit_group"
    if tied:
        if sample_attributes is None:
            raise ValueError("a shared hmm_emission_scope needs a sample sheet (sample_sheet_path)")
        if strategy != "per_group" or "sample" not in group_fields:
            raise ValueError(
                "a shared hmm_emission_scope fits transitions per sample: it needs "
                "hmm_fit_strategy 'per_group' with 'sample' in hmm_groupby"
            )
        if emission_scope(cfg) == "shared_efficiency" and not any(efficiency_columns_of(cfg)):
            raise ValueError(
                "hmm_emission_scope 'shared_efficiency' needs hmm_efficiency_covariates and/or "
                "hmm_efficiency_groupby: the technical sample-sheet columns efficiency may follow"
            )
    else:
        sample_attributes = None
    if groups and bool(_cfg_value(cfg, "hmm_distance_aware", False)):
        raise ValueError("hmm_reference_groups does not combine with hmm_distance_aware")
    if strategy == "per_group":
        for group in _group_tasks(tasks, include_sample="sample" in group_fields, groups=groups):
            plan = _make_plan(
                group,
                spec,
                cfg,
                fit_kind="PER",
                strategy=strategy,
                fields=group_fields,
                sample_attributes=sample_attributes,
            )
            base_plans.append(plan)
            for task_id in plan.apply_task_ids:
                assignments[task_id][spec.label] = plan.fit_id
        return

    shared_groups = _group_tasks(tasks, include_sample="sample" in shared_fields, groups=groups)
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
            return
        for adapt_group in _group_tasks(
            shared_group, include_sample="sample" in group_fields, groups=groups
        ):
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
