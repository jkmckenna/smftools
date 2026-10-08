"""Group-aware machine-learning split resolution without matrix access."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from smftools.informatics.molecule_identity import MOLECULE_UID_COLUMN

from .manifests import DatasetSnapshotManifest, SplitManifest
from .plan import MLPlan, SplitSpec
from .selection import MLDataSelectionPlan

_ROLES = ("train", "validation", "test")


class MLSplitPlanningError(ValueError):
    """Raised when biological groups cannot satisfy a declared split."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitCellCount:
    """Observation and group support for one split/class/modality cell."""

    split: str
    modality: str
    class_id: int
    n_observations: int
    n_groups: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable cell count."""
        return {
            "split": self.split,
            "modality": self.modality,
            "class_id": self.class_id,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
        }


@dataclass(frozen=True)
class SplitRoleSummary:
    """Auditable counts for one resolved split role."""

    split: str
    n_observations: int
    n_groups: int
    counts_by_class: Mapping[int, int]
    counts_by_modality: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counts_by_class",
            MappingProxyType(dict(sorted(self.counts_by_class.items()))),
        )
        object.__setattr__(
            self,
            "counts_by_modality",
            MappingProxyType(dict(sorted(self.counts_by_modality.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable role summary."""
        return {
            "split": self.split,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "counts_by_class": {
                str(class_id): count for class_id, count in self.counts_by_class.items()
            },
            "counts_by_modality": dict(self.counts_by_modality),
        }


@dataclass(frozen=True)
class MLSplitResolution:
    """One deterministic assignment resolved from an ML selection plan."""

    split_name: str
    fold_name: str | None
    strategy: str
    seed: int
    resolution_id: str
    selection_id: str
    plan_hash: str
    identity_digest: str
    group_by: tuple[str, ...]
    assignments: Mapping[str, str]
    group_assignments: Mapping[str, str]
    summaries: tuple[SplitRoleSummary, ...]
    class_by_modality: tuple[SplitCellCount, ...]
    warnings: tuple[str, ...]
    locked_roles: tuple[str, ...]
    # ("train", "validation") when a molecule-level validation fraction splits
    # training groups between them (`MLR-07`); test always stays isolated.
    shared_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_by", tuple(self.group_by))
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(dict(sorted(self.assignments.items()))),
        )
        object.__setattr__(
            self,
            "group_assignments",
            MappingProxyType(dict(sorted(self.group_assignments.items()))),
        )
        object.__setattr__(
            self,
            "summaries",
            tuple(sorted(self.summaries, key=lambda item: item.split)),
        )
        object.__setattr__(
            self,
            "class_by_modality",
            tuple(
                sorted(
                    self.class_by_modality,
                    key=lambda item: (item.split, item.modality, item.class_id),
                )
            ),
        )
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        object.__setattr__(self, "locked_roles", tuple(sorted(set(self.locked_roles))))

    def to_manifest(self, dataset: DatasetSnapshotManifest) -> SplitManifest:
        """Create the existing immutable split manifest after identity checks."""
        if dataset.selection.plan_hash != self.plan_hash:
            raise MLSplitPlanningError(
                "dataset snapshot plan hash does not match the split resolution"
            )
        dataset_uids = {item.molecule_uid for item in dataset.observations}
        assignment_uids = set(self.assignments)
        if dataset_uids != assignment_uids:
            missing = sorted(dataset_uids.difference(assignment_uids))
            unknown = sorted(assignment_uids.difference(dataset_uids))
            raise MLSplitPlanningError(
                "dataset snapshot membership does not match the split resolution: "
                f"missing={missing}, unknown={unknown}"
            )
        snapshot_records = []
        for observation in sorted(
            dataset.observations,
            key=lambda item: item.molecule_uid,
        ):
            snapshot_records.append(
                {
                    "molecule_uid": observation.molecule_uid,
                    "class_id": observation.class_id,
                    "modality": observation.modality,
                    "group_values": {
                        field: observation.value_for_group(field) for field in self.group_by
                    },
                }
            )
        if _sha256(snapshot_records) != self.identity_digest:
            raise MLSplitPlanningError(
                "dataset snapshot labels, modalities, or grouping metadata do not "
                "match the split resolution"
            )
        return SplitManifest.create(
            dataset=dataset,
            group_by=self.group_by,
            assignments=self.assignments,
            shared_roles=self.shared_roles,
        )

    def to_dry_run_dict(self) -> dict[str, Any]:
        """Return summaries and diagnostics without row-level assignments."""
        return {
            "split_name": self.split_name,
            "fold_name": self.fold_name,
            "strategy": self.strategy,
            "seed": self.seed,
            "resolution_id": self.resolution_id,
            "selection_id": self.selection_id,
            "plan_hash": self.plan_hash,
            "identity_digest": self.identity_digest,
            "group_by": list(self.group_by),
            "n_observations": len(self.assignments),
            "n_groups": len(self.group_assignments),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "class_by_modality": [cell.to_dict() for cell in self.class_by_modality],
            "warnings": list(self.warnings),
            "locked_roles": list(self.locked_roles),
            "shared_roles": list(self.shared_roles),
        }


@dataclass(frozen=True)
class _Group:
    group_id: str
    values: Mapping[str, str]
    token: str
    molecule_uids: tuple[str, ...]
    counts_by_class: Mapping[int, int]
    counts_by_cell: Mapping[tuple[int, str], int]

    @property
    def n_observations(self) -> int:
        return len(self.molecule_uids)


def _normalize_identity_table(
    selection: MLDataSelectionPlan,
    group_by: tuple[str, ...],
) -> pd.DataFrame:
    frame = selection.identity_table.copy(deep=False)
    required = {
        MOLECULE_UID_COLUMN,
        "class_id",
        "modality",
        *group_by,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MLSplitPlanningError(
            f"selection identity table lacks split metadata columns: {missing}"
        )
    if frame.empty:
        raise MLSplitPlanningError("cannot split an empty selection")
    if frame[MOLECULE_UID_COLUMN].astype(str).duplicated().any():
        raise MLSplitPlanningError("selection contains duplicate molecule identities")
    if frame["class_id"].isna().any():
        raise MLSplitPlanningError("group-aware training splits require complete class labels")
    normalized = frame.copy()
    normalized[MOLECULE_UID_COLUMN] = normalized[MOLECULE_UID_COLUMN].astype(str)
    normalized["modality"] = normalized["modality"].astype(str)
    normalized["class_id"] = normalized["class_id"].astype(int)
    for field in group_by:
        if normalized[field].isna().any():
            raise MLSplitPlanningError(f"grouping field {field!r} contains missing values")
        normalized[field] = normalized[field].astype(str)
    return normalized


def _display_group_token(
    rows: pd.DataFrame,
    group_by: tuple[str, ...],
    values: Mapping[str, str],
) -> str:
    parts = []
    for field in group_by:
        if field == "experiment_uid" and "experiment_id" in rows:
            aliases = tuple(sorted(set(rows["experiment_id"].astype(str))))
            if len(aliases) != 1:
                raise MLSplitPlanningError(
                    "one experiment_uid group maps to multiple experiment_id aliases"
                )
            parts.append(aliases[0])
        else:
            parts.append(values[field])
    return "/".join(parts)


def _groups(frame: pd.DataFrame, group_by: tuple[str, ...]) -> tuple[_Group, ...]:
    result = []
    token_to_values: dict[str, Mapping[str, str]] = {}
    grouper: str | list[str] = group_by[0] if len(group_by) == 1 else list(group_by)
    for raw_values, rows in frame.groupby(grouper, sort=True, dropna=False):
        value_tuple = (raw_values,) if len(group_by) == 1 else tuple(raw_values)
        values = MappingProxyType(
            {field: str(value) for field, value in zip(group_by, value_tuple)}
        )
        token = _display_group_token(rows, group_by, values)
        previous = token_to_values.setdefault(token, values)
        if dict(previous) != dict(values):
            raise MLSplitPlanningError(
                f"group display token {token!r} is ambiguous; choose different grouping fields"
            )
        cells = Counter(zip(rows["class_id"].astype(int), rows["modality"].astype(str)))
        result.append(
            _Group(
                group_id=_sha256({"group_by": list(group_by), "values": dict(values)}),
                values=values,
                token=token,
                molecule_uids=tuple(sorted(rows[MOLECULE_UID_COLUMN].astype(str))),
                counts_by_class=MappingProxyType(
                    dict(sorted(Counter(rows["class_id"].astype(int)).items()))
                ),
                counts_by_cell=MappingProxyType(dict(sorted(cells.items()))),
            )
        )
    return tuple(sorted(result, key=lambda item: item.group_id))


def _explicit_group_roles(
    groups: Sequence[_Group],
    spec: SplitSpec,
) -> dict[str, str]:
    declared = {
        "train": tuple(spec.train_groups),
        "validation": tuple(spec.validation_groups),
        "test": tuple(spec.test_groups),
    }
    by_token = {group.token: group for group in groups}
    specified = {token for tokens in declared.values() for token in tokens}
    unknown = sorted(specified.difference(by_token))
    missing = sorted(set(by_token).difference(specified))
    if missing or unknown:
        raise MLSplitPlanningError(
            "explicit split groups must cover the selected groups exactly: "
            f"missing={missing}, unknown={unknown}"
        )
    result = {}
    for role, tokens in declared.items():
        for token in tokens:
            result[by_token[token].group_id] = role
    return result


def _stratification_feasibility(
    groups: Sequence[_Group],
    *,
    n_roles: int,
) -> tuple[int, ...]:
    classes = tuple(sorted({class_id for group in groups for class_id in group.counts_by_class}))
    if len(classes) < 2:
        raise MLSplitPlanningError("stratified splitting requires at least two classes")
    if len(groups) < n_roles:
        raise MLSplitPlanningError(
            f"stratified splitting requires at least {n_roles} biological groups"
        )
    unavailable = {
        class_id: sum(class_id in group.counts_by_class for group in groups) for class_id in classes
    }
    impossible = {class_id: count for class_id, count in unavailable.items() if count < n_roles}
    if impossible:
        raise MLSplitPlanningError(
            "each class must occur in at least one biological group per split role; "
            f"group support={impossible}"
        )
    return classes


def _validate_role_class_support(
    groups: Sequence[_Group],
    group_roles: Mapping[str, str],
) -> None:
    classes = {class_id for group in groups for class_id in group.counts_by_class}
    missing = {}
    for role in sorted(set(group_roles.values())):
        present = {
            class_id
            for group in groups
            if group_roles[group.group_id] == role
            for class_id in group.counts_by_class
        }
        if present != classes:
            missing[role] = sorted(classes.difference(present))
    if missing:
        raise MLSplitPlanningError(
            f"every split role must contain every class; missing class support={missing}"
        )


def _identity_digest(
    frame: pd.DataFrame,
    group_by: tuple[str, ...],
) -> str:
    core = (MOLECULE_UID_COLUMN, "class_id", "modality")
    columns = [*core, *(field for field in group_by if field not in core)]
    ordered = frame.loc[:, columns].sort_values(MOLECULE_UID_COLUMN, kind="stable")
    records = [
        {
            "molecule_uid": str(record[MOLECULE_UID_COLUMN]),
            "class_id": int(record["class_id"]),
            "modality": str(record["modality"]),
            "group_values": {field: str(record[field]) for field in group_by},
        }
        for values in ordered.itertuples(index=False, name=None)
        for record in (dict(zip(columns, values)),)
    ]
    return _sha256(records)


def _state_score(
    totals: Mapping[str, int],
    group_totals: Mapping[str, int],
    class_totals: Mapping[str, Counter],
    cell_totals: Mapping[str, Counter],
    fractions: Mapping[str, float],
    classes: tuple[int, ...],
    cells: tuple[tuple[int, str], ...],
    *,
    n_observations: int,
    n_groups: int,
    global_classes: Counter,
    global_cells: Counter,
    added_group: _Group | None = None,
    added_role: str | None = None,
) -> tuple[int, float]:
    def increment(role: str) -> bool:
        return added_group is not None and role == added_role

    missing_classes = 0
    error = 0.0
    for role in _ROLES:
        fraction = fractions[role]
        observation_count = totals[role] + (added_group.n_observations if increment(role) else 0)
        group_count = group_totals[role] + int(increment(role))
        error += ((observation_count - n_observations * fraction) / n_observations) ** 2
        error += ((group_count - n_groups * fraction) / n_groups) ** 2
        for class_id in classes:
            class_count = class_totals[role][class_id] + (
                added_group.counts_by_class.get(class_id, 0) if increment(role) else 0
            )
            missing_classes += class_count == 0
            denominator = max(1, global_classes[class_id])
            error += ((class_count - denominator * fraction) / denominator) ** 2
        for cell in cells:
            cell_count = cell_totals[role][cell] + (
                added_group.counts_by_cell.get(cell, 0) if increment(role) else 0
            )
            denominator = max(1, global_cells[cell])
            error += ((cell_count - denominator * fraction) / denominator) ** 2
    return missing_classes, error


def _greedy_group_roles(
    groups: Sequence[_Group],
    spec: SplitSpec,
) -> dict[str, str]:
    fractions = {role: float(spec.fractions[role]) for role in _ROLES}
    classes = _stratification_feasibility(groups, n_roles=len(_ROLES))
    cells = tuple(sorted({cell for group in groups for cell in group.counts_by_cell}))
    n_observations = sum(group.n_observations for group in groups)
    n_groups = len(groups)
    global_classes = sum(
        (Counter(group.counts_by_class) for group in groups),
        Counter(),
    )
    global_cells = sum(
        (Counter(group.counts_by_cell) for group in groups),
        Counter(),
    )
    attempts = max(32, min(256, 4096 // len(groups)))
    best: tuple[tuple[int, float, str], dict[str, str]] | None = None
    for attempt in range(attempts):
        rng = np.random.default_rng(spec.seed + attempt * 104729)
        jitter = {group.group_id: float(rng.random()) for group in groups}
        role_jitter = {
            (group.group_id, role): float(rng.random()) for group in groups for role in _ROLES
        }

        def priority(group: _Group) -> tuple[float, int, float]:
            rarity = sum(
                count / max(1, global_cells[cell]) for cell, count in group.counts_by_cell.items()
            )
            return (-rarity, -group.n_observations, jitter[group.group_id])

        ordered = sorted(groups, key=priority)
        assignments: dict[str, str] = {}
        totals = {role: 0 for role in _ROLES}
        group_totals = {role: 0 for role in _ROLES}
        class_totals = {role: Counter() for role in _ROLES}
        cell_totals = {role: Counter() for role in _ROLES}
        for index, group in enumerate(ordered):
            remaining = len(ordered) - index
            represented = set(assignments.values())
            empty_roles = [role for role in _ROLES if role not in represented]
            candidates = empty_roles if remaining == len(empty_roles) else list(_ROLES)
            scored = []
            for role in candidates:
                missing, error = _state_score(
                    totals,
                    group_totals,
                    class_totals,
                    cell_totals,
                    fractions,
                    classes,
                    cells,
                    n_observations=n_observations,
                    n_groups=n_groups,
                    global_classes=global_classes,
                    global_cells=global_cells,
                    added_group=group,
                    added_role=role,
                )
                scored.append((missing, error, role_jitter[(group.group_id, role)], role))
            chosen = min(scored)[-1]
            assignments[group.group_id] = chosen
            totals[chosen] += group.n_observations
            group_totals[chosen] += 1
            class_totals[chosen].update(group.counts_by_class)
            cell_totals[chosen].update(group.counts_by_cell)
        missing, error = _state_score(
            totals,
            group_totals,
            class_totals,
            cell_totals,
            fractions,
            classes,
            cells,
            n_observations=n_observations,
            n_groups=n_groups,
            global_classes=global_classes,
            global_cells=global_cells,
        )
        tie_break = _canonical_json(dict(sorted(assignments.items())))
        candidate_score = (missing, error, tie_break)
        if best is None or candidate_score < best[0]:
            best = (candidate_score, assignments)
    if best is None or best[0][0]:
        raise MLSplitPlanningError(
            "unable to produce a group-disjoint split containing every class in "
            "train, validation, and test"
        )
    return best[1]


def _warnings_and_cells(
    frame: pd.DataFrame,
    groups: Sequence[_Group],
    group_roles: Mapping[str, str],
) -> tuple[tuple[SplitCellCount, ...], tuple[str, ...]]:
    classes = tuple(sorted(set(frame["class_id"].astype(int))))
    modalities = tuple(sorted(set(frame["modality"].astype(str))))
    all_cells = {(class_id, modality) for class_id in classes for modality in modalities}
    observed_global = set(zip(frame["class_id"].astype(int), frame["modality"].astype(str)))
    warnings = (
        [
            "dataset has absent class-by-modality cells: "
            + ", ".join(
                f"class={class_id}/modality={modality}"
                for class_id, modality in sorted(all_cells.difference(observed_global))
            )
        ]
        if all_cells.difference(observed_global)
        else []
    )
    rows = []
    group_lookup = {group.group_id: group for group in groups}
    represented_roles = tuple(sorted(set(group_roles.values())))
    for role in represented_roles:
        role_groups = [
            group_lookup[group_id]
            for group_id, assigned_role in group_roles.items()
            if assigned_role == role
        ]
        role_cells = sum(
            (Counter(group.counts_by_cell) for group in role_groups),
            Counter(),
        )
        missing = []
        for class_id, modality in sorted(all_cells):
            supporting_groups = sum(
                group.counts_by_cell.get((class_id, modality), 0) > 0 for group in role_groups
            )
            count = int(role_cells[(class_id, modality)])
            rows.append(
                SplitCellCount(
                    split=role,
                    modality=modality,
                    class_id=class_id,
                    n_observations=count,
                    n_groups=supporting_groups,
                )
            )
            if count == 0:
                missing.append(f"class={class_id}/modality={modality}")
        if missing:
            warnings.append(
                f"{role} split has absent class-by-modality cells: {', '.join(missing)}"
            )
    return tuple(rows), tuple(warnings)


def _resolution(
    *,
    split_name: str,
    fold_name: str | None,
    spec: SplitSpec,
    selection: MLDataSelectionPlan,
    frame: pd.DataFrame,
    groups: Sequence[_Group],
    group_roles: Mapping[str, str],
) -> MLSplitResolution:
    _validate_role_class_support(groups, group_roles)
    assignments = {
        molecule_uid: group_roles[group.group_id]
        for group in groups
        for molecule_uid in group.molecule_uids
    }
    summaries = []
    for role in sorted(set(group_roles.values())):
        role_groups = [group for group in groups if group_roles[group.group_id] == role]
        role_uids = {molecule_uid for group in role_groups for molecule_uid in group.molecule_uids}
        rows = frame.loc[frame[MOLECULE_UID_COLUMN].isin(role_uids)]
        summaries.append(
            SplitRoleSummary(
                split=role,
                n_observations=len(rows),
                n_groups=len(role_groups),
                counts_by_class=Counter(rows["class_id"].astype(int)),
                counts_by_modality=Counter(rows["modality"].astype(str)),
            )
        )
    cells, warnings = _warnings_and_cells(frame, groups, group_roles)
    represented = set(group_roles.values())
    resolution_id = _sha256(
        {
            "selection_id": selection.selection_id,
            "split_name": split_name,
            "fold_name": fold_name,
            "strategy": spec.strategy,
            "seed": spec.seed,
            "group_by": list(spec.group_by),
            "assignments": dict(sorted(assignments.items())),
        }
    )
    return MLSplitResolution(
        split_name=split_name,
        fold_name=fold_name,
        strategy=spec.strategy,
        seed=spec.seed,
        resolution_id=resolution_id,
        selection_id=selection.selection_id,
        plan_hash=selection.plan_hash,
        identity_digest=_identity_digest(frame, tuple(spec.group_by)),
        group_by=tuple(spec.group_by),
        assignments=assignments,
        group_assignments=group_roles,
        summaries=tuple(summaries),
        class_by_modality=cells,
        warnings=warnings,
        locked_roles=tuple(role for role in ("validation", "test") if role in represented),
    )


def plan_ml_splits(
    plan: MLPlan,
    split_name: str,
    selection: MLDataSelectionPlan,
) -> tuple[MLSplitResolution, ...]:
    """Resolve one declared split into one or more deterministic assignments.

    Explicit and stratified declarations produce one resolution. A
    leave-one-group-out declaration produces one train/test resolution per
    biological group.
    """
    if split_name not in plan.splits:
        raise MLSplitPlanningError(f"unknown split {split_name!r}")
    if selection.plan_hash != plan.plan_hash:
        raise MLSplitPlanningError("selection was resolved from a different ML plan")
    spec = plan.splits[split_name]
    group_by = tuple(spec.group_by)
    frame = _normalize_identity_table(selection, group_by)
    groups = _groups(frame, group_by)
    if spec.strategy == "explicit_groups":
        roles = _explicit_group_roles(groups, spec)
        return (
            _resolution(
                split_name=split_name,
                fold_name=None,
                spec=spec,
                selection=selection,
                frame=frame,
                groups=groups,
                group_roles=roles,
            ),
        )
    if spec.strategy == "stratified_group":
        roles = _greedy_group_roles(groups, spec)
        return (
            _resolution(
                split_name=split_name,
                fold_name=None,
                spec=spec,
                selection=selection,
                frame=frame,
                groups=groups,
                group_roles=roles,
            ),
        )

    classes = _stratification_feasibility(groups, n_roles=2)
    # `MLX-07`: groups that only ever train -- named, or (on request) every
    # group holding a single class, which cannot be scored as a test fold.
    named = set(spec.train_groups)
    unknown = sorted(named.difference(group.token for group in groups))
    if unknown:
        raise MLSplitPlanningError(f"train_groups name unknown groups: {unknown}")
    train_only = {group.group_id for group in groups if group.token in named}
    if spec.single_class_groups == "train":
        train_only |= {
            group.group_id
            for group in groups
            if len([count for count in group.counts_by_class.values() if count]) < len(classes)
        }
    held_out_groups = [group for group in groups if group.group_id not in train_only]
    if not held_out_groups:
        raise MLSplitPlanningError("leave-one-group-out has no group left to hold out")
    result = []
    for held_out in sorted(held_out_groups, key=lambda item: item.token):
        roles = {
            group.group_id: "test" if group.group_id == held_out.group_id else "train"
            for group in groups
        }
        fold_name = f"holdout={held_out.token}"
        if spec.validation_fraction is not None and spec.validation_by == "groups":
            roles = _validation_groups(groups, roles, spec, fold_name, set(classes))
        for role in ("train", "test"):
            role_classes = {
                class_id
                for group in groups
                if roles[group.group_id] == role
                for class_id in group.counts_by_class
            }
            if role_classes != set(classes):
                raise MLSplitPlanningError(
                    f"leave-one-group-out fold {held_out.token!r} cannot preserve "
                    f"all classes in {role}"
                )
        resolution = _resolution(
            split_name=split_name,
            fold_name=fold_name,
            spec=spec,
            selection=selection,
            frame=frame,
            groups=groups,
            group_roles=roles,
        )
        if spec.validation_fraction is not None and spec.validation_by == "molecules":
            resolution = with_validation_molecules(
                resolution,
                frame,
                groups,
                fraction=spec.validation_fraction,
                seed=spec.seed,
            )
        result.append(resolution)
    return tuple(result)


def _fold_rng(seed: int, fold_name: str | None) -> np.random.Generator:
    """A generator fixed by the split seed and the fold (stable across runs)."""
    digest = hashlib.sha256(f"{seed}:{fold_name}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _validation_groups(
    groups: Sequence[_Group],
    roles: Mapping[str, str],
    spec: SplitSpec,
    fold_name: str,
    classes: set[int],
) -> dict[str, str]:
    """Whole training groups as validation: ``round(fraction x training
    groups)`` of them (at least one, leaving one to train), drawn so that
    train and validation both hold every class."""
    training = sorted(
        (group for group in groups if roles[group.group_id] == "train"),
        key=lambda item: item.token,
    )
    n_validation = max(1, round(spec.validation_fraction * len(training)))
    if n_validation >= len(training):
        raise MLSplitPlanningError(
            f"{fold_name}: {len(training)} training groups cannot spare "
            f"{n_validation} for validation"
        )
    rng = _fold_rng(spec.seed, fold_name)
    for _attempt in range(200):
        order = rng.permutation(len(training))
        chosen = {training[i].group_id for i in order[:n_validation]}

        def support(members):
            return {c for group in members for c in group.counts_by_class}

        if (
            support(g for g in training if g.group_id in chosen) == classes
            and support(g for g in training if g.group_id not in chosen) == classes
        ):
            return {
                group_id: "validation" if group_id in chosen else role
                for group_id, role in roles.items()
            }
    raise MLSplitPlanningError(
        f"{fold_name}: no choice of {n_validation} validation groups keeps every class "
        "in both train and validation"
    )


def with_validation_molecules(
    resolution: MLSplitResolution,
    frame: pd.DataFrame,
    groups: Sequence[_Group],
    *,
    fraction: float,
    seed: int,
) -> MLSplitResolution:
    """Move ``fraction`` of the training molecules to validation, stratified by
    group x class (each cell rounded), seeded by the split seed and fold.

    Train and validation then share groups (recorded as ``shared_roles``);
    test is untouched.
    """
    group_of = {uid: group.group_id for group in groups for uid in group.molecule_uids}
    rows = frame.set_index(MOLECULE_UID_COLUMN)
    assignments = _molecule_validation(
        resolution.assignments, rows, group_of, fraction, seed, resolution.fold_name
    )
    summaries, cells_out = _molecule_summaries(rows, assignments, group_of)
    resolution_id = _sha256(
        {
            "resolution_id": resolution.resolution_id,
            "validation_fraction": fraction,
            "validation_by": "molecules",
            "assignments": dict(sorted(assignments.items())),
        }
    )
    return replace(
        resolution,
        resolution_id=resolution_id,
        assignments=assignments,
        summaries=summaries,
        class_by_modality=cells_out,
        locked_roles=tuple(sorted({*resolution.locked_roles, "validation"})),
        shared_roles=("train", "validation"),
    )


def _molecule_validation(
    assignments: Mapping[str, str],
    rows: pd.DataFrame,
    group_of: Mapping[str, str],
    fraction: float,
    seed: int,
    fold_name: str | None,
) -> dict[str, str]:
    training = sorted(uid for uid, role in assignments.items() if role == "train")
    cells = (
        pd.DataFrame(
            {
                "uid": training,
                "group": [group_of[uid] for uid in training],
                "class_id": rows.loc[training, "class_id"].astype(int).to_numpy(),
            }
        )
        .groupby(["group", "class_id"], sort=True)["uid"]
        .apply(list)
    )
    rng = _fold_rng(seed, fold_name)
    chosen: set[str] = set()
    for members in cells:
        take = int(round(fraction * len(members)))
        if take:
            chosen.update(rng.choice(np.asarray(members), size=take, replace=False).tolist())
    result = {uid: "validation" if uid in chosen else role for uid, role in assignments.items()}
    classes = set(rows["class_id"].astype(int))
    for role in ("train", "validation"):
        present = {int(rows.at[uid, "class_id"]) for uid, r in result.items() if r == role}
        if present != classes:
            raise MLSplitPlanningError(
                f"{fold_name}: validation fraction {fraction} leaves {role} "
                f"without classes {sorted(classes - present)}"
            )
    return result


FINAL_SPLIT_NAME = "final"


def final_split_assignments(
    plan: MLPlan, split_name: str, selection: MLDataSelectionPlan
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Roles for a final model fit on every selected row (`MLR-02`): all
    train, or -- when the split declares a validation fraction (`MLR-07`) --
    with that validation taken as in each fold (molecules by group x class,
    or whole groups). Returns the assignments and the shared roles."""
    spec = plan.splits[split_name]
    group_by = tuple(spec.group_by)
    frame = _normalize_identity_table(selection, group_by)
    groups = _groups(frame, group_by)
    roles = {group.group_id: "train" for group in groups}
    if spec.validation_fraction is not None and spec.validation_by == "groups":
        classes = {class_id for group in groups for class_id in group.counts_by_class}
        roles = _validation_groups(groups, roles, spec, FINAL_SPLIT_NAME, classes)
    assignments = {uid: roles[group.group_id] for group in groups for uid in group.molecule_uids}
    if spec.validation_fraction is not None and spec.validation_by == "molecules":
        group_of = {uid: group.group_id for group in groups for uid in group.molecule_uids}
        assignments = _molecule_validation(
            assignments,
            frame.set_index(MOLECULE_UID_COLUMN),
            group_of,
            spec.validation_fraction,
            spec.seed,
            FINAL_SPLIT_NAME,
        )
        return assignments, ("train", "validation")
    return assignments, ()


def _molecule_summaries(
    rows: pd.DataFrame, assignments: Mapping[str, str], group_of: Mapping[str, str]
) -> tuple[tuple[SplitRoleSummary, ...], tuple[SplitCellCount, ...]]:
    frame = pd.DataFrame(
        {
            "uid": list(assignments),
            "role": list(assignments.values()),
            "group": [group_of[uid] for uid in assignments],
            "class_id": rows.loc[list(assignments), "class_id"].astype(int).to_numpy(),
            "modality": rows.loc[list(assignments), "modality"].astype(str).to_numpy(),
        }
    )
    summaries = tuple(
        SplitRoleSummary(
            split=role,
            n_observations=len(part),
            n_groups=part["group"].nunique(),
            counts_by_class=Counter(part["class_id"].tolist()),
            counts_by_modality=Counter(part["modality"].tolist()),
        )
        for role, part in frame.groupby("role", sort=True)
    )
    cells = []
    for role, part in frame.groupby("role", sort=True):
        for class_id in sorted(frame["class_id"].unique()):
            for modality in sorted(frame["modality"].unique()):
                cell = part[(part["class_id"] == class_id) & (part["modality"] == modality)]
                cells.append(
                    SplitCellCount(
                        split=role,
                        modality=modality,
                        class_id=int(class_id),
                        n_observations=len(cell),
                        n_groups=cell["group"].nunique(),
                    )
                )
    return summaries, tuple(cells)
