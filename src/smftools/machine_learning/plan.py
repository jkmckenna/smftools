"""Strict, versioned declarations for smftools machine-learning workflows.

The plan describes user intent only. Loading and validating a plan does not read
experiment matrices, resolve output paths, fit models, or import an orchestration
framework.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

ML_PLAN_SCHEMA_VERSION = 1
SUPPORTED_MODALITIES = frozenset({"conversion", "deaminase", "direct"})
SUPPORTED_JOB_ACTIONS = frozenset({"apply", "evaluate", "explain", "plot", "train"})


class MLPlanValidationError(ValueError):
    """Raised when an ML plan does not satisfy its declared schema."""


def _fail(path: str, message: str) -> None:
    raise MLPlanValidationError(f"{path}: {message}")


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(path, "keys must be strings")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        _fail(path, f"contains unknown fields: {unknown}")
    missing = sorted(required.difference(value))
    if missing:
        _fail(path, f"is missing required fields: {missing}")


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        _fail(f"{path}.{key}", "must be a non-empty string")
    return item.strip()


def _optional_string(value: Mapping[str, Any], key: str, path: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        _fail(f"{path}.{key}", "must be a non-empty string or null")
    return item.strip()


def _string_tuple(value: Any, path: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not all(isinstance(item, str) for item in value):
            _fail(path, "must contain only strings")
        result = tuple(item.strip() for item in value)
    else:
        _fail(path, "must be a sequence of strings")
    if any(not item for item in result):
        _fail(path, "cannot contain empty values")
    if len(set(result)) != len(result):
        _fail(path, "cannot contain duplicate values")
    if required and not result:
        _fail(path, "must contain at least one value")
    return result


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            _fail(path, "mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    _fail(path, f"contains a non-JSON value of type {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _thaw(getattr(value, field.name)) for field in fields(value)}
    return value


@dataclass(frozen=True)
class ScopeSpec:
    """Experiment or project ownership declared independently of filesystem paths."""

    kind: str
    set_name: str | None = None


@dataclass(frozen=True)
class SelectionSpec:
    """Stable identifiers included in or excluded from a dataset."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicalChannelSource:
    """One modality-specific physical matrix source for a biological channel."""

    modality: str
    stage: str
    layer: str
    site_context: str


@dataclass(frozen=True)
class ChannelSpec:
    """One ordered biological input channel and its physical source mappings."""

    name: str
    biological_role: str
    sources: tuple[PhysicalChannelSource, ...]


@dataclass(frozen=True)
class LabelSpec:
    """Explicit observation-label vocabulary for supervised jobs."""

    column: str
    classes: Mapping[str, int]
    source: str = "obs"
    missing: str = "drop"
    positive_class: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Named selection and ordered input-channel declaration."""

    modalities: tuple[str, ...]
    channels: tuple[ChannelSpec, ...]
    channel_policy: str
    experiments: SelectionSpec = SelectionSpec()
    samples: SelectionSpec = SelectionSpec()
    references: tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    labels: LabelSpec | None = None


@dataclass(frozen=True)
class SplitSpec:
    """Biological-group split policy, separate from balancing."""

    strategy: str
    group_by: tuple[str, ...]
    train_groups: tuple[str, ...] = ()
    validation_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()
    fractions: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    seed: int = 0


@dataclass(frozen=True)
class BalanceRoleSpec:
    """Balancing method for one split role."""

    method: str


@dataclass(frozen=True)
class BalancingSpec:
    """Training balancing with natural validation and test prevalence."""

    train: BalanceRoleSpec = BalanceRoleSpec("natural")
    validation: BalanceRoleSpec = BalanceRoleSpec("natural")
    test: BalanceRoleSpec = BalanceRoleSpec("natural")


@dataclass(frozen=True)
class ModelSpec:
    """Backend-specific estimator family or neural recipe declaration."""

    backend: str
    family: str | None = None
    recipe: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    overrides: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    initialization: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({"kind": "scratch"})
    )


@dataclass(frozen=True)
class JobSpec:
    """One train, apply, evaluate, explain, or plot request."""

    action: str
    dataset: str | None = None
    split: str | None = None
    balancing: str | None = None
    models: tuple[str, ...] = ()
    model: str | None = None
    source_job: str | None = None
    runs: tuple[str, ...] = ()
    evaluate: tuple[str, ...] = ()
    explain: tuple[str, ...] = ()
    plots: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class TrackingSpec:
    """Optional tracker request; local artifacts remain authoritative."""

    provider: str = "none"
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class MLPlan:
    """Fully resolved, immutable version-1 ML plan."""

    schema_version: int
    scope: ScopeSpec
    datasets: Mapping[str, DatasetSpec]
    splits: Mapping[str, SplitSpec]
    balancing: Mapping[str, BalancingSpec]
    models: Mapping[str, ModelSpec]
    jobs: Mapping[str, JobSpec]
    tracking: TrackingSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable resolved representation."""
        payload = _thaw(self)
        payload["scope"]["set"] = payload["scope"].pop("set_name")
        if payload["tracking"] is None:
            payload.pop("tracking")
        return payload

    def canonical_json(self) -> str:
        """Return stable canonical JSON for identity and persistence."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def plan_hash(self) -> str:
        """Return the SHA-256 identity of the resolved plan."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _parse_scope(raw: Any) -> ScopeSpec:
    path = "scope"
    value = _as_mapping(raw, path)
    _check_keys(value, path=path, allowed={"kind", "set"}, required={"kind"})
    kind = _required_string(value, "kind", path).lower()
    if kind not in {"experiment", "project"}:
        _fail(f"{path}.kind", "must be 'experiment' or 'project'")
    set_name = _optional_string(value, "set", path)
    if kind == "experiment" and set_name is not None:
        _fail(f"{path}.set", "is only valid for project scope")
    return ScopeSpec(kind=kind, set_name=set_name)


def _parse_selection(raw: Any, path: str) -> SelectionSpec:
    if raw is None:
        return SelectionSpec()
    value = _as_mapping(raw, path)
    _check_keys(value, path=path, allowed={"include", "exclude"})
    include = _string_tuple(value.get("include"), f"{path}.include")
    exclude = _string_tuple(value.get("exclude"), f"{path}.exclude")
    overlap = sorted(set(include).intersection(exclude))
    if overlap:
        _fail(path, f"includes and excludes the same identifiers: {overlap}")
    return SelectionSpec(include=include, exclude=exclude)


def _parse_source(raw: Any, path: str) -> PhysicalChannelSource:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={"modality", "stage", "layer", "site_context"},
        required={"modality", "stage", "layer", "site_context"},
    )
    modality = _required_string(value, "modality", path).lower()
    if modality not in SUPPORTED_MODALITIES:
        _fail(f"{path}.modality", f"must be one of {sorted(SUPPORTED_MODALITIES)}")
    return PhysicalChannelSource(
        modality=modality,
        stage=_required_string(value, "stage", path),
        layer=_required_string(value, "layer", path),
        site_context=_required_string(value, "site_context", path),
    )


def _parse_channel(raw: Any, path: str) -> ChannelSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={"name", "biological_role", "sources"},
        required={"name", "biological_role", "sources"},
    )
    sources_raw = value["sources"]
    if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
        _fail(f"{path}.sources", "must be a sequence")
    sources = tuple(
        _parse_source(source, f"{path}.sources[{index}]")
        for index, source in enumerate(sources_raw)
    )
    if not sources:
        _fail(f"{path}.sources", "must contain at least one physical source")
    identities = [(source.modality, source.stage, source.layer) for source in sources]
    if len(set(identities)) != len(identities):
        _fail(f"{path}.sources", "contains duplicate physical source mappings")
    return ChannelSpec(
        name=_required_string(value, "name", path),
        biological_role=_required_string(value, "biological_role", path),
        sources=sources,
    )


def _default_channels(modality: str) -> tuple[ChannelSpec, ...]:
    if modality == "deaminase":
        return (
            ChannelSpec(
                name="accessibility",
                biological_role="accessibility",
                sources=(
                    PhysicalChannelSource(
                        modality="deaminase",
                        stage="preprocess",
                        layer="C_site_binary",
                        site_context="C",
                    ),
                ),
            ),
        )
    if modality == "conversion":
        return (
            ChannelSpec(
                name="accessibility",
                biological_role="accessibility",
                sources=(
                    PhysicalChannelSource(
                        modality="conversion",
                        stage="preprocess",
                        layer="GpC_site_binary",
                        site_context="GpC",
                    ),
                ),
            ),
            ChannelSpec(
                name="endogenous_methylation",
                biological_role="endogenous_methylation",
                sources=(
                    PhysicalChannelSource(
                        modality="conversion",
                        stage="preprocess",
                        layer="CpG_site_binary",
                        site_context="CpG",
                    ),
                ),
            ),
        )
    _fail(
        "datasets.channels",
        "direct-modality channels must be declared explicitly because A, GpC, and CpG roles vary",
    )


def _parse_label(raw: Any, path: str) -> LabelSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={"source", "column", "classes", "missing", "positive_class"},
        required={"column", "classes"},
    )
    source = value.get("source", "obs")
    if not isinstance(source, str):
        _fail(f"{path}.source", "must be a string")
    source = source.strip()
    if source != "obs":
        _fail(f"{path}.source", "only 'obs' is currently supported")
    classes_raw = _as_mapping(value["classes"], f"{path}.classes")
    if len(classes_raw) < 2:
        _fail(f"{path}.classes", "must define at least two classes")
    classes: dict[str, int] = {}
    for name, class_id in classes_raw.items():
        if not name.strip():
            _fail(f"{path}.classes", "class names cannot be empty")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            _fail(f"{path}.classes.{name}", "class IDs must be integers")
        classes[name] = class_id
    if len(set(classes.values())) != len(classes):
        _fail(f"{path}.classes", "class IDs must be unique")
    missing = value.get("missing", "drop")
    if not isinstance(missing, str):
        _fail(f"{path}.missing", "must be a string")
    missing = missing.strip().lower()
    if missing not in {"drop", "error"}:
        _fail(f"{path}.missing", "must be 'drop' or 'error'")
    positive_class = _optional_string(value, "positive_class", path)
    if positive_class is not None and positive_class not in classes:
        _fail(f"{path}.positive_class", "must name one of the declared classes")
    return LabelSpec(
        column=_required_string(value, "column", path),
        classes=MappingProxyType(classes),
        source=source,
        missing=missing,
        positive_class=positive_class,
    )


def _parse_dataset(raw: Any, path: str) -> DatasetSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={
            "modalities",
            "channel_policy",
            "channels",
            "experiments",
            "samples",
            "references",
            "filters",
            "labels",
        },
        required={"modalities"},
    )
    modalities = tuple(
        modality.lower()
        for modality in _string_tuple(value["modalities"], f"{path}.modalities", required=True)
    )
    unknown_modalities = sorted(set(modalities).difference(SUPPORTED_MODALITIES))
    if unknown_modalities:
        _fail(f"{path}.modalities", f"contains unsupported modalities: {unknown_modalities}")

    policy_raw = value.get("channel_policy")
    if policy_raw is None:
        if len(modalities) > 1:
            _fail(
                f"{path}.channel_policy",
                "is required for mixed-modality datasets ('harmonized' or 'union')",
            )
        channel_policy = "single_modality"
    else:
        if not isinstance(policy_raw, str):
            _fail(f"{path}.channel_policy", "must be a string")
        channel_policy = policy_raw.strip().lower()
    allowed_policies = {"single_modality"} if len(modalities) == 1 else {"harmonized", "union"}
    if channel_policy not in allowed_policies:
        _fail(f"{path}.channel_policy", f"must be one of {sorted(allowed_policies)}")

    channels_raw = value.get("channels")
    if channels_raw is None:
        if len(modalities) != 1:
            _fail(f"{path}.channels", "must be explicit for mixed-modality datasets")
        channels = _default_channels(modalities[0])
    else:
        if not isinstance(channels_raw, Sequence) or isinstance(channels_raw, (str, bytes)):
            _fail(f"{path}.channels", "must be an ordered sequence")
        channels = tuple(
            _parse_channel(channel, f"{path}.channels[{index}]")
            for index, channel in enumerate(channels_raw)
        )
        if not channels:
            _fail(f"{path}.channels", "must contain at least one channel")
    channel_names = [channel.name for channel in channels]
    if len(set(channel_names)) != len(channel_names):
        _fail(f"{path}.channels", "channel names must be unique")
    selected_modalities = set(modalities)
    for index, channel in enumerate(channels):
        source_modalities = {source.modality for source in channel.sources}
        outside = sorted(source_modalities.difference(selected_modalities))
        if outside:
            _fail(
                f"{path}.channels[{index}].sources",
                f"references modalities not selected by the dataset: {outside}",
            )
        if channel_policy == "harmonized" and source_modalities != selected_modalities:
            missing = sorted(selected_modalities.difference(source_modalities))
            _fail(
                f"{path}.channels[{index}].sources",
                f"harmonized channels require a physical source for every modality; missing {missing}",
            )
    represented_modalities = {source.modality for channel in channels for source in channel.sources}
    missing_modalities = sorted(selected_modalities.difference(represented_modalities))
    if missing_modalities:
        _fail(
            f"{path}.channels",
            f"do not define any physical source for selected modalities: {missing_modalities}",
        )

    filters = _freeze_json(value.get("filters", {}), f"{path}.filters")
    if not isinstance(filters, Mapping):
        _fail(f"{path}.filters", "must be a mapping")
    labels = (
        _parse_label(value["labels"], f"{path}.labels") if value.get("labels") is not None else None
    )
    return DatasetSpec(
        modalities=modalities,
        channels=channels,
        channel_policy=channel_policy,
        experiments=_parse_selection(value.get("experiments"), f"{path}.experiments"),
        samples=_parse_selection(value.get("samples"), f"{path}.samples"),
        references=_string_tuple(value.get("references"), f"{path}.references"),
        filters=filters,
        labels=labels,
    )


def _parse_split(raw: Any, path: str) -> SplitSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={
            "strategy",
            "group_by",
            "train_groups",
            "validation_groups",
            "test_groups",
            "fractions",
            "seed",
        },
        required={"strategy", "group_by"},
    )
    strategy = _required_string(value, "strategy", path).lower()
    if strategy not in {
        "explicit_groups",
        "leave_one_group_out",
        "stratified_group",
    }:
        _fail(
            f"{path}.strategy",
            "must be 'explicit_groups', 'leave_one_group_out', or 'stratified_group'",
        )
    seed = value.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail(f"{path}.seed", "must be an integer")
    role_groups = {
        "train": _string_tuple(value.get("train_groups"), f"{path}.train_groups"),
        "validation": _string_tuple(value.get("validation_groups"), f"{path}.validation_groups"),
        "test": _string_tuple(value.get("test_groups"), f"{path}.test_groups"),
    }
    seen: dict[str, str] = {}
    for role, groups in role_groups.items():
        for group in groups:
            if group in seen:
                _fail(path, f"group {group!r} appears in both {seen[group]} and {role}")
            seen[group] = role

    fractions_raw = value.get("fractions", {})
    fractions_map = _as_mapping(fractions_raw, f"{path}.fractions")
    _check_keys(
        fractions_map,
        path=f"{path}.fractions",
        allowed={"train", "validation", "test"},
    )
    fractions: dict[str, float] = {}
    for role, fraction in fractions_map.items():
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            _fail(f"{path}.fractions.{role}", "must be numeric")
        fractions[role] = float(fraction)
    if strategy == "explicit_groups":
        if fractions:
            _fail(f"{path}.fractions", "cannot be combined with explicit_groups")
        if not all(role_groups.values()):
            _fail(path, "explicit_groups requires non-empty train, validation, and test groups")
    elif strategy == "stratified_group":
        if any(role_groups.values()):
            _fail(path, "stratified_group cannot include explicit role groups")
        if not fractions:
            fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
        if set(fractions) != {"train", "validation", "test"}:
            _fail(f"{path}.fractions", "must define train, validation, and test")
        if any(fraction <= 0 or fraction >= 1 for fraction in fractions.values()):
            _fail(f"{path}.fractions", "each fraction must be between zero and one")
        if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            _fail(f"{path}.fractions", "must sum to 1.0")
    else:
        if any(role_groups.values()) or fractions:
            _fail(
                path,
                "leave_one_group_out cannot include explicit role groups or fractions",
            )
    return SplitSpec(
        strategy=strategy,
        group_by=_string_tuple(value["group_by"], f"{path}.group_by", required=True),
        train_groups=role_groups["train"],
        validation_groups=role_groups["validation"],
        test_groups=role_groups["test"],
        fractions=MappingProxyType(fractions),
        seed=seed,
    )


def _parse_balance_role(raw: Any, path: str, *, training: bool) -> BalanceRoleSpec:
    value = _as_mapping(raw, path)
    _check_keys(value, path=path, allowed={"method"}, required={"method"})
    method = _required_string(value, "method", path).lower()
    allowed = (
        {"natural", "class_weight", "weighted_sampler", "downsample", "upsample"}
        if training
        else {"natural"}
    )
    if method not in allowed:
        _fail(f"{path}.method", f"must be one of {sorted(allowed)}")
    return BalanceRoleSpec(method=method)


def _parse_balancing(raw: Any, path: str) -> BalancingSpec:
    value = _as_mapping(raw, path)
    _check_keys(value, path=path, allowed={"train", "validation", "test"})
    return BalancingSpec(
        train=_parse_balance_role(
            value.get("train", {"method": "natural"}), f"{path}.train", training=True
        ),
        validation=_parse_balance_role(
            value.get("validation", {"method": "natural"}),
            f"{path}.validation",
            training=False,
        ),
        test=_parse_balance_role(
            value.get("test", {"method": "natural"}), f"{path}.test", training=False
        ),
    )


def _parse_model(raw: Any, path: str) -> ModelSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={"backend", "family", "recipe", "parameters", "overrides", "initialization"},
        required={"backend"},
    )
    backend = _required_string(value, "backend", path).lower()
    if backend not in {"sklearn", "torch"}:
        _fail(f"{path}.backend", "must be 'sklearn' or 'torch'")
    family = _optional_string(value, "family", path)
    recipe = _optional_string(value, "recipe", path)
    if backend == "sklearn" and (family is None or recipe is not None):
        _fail(path, "sklearn models require 'family' and cannot declare 'recipe'")
    if backend == "torch" and (recipe is None or family is not None):
        _fail(path, "torch models require 'recipe' and cannot declare 'family'")
    parameters = _freeze_json(value.get("parameters", {}), f"{path}.parameters")
    overrides = _freeze_json(value.get("overrides", {}), f"{path}.overrides")
    initialization = _freeze_json(
        value.get("initialization", {"kind": "scratch"}), f"{path}.initialization"
    )
    for key, item in {
        "parameters": parameters,
        "overrides": overrides,
        "initialization": initialization,
    }.items():
        if not isinstance(item, Mapping):
            _fail(f"{path}.{key}", "must be a mapping")
    return ModelSpec(
        backend=backend,
        family=family,
        recipe=recipe,
        parameters=parameters,
        overrides=overrides,
        initialization=initialization,
    )


def _parse_job(raw: Any, path: str) -> JobSpec:
    value = _as_mapping(raw, path)
    _check_keys(
        value,
        path=path,
        allowed={
            "action",
            "dataset",
            "split",
            "balancing",
            "models",
            "model",
            "source_job",
            "runs",
            "evaluate",
            "explain",
            "plots",
            "parameters",
        },
        required={"action"},
    )
    action = _required_string(value, "action", path).lower()
    if action not in SUPPORTED_JOB_ACTIONS:
        _fail(f"{path}.action", f"must be one of {sorted(SUPPORTED_JOB_ACTIONS)}")
    parameters = _freeze_json(value.get("parameters", {}), f"{path}.parameters")
    if not isinstance(parameters, Mapping):
        _fail(f"{path}.parameters", "must be a mapping")
    return JobSpec(
        action=action,
        dataset=_optional_string(value, "dataset", path),
        split=_optional_string(value, "split", path),
        balancing=_optional_string(value, "balancing", path),
        models=_string_tuple(value.get("models"), f"{path}.models"),
        model=_optional_string(value, "model", path),
        source_job=_optional_string(value, "source_job", path),
        runs=_string_tuple(value.get("runs"), f"{path}.runs"),
        evaluate=_string_tuple(value.get("evaluate"), f"{path}.evaluate"),
        explain=_string_tuple(value.get("explain"), f"{path}.explain"),
        plots=_string_tuple(value.get("plots"), f"{path}.plots"),
        parameters=parameters,
    )


def _parse_tracking(raw: Any) -> TrackingSpec:
    path = "tracking"
    value = _as_mapping(raw, path)
    _check_keys(value, path=path, allowed={"provider", "parameters"})
    provider = value.get("provider", "none")
    if not isinstance(provider, str) or not provider.strip():
        _fail(f"{path}.provider", "must be a non-empty string")
    provider = provider.strip().lower()
    parameters = _freeze_json(value.get("parameters", {}), f"{path}.parameters")
    if not isinstance(parameters, Mapping):
        _fail(f"{path}.parameters", "must be a mapping")
    return TrackingSpec(provider=provider, parameters=parameters)


def _parse_named(
    raw: Any,
    path: str,
    parser: Callable[[Any, str], Any],
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    value = _as_mapping(raw, path)
    if required and not value:
        _fail(path, "must contain at least one named declaration")
    parsed: dict[str, Any] = {}
    for name, item in value.items():
        if not name.strip():
            _fail(path, "names cannot be empty")
        parsed[name] = parser(item, f"{path}.{name}")
    return MappingProxyType(parsed)


def _validate_job_references(plan: MLPlan) -> None:
    for name, job in plan.jobs.items():
        path = f"jobs.{name}"
        if job.dataset is not None and job.dataset not in plan.datasets:
            _fail(f"{path}.dataset", f"references unknown dataset {job.dataset!r}")
        if job.split is not None and job.split not in plan.splits:
            _fail(f"{path}.split", f"references unknown split {job.split!r}")
        if job.balancing is not None and job.balancing not in plan.balancing:
            _fail(f"{path}.balancing", f"references unknown balancing policy {job.balancing!r}")
        for model in job.models:
            if model not in plan.models:
                _fail(f"{path}.models", f"references unknown model {model!r}")
        if (
            job.model is not None
            and not job.model.startswith("model:")
            and job.model not in plan.models
        ):
            _fail(f"{path}.model", f"references unknown model {job.model!r}")
        if job.source_job is not None:
            if job.source_job not in plan.jobs:
                _fail(f"{path}.source_job", f"references unknown job {job.source_job!r}")
            if job.source_job == name:
                _fail(f"{path}.source_job", "cannot reference itself")
        for run in job.runs:
            if not run.startswith("run:") and run not in plan.jobs:
                _fail(f"{path}.runs", f"references unknown run or job {run!r}")
            if run == name:
                _fail(f"{path}.runs", "cannot reference the job itself")

        if job.action == "train":
            if job.dataset is None or job.split is None or not job.models:
                _fail(path, "train requires dataset, split, and at least one model")
            if job.model is not None or job.source_job is not None or job.runs or job.plots:
                _fail(path, "train cannot declare model, source_job, runs, or plots")
            if plan.datasets[job.dataset].labels is None:
                _fail(f"{path}.dataset", "train requires a dataset with labels")
        elif job.action == "apply":
            if job.dataset is None or job.model is None:
                _fail(path, "apply requires dataset and model")
            if job.split or job.balancing or job.models or job.runs or job.evaluate or job.plots:
                _fail(
                    path,
                    "apply cannot declare split, balancing, models, runs, evaluate, or plots",
                )
            if job.source_job is not None and plan.jobs[job.source_job].action != "train":
                _fail(f"{path}.source_job", "must reference a train job")
        elif job.action == "evaluate":
            if job.dataset is None or job.source_job is None:
                _fail(path, "evaluate requires dataset and source_job")
            if plan.jobs[job.source_job].action not in {"apply", "train"}:
                _fail(f"{path}.source_job", "must reference an apply or train job")
            if job.split or job.balancing or job.models or job.model or job.runs or job.plots:
                _fail(
                    path,
                    "evaluate cannot declare split, balancing, models, model, runs, or plots",
                )
            if plan.datasets[job.dataset].labels is None:
                _fail(f"{path}.dataset", "evaluate requires a dataset with labels")
        elif job.action == "explain":
            if job.dataset is None or job.model is None or not job.explain:
                _fail(path, "explain requires dataset, model, and at least one explain method")
            if job.split or job.balancing or job.models or job.runs or job.evaluate or job.plots:
                _fail(
                    path,
                    "explain cannot declare split, balancing, models, runs, evaluate, or plots",
                )
            if job.source_job is not None and plan.jobs[job.source_job].action not in {
                "apply",
                "train",
            }:
                _fail(f"{path}.source_job", "must reference an apply or train job")
        elif job.action == "plot":
            if not job.runs or not job.plots:
                _fail(path, "plot requires at least one run and plot type")
            if (
                job.dataset
                or job.split
                or job.balancing
                or job.models
                or job.model
                or job.source_job
                or job.evaluate
                or job.explain
            ):
                _fail(
                    path,
                    "plot cannot declare dataset, split, balancing, models, model, "
                    "source_job, evaluate, or explain",
                )
            for run in job.runs:
                if not run.startswith("run:") and plan.jobs[run].action == "plot":
                    _fail(f"{path}.runs", "cannot reference another plot job as a run")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(previous, value)
        else:
            merged[key] = value
    return merged


def parse_ml_plan(
    raw: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> MLPlan:
    """Validate a plan mapping with ``defaults < raw < overrides`` precedence.

    Nested mappings are merged recursively. Sequences and scalar values in
    ``overrides`` replace the corresponding file values.
    """
    value = _as_mapping(raw, "plan")
    if overrides is not None:
        value = _deep_merge(value, _as_mapping(overrides, "overrides"))
    _check_keys(
        value,
        path="plan",
        allowed={
            "schema_version",
            "scope",
            "datasets",
            "splits",
            "balancing",
            "models",
            "jobs",
            "tracking",
        },
        required={"schema_version", "scope", "datasets", "splits", "models", "jobs"},
    )
    schema_version = value["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        _fail("schema_version", "must be an integer")
    if schema_version != ML_PLAN_SCHEMA_VERSION:
        _fail(
            "schema_version",
            f"unsupported version {schema_version}; supported version is {ML_PLAN_SCHEMA_VERSION}",
        )
    plan = MLPlan(
        schema_version=schema_version,
        scope=_parse_scope(value["scope"]),
        datasets=_parse_named(value["datasets"], "datasets", _parse_dataset),
        splits=_parse_named(value["splits"], "splits", _parse_split),
        balancing=_parse_named(
            value.get("balancing", {}),
            "balancing",
            _parse_balancing,
            required=False,
        ),
        models=_parse_named(value["models"], "models", _parse_model),
        jobs=_parse_named(value["jobs"], "jobs", _parse_job),
        tracking=(
            _parse_tracking(value["tracking"]) if value.get("tracking") is not None else None
        ),
    )
    _validate_job_references(plan)
    return plan


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("plan", f"contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Loading YAML ML plans requires PyYAML; install pyyaml or use JSON."
        ) from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        pairs = loader.construct_pairs(node, deep=deep)
        return _reject_duplicate_pairs(pairs)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except MLPlanValidationError:
        raise
    except yaml.YAMLError as exc:
        raise MLPlanValidationError(f"cannot parse YAML ML plan: {exc}") from exc


def load_ml_plan(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> MLPlan:
    """Load and validate a JSON or YAML ML plan.

    Args:
        path: Plan ending in ``.json``, ``.yaml``, or ``.yml``.
        overrides: Explicit values applied after file values. Nested mappings
            merge recursively; sequences and scalars replace file values.

    Returns:
        Fully resolved immutable plan.

    Raises:
        MLPlanValidationError: If parsing or schema validation fails.
        ImportError: If YAML is requested without PyYAML.
    """
    plan_path = Path(path)
    try:
        text = plan_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MLPlanValidationError(f"cannot read ML plan {plan_path}: {exc}") from exc
    suffix = plan_path.suffix.lower()
    try:
        if suffix == ".json":
            raw = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        elif suffix in {".yaml", ".yml"}:
            raw = _load_yaml(text)
        else:
            _fail("plan", "file extension must be .json, .yaml, or .yml")
    except MLPlanValidationError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise MLPlanValidationError(f"cannot parse ML plan {plan_path}: {exc}") from exc
    return parse_ml_plan(_as_mapping(raw, "plan"), overrides=overrides)
