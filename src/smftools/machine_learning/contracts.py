"""Versioned ML input, label, mask, and predictor capability contracts.

These immutable contracts describe resolved scientific interfaces. They do not
materialize matrices, fit models, resolve workspaces, or adapt legacy models.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .plan import ChannelSpec, DatasetSpec, LabelSpec, PhysicalChannelSource

ML_INPUT_SCHEMA_VERSION = 1
ML_LABEL_SCHEMA_VERSION = 1
ML_CAPABILITY_SCHEMA_VERSION = 1

INPUT_AXES = ("observation", "position", "channel")
MASK_KINDS = frozenset(
    {"observed", "availability", "design", "padding", "attention", "corruption", "loss"}
)
MASK_KIND_ORDER = (
    "observed",
    "availability",
    "design",
    "padding",
    "attention",
    "corruption",
    "loss",
)
EXECUTION_PHASES = frozenset({"train", "validation", "test", "inference"})
MASK_CONSUMERS = frozenset({"materializer", "predictor", "pretraining_task", "loss"})

_MASK_CONTRACTS = {
    "observed": {
        "true_means": "value_is_measured",
        "consumers": ("materializer", "predictor"),
        "phases": ("train", "validation", "test", "inference"),
        "default_axes": INPUT_AXES,
        "allowed_axes": {INPUT_AXES},
    },
    "availability": {
        "true_means": "channel_is_applicable",
        "consumers": ("materializer", "predictor"),
        "phases": ("train", "validation", "test", "inference"),
        "default_axes": ("observation", "channel"),
        "allowed_axes": {
            ("observation", "channel"),
            INPUT_AXES,
        },
    },
    "design": {
        "true_means": "value_is_enabled_by_design",
        "consumers": ("materializer", "predictor"),
        "phases": ("train", "validation", "test", "inference"),
        "default_axes": ("position", "channel"),
        "allowed_axes": {
            ("position", "channel"),
            INPUT_AXES,
        },
    },
    "padding": {
        "true_means": "position_is_padding",
        "consumers": ("predictor",),
        "phases": ("train", "validation", "test", "inference"),
        "default_axes": ("observation", "position"),
        "allowed_axes": {("observation", "position")},
    },
    "attention": {
        "true_means": "position_may_be_attended",
        "consumers": ("predictor",),
        "phases": ("train", "validation", "test", "inference"),
        "default_axes": ("observation", "position"),
        "allowed_axes": {("observation", "position")},
    },
    "corruption": {
        "true_means": "value_was_intentionally_corrupted",
        "consumers": ("pretraining_task",),
        "phases": ("train",),
        "default_axes": INPUT_AXES,
        "allowed_axes": {INPUT_AXES},
    },
    "loss": {
        "true_means": "element_contributes_to_loss",
        "consumers": ("loss",),
        "phases": ("train", "validation", "test"),
        "default_axes": INPUT_AXES,
        "allowed_axes": {
            ("observation",),
            ("observation", "position"),
            INPUT_AXES,
        },
    },
}


class MLContractError(ValueError):
    """Raised when a resolved ML contract is invalid."""


class InputCompatibilityError(MLContractError):
    """Raised before execution when two input schemas are incompatible."""


def _fail(path: str, message: str) -> None:
    raise MLContractError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(path, "keys must be strings")
    return value


def _keys(
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


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value.strip()


def _strings(value: Any, path: str, *, required: bool = False) -> tuple[str, ...]:
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
    if len(result) != len(set(result)):
        _fail(path, "cannot contain duplicates")
    if required and not result:
        _fail(path, "must contain at least one value")
    return result


def _version(value: Any, expected: int, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if value != expected:
        _fail(path, f"unsupported version {value}; supported version is {expected}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_to_dict(source: PhysicalChannelSource) -> dict[str, str]:
    return {
        "modality": source.modality,
        "stage": source.stage,
        "layer": source.layer,
        "site_context": source.site_context,
    }


def _source_from_dict(raw: Any, path: str) -> PhysicalChannelSource:
    value = _mapping(raw, path)
    _keys(
        value,
        path=path,
        allowed={"modality", "stage", "layer", "site_context"},
        required={"modality", "stage", "layer", "site_context"},
    )
    return PhysicalChannelSource(
        modality=_string(value["modality"], f"{path}.modality"),
        stage=_string(value["stage"], f"{path}.stage"),
        layer=_string(value["layer"], f"{path}.layer"),
        site_context=_string(value["site_context"], f"{path}.site_context"),
    )


@dataclass(frozen=True)
class InputChannelSchema:
    """One ordered biological channel in a resolved model tensor."""

    name: str
    biological_role: str
    sources: tuple[PhysicalChannelSource, ...]
    dtype: str
    transform_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sources",
            tuple(
                sorted(
                    self.sources,
                    key=lambda source: (
                        source.modality,
                        source.stage,
                        source.layer,
                        source.site_context,
                    ),
                )
            ),
        )
        _validate_channel(self, "channel")

    @classmethod
    def from_plan_channel(
        cls,
        channel: ChannelSpec,
        *,
        dtype: str = "float32",
        transform_id: str = "identity",
    ) -> "InputChannelSchema":
        """Resolve one ML-plan channel into a typed input channel."""
        result = cls(
            name=channel.name,
            biological_role=channel.biological_role,
            sources=channel.sources,
            dtype=dtype,
            transform_id=transform_id,
        )
        _validate_channel(result, "channel")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable channel record."""
        return {
            "name": self.name,
            "biological_role": self.biological_role,
            "sources": [_source_to_dict(source) for source in self.sources],
            "dtype": self.dtype,
            "transform_id": self.transform_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InputChannelSchema":
        """Validate and restore a serialized channel record."""
        path = "channel"
        value = _mapping(raw, path)
        _keys(
            value,
            path=path,
            allowed={"name", "biological_role", "sources", "dtype", "transform_id"},
            required={"name", "biological_role", "sources", "dtype", "transform_id"},
        )
        sources_raw = value["sources"]
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
            _fail(f"{path}.sources", "must be a sequence")
        result = cls(
            name=_string(value["name"], f"{path}.name"),
            biological_role=_string(value["biological_role"], f"{path}.biological_role"),
            sources=tuple(
                _source_from_dict(source, f"{path}.sources[{index}]")
                for index, source in enumerate(sources_raw)
            ),
            dtype=_string(value["dtype"], f"{path}.dtype"),
            transform_id=_string(value["transform_id"], f"{path}.transform_id"),
        )
        _validate_channel(result, path)
        return result


def _validate_channel(channel: InputChannelSchema, path: str) -> None:
    _string(channel.name, f"{path}.name")
    _string(channel.biological_role, f"{path}.biological_role")
    _string(channel.dtype, f"{path}.dtype")
    _string(channel.transform_id, f"{path}.transform_id")
    try:
        canonical_dtype = np.dtype(channel.dtype).name
    except TypeError as exc:
        _fail(f"{path}.dtype", f"is not a valid NumPy dtype: {exc}")
    if channel.dtype != canonical_dtype:
        _fail(f"{path}.dtype", f"must use canonical dtype name {canonical_dtype!r}")
    if not channel.sources:
        _fail(f"{path}.sources", "must contain at least one physical source")
    for index, source in enumerate(channel.sources):
        _string(source.modality, f"{path}.sources[{index}].modality")
        _string(source.stage, f"{path}.sources[{index}].stage")
        _string(source.layer, f"{path}.sources[{index}].layer")
        _string(source.site_context, f"{path}.sources[{index}].site_context")
    identities = [
        (source.modality, source.stage, source.layer, source.site_context)
        for source in channel.sources
    ]
    if len(identities) != len(set(identities)):
        _fail(f"{path}.sources", "cannot contain duplicate physical sources")


@dataclass(frozen=True)
class MaskSpec:
    """Named boolean-mask semantics with an unambiguous true polarity."""

    name: str
    kind: str
    axes: tuple[str, ...]
    true_means: str
    consumers: tuple[str, ...]
    phases: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "consumers", tuple(self.consumers))
        object.__setattr__(self, "phases", tuple(self.phases))
        _validate_mask(self, "mask")

    @classmethod
    def standard(
        cls,
        kind: str,
        *,
        name: str | None = None,
        axes: Sequence[str] | None = None,
    ) -> "MaskSpec":
        """Construct one canonical mask definition."""
        kind = _string(kind, "mask.kind").lower()
        if kind not in MASK_KINDS:
            _fail("mask.kind", f"must be one of {sorted(MASK_KINDS)}")
        contract = _MASK_CONTRACTS[kind]
        result = cls(
            name=name or kind,
            kind=kind,
            axes=tuple(axes) if axes is not None else contract["default_axes"],
            true_means=contract["true_means"],
            consumers=contract["consumers"],
            phases=contract["phases"],
        )
        _validate_mask(result, "mask")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mask record."""
        return {
            "name": self.name,
            "kind": self.kind,
            "axes": list(self.axes),
            "true_means": self.true_means,
            "consumers": list(self.consumers),
            "phases": list(self.phases),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MaskSpec":
        """Validate and restore a serialized mask record."""
        path = "mask"
        value = _mapping(raw, path)
        _keys(
            value,
            path=path,
            allowed={"name", "kind", "axes", "true_means", "consumers", "phases"},
            required={"name", "kind", "axes", "true_means", "consumers", "phases"},
        )
        result = cls(
            name=_string(value["name"], f"{path}.name"),
            kind=_string(value["kind"], f"{path}.kind").lower(),
            axes=_strings(value["axes"], f"{path}.axes", required=True),
            true_means=_string(value["true_means"], f"{path}.true_means"),
            consumers=_strings(value["consumers"], f"{path}.consumers", required=True),
            phases=_strings(value["phases"], f"{path}.phases", required=True),
        )
        _validate_mask(result, path)
        return result


def _validate_mask(mask: MaskSpec, path: str) -> None:
    _string(mask.name, f"{path}.name")
    if mask.kind not in MASK_KINDS:
        _fail(f"{path}.kind", f"must be one of {sorted(MASK_KINDS)}")
    contract = _MASK_CONTRACTS[mask.kind]
    if mask.axes not in contract["allowed_axes"]:
        _fail(
            f"{path}.axes",
            f"invalid axes for {mask.kind!r}; allowed: {sorted(contract['allowed_axes'])}",
        )
    if mask.true_means != contract["true_means"]:
        _fail(
            f"{path}.true_means",
            f"must be {contract['true_means']!r} for {mask.kind!r}",
        )
    if mask.consumers != contract["consumers"]:
        _fail(
            f"{path}.consumers",
            f"must be {list(contract['consumers'])} for {mask.kind!r}",
        )
    if mask.phases != contract["phases"]:
        _fail(
            f"{path}.phases",
            f"must be {list(contract['phases'])} for {mask.kind!r}",
        )


@dataclass(frozen=True)
class InputSchema:
    """Resolved tensor identity used for model compatibility checks."""

    schema_version: int
    modalities: tuple[str, ...]
    reference: str
    coordinate_system: str
    axes: tuple[str, ...]
    n_positions: int
    channels: tuple[InputChannelSchema, ...]
    masks: tuple[MaskSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", tuple(sorted(self.modalities)))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "channels", tuple(self.channels))
        order = {kind: index for index, kind in enumerate(MASK_KIND_ORDER)}
        object.__setattr__(
            self,
            "masks",
            tuple(
                sorted(
                    self.masks,
                    key=lambda mask: (order.get(mask.kind, len(order)), mask.name),
                )
            ),
        )
        _validate_input_schema(self)

    @classmethod
    def from_dataset(
        cls,
        dataset: DatasetSpec,
        *,
        reference: str,
        n_positions: int,
        coordinate_system: str = "reference_0_based",
        dtype: str = "float32",
        transforms: Mapping[str, str] | None = None,
        masks: Sequence[MaskSpec] | None = None,
    ) -> "InputSchema":
        """Resolve an ML-plan dataset declaration into a model input schema."""
        transforms = dict(transforms or {})
        unknown = sorted(set(transforms).difference(channel.name for channel in dataset.channels))
        if unknown:
            _fail("transforms", f"references unknown channels: {unknown}")
        channels = tuple(
            InputChannelSchema.from_plan_channel(
                channel,
                dtype=dtype,
                transform_id=transforms.get(channel.name, "identity"),
            )
            for channel in dataset.channels
        )
        result = cls(
            schema_version=ML_INPUT_SCHEMA_VERSION,
            modalities=tuple(sorted(dataset.modalities)),
            reference=reference,
            coordinate_system=coordinate_system,
            axes=INPUT_AXES,
            n_positions=n_positions,
            channels=channels,
            masks=tuple(
                masks
                if masks is not None
                else (
                    MaskSpec.standard("observed"),
                    MaskSpec.standard("availability"),
                    MaskSpec.standard(
                        "design",
                        axes=(
                            INPUT_AXES
                            if dataset.channel_policy == "union"
                            else ("position", "channel")
                        ),
                    ),
                )
            ),
        )
        _validate_input_schema(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable input schema."""
        return {
            "schema_version": self.schema_version,
            "modalities": list(self.modalities),
            "reference": self.reference,
            "coordinate_system": self.coordinate_system,
            "axes": list(self.axes),
            "n_positions": self.n_positions,
            "channels": [channel.to_dict() for channel in self.channels],
            "masks": [mask.to_dict() for mask in self.masks],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InputSchema":
        """Validate and restore a serialized input schema."""
        path = "input_schema"
        value = _mapping(raw, path)
        _keys(
            value,
            path=path,
            allowed={
                "schema_version",
                "modalities",
                "reference",
                "coordinate_system",
                "axes",
                "n_positions",
                "channels",
                "masks",
            },
            required={
                "schema_version",
                "modalities",
                "reference",
                "coordinate_system",
                "axes",
                "n_positions",
                "channels",
                "masks",
            },
        )
        channels_raw = value["channels"]
        masks_raw = value["masks"]
        if not isinstance(channels_raw, Sequence) or isinstance(channels_raw, (str, bytes)):
            _fail(f"{path}.channels", "must be a sequence")
        if not isinstance(masks_raw, Sequence) or isinstance(masks_raw, (str, bytes)):
            _fail(f"{path}.masks", "must be a sequence")
        n_positions = value["n_positions"]
        if isinstance(n_positions, bool) or not isinstance(n_positions, int):
            _fail(f"{path}.n_positions", "must be an integer")
        result = cls(
            schema_version=_version(
                value["schema_version"],
                ML_INPUT_SCHEMA_VERSION,
                f"{path}.schema_version",
            ),
            modalities=tuple(
                sorted(_strings(value["modalities"], f"{path}.modalities", required=True))
            ),
            reference=_string(value["reference"], f"{path}.reference"),
            coordinate_system=_string(value["coordinate_system"], f"{path}.coordinate_system"),
            axes=_strings(value["axes"], f"{path}.axes", required=True),
            n_positions=n_positions,
            channels=tuple(InputChannelSchema.from_dict(channel) for channel in channels_raw),
            masks=tuple(MaskSpec.from_dict(mask) for mask in masks_raw),
        )
        _validate_input_schema(result)
        return result

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return _canonical_json(self.to_dict())

    @property
    def schema_hash(self) -> str:
        """Return the SHA-256 identity of the complete schema."""
        return _sha256(self.to_dict())


def _validate_input_schema(schema: InputSchema) -> None:
    _version(schema.schema_version, ML_INPUT_SCHEMA_VERSION, "input_schema.schema_version")
    if not schema.modalities:
        _fail("input_schema.modalities", "must contain at least one modality")
    if len(schema.modalities) != len(set(schema.modalities)):
        _fail("input_schema.modalities", "cannot contain duplicates")
    if schema.modalities != tuple(sorted(schema.modalities)):
        _fail("input_schema.modalities", "must be sorted for canonical identity")
    if schema.axes != INPUT_AXES:
        _fail("input_schema.axes", f"must be {list(INPUT_AXES)}")
    if (
        isinstance(schema.n_positions, bool)
        or not isinstance(schema.n_positions, int)
        or schema.n_positions <= 0
    ):
        _fail("input_schema.n_positions", "must be a positive integer")
    _string(schema.reference, "input_schema.reference")
    _string(schema.coordinate_system, "input_schema.coordinate_system")
    if not schema.channels:
        _fail("input_schema.channels", "must contain at least one ordered channel")
    names = [channel.name for channel in schema.channels]
    if len(names) != len(set(names)):
        _fail("input_schema.channels", "channel names must be unique")
    selected_modalities = set(schema.modalities)
    represented_modalities: set[str] = set()
    for index, channel in enumerate(schema.channels):
        _validate_channel(channel, f"input_schema.channels[{index}]")
        source_modalities = {source.modality for source in channel.sources}
        outside = sorted(source_modalities.difference(selected_modalities))
        if outside:
            _fail(
                f"input_schema.channels[{index}].sources",
                f"references unselected modalities: {outside}",
            )
        represented_modalities.update(source_modalities)
    missing_modalities = sorted(selected_modalities.difference(represented_modalities))
    if missing_modalities:
        _fail(
            "input_schema.channels",
            f"have no physical source for selected modalities: {missing_modalities}",
        )
    mask_names = [mask.name for mask in schema.masks]
    mask_kinds = [mask.kind for mask in schema.masks]
    if len(mask_names) != len(set(mask_names)):
        _fail("input_schema.masks", "mask names must be unique")
    if len(mask_kinds) != len(set(mask_kinds)):
        _fail("input_schema.masks", "mask kinds must be unique")
    for index, mask in enumerate(schema.masks):
        _validate_mask(mask, f"input_schema.masks[{index}]")
    for required_kind in ("observed", "availability", "design"):
        if required_kind not in mask_kinds:
            _fail(
                "input_schema.masks",
                f"must declare the canonical {required_kind!r} mask",
            )
    has_inapplicable_channel = any(
        {source.modality for source in channel.sources} != selected_modalities
        for channel in schema.channels
    )
    if has_inapplicable_channel and "availability" not in mask_kinds:
        _fail(
            "input_schema.masks",
            "union-channel schemas require an availability mask",
        )


@dataclass(frozen=True)
class LabelSchema:
    """Explicit, ordered classification vocabulary independent of pandas categories."""

    schema_version: int
    source: str
    field: str
    task_type: str
    value_to_class: Mapping[str, int]
    class_order: tuple[str, ...]
    positive_class: str | None
    missing_action: str
    unknown_action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_to_class",
            MappingProxyType(dict(self.value_to_class)),
        )
        object.__setattr__(self, "class_order", tuple(self.class_order))
        _validate_label_schema(self)

    @classmethod
    def from_plan_label(cls, label: LabelSpec) -> "LabelSchema":
        """Resolve an ML-plan label declaration into a classification schema."""
        ordered = tuple(name for name, _ in sorted(label.classes.items(), key=lambda item: item[1]))
        positive_class = label.positive_class
        if len(ordered) == 2 and positive_class is None:
            positive_class = ordered[1]
        result = cls(
            schema_version=ML_LABEL_SCHEMA_VERSION,
            source=label.source,
            field=label.column,
            task_type="binary_classification" if len(ordered) == 2 else "multiclass_classification",
            value_to_class=MappingProxyType(dict(label.classes)),
            class_order=ordered,
            positive_class=positive_class,
            missing_action=label.missing,
            unknown_action="error",
        )
        _validate_label_schema(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable label schema."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "field": self.field,
            "task_type": self.task_type,
            "value_to_class": dict(self.value_to_class),
            "class_order": list(self.class_order),
            "positive_class": self.positive_class,
            "missing_action": self.missing_action,
            "unknown_action": self.unknown_action,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelSchema":
        """Validate and restore a serialized label schema."""
        path = "label_schema"
        value = _mapping(raw, path)
        _keys(
            value,
            path=path,
            allowed={
                "schema_version",
                "source",
                "field",
                "task_type",
                "value_to_class",
                "class_order",
                "positive_class",
                "missing_action",
                "unknown_action",
            },
            required={
                "schema_version",
                "source",
                "field",
                "task_type",
                "value_to_class",
                "class_order",
                "positive_class",
                "missing_action",
                "unknown_action",
            },
        )
        mapping = _mapping(value["value_to_class"], f"{path}.value_to_class")
        classes: dict[str, int] = {}
        for name, class_id in mapping.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                _fail(f"{path}.value_to_class.{name}", "class IDs must be integers")
            classes[name] = class_id
        positive = value["positive_class"]
        if positive is not None:
            positive = _string(positive, f"{path}.positive_class")
        result = cls(
            schema_version=_version(
                value["schema_version"],
                ML_LABEL_SCHEMA_VERSION,
                f"{path}.schema_version",
            ),
            source=_string(value["source"], f"{path}.source"),
            field=_string(value["field"], f"{path}.field"),
            task_type=_string(value["task_type"], f"{path}.task_type"),
            value_to_class=MappingProxyType(classes),
            class_order=_strings(value["class_order"], f"{path}.class_order", required=True),
            positive_class=positive,
            missing_action=_string(value["missing_action"], f"{path}.missing_action"),
            unknown_action=_string(value["unknown_action"], f"{path}.unknown_action"),
        )
        _validate_label_schema(result)
        return result

    def encode(self, values: Iterable[Any]) -> tuple[int | None, ...]:
        """Encode exact source values without deriving categorical codes."""
        encoded: list[int | None] = []
        for value in values:
            missing = value is None or (
                isinstance(value, (float, np.floating)) and math.isnan(float(value))
            )
            if missing:
                if self.missing_action == "error":
                    _fail("labels", "encountered a missing value")
                encoded.append(None)
                continue
            if not isinstance(value, str) or value not in self.value_to_class:
                if self.unknown_action == "error":
                    _fail("labels", f"encountered unknown value {value!r}")
                encoded.append(None)
                continue
            encoded.append(self.value_to_class[value])
        return tuple(encoded)

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return _canonical_json(self.to_dict())

    @property
    def schema_hash(self) -> str:
        """Return the SHA-256 identity of the complete label schema."""
        return _sha256(self.to_dict())


def _validate_label_schema(schema: LabelSchema) -> None:
    _version(schema.schema_version, ML_LABEL_SCHEMA_VERSION, "label_schema.schema_version")
    if schema.source != "obs":
        _fail("label_schema.source", "only 'obs' is currently supported")
    _string(schema.field, "label_schema.field")
    expected_task = (
        "binary_classification" if len(schema.class_order) == 2 else "multiclass_classification"
    )
    if len(schema.class_order) < 2:
        _fail("label_schema.class_order", "must contain at least two classes")
    if len(schema.class_order) != len(set(schema.class_order)):
        _fail("label_schema.class_order", "cannot contain duplicates")
    if not all(isinstance(name, str) and name for name in schema.value_to_class):
        _fail("label_schema.value_to_class", "keys must be non-empty strings")
    if schema.task_type != expected_task:
        _fail("label_schema.task_type", f"must be {expected_task!r}")
    if set(schema.value_to_class) != set(schema.class_order):
        _fail(
            "label_schema.value_to_class",
            "keys must exactly match class_order",
        )
    expected_ids = {name: index for index, name in enumerate(schema.class_order)}
    if dict(schema.value_to_class) != expected_ids:
        _fail(
            "label_schema.value_to_class",
            "class IDs must be contiguous and match class_order",
        )
    if schema.task_type == "binary_classification":
        if schema.positive_class not in schema.class_order:
            _fail(
                "label_schema.positive_class",
                "binary classification requires a declared class",
            )
    elif schema.positive_class is not None and schema.positive_class not in schema.class_order:
        _fail("label_schema.positive_class", "must name a declared class or be null")
    if schema.missing_action not in {"drop", "error"}:
        _fail("label_schema.missing_action", "must be 'drop' or 'error'")
    if schema.unknown_action not in {"drop", "error"}:
        _fail("label_schema.unknown_action", "must be 'drop' or 'error'")


@dataclass(frozen=True)
class PredictorCapabilities:
    """Backend-neutral feature flags used before model dispatch."""

    schema_version: int
    backend: str
    probability_output: bool
    incremental_fit: bool
    sample_weights: bool
    position_masks: bool
    gradients: bool
    convolutional_layers: bool
    attention_data: bool
    supported_mask_kinds: tuple[str, ...] = ()
    required_mask_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_mask_kinds",
            tuple(sorted(self.supported_mask_kinds)),
        )
        object.__setattr__(
            self,
            "required_mask_kinds",
            tuple(sorted(self.required_mask_kinds)),
        )
        _validate_capabilities(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable capability record."""
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "probability_output": self.probability_output,
            "incremental_fit": self.incremental_fit,
            "sample_weights": self.sample_weights,
            "position_masks": self.position_masks,
            "gradients": self.gradients,
            "convolutional_layers": self.convolutional_layers,
            "attention_data": self.attention_data,
            "supported_mask_kinds": list(self.supported_mask_kinds),
            "required_mask_kinds": list(self.required_mask_kinds),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredictorCapabilities":
        """Validate and restore a serialized capability record."""
        path = "capabilities"
        value = _mapping(raw, path)
        boolean_fields = {
            "probability_output",
            "incremental_fit",
            "sample_weights",
            "position_masks",
            "gradients",
            "convolutional_layers",
            "attention_data",
        }
        _keys(
            value,
            path=path,
            allowed={
                "schema_version",
                "backend",
                "supported_mask_kinds",
                "required_mask_kinds",
                *boolean_fields,
            },
            required={
                "schema_version",
                "backend",
                "supported_mask_kinds",
                "required_mask_kinds",
                *boolean_fields,
            },
        )
        for field in boolean_fields:
            if not isinstance(value[field], bool):
                _fail(f"{path}.{field}", "must be boolean")
        result = cls(
            schema_version=_version(
                value["schema_version"],
                ML_CAPABILITY_SCHEMA_VERSION,
                f"{path}.schema_version",
            ),
            backend=_string(value["backend"], f"{path}.backend"),
            probability_output=value["probability_output"],
            incremental_fit=value["incremental_fit"],
            sample_weights=value["sample_weights"],
            position_masks=value["position_masks"],
            gradients=value["gradients"],
            convolutional_layers=value["convolutional_layers"],
            attention_data=value["attention_data"],
            supported_mask_kinds=_strings(
                value["supported_mask_kinds"],
                f"{path}.supported_mask_kinds",
            ),
            required_mask_kinds=_strings(
                value["required_mask_kinds"],
                f"{path}.required_mask_kinds",
            ),
        )
        _validate_capabilities(result)
        return result

    def canonical_json(self) -> str:
        """Return stable canonical JSON."""
        return _canonical_json(self.to_dict())

    @property
    def schema_hash(self) -> str:
        """Return the SHA-256 identity of the capability declaration."""
        return _sha256(self.to_dict())


def _validate_capabilities(capabilities: PredictorCapabilities) -> None:
    _version(
        capabilities.schema_version,
        ML_CAPABILITY_SCHEMA_VERSION,
        "capabilities.schema_version",
    )
    _string(capabilities.backend, "capabilities.backend")
    for field in (
        "probability_output",
        "incremental_fit",
        "sample_weights",
        "position_masks",
        "gradients",
        "convolutional_layers",
        "attention_data",
    ):
        if not isinstance(getattr(capabilities, field), bool):
            _fail(f"capabilities.{field}", "must be boolean")
    if len(capabilities.supported_mask_kinds) != len(set(capabilities.supported_mask_kinds)):
        _fail("capabilities.supported_mask_kinds", "cannot contain duplicates")
    if len(capabilities.required_mask_kinds) != len(set(capabilities.required_mask_kinds)):
        _fail("capabilities.required_mask_kinds", "cannot contain duplicates")
    supported = set(capabilities.supported_mask_kinds)
    required = set(capabilities.required_mask_kinds)
    unknown = sorted(supported.difference(MASK_KINDS))
    if unknown:
        _fail("capabilities.supported_mask_kinds", f"contains unknown kinds: {unknown}")
    if not required.issubset(supported):
        _fail(
            "capabilities.required_mask_kinds",
            f"must be a subset of supported masks; unsupported: {sorted(required - supported)}",
        )
    position_kinds = {"observed", "availability", "design", "padding", "attention"}
    if supported.intersection(position_kinds) and not capabilities.position_masks:
        _fail(
            "capabilities.position_masks",
            "must be true when position-related masks are supported",
        )


def assert_input_compatible(expected: InputSchema, actual: InputSchema) -> None:
    """Raise an actionable error unless two resolved schemas match exactly."""
    if expected.schema_hash == actual.schema_hash:
        return
    differences: list[str] = []
    for field in (
        "schema_version",
        "modalities",
        "reference",
        "coordinate_system",
        "axes",
        "n_positions",
    ):
        if getattr(expected, field) != getattr(actual, field):
            differences.append(field)
    if expected.channels != actual.channels:
        differences.append("ordered channel/source/role/dtype/transform definitions")
    if expected.masks != actual.masks:
        differences.append("mask definitions")
    raise InputCompatibilityError(
        "input schema is incompatible; differing fields: " + ", ".join(differences)
    )


def validate_mask_usage(
    schema: InputSchema,
    mask_names: Iterable[str],
    *,
    consumer: str,
    phase: str,
) -> tuple[MaskSpec, ...]:
    """Validate that named masks may be consumed in one execution phase."""
    if consumer not in MASK_CONSUMERS:
        _fail("consumer", f"must be one of {sorted(MASK_CONSUMERS)}")
    if phase not in EXECUTION_PHASES:
        _fail("phase", f"must be one of {sorted(EXECUTION_PHASES)}")
    by_name = {mask.name: mask for mask in schema.masks}
    names = tuple(mask_names)
    if len(names) != len(set(names)):
        _fail("mask_names", "cannot contain duplicates")
    selected: list[MaskSpec] = []
    for name in names:
        if name not in by_name:
            _fail("mask_names", f"references undeclared mask {name!r}")
        mask = by_name[name]
        if consumer not in mask.consumers:
            _fail(
                f"mask_names.{name}",
                f"{mask.kind!r} masks are not consumed by {consumer!r}",
            )
        if phase not in mask.phases:
            _fail(
                f"mask_names.{name}",
                f"{mask.kind!r} masks are not valid during {phase!r}",
            )
        selected.append(mask)
    return tuple(selected)


def validate_predictor_masks(
    schema: InputSchema,
    capabilities: PredictorCapabilities,
    mask_names: Iterable[str],
    *,
    phase: str,
) -> tuple[MaskSpec, ...]:
    """Reject masks a predictor cannot consume or required masks it did not receive."""
    _validate_capabilities(capabilities)
    selected = validate_mask_usage(
        schema,
        mask_names,
        consumer="predictor",
        phase=phase,
    )
    selected_kinds = {mask.kind for mask in selected}
    supported = set(capabilities.supported_mask_kinds)
    unsupported = sorted(selected_kinds.difference(supported))
    if unsupported:
        _fail(
            "predictor_masks",
            f"backend {capabilities.backend!r} does not support masks: {unsupported}",
        )
    missing = sorted(set(capabilities.required_mask_kinds).difference(selected_kinds))
    if missing:
        _fail("predictor_masks", f"required masks were not provided: {missing}")
    return selected


def validate_mask_arrays(
    schema: InputSchema,
    arrays: Mapping[str, Any],
    *,
    batch_size: int,
    require_all: bool = True,
) -> None:
    """Validate declared mask names, boolean dtypes, and axis-derived shapes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        _fail("batch_size", "must be a positive integer")
    by_name = {mask.name: mask for mask in schema.masks}
    unknown = sorted(set(arrays).difference(by_name))
    if unknown:
        _fail("mask_arrays", f"contains undeclared masks: {unknown}")
    if require_all:
        missing = sorted(set(by_name).difference(arrays))
        if missing:
            _fail("mask_arrays", f"is missing declared masks: {missing}")
    axis_sizes = {
        "observation": batch_size,
        "position": schema.n_positions,
        "channel": len(schema.channels),
    }
    for name, array in arrays.items():
        mask = by_name[name]
        expected_shape = tuple(axis_sizes[axis] for axis in mask.axes)
        shape = tuple(getattr(array, "shape", ()))
        if shape != expected_shape:
            _fail(
                f"mask_arrays.{name}",
                f"shape {shape} does not match axes {mask.axes} -> {expected_shape}",
            )
        dtype = str(getattr(array, "dtype", ""))
        if dtype != "bool" and not dtype.endswith(".bool"):
            _fail(f"mask_arrays.{name}", f"dtype must be boolean, got {dtype!r}")


def validate_mask_relationships(schema: InputSchema, arrays: Mapping[str, Any]) -> None:
    """Validate availability, observation, corruption, padding, and attention invariants."""
    by_kind = {mask.kind: mask for mask in schema.masks}

    def array_for(kind: str) -> np.ndarray | None:
        mask = by_kind.get(kind)
        if mask is None or mask.name not in arrays:
            return None
        return np.asarray(arrays[mask.name], dtype=bool)

    observed = array_for("observed")
    availability = array_for("availability")
    if observed is not None and availability is not None:
        if availability.ndim == 2:
            availability = availability[:, np.newaxis, :]
        try:
            available_at_values = np.broadcast_to(availability, observed.shape)
        except ValueError as exc:
            _fail("mask_arrays", f"availability cannot broadcast to observed: {exc}")
        if np.any(observed & ~available_at_values):
            _fail(
                "mask_arrays",
                "an unavailable modality channel cannot be marked observed",
            )

    corruption = array_for("corruption")
    if corruption is not None and observed is not None:
        if corruption.shape != observed.shape:
            _fail("mask_arrays", "corruption and observed masks must have the same shape")
        if np.any(corruption & ~observed):
            _fail("mask_arrays", "corruption can only select observed values")

    padding = array_for("padding")
    attention = array_for("attention")
    if padding is not None and attention is not None:
        if padding.shape != attention.shape:
            _fail("mask_arrays", "padding and attention masks must have the same shape")
        if np.any(padding & attention):
            _fail("mask_arrays", "padding positions cannot be attendable")
