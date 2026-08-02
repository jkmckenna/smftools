"""Backend-neutral requests and aligned results for model explanations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..artifacts import ExplanationBaseline, ExplanationMaskPolicy, ExplanationTarget
from ..contracts import InputChannelSchema, InputSchema, LabelSchema, PredictorCapabilities

INTERPRETABILITY_SCHEMA_VERSION = 1
EXPLANATION_SPLITS = frozenset({"train", "validation", "test", "inference"})
AGGREGATION_REDUCTIONS = frozenset({"none", "mean", "mean_absolute", "sum", "sum_absolute"})


class InterpretabilityContractError(ValueError):
    """Raised before explanation execution when a request is unsafe or incompatible."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterpretabilityContractError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _digest(value: Any, path: str) -> str:
    result = _string(value, path).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise InterpretabilityContractError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InterpretabilityContractError(f"{path} must be an integer >= {minimum}")
    return value


def _strings(values: Sequence[str], path: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise InterpretabilityContractError(f"{path} must be a sequence of strings")
    result = tuple(_string(value, f"{path}[]") for value in values)
    if len(result) != len(set(result)):
        raise InterpretabilityContractError(f"{path} cannot contain duplicates")
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpretabilityContractError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise InterpretabilityContractError(f"{path} keys must be strings")
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    if set(value) != fields:
        raise InterpretabilityContractError(f"{path} fields must be exactly {sorted(fields)}")


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise InterpretabilityContractError(f"{path} must contain finite values")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key for key in value):
            raise InterpretabilityContractError(f"{path} keys must be non-empty strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise InterpretabilityContractError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InterpretabilityContractError(f"value is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExplanationMethodContract:
    """Canonical method name and the capabilities it requires."""

    name: str
    version: str
    backends: tuple[str, ...]
    families: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    baseline_policy: str = "optional"
    layer_policy: str = "forbidden"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "method.name"))
        object.__setattr__(self, "version", _string(self.version, "method.version"))
        object.__setattr__(self, "backends", _strings(self.backends, "method.backends"))
        object.__setattr__(self, "families", _strings(self.families, "method.families"))
        object.__setattr__(
            self,
            "required_capabilities",
            _strings(self.required_capabilities, "method.required_capabilities"),
        )
        if self.baseline_policy not in {"required", "optional", "forbidden"}:
            raise InterpretabilityContractError("method.baseline_policy is invalid")
        if self.layer_policy not in {"required", "optional", "forbidden"}:
            raise InterpretabilityContractError("method.layer_policy is invalid")


METHOD_CONTRACTS = MappingProxyType(
    {
        method.name: method
        for method in (
            ExplanationMethodContract(
                "NaiveBayesLogOdds",
                "1",
                ("sklearn",),
                families=("bernoulli_nb",),
                baseline_policy="forbidden",
            ),
            ExplanationMethodContract(
                "LinearCoefficients",
                "1",
                ("sklearn",),
                families=("logistic_regression",),
                baseline_policy="forbidden",
            ),
            ExplanationMethodContract(
                "PermutationImportance",
                "1",
                ("sklearn", "torch"),
                required_capabilities=("probability_output",),
                baseline_policy="forbidden",
            ),
            ExplanationMethodContract(
                "TreeSHAP",
                "1",
                ("sklearn",),
                families=("random_forest",),
            ),
            ExplanationMethodContract(
                "KernelSHAP",
                "1",
                ("sklearn", "torch"),
                required_capabilities=("probability_output",),
                baseline_policy="required",
            ),
            ExplanationMethodContract(
                "Saliency",
                "1",
                ("torch",),
                required_capabilities=("gradients",),
                baseline_policy="forbidden",
            ),
            ExplanationMethodContract(
                "InputXGradient",
                "1",
                ("torch",),
                required_capabilities=("gradients",),
                baseline_policy="forbidden",
            ),
            ExplanationMethodContract(
                "IntegratedGradients",
                "1",
                ("torch",),
                required_capabilities=("gradients",),
                baseline_policy="required",
            ),
            ExplanationMethodContract(
                "DeepLift",
                "1",
                ("torch",),
                required_capabilities=("gradients",),
                baseline_policy="required",
            ),
            ExplanationMethodContract(
                "GradientSHAP",
                "1",
                ("torch",),
                required_capabilities=("gradients",),
                baseline_policy="required",
            ),
            ExplanationMethodContract(
                "LayerGradCam",
                "1",
                ("torch",),
                required_capabilities=("gradients", "convolutional_layers"),
                baseline_policy="forbidden",
                layer_policy="required",
            ),
            ExplanationMethodContract(
                "GuidedGradCam",
                "1",
                ("torch",),
                required_capabilities=("gradients", "convolutional_layers"),
                baseline_policy="forbidden",
                layer_policy="required",
            ),
            ExplanationMethodContract(
                "AttentionRollout",
                "1",
                ("torch",),
                required_capabilities=("attention_data",),
                baseline_policy="forbidden",
                layer_policy="optional",
            ),
            ExplanationMethodContract(
                "AttentionGradient",
                "1",
                ("torch",),
                required_capabilities=("gradients", "attention_data"),
                baseline_policy="forbidden",
                layer_policy="optional",
            ),
        )
    }
)


@dataclass(frozen=True)
class ExplanationDecisionProvenance:
    """Where method, layer, baseline, and parameter choices were selected."""

    kind: str
    split_role: str | None = None
    cohort: str | None = None

    def __post_init__(self) -> None:
        kind = _string(self.kind, "decision.kind")
        if kind not in {"fixed", "selected"}:
            raise InterpretabilityContractError("decision.kind must be 'fixed' or 'selected'")
        object.__setattr__(self, "kind", kind)
        if kind == "fixed":
            if self.split_role is not None or self.cohort is not None:
                raise InterpretabilityContractError(
                    "fixed explanation choices cannot declare a selection cohort"
                )
            return
        if self.split_role not in {"train", "validation"}:
            raise InterpretabilityContractError(
                "explanation choices may only be selected on train or validation data"
            )
        object.__setattr__(self, "cohort", _string(self.cohort, "decision.cohort"))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible decision provenance."""
        return {"kind": self.kind, "split_role": self.split_role, "cohort": self.cohort}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExplanationDecisionProvenance:
        """Strictly restore explanation-choice provenance."""
        value = _mapping(raw, "decision")
        _exact_fields(value, {"kind", "split_role", "cohort"}, "decision")
        return cls(
            kind=value["kind"],
            split_role=value["split_role"],
            cohort=value["cohort"],
        )


@dataclass(frozen=True)
class AttributionAggregation:
    """Requested reproducible summary of the retained raw attribution tensor."""

    reduction: str = "none"
    axes: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reduction = _string(self.reduction, "aggregation.reduction")
        if reduction not in AGGREGATION_REDUCTIONS:
            raise InterpretabilityContractError(
                f"aggregation.reduction must be one of {sorted(AGGREGATION_REDUCTIONS)}"
            )
        axes = _strings(self.axes, "aggregation.axes")
        unknown = sorted(set(axes).difference({"observation", "position", "channel"}))
        if unknown:
            raise InterpretabilityContractError(
                f"aggregation.axes contains unknown attribution axes: {unknown}"
            )
        groups = _strings(self.group_by, "aggregation.group_by")
        if reduction == "none" and (axes or groups):
            raise InterpretabilityContractError(
                "aggregation axes/group_by require a non-'none' reduction"
            )
        if reduction != "none" and not axes:
            raise InterpretabilityContractError("an aggregation reduction requires axes")
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "group_by", groups)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible aggregation semantics."""
        return {
            "reduction": self.reduction,
            "axes": list(self.axes),
            "group_by": list(self.group_by),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AttributionAggregation:
        """Strictly restore requested aggregation semantics."""
        value = _mapping(raw, "aggregation")
        _exact_fields(value, {"reduction", "axes", "group_by"}, "aggregation")
        return cls(
            reduction=value["reduction"],
            axes=value["axes"],
            group_by=value["group_by"],
        )


@dataclass(frozen=True)
class InterpretabilityRequest:
    """Complete immutable intent for one explanation computation."""

    schema_version: int
    request_id: str
    method: str
    model_id: str
    dataset_snapshot_id: str
    input_schema_hash: str
    split_role: str
    cohort: str
    observation_uids: tuple[str, ...]
    target: ExplanationTarget
    baseline: ExplanationBaseline | None
    layer: str | None
    mask_policy: ExplanationMaskPolicy
    aggregation: AttributionAggregation
    decision: ExplanationDecisionProvenance
    parameters: Mapping[str, Any]
    random_seed: int

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "request.schema_version", minimum=1)
        if schema_version != INTERPRETABILITY_SCHEMA_VERSION:
            raise InterpretabilityContractError(
                f"unsupported interpretability schema version {self.schema_version}"
            )
        method = _string(self.method, "request.method")
        if method not in METHOD_CONTRACTS:
            raise InterpretabilityContractError(
                f"method must be one of {sorted(METHOD_CONTRACTS)}; aliases are not accepted"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "request_id", _digest(self.request_id, "request.request_id"))
        object.__setattr__(self, "model_id", _digest(self.model_id, "request.model_id"))
        object.__setattr__(
            self,
            "dataset_snapshot_id",
            _digest(self.dataset_snapshot_id, "request.dataset_snapshot_id"),
        )
        object.__setattr__(
            self,
            "input_schema_hash",
            _digest(self.input_schema_hash, "request.input_schema_hash"),
        )
        split_role = _string(self.split_role, "request.split_role")
        if split_role not in EXPLANATION_SPLITS:
            raise InterpretabilityContractError(
                f"request.split_role must be one of {sorted(EXPLANATION_SPLITS)}"
            )
        object.__setattr__(self, "split_role", split_role)
        object.__setattr__(self, "cohort", _string(self.cohort, "request.cohort"))
        observations = _strings(self.observation_uids, "request.observation_uids")
        if not observations:
            raise InterpretabilityContractError("request.observation_uids cannot be empty")
        object.__setattr__(self, "observation_uids", observations)
        if not isinstance(self.target, ExplanationTarget):
            raise InterpretabilityContractError("request.target must be an ExplanationTarget")
        if self.baseline is not None and not isinstance(self.baseline, ExplanationBaseline):
            raise InterpretabilityContractError(
                "request.baseline must be an ExplanationBaseline or null"
            )
        if not isinstance(self.mask_policy, ExplanationMaskPolicy):
            raise InterpretabilityContractError(
                "request.mask_policy must be an ExplanationMaskPolicy"
            )
        if not isinstance(self.aggregation, AttributionAggregation):
            raise InterpretabilityContractError(
                "request.aggregation must be an AttributionAggregation"
            )
        if not isinstance(self.decision, ExplanationDecisionProvenance):
            raise InterpretabilityContractError(
                "request.decision must be an ExplanationDecisionProvenance"
            )
        object.__setattr__(self, "layer", _optional_string(self.layer, "request.layer"))
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(self.parameters, "request.parameters"),
        )
        _integer(self.random_seed, "request.random_seed")
        expected = _sha256(self._identity_dict())
        if self.request_id != expected:
            raise InterpretabilityContractError("request.request_id does not match request content")

    @classmethod
    def create(
        cls,
        *,
        method: str,
        model_id: str,
        dataset_snapshot_id: str,
        input_schema_hash: str,
        split_role: str,
        cohort: str,
        observation_uids: Sequence[str],
        target: ExplanationTarget,
        mask_policy: ExplanationMaskPolicy,
        baseline: ExplanationBaseline | None = None,
        layer: str | None = None,
        aggregation: AttributionAggregation | None = None,
        decision: ExplanationDecisionProvenance | None = None,
        parameters: Mapping[str, Any] | None = None,
        random_seed: int = 0,
    ) -> InterpretabilityRequest:
        """Create a content-addressed explanation request."""
        values = {
            "schema_version": INTERPRETABILITY_SCHEMA_VERSION,
            "method": method,
            "model_id": model_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "input_schema_hash": input_schema_hash,
            "split_role": split_role,
            "cohort": cohort,
            "observation_uids": tuple(observation_uids),
            "target": target,
            "baseline": baseline,
            "layer": layer,
            "mask_policy": mask_policy,
            "aggregation": aggregation or AttributionAggregation(),
            "decision": decision or ExplanationDecisionProvenance("fixed"),
            "parameters": parameters or {},
            "random_seed": random_seed,
        }
        identity = {
            "schema_version": INTERPRETABILITY_SCHEMA_VERSION,
            "method": method,
            "model_id": model_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "input_schema_hash": input_schema_hash,
            "split_role": split_role,
            "cohort": cohort,
            "observation_uids": list(observation_uids),
            "target": target.to_dict(),
            "baseline": None if baseline is None else baseline.to_dict(),
            "layer": layer,
            "mask_policy": mask_policy.to_dict(),
            "aggregation": values["aggregation"].to_dict(),
            "decision": values["decision"].to_dict(),
            "parameters": _thaw_json(_freeze_json(values["parameters"], "request.parameters")),
            "random_seed": random_seed,
        }
        return cls(request_id=_sha256(identity), **values)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "model_id": self.model_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "input_schema_hash": self.input_schema_hash,
            "split_role": self.split_role,
            "cohort": self.cohort,
            "observation_uids": list(self.observation_uids),
            "target": self.target.to_dict(),
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "layer": self.layer,
            "mask_policy": self.mask_policy.to_dict(),
            "aggregation": self.aggregation.to_dict(),
            "decision": self.decision.to_dict(),
            "parameters": _thaw_json(self.parameters),
            "random_seed": self.random_seed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible explanation request."""
        return {"request_id": self.request_id, **self._identity_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> InterpretabilityRequest:
        """Strictly validate and restore a version-1 explanation request."""
        value = _mapping(raw, "request")
        fields = {
            "schema_version",
            "request_id",
            "method",
            "model_id",
            "dataset_snapshot_id",
            "input_schema_hash",
            "split_role",
            "cohort",
            "observation_uids",
            "target",
            "baseline",
            "layer",
            "mask_policy",
            "aggregation",
            "decision",
            "parameters",
            "random_seed",
        }
        _exact_fields(value, fields, "request")
        baseline = value["baseline"]
        return cls(
            schema_version=value["schema_version"],
            request_id=value["request_id"],
            method=value["method"],
            model_id=value["model_id"],
            dataset_snapshot_id=value["dataset_snapshot_id"],
            input_schema_hash=value["input_schema_hash"],
            split_role=value["split_role"],
            cohort=value["cohort"],
            observation_uids=value["observation_uids"],
            target=ExplanationTarget.from_dict(_mapping(value["target"], "request.target")),
            baseline=(
                None
                if baseline is None
                else ExplanationBaseline.from_dict(_mapping(baseline, "request.baseline"))
            ),
            layer=value["layer"],
            mask_policy=ExplanationMaskPolicy.from_dict(
                _mapping(value["mask_policy"], "request.mask_policy")
            ),
            aggregation=AttributionAggregation.from_dict(
                _mapping(value["aggregation"], "request.aggregation")
            ),
            decision=ExplanationDecisionProvenance.from_dict(
                _mapping(value["decision"], "request.decision")
            ),
            parameters=_mapping(value["parameters"], "request.parameters"),
            random_seed=value["random_seed"],
        )


def validate_interpretability_request(
    request: InterpretabilityRequest,
    *,
    family: str,
    capabilities: PredictorCapabilities,
    input_schema: InputSchema,
    label_schema: LabelSchema,
    available_layers: Sequence[str] | None = None,
) -> ExplanationMethodContract:
    """Reject unsupported requests before importing or executing an explainer."""
    method = METHOD_CONTRACTS[request.method]
    family = _string(family, "family")
    if capabilities.backend not in method.backends:
        raise InterpretabilityContractError(
            f"{request.method} does not support backend {capabilities.backend!r}"
        )
    if method.families and family not in method.families:
        raise InterpretabilityContractError(
            f"{request.method} does not support model family {family!r}"
        )
    missing = [
        name for name in method.required_capabilities if not getattr(capabilities, name, False)
    ]
    if missing:
        raise InterpretabilityContractError(
            f"{request.method} requires model capabilities: {missing}"
        )
    if method.baseline_policy == "required" and request.baseline is None:
        raise InterpretabilityContractError(f"{request.method} requires a baseline/background")
    if method.baseline_policy == "forbidden" and request.baseline is not None:
        raise InterpretabilityContractError(f"{request.method} does not consume a baseline")
    if method.layer_policy == "required" and request.layer is None:
        raise InterpretabilityContractError(f"{request.method} requires a target layer")
    if method.layer_policy == "forbidden" and request.layer is not None:
        raise InterpretabilityContractError(f"{request.method} does not consume a target layer")
    if request.layer is not None and available_layers is not None:
        layers = _strings(available_layers, "available_layers")
        if request.layer not in layers:
            raise InterpretabilityContractError(
                f"target layer {request.layer!r} is not exposed by the model adapter"
            )
    if request.baseline is not None and request.baseline.dataset_snapshot_id is not None:
        if request.baseline.cohort != "train":
            raise InterpretabilityContractError(
                "data-derived explanation baselines/backgrounds must come from the train split"
            )
    if request.baseline is not None and (
        (request.baseline.dataset_snapshot_id is None) != (request.baseline.cohort is None)
    ):
        raise InterpretabilityContractError(
            "baseline dataset_snapshot_id and cohort must either both be set or both be null"
        )
    if request.input_schema_hash != input_schema.schema_hash:
        raise InterpretabilityContractError("request input_schema_hash differs from input schema")
    if request.target.class_id is None or request.target.class_name is None:
        raise InterpretabilityContractError("classification explanations require a target class")
    if (
        request.target.class_id < 0
        or request.target.class_id >= len(label_schema.class_order)
        or label_schema.class_order[request.target.class_id] != request.target.class_name
    ):
        raise InterpretabilityContractError("request target differs from persisted label schema")
    declared_masks = {mask.kind for mask in input_schema.masks}
    requested_masks = set(request.mask_policy.mask_kinds)
    unknown = sorted(requested_masks.difference(declared_masks))
    if unknown:
        raise InterpretabilityContractError(
            f"request mask policy references masks absent from input schema: {unknown}"
        )
    unsupported = sorted(requested_masks.difference(capabilities.supported_mask_kinds))
    if unsupported:
        raise InterpretabilityContractError(
            f"model cannot apply requested explanation masks: {unsupported}"
        )
    return method


@dataclass(frozen=True)
class AttributionFeature:
    """One transformed feature with its physical and biological meaning."""

    name: str
    kind: str
    coordinate: int
    channel: InputChannelSchema

    def __post_init__(self) -> None:
        name = _string(self.name, "feature.name")
        kind = _string(self.kind, "feature.kind")
        if kind not in {"signal", "observed", "design", "availability", "padding"}:
            raise InterpretabilityContractError(
                "feature.kind must be signal or a canonical mask-indicator kind"
            )
        if isinstance(self.coordinate, bool) or not isinstance(self.coordinate, int):
            raise InterpretabilityContractError("feature.coordinate must be an integer")
        if not isinstance(self.channel, InputChannelSchema):
            raise InterpretabilityContractError("feature.channel must be an InputChannelSchema")
        expected_name = f"{kind}:{self.channel.name}@{self.coordinate}"
        if name != expected_name:
            raise InterpretabilityContractError(
                f"feature.name must match its kind, channel, and coordinate: {expected_name!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible transformed-feature metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "coordinate": self.coordinate,
            "channel": self.channel.to_dict(),
        }


@dataclass(frozen=True)
class AttributionResult:
    """Immutable attribution values aligned to the resolved biological input axes."""

    result_id: str
    request: InterpretabilityRequest
    axes: tuple[str, ...]
    values: np.ndarray
    observation_uids: tuple[str, ...]
    coordinates: np.ndarray
    channels: tuple[InputChannelSchema, ...]
    features: tuple[AttributionFeature, ...] = ()
    convergence_delta: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _digest(self.result_id, "result.result_id"))
        if not isinstance(self.request, InterpretabilityRequest):
            raise InterpretabilityContractError("result.request must be an InterpretabilityRequest")
        axes = _strings(self.axes, "result.axes")
        if axes not in {
            ("observation", "position", "channel"),
            ("position", "channel"),
            ("observation", "position"),
            ("position",),
            ("observation", "feature"),
            ("feature",),
        }:
            raise InterpretabilityContractError(
                "result axes must describe aligned position/channel or transformed-feature values"
            )
        object.__setattr__(self, "axes", axes)
        observations = _strings(self.observation_uids, "result.observation_uids")
        if "observation" in axes:
            if observations != self.request.observation_uids:
                raise InterpretabilityContractError(
                    "result observation axis differs from requested observation order"
                )
        elif observations:
            raise InterpretabilityContractError(
                "global attribution results cannot declare an observation axis"
            )
        object.__setattr__(self, "observation_uids", observations)
        raw_coordinates = np.asarray(self.coordinates)
        coordinates = raw_coordinates.astype(np.int64)
        if (
            coordinates.ndim != 1
            or coordinates.size == 0
            or not np.array_equal(raw_coordinates, coordinates)
            or len(np.unique(coordinates)) != len(coordinates)
        ):
            raise InterpretabilityContractError("result coordinates must be a non-empty vector")
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        channels = tuple(self.channels)
        features = tuple(self.features)
        if "feature" in axes:
            if not features:
                raise InterpretabilityContractError(
                    "results with a feature axis must declare transformed-feature metadata"
                )
            if not all(isinstance(feature, AttributionFeature) for feature in features):
                raise InterpretabilityContractError(
                    "result features must contain AttributionFeature records"
                )
            if channels:
                raise InterpretabilityContractError(
                    "transformed-feature results declare channels through each feature"
                )
            names = tuple(feature.name for feature in features)
            if len(names) != len(set(names)):
                raise InterpretabilityContractError("result feature names must be unique")
        elif features:
            raise InterpretabilityContractError(
                "position-aligned results cannot declare transformed-feature metadata"
            )
        if "channel" in axes and not channels:
            raise InterpretabilityContractError(
                "results with a channel axis must declare channel metadata"
            )
        if "channel" not in axes and "feature" not in axes and channels:
            raise InterpretabilityContractError(
                "position-only results cannot declare a channel axis"
            )
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "features", features)
        axis_sizes = {
            "observation": len(observations),
            "position": len(coordinates),
            "channel": len(channels),
            "feature": len(features),
        }
        expected_shape = tuple(axis_sizes[axis] for axis in axes)
        values = np.asarray(self.values, dtype=np.float64).copy()
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise InterpretabilityContractError(
                f"result values must be finite with shape {expected_shape}"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        if self.convergence_delta is not None:
            delta = np.asarray(self.convergence_delta, dtype=np.float64).copy()
            expected_delta = (len(observations),) if "observation" in axes else (1,)
            if delta.shape != expected_delta or not np.isfinite(delta).all():
                raise InterpretabilityContractError(
                    f"convergence_delta must be finite with shape {expected_delta}"
                )
            delta.setflags(write=False)
            object.__setattr__(self, "convergence_delta", delta)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "result.metadata"))
        expected = _sha256(self._identity_dict())
        if self.result_id != expected:
            raise InterpretabilityContractError("result.result_id does not match result content")

    @classmethod
    def create(
        cls,
        *,
        request: InterpretabilityRequest,
        axes: Sequence[str],
        values: Any,
        observation_uids: Sequence[str],
        coordinates: Any,
        channels: Sequence[InputChannelSchema],
        features: Sequence[AttributionFeature] = (),
        convergence_delta: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AttributionResult:
        """Create a content-addressed aligned attribution result."""
        fields = {
            "request": request,
            "axes": tuple(axes),
            "values": values,
            "observation_uids": tuple(observation_uids),
            "coordinates": coordinates,
            "channels": tuple(channels),
            "features": tuple(features),
            "convergence_delta": convergence_delta,
            "metadata": metadata or {},
        }
        normalized_values = np.asarray(values, dtype=np.float64)
        normalized_coordinates = np.asarray(coordinates, dtype=np.int64)
        normalized_delta = (
            None if convergence_delta is None else np.asarray(convergence_delta, dtype=np.float64)
        )
        identity = {
            "request_id": request.request_id,
            "axes": list(axes),
            "values": _array_digest(normalized_values),
            "observation_uids": list(observation_uids),
            "coordinates": _array_digest(normalized_coordinates),
            "channels": [channel.to_dict() for channel in channels],
            "features": [feature.to_dict() for feature in features],
            "convergence_delta": (
                None if normalized_delta is None else _array_digest(normalized_delta)
            ),
            "metadata": _thaw_json(_freeze_json(metadata or {}, "result.metadata")),
        }
        return cls(result_id=_sha256(identity), **fields)

    def validate_against(self, input_schema: InputSchema) -> None:
        """Validate axis widths and biological channel/source metadata."""
        if self.request.input_schema_hash != input_schema.schema_hash:
            raise InterpretabilityContractError("result request differs from input schema")
        if len(self.coordinates) != input_schema.n_positions:
            raise InterpretabilityContractError(
                "result position axis width differs from input schema"
            )
        if "channel" in self.axes and self.channels != input_schema.channels:
            raise InterpretabilityContractError(
                "result channel names, biological roles, or physical site contexts differ "
                "from input schema"
            )
        if "feature" in self.axes:
            channels = {channel.name: channel for channel in input_schema.channels}
            for feature in self.features:
                if feature.coordinate not in self.coordinates:
                    raise InterpretabilityContractError(
                        "result transformed feature references an unknown coordinate"
                    )
                if channels.get(feature.channel.name) != feature.channel:
                    raise InterpretabilityContractError(
                        "result transformed feature differs from input biological channel metadata"
                    )
            expected_signal = {
                (int(coordinate), channel.name)
                for coordinate in self.coordinates
                for channel in input_schema.channels
            }
            actual_signal = {
                (feature.coordinate, feature.channel.name)
                for feature in self.features
                if feature.kind == "signal"
            }
            if actual_signal != expected_signal:
                raise InterpretabilityContractError(
                    "result transformed features must contain every schema-aligned signal feature"
                )

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "axes": list(self.axes),
            "values": _array_digest(np.asarray(self.values)),
            "observation_uids": list(self.observation_uids),
            "coordinates": _array_digest(np.asarray(self.coordinates)),
            "channels": [channel.to_dict() for channel in self.channels],
            "features": [feature.to_dict() for feature in self.features],
            "convergence_delta": (
                None
                if self.convergence_delta is None
                else _array_digest(np.asarray(self.convergence_delta))
            ),
            "metadata": _thaw_json(self.metadata),
        }
