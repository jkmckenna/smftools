"""Bounded, mask-aware explanations for fitted plain-Torch classifiers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from smftools.optional_imports import require

from ..contracts import validate_mask_arrays, validate_mask_relationships
from ..data.partition_dataset import MLMaterializedPartitionData, MLPartitionBatch
from ..data.transforms import TorchFeatureTransform, TorchTransformedBatch
from ..training.torch_backend import FittedTorchModel
from .background import BackgroundReference
from .contracts import (
    AttributionResult,
    InterpretabilityContractError,
    InterpretabilityRequest,
    _integer,
    _string,
    _thaw_json,
    validate_interpretability_request,
)

_NeuralData = MLMaterializedPartitionData | MLPartitionBatch
_INPUT_METHODS = frozenset(
    {"Saliency", "InputXGradient", "IntegratedGradients", "DeepLift", "GradientSHAP"}
)
_LAYER_METHODS = frozenset({"LayerGradCam", "GuidedGradCam"})
_INTEGRATION_METHODS = frozenset(
    {"gausslegendre", "riemann_left", "riemann_middle", "riemann_right", "riemann_trapezoid"}
)
_INTERPOLATION_MODES = frozenset({"area", "linear", "nearest"})


def _exact_parameters(request: InterpretabilityRequest, expected: set[str]) -> dict[str, Any]:
    values = _thaw_json(request.parameters)
    if set(values) != expected:
        raise InterpretabilityContractError(
            f"{request.method} parameters must be exactly {sorted(expected)}"
        )
    return values


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise InterpretabilityContractError(f"{path} must be boolean")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterpretabilityContractError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise InterpretabilityContractError(f"{path} must be finite and >= {minimum}")
    return result


def _available_layers(model: FittedTorchModel) -> tuple[str, ...]:
    return ("attribution_layer",) if hasattr(model.model, "attribution_layer") else ()


def _validate_data(
    model: FittedTorchModel,
    data: _NeuralData,
    request: InterpretabilityRequest,
) -> TorchTransformedBatch:
    validate_interpretability_request(
        request,
        family=model.family,
        capabilities=model.architecture.capabilities,
        input_schema=model.input_schema,
        label_schema=model.label_schema,
        available_layers=_available_layers(model),
    )
    if request.dataset_snapshot_id != model.dataset_snapshot_id:
        raise InterpretabilityContractError(
            "explanation request dataset differs from the fitted Torch model"
        )
    data_split = getattr(data, "split", request.split_role)
    if request.split_role != data_split:
        raise InterpretabilityContractError(
            "explanation request split differs from the supplied data"
        )
    if request.observation_uids != tuple(data.molecule_uids):
        raise InterpretabilityContractError(
            "explanation request observations differ from the supplied data order"
        )
    if model.transform.spec.indicators:
        raise InterpretabilityContractError(
            "Torch explanations require transforms that keep masks separate from signals"
        )
    if tuple(data.channel_names) != tuple(channel.name for channel in model.input_schema.channels):
        raise InterpretabilityContractError(
            "explanation data channel order differs from the fitted model schema"
        )
    if tuple(map(int, data.coordinates)) != model.transform.coordinates:
        raise InterpretabilityContractError(
            "explanation data coordinates differ from the fitted feature transform"
        )
    by_kind = {
        "observed": data.observed_mask,
        "availability": data.availability_mask,
        "design": data.design_mask,
        "padding": data.padding_mask,
    }
    masks = {mask.name: by_kind[mask.kind] for mask in model.input_schema.masks}
    validate_mask_arrays(
        model.input_schema,
        masks,
        batch_size=len(data.molecule_uids),
        require_all=True,
    )
    validate_mask_relationships(model.input_schema, masks)
    if not any(mask.kind == "padding" for mask in model.input_schema.masks) and np.any(
        data.padding_mask
    ):
        raise InterpretabilityContractError(
            "explanation data contains padding but the input schema does not declare it"
        )
    transformed = TorchFeatureTransform(model.transform, device=model.resolved_device)(data)
    if not bool(transformed.values.isfinite().all().item()):
        raise InterpretabilityContractError("transformed neural explanation values must be finite")
    return transformed


def _expanded_masks(batch: TorchTransformedBatch) -> tuple[Any, Any, Any, Any]:
    observed = batch.observed_mask
    availability = batch.availability_mask
    design = batch.design_mask
    padding = batch.padding_mask
    if design.ndim == 2:
        design = design.unsqueeze(0).expand(len(observed), -1, -1)
    return observed, availability, design, padding


def _valid_values(batch: TorchTransformedBatch) -> Any:
    observed, availability, design, padding = _expanded_masks(batch)
    return observed & availability[:, :, None] & design & ~padding[:, None, :]


def _slice_masks(masks: tuple[Any, Any, Any, Any], start: int, stop: int) -> tuple[Any, ...]:
    return tuple(mask[start:stop] for mask in masks)


def _target_wrapper(torch: Any, model: FittedTorchModel, target_class: int) -> Any:
    class TargetOutput(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model.model

        def forward(self, values, observed, availability, design, padding):
            logits = self.inner(
                values,
                observed_mask=observed,
                availability_mask=availability,
                design_mask=design,
                padding_mask=padding,
            )
            if logits.ndim != 2:
                raise InterpretabilityContractError(
                    "Torch explanation model must return a two-dimensional logit matrix"
                )
            if logits.shape[1] == 1:
                if len(model.label_schema.class_order) != 2:
                    raise InterpretabilityContractError(
                        "single-logit explanation requires a binary label schema"
                    )
                return logits[:, 0] if target_class == 1 else -logits[:, 0]
            if logits.shape[1] != len(model.label_schema.class_order):
                raise InterpretabilityContractError(
                    "Torch explanation logits differ from the persisted class order"
                )
            return logits[:, target_class]

    return TargetOutput()


def _background_values(
    model: FittedTorchModel,
    request: InterpretabilityRequest,
    background: BackgroundReference | None,
) -> Any | None:
    if request.baseline is None:
        if background is not None:
            raise InterpretabilityContractError(
                f"{request.method} request does not declare a background"
            )
        return None
    if background is None:
        raise InterpretabilityContractError(
            f"{request.method} requires its checksummed training background values"
        )
    background.validate_against(model.input_schema)
    if request.baseline.baseline_hash != background.background_hash:
        raise InterpretabilityContractError(
            "neural explanation background differs from the request baseline"
        )
    return TorchFeatureTransform(model.transform, device=model.resolved_device)(
        background.as_materialized(model.input_schema)
    ).values


@contextmanager
def _seeded_captum(torch: Any, seed: int, device: Any):
    devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _input_parameters(request: InterpretabilityRequest) -> dict[str, Any]:
    expected = {
        "Saliency": {"absolute", "example_batch_size"},
        "InputXGradient": {"example_batch_size"},
        "IntegratedGradients": {
            "baseline_reduction",
            "example_batch_size",
            "integration_method",
            "internal_batch_size",
            "n_steps",
        },
        "DeepLift": {"baseline_reduction", "example_batch_size"},
        "GradientSHAP": {"example_batch_size", "n_samples", "stdevs"},
    }[request.method]
    parameters = _exact_parameters(request, expected)
    parameters["example_batch_size"] = _integer(
        parameters["example_batch_size"], "example_batch_size", minimum=1
    )
    if request.method == "Saliency":
        parameters["absolute"] = _boolean(parameters["absolute"], "absolute")
    elif request.method == "IntegratedGradients":
        parameters["n_steps"] = _integer(parameters["n_steps"], "n_steps", minimum=2)
        parameters["internal_batch_size"] = _integer(
            parameters["internal_batch_size"], "internal_batch_size", minimum=1
        )
        method = _string(parameters["integration_method"], "integration_method")
        if method not in _INTEGRATION_METHODS:
            raise InterpretabilityContractError(
                f"integration_method must be one of {sorted(_INTEGRATION_METHODS)}"
            )
        parameters["integration_method"] = method
    elif request.method == "GradientSHAP":
        parameters["n_samples"] = _integer(parameters["n_samples"], "n_samples", minimum=1)
        parameters["stdevs"] = _number(parameters["stdevs"], "stdevs")
    if "baseline_reduction" in parameters:
        reduction = _string(parameters["baseline_reduction"], "baseline_reduction")
        if reduction != "mean":
            raise InterpretabilityContractError(
                "baseline_reduction currently supports only the training-background mean"
            )
        parameters["baseline_reduction"] = reduction
    return parameters


def _layer_parameters(request: InterpretabilityRequest) -> dict[str, Any]:
    expected = (
        {
            "attribute_to_layer_input",
            "example_batch_size",
            "interpolate_mode",
            "relu_attributions",
        }
        if request.method == "LayerGradCam"
        else {"attribute_to_layer_input", "example_batch_size", "interpolate_mode"}
    )
    parameters = _exact_parameters(request, expected)
    parameters["example_batch_size"] = _integer(
        parameters["example_batch_size"], "example_batch_size", minimum=1
    )
    parameters["attribute_to_layer_input"] = _boolean(
        parameters["attribute_to_layer_input"], "attribute_to_layer_input"
    )
    if "relu_attributions" in parameters:
        parameters["relu_attributions"] = _boolean(
            parameters["relu_attributions"], "relu_attributions"
        )
    mode = _string(parameters["interpolate_mode"], "interpolate_mode")
    if mode not in _INTERPOLATION_MODES:
        raise InterpretabilityContractError(
            f"interpolate_mode must be one of {sorted(_INTERPOLATION_MODES)}"
        )
    parameters["interpolate_mode"] = mode
    return parameters


def _captum_classes() -> tuple[Any, ...]:
    captum_attr = require("captum.attr", extra="ml-extended", purpose="neural explanations")
    captum = require("captum", extra="ml-extended", purpose="neural explanations")
    return (
        captum,
        captum_attr.Saliency,
        captum_attr.InputXGradient,
        captum_attr.IntegratedGradients,
        captum_attr.DeepLift,
        captum_attr.GradientShap,
        captum_attr.LayerGradCam,
        captum_attr.GuidedGradCam,
        captum_attr.LayerAttribution,
    )


def _input_attributions(
    model: FittedTorchModel,
    request: InterpretabilityRequest,
    batch: TorchTransformedBatch,
    background_values: Any | None,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    torch = require("torch", extra="ml-base", purpose="neural explanations")
    (
        _captum,
        Saliency,
        InputXGradient,
        IntegratedGradients,
        DeepLift,
        GradientShap,
        _LayerGradCam,
        _GuidedGradCam,
        _LayerAttribution,
    ) = _captum_classes()
    target = request.target.class_id
    assert target is not None
    wrapped = _target_wrapper(torch, model, target)
    masks = _expanded_masks(batch)
    validity = _valid_values(batch)
    baseline_mean = (
        None if background_values is None else background_values.mean(dim=0, keepdim=True)
    )
    chunks: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    batch_size = parameters["example_batch_size"]
    for start in range(0, len(batch.values), batch_size):
        stop = min(start + batch_size, len(batch.values))
        values = batch.values[start:stop].detach().clone().requires_grad_(True)
        forward_args = _slice_masks(masks, start, stop)
        if request.method == "Saliency":
            attrs = Saliency(wrapped).attribute(
                values,
                additional_forward_args=forward_args,
                abs=parameters["absolute"],
            )
        elif request.method == "InputXGradient":
            attrs = InputXGradient(wrapped).attribute(
                values,
                additional_forward_args=forward_args,
            )
        elif request.method == "IntegratedGradients":
            internal = max(parameters["internal_batch_size"], len(values))
            attrs, delta = IntegratedGradients(wrapped).attribute(
                values,
                baselines=baseline_mean,
                additional_forward_args=forward_args,
                n_steps=parameters["n_steps"],
                method=parameters["integration_method"],
                internal_batch_size=internal,
                return_convergence_delta=True,
            )
            deltas.append(delta.detach().cpu().numpy())
        elif request.method == "DeepLift":
            attrs, delta = DeepLift(wrapped).attribute(
                values,
                baselines=baseline_mean,
                additional_forward_args=forward_args,
                return_convergence_delta=True,
            )
            deltas.append(delta.detach().cpu().numpy())
        else:
            attrs, delta = GradientShap(wrapped).attribute(
                values,
                baselines=background_values,
                additional_forward_args=forward_args,
                n_samples=parameters["n_samples"],
                stdevs=parameters["stdevs"],
                return_convergence_delta=True,
            )
            raw_delta = delta.detach().cpu().numpy().reshape(len(values), -1)
            deltas.append(np.mean(np.abs(raw_delta), axis=1))
        attrs = attrs.masked_fill(~validity[start:stop], 0.0)
        chunks.append(attrs.detach().cpu().numpy().transpose(0, 2, 1))
        model.model.zero_grad(set_to_none=True)
    convergence = None if not deltas else np.concatenate(deltas).astype(np.float64)
    delta_policy = (
        "not_available"
        if convergence is None
        else (
            "mean_absolute_over_gradient_samples"
            if request.method == "GradientSHAP"
            else "captum_convergence_delta"
        )
    )
    return np.concatenate(chunks), convergence, delta_policy


def _layer_attributions(
    model: FittedTorchModel,
    request: InterpretabilityRequest,
    batch: TorchTransformedBatch,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...], tuple[Any, ...]]:
    torch = require("torch", extra="ml-base", purpose="neural explanations")
    (
        _captum,
        _Saliency,
        _InputXGradient,
        _IntegratedGradients,
        _DeepLift,
        _GradientShap,
        LayerGradCam,
        GuidedGradCam,
        LayerAttribution,
    ) = _captum_classes()
    target = request.target.class_id
    assert target is not None
    wrapped = _target_wrapper(torch, model, target)
    layer = getattr(model.model, request.layer)
    masks = _expanded_masks(batch)
    validity = _valid_values(batch)
    chunks: list[np.ndarray] = []
    batch_size = parameters["example_batch_size"]
    for start in range(0, len(batch.values), batch_size):
        stop = min(start + batch_size, len(batch.values))
        values = batch.values[start:stop].detach().clone().requires_grad_(True)
        forward_args = _slice_masks(masks, start, stop)
        if request.method == "LayerGradCam":
            attrs = LayerGradCam(wrapped, layer).attribute(
                values,
                additional_forward_args=forward_args,
                attribute_to_layer_input=parameters["attribute_to_layer_input"],
                relu_attributions=parameters["relu_attributions"],
                attr_dim_summation=True,
            )
            attrs = LayerAttribution.interpolate(
                attrs,
                (len(model.transform.coordinates),),
                interpolate_mode=parameters["interpolate_mode"],
            ).squeeze(1)
            attrs = attrs.masked_fill(~validity[start:stop].any(dim=1), 0.0)
            chunks.append(attrs.detach().cpu().numpy())
        else:
            attrs = GuidedGradCam(wrapped, layer).attribute(
                values,
                additional_forward_args=forward_args,
                interpolate_mode=parameters["interpolate_mode"],
                attribute_to_layer_input=parameters["attribute_to_layer_input"],
            )
            attrs = attrs.masked_fill(~validity[start:stop], 0.0)
            chunks.append(attrs.detach().cpu().numpy().transpose(0, 2, 1))
        model.model.zero_grad(set_to_none=True)
    if request.method == "LayerGradCam":
        return np.concatenate(chunks), ("observation", "position"), ()
    return (
        np.concatenate(chunks),
        ("observation", "position", "channel"),
        model.input_schema.channels,
    )


def explain_torch_model(
    model: FittedTorchModel,
    data: _NeuralData,
    request: InterpretabilityRequest,
    *,
    background: BackgroundReference | None = None,
) -> AttributionResult:
    """Execute one bounded neural explanation for an already-loaded Torch model.

    Captum is imported only after the model, request, masks, layer, parameters,
    and optional training background have passed preflight validation.

    Args:
        model: Trusted canonical fitted Torch model.
        data: Materialized or bounded-batch observations in request order.
        request: Complete explanation intent and bounded method parameters.
        background: Checksummed training background for baseline-dependent methods.

    Returns:
        Immutable attribution values aligned to input or genomic-position axes.
    """
    if not isinstance(model, FittedTorchModel):
        raise InterpretabilityContractError(
            "neural explanations require a canonical FittedTorchModel"
        )
    batch = _validate_data(model, data, request)
    background_values = _background_values(model, request, background)
    if request.method in _INPUT_METHODS:
        parameters = _input_parameters(request)
    elif request.method in _LAYER_METHODS:
        parameters = _layer_parameters(request)
    else:
        raise InterpretabilityContractError(
            f"method {request.method!r} is not implemented by the Torch explanation adapter"
        )
    torch = require("torch", extra="ml-base", purpose="neural explanations")
    device = batch.values.device
    was_training = bool(model.model.training)
    model.model.eval()
    try:
        with _seeded_captum(torch, request.random_seed, device), torch.enable_grad():
            if request.method in _INPUT_METHODS:
                values, convergence, delta_policy = _input_attributions(
                    model,
                    request,
                    batch,
                    background_values,
                    parameters,
                )
                axes = ("observation", "position", "channel")
                channels = model.input_schema.channels
            else:
                values, axes, channels = _layer_attributions(
                    model,
                    request,
                    batch,
                    parameters,
                )
                convergence = None
                delta_policy = "not_available"
    finally:
        model.model.zero_grad(set_to_none=True)
        model.model.train(was_training)
    result = AttributionResult.create(
        request=request,
        axes=axes,
        values=values,
        observation_uids=request.observation_uids,
        coordinates=data.coordinates,
        channels=channels,
        convergence_delta=convergence,
        metadata={
            "backend": "torch",
            "family": model.family,
            "implementation": f"captum.{request.method}",
            "implementation_version": str(_captum_classes()[0].__version__),
            "transform_id": model.transform.transform_id,
            "target_space": (
                "target_vs_reference_logit"
                if len(model.label_schema.class_order) == 2
                else "class_logit"
            ),
            "mask_enforcement": "model_forward_and_explicit_zeroing",
            "background_hash": None if background is None else background.background_hash,
            "example_batch_size": parameters["example_batch_size"],
            "maximum_executed_batch_size": min(
                parameters["example_batch_size"], len(request.observation_uids)
            ),
            "convergence_delta_policy": delta_policy,
        },
    )
    result.validate_against(model.input_schema)
    return result
