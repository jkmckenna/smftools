"""Tests for bounded, mask-aware neural explanation adapters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smftools.machine_learning.artifacts import ExplanationMaskPolicy, ExplanationTarget
from smftools.machine_learning.contracts import InputSchema, LabelSchema, MaskSpec
from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.data.transforms import FeatureTransformSpec, fit_feature_transform
from smftools.machine_learning.interpretability import (
    ExplanationDecisionProvenance,
    InterpretabilityContractError,
    InterpretabilityRequest,
    explain_torch_model,
    sample_training_background,
)
from smftools.machine_learning.models import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.training import (
    FittedTorchModel,
    TorchEpochRecord,
    TorchTrainingConfig,
)

pytestmark = pytest.mark.unit

MODEL_ID = "1" * 64
DATASET_ID = "2" * 64
SPLIT_ID = "3" * 64


def _schemas() -> tuple[InputSchema, LabelSchema]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["conversion"],
                    "channels": [
                        {
                            "name": "gpc_accessibility",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "conversion",
                                    "stage": "preprocessed",
                                    "layer": "GpC_site_binary",
                                    "site_context": "GpC",
                                }
                            ],
                        },
                        {
                            "name": "cpg_methylation",
                            "biological_role": "endogenous_methylation",
                            "sources": [
                                {
                                    "modality": "conversion",
                                    "stage": "preprocessed",
                                    "layer": "CpG_site_binary",
                                    "site_context": "CpG",
                                }
                            ],
                        },
                    ],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample"],
                    "train_groups": ["sample-a"],
                    "validation_groups": ["sample-b"],
                    "test_groups": ["sample-c"],
                }
            },
            "models": {"cnn": {"backend": "torch", "recipe": "residual_dilated_cnn_v1"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["cnn"],
                }
            },
        }
    )
    dataset = plan.datasets["reads"]
    return (
        InputSchema.from_dataset(
            dataset,
            reference="locus",
            n_positions=6,
            masks=(
                MaskSpec.standard("observed"),
                MaskSpec.standard("availability"),
                MaskSpec.standard("design"),
                MaskSpec.standard("padding"),
            ),
        ),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _data(split: str, *, offset: int = 0) -> MLMaterializedPartitionData:
    values = np.asarray(
        [
            [[0, 0], [0, 1], [0, 0], [1, 0], [0, 0], [0, 1]],
            [[1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0]],
            [[0, 1], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0]],
            [[1, 0], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1]],
        ],
        dtype=np.float32,
    )
    observed = np.ones_like(values, dtype=bool)
    observed[0, 2, 0] = False
    values[0, 2, 0] = np.nan
    availability = np.ones((len(values), 2), dtype=bool)
    availability[1, 1] = False
    observed[1, :, 1] = False
    design = np.ones((6, 2), dtype=bool)
    design[4, 0] = False
    observed[:, 4, 0] = False
    padding = np.zeros((len(values), 6), dtype=bool)
    padding[2, 5] = True
    observed[2, 5, :] = False
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"{split}-molecule-{offset + index}" for index in range(len(values))),
        read_ids=tuple(f"{split}-read-{offset + index}" for index in range(len(values))),
        experiment_uids=(f"{split}-experiment",) * len(values),
        modalities=("conversion",) * len(values),
        coordinates=np.arange(100, 106, dtype=np.int64),
        channel_names=("gpc_accessibility", "cpg_methylation"),
        values=values,
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        observed_mask=observed,
        availability_mask=availability,
        design_mask=design,
        padding_mask=padding,
    )


def _fitted_model():
    input_schema, label_schema = _schemas()
    train = _data("train")
    transform = fit_feature_transform(
        train,
        FeatureTransformSpec(imputation="mean", scaling="standard", indicators=()),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
    )
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=input_schema,
        parameters={
            "in_channels": 2,
            "stem_channels": 4,
            "block_channels": [4],
            "dilations": [1],
            "stem_kernel_size": 3,
            "kernel_size": 3,
            "dropout": 0.0,
            "hidden_dim": 4,
            "use_se": False,
            "use_attention_pool": False,
        },
    )
    torch.manual_seed(9)
    module = BUILTIN_MODEL_REGISTRY.build(resolved)
    fitted = FittedTorchModel(
        family=resolved.family,
        architecture=resolved,
        model=module,
        transform=transform,
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        training_config=TorchTrainingConfig(
            max_epochs=1,
            batch_size=2,
            patience=1,
            device="cpu",
        ),
        resolved_device="cpu",
        best_epoch=1,
        history=(TorchEpochRecord(epoch=1, train_loss=0.5, validation_loss=0.5),),
        validation_loss=0.5,
        test_loss=0.5,
    )
    return train, fitted


def _background(train, model):
    return sample_training_background(
        train,
        model.input_schema,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        max_observations=3,
        random_seed=5,
    )


def _request(
    model,
    data,
    method: str,
    parameters: dict,
    *,
    background=None,
    layer: str | None = None,
) -> InterpretabilityRequest:
    mask_kinds = tuple(
        mask.kind
        for mask in model.input_schema.masks
        if mask.kind in model.architecture.capabilities.supported_mask_kinds
    )
    return InterpretabilityRequest.create(
        method=method,
        model_id=MODEL_ID,
        dataset_snapshot_id=DATASET_ID,
        input_schema_hash=model.input_schema.schema_hash,
        split_role=data.split,
        cohort=f"{data.split}-natural",
        observation_uids=data.molecule_uids,
        target=ExplanationTarget(
            output_name="activity_target_logit",
            class_id=1,
            class_name="active",
        ),
        baseline=None if background is None else background.to_baseline(),
        layer=layer,
        mask_policy=ExplanationMaskPolicy.create(
            mask_kinds=mask_kinds,
            handling="forward masks through the model and zero invalid input attributions",
        ),
        decision=ExplanationDecisionProvenance("fixed"),
        parameters=parameters,
        random_seed=13,
    )


def _valid(data) -> np.ndarray:
    design = np.broadcast_to(data.design_mask, data.values.shape)
    return (
        data.observed_mask
        & data.availability_mask[:, None, :]
        & design
        & ~data.padding_mask[:, :, None]
    )


@pytest.mark.parametrize(
    ("method", "parameters", "needs_background"),
    [
        ("Saliency", {"absolute": False, "example_batch_size": 2}, False),
        ("InputXGradient", {"example_batch_size": 2}, False),
        (
            "IntegratedGradients",
            {
                "baseline_reduction": "mean",
                "example_batch_size": 2,
                "integration_method": "gausslegendre",
                "internal_batch_size": 8,
                "n_steps": 24,
            },
            True,
        ),
        ("DeepLift", {"baseline_reduction": "mean", "example_batch_size": 2}, True),
        (
            "GradientSHAP",
            {"example_batch_size": 2, "n_samples": 5, "stdevs": 0.0},
            True,
        ),
    ],
)
def test_input_methods_are_bounded_masked_and_schema_aligned(
    method: str,
    parameters: dict,
    needs_background: bool,
) -> None:
    train, model = _fitted_model()
    test = _data("test", offset=100)
    background = _background(train, model) if needs_background else None
    request = _request(model, test, method, parameters, background=background)
    model.model.train()

    result = explain_torch_model(model, test, request, background=background)

    assert result.axes == ("observation", "position", "channel")
    assert result.values.shape == test.values.shape
    assert np.all(result.values[~_valid(test)] == 0.0)
    assert np.any(result.values[_valid(test)] != 0.0)
    assert result.channels[0].biological_role == "accessibility"
    assert result.channels[0].sources[0].site_context == "GpC"
    assert result.channels[1].biological_role == "endogenous_methylation"
    assert result.metadata["maximum_executed_batch_size"] == 2
    assert result.metadata["mask_enforcement"] == "model_forward_and_explicit_zeroing"
    assert model.model.training is True
    if method in {"IntegratedGradients", "DeepLift", "GradientSHAP"}:
        assert result.convergence_delta is not None
        assert result.convergence_delta.shape == (len(test.molecule_uids),)
        assert result.metadata["background_hash"] == background.background_hash
    else:
        assert result.convergence_delta is None


def test_integrated_gradients_completeness_delta_is_bounded() -> None:
    train, model = _fitted_model()
    test = _data("test", offset=100)
    background = _background(train, model)
    request = _request(
        model,
        test,
        "IntegratedGradients",
        {
            "baseline_reduction": "mean",
            "example_batch_size": 2,
            "integration_method": "gausslegendre",
            "internal_batch_size": 16,
            "n_steps": 64,
        },
        background=background,
    )

    result = explain_torch_model(model, test, request, background=background)

    assert np.max(np.abs(result.convergence_delta)) < 0.15
    assert result.metadata["convergence_delta_policy"] == "captum_convergence_delta"


@pytest.mark.parametrize(
    ("method", "parameters", "expected_axes"),
    [
        (
            "LayerGradCam",
            {
                "attribute_to_layer_input": False,
                "example_batch_size": 2,
                "interpolate_mode": "linear",
                "relu_attributions": False,
            },
            ("observation", "position"),
        ),
        (
            "GuidedGradCam",
            {
                "attribute_to_layer_input": False,
                "example_batch_size": 2,
                "interpolate_mode": "linear",
            },
            ("observation", "position", "channel"),
        ),
    ],
)
def test_declared_layer_gradcam_is_bounded_and_masked(
    method: str,
    parameters: dict,
    expected_axes: tuple[str, ...],
) -> None:
    _train, model = _fitted_model()
    validation = _data("validation", offset=200)
    request = _request(
        model,
        validation,
        method,
        parameters,
        layer="attribution_layer",
    )

    result = explain_torch_model(model, validation, request)

    assert result.axes == expected_axes
    if method == "LayerGradCam":
        assert np.all(result.values[~_valid(validation).any(axis=2)] == 0.0)
        assert result.channels == ()
    else:
        assert np.all(result.values[~_valid(validation)] == 0.0)
        assert result.channels == model.input_schema.channels
    assert result.request.layer == "attribution_layer"


def test_invalid_parameters_fail_before_captum_import(monkeypatch) -> None:
    _train, model = _fitted_model()
    validation = _data("validation", offset=200)
    request = _request(
        model,
        validation,
        "Saliency",
        {"absolute": False, "example_batch_size": 2, "unknown": True},
    )

    def fail_import():
        raise AssertionError("Captum should not be imported")

    monkeypatch.setattr(
        "smftools.machine_learning.interpretability.neural._captum_classes",
        fail_import,
    )
    with pytest.raises(InterpretabilityContractError, match="parameters must be exactly"):
        explain_torch_model(model, validation, request)


def test_baseline_and_layer_requirements_fail_before_execution() -> None:
    train, model = _fitted_model()
    validation = _data("validation", offset=200)
    background = _background(train, model)
    baseline_request = _request(
        model,
        validation,
        "DeepLift",
        {"baseline_reduction": "mean", "example_batch_size": 2},
        background=background,
    )
    with pytest.raises(InterpretabilityContractError, match="checksummed training background"):
        explain_torch_model(model, validation, baseline_request)

    invalid_layer = _request(
        model,
        validation,
        "LayerGradCam",
        {
            "attribute_to_layer_input": False,
            "example_batch_size": 2,
            "interpolate_mode": "linear",
            "relu_attributions": False,
        },
        layer="backbone.0.conv2",
    )
    with pytest.raises(InterpretabilityContractError, match="not exposed"):
        explain_torch_model(model, validation, invalid_layer)


def test_gradient_shap_is_deterministic_for_a_fixed_request_seed() -> None:
    train, model = _fitted_model()
    test = _data("test", offset=100)
    background = _background(train, model)
    request = _request(
        model,
        test,
        "GradientSHAP",
        {"example_batch_size": 2, "n_samples": 5, "stdevs": 0.05},
        background=background,
    )

    first = explain_torch_model(model, test, request, background=background)
    second = explain_torch_model(model, test, request, background=background)

    assert first.result_id == second.result_id
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.convergence_delta, second.convergence_delta)
