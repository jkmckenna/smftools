from __future__ import annotations

import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch

from smftools.analysis.compute.ml_cnn import (
    CNNConfig as LegacyCNNConfig,
)
from smftools.analysis.compute.ml_cnn import (
    ResidualDilatedCNN1d as LegacyResidualCNN,
)
from smftools.analysis.compute.ml_cnn import (
    build_cnn_model,
    cnn_config_from_dict,
)
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.models import (
    BUILTIN_MODEL_REGISTRY,
    ResidualCNNConfig,
    ResidualCNNConfigError,
    ResidualDilatedCNN1d,
    TorchPredictor,
    build_residual_cnn,
)
from smftools.machine_learning.models.registry import ModelRegistryError
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit


def _schemas(*, modality: str = "deaminase") -> tuple[InputSchema, LabelSchema]:
    dataset: dict = {
        "modalities": [modality],
        "labels": {
            "column": "activity",
            "classes": {"inactive": 0, "active": 1},
        },
    }
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {"reads": dataset},
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["a"],
                    "validation_groups": ["b"],
                    "test_groups": ["c"],
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
    spec = plan.datasets["reads"]
    return (
        InputSchema.from_dataset(spec, reference="locus", n_positions=8),
        LabelSchema.from_plan_label(spec.labels),
    )


def _small_config(**overrides) -> ResidualCNNConfig:
    parameters = {
        "in_channels": 2,
        "stem_channels": 4,
        "block_channels": (4, 8),
        "dilations": (1, 2),
        "stem_kernel_size": 3,
        "kernel_size": 3,
        "dropout": 0.0,
        "hidden_dim": 6,
        "output_dim": 1,
        "use_se": False,
        "use_attention_pool": True,
    }
    parameters.update(overrides)
    return ResidualCNNConfig(**parameters)


def test_config_round_trip_is_strict_and_deeply_immutable() -> None:
    config = _small_config()
    restored = ResidualCNNConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.to_dict() == config.to_dict()
    assert isinstance(restored.block_channels, tuple)

    tampered = deepcopy(config.to_dict())
    tampered["unknown"] = 1
    with pytest.raises(ResidualCNNConfigError, match="fields must be exactly"):
        ResidualCNNConfig.from_dict(tampered)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"block_channels": (4, 8), "dilations": (1,)}, "same number"),
        ({"kernel_size": 4}, "must be odd"),
        ({"dropout": 1.0}, r"\[0, 1\)"),
        ({"hidden_dim": 0}, "positive integer"),
        ({"use_se": 1}, "must be boolean"),
    ],
)
def test_invalid_architecture_combinations_fail_on_construction(overrides, message) -> None:
    with pytest.raises(ResidualCNNConfigError, match=message):
        _small_config(**overrides)


@pytest.mark.parametrize(
    "config",
    [
        _small_config(block_channels=(4,), dilations=(1,), output_dim=1),
        _small_config(block_channels=(3, 5, 7), dilations=(1, 3, 9), output_dim=4),
    ],
)
def test_supported_depth_width_and_output_dimensions_have_stable_shapes(config) -> None:
    model = build_residual_cnn(config)
    values = torch.rand(5, config.in_channels, 17)

    logits = model(values)
    features = model.forward_features(values)

    assert logits.shape == (5, config.output_dim)
    assert features.shape == (5, config.block_channels[-1], 17)
    assert model.attribution_layer is model.backbone[-1].conv2


def test_separate_masks_exclude_invalid_values_from_features_and_pooling() -> None:
    torch.manual_seed(4)
    model = build_residual_cnn(_small_config()).eval()
    values = torch.rand(2, 2, 8)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 0, 2] = False
    availability = torch.tensor([[True, False], [True, True]])
    design = torch.ones(2, 8, dtype=torch.bool)
    design[:, 4] = False
    padding = torch.tensor(
        [[False, False, False, False, False, False, True, True]] * 2,
        dtype=torch.bool,
    )
    valid = observed & availability[:, :, None] & design[None, :, :] & ~padding[:, None, :]
    changed = values.clone()
    changed[~valid] = 10_000.0

    with torch.no_grad():
        expected = model(
            values,
            observed_mask=observed,
            availability_mask=availability,
            design_mask=design,
            padding_mask=padding,
        )
        observed_logits = model(
            changed,
            observed_mask=observed,
            availability_mask=availability,
            design_mask=design,
            padding_mask=padding,
        )

    torch.testing.assert_close(observed_logits, expected)


def test_mask_and_input_shape_errors_fail_before_convolution() -> None:
    model = build_residual_cnn(_small_config())
    values = torch.rand(2, 2, 8)

    with pytest.raises(ValueError, match="expected 2 channels"):
        model(torch.rand(2, 1, 8))
    with pytest.raises(ValueError, match="observed_mask"):
        model(values, observed_mask=torch.ones(2, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="must be boolean"):
        model(values, observed_mask=torch.ones_like(values))
    invalid = values.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        model(invalid)
    with pytest.raises(ValueError, match="at least one valid position"):
        model(values, observed_mask=torch.zeros_like(values, dtype=torch.bool))


def test_resolved_recipe_reconstructs_exact_architecture_and_state_dict() -> None:
    input_schema, _label_schema = _schemas()
    overrides = {
        "stem_channels": 4,
        "block_channels": [4, 8],
        "dilations": [1, 2],
        "stem_kernel_size": 3,
        "kernel_size": 3,
        "dropout": 0.0,
        "hidden_dim": 6,
        "use_se": False,
        "use_attention_pool": False,
    }
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=input_schema,
        parameters=overrides,
    )
    torch.manual_seed(8)
    source = BUILTIN_MODEL_REGISTRY.build(resolved).eval()
    restored_config = ResidualCNNConfig.from_dict(resolved.architecture.parameters)
    restored = build_residual_cnn(restored_config).eval()
    restored.load_state_dict(source.state_dict())
    values = torch.rand(3, 1, 8)

    with torch.no_grad():
        expected = source(values)
        actual = restored(values)

    assert restored_config == resolved.config
    assert tuple(source.state_dict()) == tuple(restored.state_dict())
    torch.testing.assert_close(actual, expected)


def test_recipe_rejects_channel_count_mismatch_before_build() -> None:
    input_schema, _label_schema = _schemas(modality="conversion")

    with pytest.raises(ModelRegistryError, match="input channels"):
        BUILTIN_MODEL_REGISTRY.resolve(
            "residual_dilated_cnn",
            input_schema=input_schema,
        )

    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=input_schema,
        parameters={"in_channels": 2},
    )
    assert resolved.config.in_channels == 2


def test_torch_predictor_forwards_contract_masks_in_channel_first_layout() -> None:
    input_schema, label_schema = _schemas()
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=input_schema,
        parameters={
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
    predictor = TorchPredictor(
        model=BUILTIN_MODEL_REGISTRY.build(resolved),
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=resolved.capabilities,
    )
    values = np.ones((2, 1, 8), dtype=np.float32)
    masks = {
        "observed": np.ones((2, 8, 1), dtype=bool),
        "availability": np.ones((2, 1), dtype=bool),
        "design": np.ones((8, 1), dtype=bool),
    }

    scores = predictor.predict_scores(values, masks=masks)

    assert scores.shape == (2, 2)


def test_analysis_cnn_symbols_are_compatibility_imports_of_canonical_family() -> None:
    assert LegacyCNNConfig is ResidualCNNConfig
    assert LegacyResidualCNN is ResidualDilatedCNN1d
    legacy_payload = _small_config().to_dict()
    legacy_payload.pop("output_dim")
    restored = cnn_config_from_dict(legacy_payload)
    model = build_cnn_model(restored)

    assert restored.output_dim == 1
    assert isinstance(model, ResidualDilatedCNN1d)
    assert not hasattr(model, "trainer")
    assert not hasattr(model, "adata")


def test_importing_plain_torch_family_does_not_import_lightning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import smftools.machine_learning.models.residual_cnn; "
                "assert 'pytorch_lightning' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
