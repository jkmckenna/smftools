"""MLR-09: small, interpretable convolutional scanners."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smftools.machine_learning.models.conv_scanner import (  # noqa: E402
    ConvScannerConfig,
    ConvScannerConfigError,
    build_conv_scanner,
)
from smftools.machine_learning.models.registry import (  # noqa: E402
    BUILTIN_MODEL_REGISTRY,
    CONV_SCANNER_RECIPES,
)
from smftools.machine_learning.models.residual_cnn import effective_span  # noqa: E402

pytestmark = pytest.mark.unit


def _model(**fields):
    torch.manual_seed(0)
    model = build_conv_scanner(ConvScannerConfig(in_channels=1, **fields))
    model.eval()
    return model


def _inputs(values: np.ndarray):
    tensor = torch.as_tensor(values[:, None, :], dtype=torch.float32)
    return tensor, torch.ones_like(tensor, dtype=torch.bool)


def test_receptive_field_and_stride() -> None:
    one = ConvScannerConfig(in_channels=1, filters=(8,), kernel_sizes=(21,))
    assert (one.receptive_field, one.feature_stride) == (21, 1)
    two = ConvScannerConfig(in_channels=1, filters=(16, 16), kernel_sizes=(15, 9), downsample=4)
    # 15, then a 4-wide pool (+3), then 9 taps four positions apart (+32).
    assert (two.receptive_field, two.feature_stride) == (50, 4)
    dilated = ConvScannerConfig(in_channels=1, filters=(4,), kernel_sizes=(5,), dilations=(3,))
    assert dilated.receptive_field == 13


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kernel_sizes": (20,)}, "odd"),
        ({"filters": (8, 8), "kernel_sizes": (9,)}, "one value per layer"),
        ({"pooling": ("median",)}, "pooling"),
        ({"pooling": ("max", "max")}, "pooling"),
        ({"kernel_sizes": (51,), "max_receptive_field": 21}, "exceeds"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_configs_are_refused(fields, message) -> None:
    with pytest.raises(ConvScannerConfigError, match=message):
        ConvScannerConfig(in_channels=1, **fields)


def test_config_round_trips() -> None:
    config = ConvScannerConfig(
        in_channels=2, filters=(16, 8), kernel_sizes=(15, 9), downsample=4, pooling=("max", "avg")
    )
    assert ConvScannerConfig.from_dict(config.to_dict()) == config


def test_shapes_with_and_without_downsampling() -> None:
    values, observed = _inputs(np.random.default_rng(0).random((3, 101)).round())
    flat = _model(filters=(4,), kernel_sizes=(9,))
    assert flat(values, observed_mask=observed).shape == (3, 1)
    assert flat.forward_features(values, observed_mask=observed).shape == (3, 4, 101)
    stacked = _model(filters=(4, 4), kernel_sizes=(9, 5), downsample=4)
    assert stacked(values, observed_mask=observed).shape == (3, 1)
    assert stacked.forward_features(values, observed_mask=observed).shape == (3, 4, 26)


def test_global_max_ignores_where_a_pattern_is_adaptive_bins_do_not() -> None:
    pattern = np.array([1, 1, 0, 0, 1, 1], dtype=float)

    def molecule(start):
        values = np.zeros((1, 200))
        values[0, start : start + pattern.size] = pattern
        return _inputs(values)

    scanner = _model(filters=(4,), kernel_sizes=(9,))
    binned = _model(filters=(4,), kernel_sizes=(9,), adaptive_bins=8)
    with torch.no_grad():
        a, b = (scanner(*molecule(s)[:1], observed_mask=molecule(s)[1]) for s in (40, 140))
        torch.testing.assert_close(a, b)
        c, d = (binned(*molecule(s)[:1], observed_mask=molecule(s)[1]) for s in (40, 140))
        assert not torch.allclose(c, d)


def test_effective_span_respects_downsampling() -> None:
    values, observed = _inputs(np.random.default_rng(1).random((4, 400)).round())
    model = _model(filters=(4, 4), kernel_sizes=(9, 5), downsample=4)
    measured = effective_span(model, values, observed_mask=observed)
    assert measured["receptive_field"] == model.config.receptive_field
    assert 1 <= measured["effective_span_50"] <= measured["effective_span_90"]
    assert measured["effective_span_90"] <= model.config.receptive_field + model.feature_stride


def test_registry_family_and_recipes() -> None:
    definition = BUILTIN_MODEL_REGISTRY.definition("conv_scanner")
    assert definition.model_class == "spatial"
    assert definition.default_explanation == "IntegratedGradients"
    for name in CONV_SCANNER_RECIPES:
        config = ConvScannerConfig.from_dict(BUILTIN_MODEL_REGISTRY.recipe(f"{name}_v1").parameters)
        model = build_conv_scanner(config)
        assert sum(p.numel() for p in model.parameters()) < 40_000


@pytest.mark.parametrize("family", ["scanner", "scanner_attention", "residual"])
def test_a_molecule_with_no_valid_position_scores_from_the_bias(family) -> None:
    from smftools.machine_learning.models.residual_cnn import (
        ResidualCNNConfig,
        build_residual_cnn,
    )

    torch.manual_seed(0)
    if family == "residual":
        model = build_residual_cnn(ResidualCNNConfig(in_channels=1))
    else:
        pooling = ("max", "avg", "attention") if family == "scanner_attention" else ("max",)
        model = build_conv_scanner(
            ConvScannerConfig(in_channels=1, filters=(4,), kernel_sizes=(9,), pooling=pooling)
        )
    model.eval()
    values, observed = _inputs(np.random.default_rng(2).random((3, 60)).round())
    observed[1] = False  # no observed site in the region
    logits = model(values, observed_mask=observed)
    assert torch.isfinite(logits).all()
    alone = model(values[[0, 2]], observed_mask=observed[[0, 2]])
    torch.testing.assert_close(logits[[0, 2]], alone)
    # An empty molecule's score does not depend on its (unobserved) values.
    other = values.clone()
    other[1] = 1 - other[1]
    torch.testing.assert_close(model(other, observed_mask=observed)[1], logits[1])
    model.train()
    model(values, observed_mask=observed).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
