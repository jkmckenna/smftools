"""MLR-08: residual CNNs with a stated, enforced detector span."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smftools.machine_learning.models.registry import (  # noqa: E402
    BUILTIN_MODEL_REGISTRY,
    DETECTOR_SCALE_DILATIONS,
)
from smftools.machine_learning.models.residual_cnn import (  # noqa: E402
    ResidualCNNConfig,
    ResidualCNNConfigError,
    build_residual_cnn,
    effective_span,
)

pytestmark = pytest.mark.unit

LADDER = {
    "rcnn_subnucleosome": 113,
    "rcnn_2_3_nucleosomes": 513,
    "rcnn_4_6_nucleosomes": 1025,
    "rcnn_full_locus": 5121,
}


def _small(**overrides) -> ResidualCNNConfig:
    """A narrow bounded model: dilations 1, 2 -> receptive field 1 + 8 + 8 + 16 = 33."""
    parameters = {
        "in_channels": 1,
        "stem_channels": 4,
        "block_channels": (4, 4),
        "dilations": (1, 2),
        "use_se": False,
        "mask_channels": True,
        "span_masking": True,
        "max_receptive_field": 33,
        "dropout": 0.0,
    }
    parameters.update(overrides)
    return ResidualCNNConfig(**parameters)


def _model(config: ResidualCNNConfig):
    torch.manual_seed(0)
    model = build_residual_cnn(config)
    model.eval()
    return model


@pytest.mark.parametrize(("name", "span"), sorted(LADDER.items()))
def test_recipes_have_their_stated_receptive_fields(name: str, span: int) -> None:
    config = ResidualCNNConfig.from_dict(BUILTIN_MODEL_REGISTRY.recipe(f"{name}_v1").parameters)
    assert config.receptive_field == span == config.max_receptive_field
    assert config.dilations == DETECTOR_SCALE_DILATIONS[name]
    assert not config.use_se and config.mask_channels and config.span_masking
    assert set(config.block_channels) == {64}


def test_receptive_field_formula() -> None:
    config = ResidualCNNConfig(in_channels=1)  # the original recipe
    assert config.receptive_field == 1 + (9 - 1) + sum(2 * (5 - 1) * d for d in config.dilations)


def test_an_over_wide_or_squeeze_excite_model_is_refused() -> None:
    with pytest.raises(ResidualCNNConfigError, match="exceeds max_receptive_field"):
        _small(max_receptive_field=20)
    with pytest.raises(ResidualCNNConfigError, match="use_se=False"):
        _small(use_se=True)


def test_new_fields_are_left_out_when_unset() -> None:
    plain = ResidualCNNConfig(in_channels=2)
    assert {"mask_channels", "span_masking", "max_receptive_field"}.isdisjoint(plain.to_dict())
    assert ResidualCNNConfig.from_dict(plain.to_dict()) == plain
    bounded = _small()
    assert ResidualCNNConfig.from_dict(bounded.to_dict()) == bounded


def _features(model, values, observed):
    with torch.no_grad():
        return model.forward_features(
            torch.as_tensor(values, dtype=torch.float32),
            observed_mask=torch.as_tensor(observed),
        ).numpy()


def test_a_perturbation_moves_features_only_within_half_the_receptive_field() -> None:
    config = _small()
    model = _model(config)
    rng = np.random.default_rng(0)
    values = (rng.random((2, 1, 200)) < 0.5).astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    before = _features(model, values, observed)
    position = 100
    values[:, 0, position] = 1 - values[:, 0, position]
    changed = np.flatnonzero(
        np.abs(_features(model, values, observed) - before).sum(axis=(0, 1)) > 1e-6
    )
    half = config.receptive_field // 2
    assert changed.size and changed.min() >= position - half and changed.max() <= position + half


def test_squeeze_excite_lets_a_perturbation_reach_everywhere() -> None:
    model = _model(_small(use_se=True, max_receptive_field=None))
    rng = np.random.default_rng(0)
    values = (rng.random((2, 1, 200)) < 0.5).astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    before = _features(model, values, observed)
    values[:, 0, 100] = 1 - values[:, 0, 100]
    changed = np.abs(_features(model, values, observed) - before).sum(axis=(0, 1)) > 1e-6
    assert changed[:30].any() and changed[-30:].any()  # far outside the 33-position window


def test_mask_channels_tell_no_site_from_an_unmodified_site() -> None:
    model = _model(_small())
    assert model.stem[0].in_channels == 2  # signal + validity
    values = np.zeros((1, 1, 60), dtype=np.float32)
    observed = np.ones_like(values, dtype=bool)
    unobserved = observed.copy()
    unobserved[:, :, 25:35] = False  # the same zeros, now "no site"
    near = slice(20, 40)
    a, b = _features(model, values, observed), _features(model, values, unobserved)
    assert not np.allclose(a[:, :, near], b[:, :, near])


def test_span_masking_carries_features_between_sparse_sites() -> None:
    values = np.zeros((1, 1, 100), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    observed[:, :, 10:90:10] = True  # a site every 10 positions
    values[:, :, 10:90:10] = 1.0
    between = [15, 25, 45]
    spanned = _features(_model(_small()), values, observed)
    assert np.abs(spanned[0][:, between]).sum() > 0
    sites_only = _features(_model(_small(span_masking=False)), values, observed)
    assert np.abs(sites_only[0][:, between]).sum() == 0
    # Outside the read (before its first site) features stay zero either way.
    assert np.abs(spanned[0][:, :10]).sum() == 0


def test_predictions_do_not_depend_on_where_a_pattern_sits() -> None:
    model = _model(_small())
    pattern = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.float32)

    def logits(start):
        values = np.zeros((1, 1, 300), dtype=np.float32)
        values[0, 0, start : start + pattern.size] = pattern
        observed = np.ones_like(values, dtype=bool)
        with torch.no_grad():
            return model(torch.as_tensor(values), observed_mask=torch.as_tensor(observed)).numpy()

    # Far from the edges, a translated pattern gives the same pooled detectors.
    np.testing.assert_allclose(logits(100), logits(180), atol=1e-5)


def test_effective_span_is_bounded_and_can_be_much_narrower() -> None:
    config = _small()
    model = _model(config)
    rng = np.random.default_rng(1)
    values = torch.as_tensor((rng.random((4, 1, 200)) < 0.5).astype(np.float32))
    observed = torch.ones_like(values, dtype=torch.bool)
    measured = effective_span(model, values, observed_mask=observed)
    assert measured["receptive_field"] == 33
    assert 1 <= measured["effective_span_50"] <= measured["effective_span_90"] <= 33

    # Only centre taps: every detector sees its own position alone.
    centred = _model(config)
    with torch.no_grad():
        for module in centred.modules():
            if isinstance(module, torch.nn.Conv1d) and module.kernel_size[0] > 1:
                weight = module.weight
                middle = weight.shape[-1] // 2
                weight[..., :middle] = 0
                weight[..., middle + 1 :] = 0
    narrow = effective_span(centred, values, observed_mask=observed)
    assert narrow["effective_span_90"] == 1 < narrow["receptive_field"]
