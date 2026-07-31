"""Passing behavioral contracts worth preserving during ML migration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as torch_functional

from smftools import optional_imports
from smftools.analysis.compute.ml_cnn import (
    CNNConfig,
    build_cnn_input,
    build_cnn_model,
)
from smftools.analysis.compute.ml_metrics import (
    build_binary_classifier,
    fit_classifier,
    predict_binary_scores,
)
from smftools.analysis.compute.ml_splits import build_leave_one_group_out_splits

pytestmark = pytest.mark.unit


def _small_cnn_config() -> CNNConfig:
    return CNNConfig(
        in_channels=2,
        stem_channels=4,
        block_channels=(4,),
        dilations=(1,),
        stem_kernel_size=3,
        kernel_size=3,
        dropout=0.0,
        hidden_dim=4,
        use_se=False,
        use_attention_pool=False,
    )


def test_cnn_input_preserves_signal_observed_and_design_masks():
    matrix = np.array(
        [[1.0, np.nan, 0.0, 1.0], [np.nan, 1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    learnable = np.array([True, False, True, True])

    encoded = build_cnn_input(
        matrix,
        feature_labels=np.array(["ref:10", "ref:20", "ref:30", "ref:40"]),
        include_positional=True,
        include_spacing=True,
        learnable_mask=learnable,
        include_design_mask=True,
    )

    assert encoded.shape == (2, 6, 4)
    np.testing.assert_array_equal(encoded[:, 0, 1], np.zeros(2))
    np.testing.assert_array_equal(encoded[:, 1, 1], np.zeros(2))
    np.testing.assert_array_equal(encoded[:, 2, :], np.broadcast_to(~learnable, (2, 4)))
    assert set(np.unique(encoded[:, 1, :])) <= {0.0, 1.0}


def test_leave_one_group_out_fixture_has_disjoint_three_group_folds():
    metadata = pd.DataFrame(
        {
            "group": np.repeat(["sample-a", "sample-b", "sample-c"], 4),
            "label": [0, 1, 0, 1] * 3,
        }
    )

    folds = build_leave_one_group_out_splits(metadata, group_col="group", label_col="label")

    assert len(folds) == 3
    for fold in folds:
        train_groups = set(metadata.iloc[fold["train_idx"]]["group"])
        test_groups = set(metadata.iloc[fold["test_idx"]]["group"])
        assert train_groups.isdisjoint(test_groups)


def test_leave_one_group_out_multiclass_fixture_preserves_all_classes():
    metadata = pd.DataFrame(
        {
            "group": np.repeat(["sample-a", "sample-b", "sample-c"], 3),
            "label": [0, 1, 2] * 3,
        }
    )

    folds = build_leave_one_group_out_splits(metadata, group_col="group", label_col="label")

    assert len(folds) == 3
    for fold in folds:
        assert set(metadata.iloc[fold["train_idx"]]["label"]) == {0, 1, 2}
        assert set(metadata.iloc[fold["test_idx"]]["label"]) == {0, 1, 2}


def test_sklearn_validation_data_does_not_refit_training_imputer():
    estimator = build_binary_classifier("bernoulli_nb")
    train_features = np.array(
        [[0.0, np.nan], [0.0, 1.0], [1.0, 1.0], [1.0, np.nan]],
        dtype=np.float32,
    )
    train_labels = np.array([0, 0, 1, 1])
    validation_features = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    fitted = fit_classifier(estimator, train_features, train_labels)
    before = fitted.named_steps["imputer"].statistics_.copy()

    scores = predict_binary_scores(fitted, validation_features)

    np.testing.assert_array_equal(fitted.named_steps["imputer"].statistics_, before)
    assert scores.shape == (2,)


def test_plain_torch_model_completes_one_optimization_step():
    torch.manual_seed(7)
    model = build_cnn_model(_small_cnn_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    features = torch.rand(6, 2, 8)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad()
    logits = model(features).squeeze(1)
    loss = torch_functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        not torch.equal(previous, current) for previous, current in zip(before, model.parameters())
    )


def test_plain_torch_state_dict_round_trip_preserves_predictions(tmp_path):
    torch.manual_seed(11)
    config = _small_cnn_config()
    source = build_cnn_model(config).eval()
    features = torch.rand(3, 2, 8)
    with torch.no_grad():
        expected = source(features)
    checkpoint = tmp_path / "weights.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = build_cnn_model(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        observed = restored(features)

    torch.testing.assert_close(observed, expected)


def test_optional_import_error_names_install_extra(monkeypatch):
    def _missing_import(_package):
        raise ModuleNotFoundError("missing dependency")

    monkeypatch.setattr(optional_imports, "import_module", _missing_import)

    with pytest.raises(ModuleNotFoundError, match=r"smftools\[ml-extended\]"):
        optional_imports.require(
            "missing_ml_dependency",
            extra="ml-extended",
            purpose="ML behavior baseline",
        )
