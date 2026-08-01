from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB

from smftools.machine_learning.contracts import (
    ML_CAPABILITY_SCHEMA_VERSION,
    InputSchema,
    LabelSchema,
    PredictorCapabilities,
)
from smftools.machine_learning.models.protocols import (
    PredictorError,
    PredictorProtocol,
    SklearnPredictor,
    TorchPredictor,
    adapt_loaded_predictor,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit


def _schemas() -> tuple[InputSchema, LabelSchema]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["a"],
                    "validation_groups": ["b"],
                    "test_groups": ["c"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                }
            },
        }
    )
    dataset = plan.datasets["reads"]
    return (
        InputSchema.from_dataset(dataset, reference="locus", n_positions=2),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _capabilities(
    backend: str,
    *,
    probability: bool = True,
    supported_masks: tuple[str, ...] = (),
    required_masks: tuple[str, ...] = (),
) -> PredictorCapabilities:
    return PredictorCapabilities(
        schema_version=ML_CAPABILITY_SCHEMA_VERSION,
        backend=backend,
        probability_output=probability,
        incremental_fit=False,
        sample_weights=False,
        position_masks=bool(supported_masks),
        gradients=backend == "torch",
        convolutional_layers=False,
        attention_data=False,
        supported_mask_kinds=supported_masks,
        required_mask_kinds=required_masks,
    )


def test_sklearn_adapter_returns_ordered_backend_neutral_outputs() -> None:
    input_schema, label_schema = _schemas()
    features = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.asarray([0, 0, 1, 1])
    model = BernoulliNB().fit(features, labels)
    predictor = SklearnPredictor(
        model=model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("sklearn"),
    )

    assert isinstance(predictor, PredictorProtocol)
    np.testing.assert_array_equal(predictor.predict(features), model.predict(features))
    np.testing.assert_allclose(
        predictor.predict_probabilities(features), model.predict_proba(features)
    )
    assert predictor.predict_scores(features).shape == (4, 2)


def test_sklearn_adapter_rejects_wrong_class_order_and_masks() -> None:
    input_schema, label_schema = _schemas()
    features = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    model = BernoulliNB().fit(features, np.asarray([0, 1]))
    model.classes_ = np.asarray([1, 0])

    with pytest.raises(PredictorError, match="classes_ differ"):
        SklearnPredictor(
            model=model,
            input_schema=input_schema,
            label_schema=label_schema,
            capabilities=_capabilities("sklearn"),
        )

    model.classes_ = np.asarray([0, 1])
    predictor = SklearnPredictor(
        model=model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("sklearn"),
    )
    observed = np.ones((2, 2, 1), dtype=bool)
    with pytest.raises(ValueError, match="does not support masks"):
        predictor.predict(features, masks={"observed": observed})


def test_probability_and_explanation_capabilities_fail_before_execution() -> None:
    input_schema, label_schema = _schemas()
    features = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    model = BernoulliNB().fit(features, np.asarray([0, 1]))
    predictor = SklearnPredictor(
        model=model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("sklearn", probability=False),
    )

    with pytest.raises(PredictorError, match="probability_output"):
        predictor.predict_probabilities(features)
    with pytest.raises(PredictorError, match="gradients"):
        predictor.require_capabilities("gradients")


def test_adapter_rejects_incompatible_input_schema() -> None:
    input_schema, label_schema = _schemas()
    features = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    model = BernoulliNB().fit(features, np.asarray([0, 1]))
    predictor = SklearnPredictor(
        model=model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("sklearn"),
    )
    raw = input_schema.to_dict()
    raw["reference"] = "different-locus"

    with pytest.raises(ValueError, match="input schema is incompatible"):
        predictor.predict(features, input_schema=InputSchema.from_dict(raw))


def test_torch_adapter_normalizes_binary_logits_and_restores_training_mode() -> None:
    import torch

    class BinaryModule(torch.nn.Module):
        def forward(self, values):
            return values.mean(dim=(1, 2), keepdim=False).unsqueeze(1) - 0.5

    input_schema, label_schema = _schemas()
    model = BinaryModule().train()
    predictor = TorchPredictor(
        model=model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("torch"),
    )
    values = np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)

    scores = predictor.predict_scores(values)
    probabilities = predictor.predict_probabilities(values)

    assert scores.shape == (2, 2)
    np.testing.assert_array_equal(predictor.predict(values), [0, 1])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 1] < 0.5 < probabilities[1, 1]
    assert model.training


def test_torch_adapter_requires_and_forwards_declared_masks() -> None:
    import torch

    class MaskedModule(torch.nn.Module):
        def forward(self, values, observed_mask):
            return (values * observed_mask).mean(dim=(1, 2), keepdim=False).unsqueeze(1)

    input_schema, label_schema = _schemas()
    predictor = TorchPredictor(
        model=MaskedModule(),
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities(
            "torch",
            supported_masks=("observed",),
            required_masks=("observed",),
        ),
    )
    values = np.ones((2, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="required masks were not provided"):
        predictor.predict(values)

    predictions = predictor.predict(
        values,
        masks={"observed": np.ones((2, 2, 1), dtype=bool)},
    )
    np.testing.assert_array_equal(predictions, [1, 1])


def test_torch_adapter_rejects_non_channel_first_values() -> None:
    import torch

    class BinaryModule(torch.nn.Module):
        def forward(self, values):
            return values.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

    input_schema, label_schema = _schemas()
    predictor = TorchPredictor(
        model=BinaryModule(),
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("torch"),
    )

    with pytest.raises(PredictorError, match="channel-first"):
        predictor.predict(np.ones((2, 2, 1), dtype=np.float32))


def test_loaded_backend_adapter_dispatch_is_confined_to_construction() -> None:
    input_schema, label_schema = _schemas()
    features = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    model = BernoulliNB().fit(features, np.asarray([0, 1]))

    predictor = adapt_loaded_predictor(
        model,
        input_schema=input_schema,
        label_schema=label_schema,
        capabilities=_capabilities("sklearn"),
    )

    assert isinstance(predictor, PredictorProtocol)
    assert predictor.backend == "sklearn"
