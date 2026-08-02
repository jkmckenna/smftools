"""Tests for backend-neutral prediction and evaluation records."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smftools.machine_learning.evaluation import (
    EvaluationContractError,
    PredictionResult,
    ThresholdProvenance,
    TrainingEvent,
    TrainingHistory,
    aggregate_fold_metrics,
    evaluate_predictions,
    fit_binary_threshold,
    sklearn_training_history,
    torch_training_history,
)
from smftools.machine_learning.inference import (
    SklearnPredictionResult,
    TorchPredictionResult,
)

pytestmark = pytest.mark.unit


def _binary_predictions(
    *,
    split: str = "validation",
    probabilities: np.ndarray | None = None,
) -> PredictionResult:
    truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    if probabilities is None:
        positive = np.asarray([0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.6])
        probabilities = np.column_stack((1 - positive, positive))
    class_ids = np.argmax(probabilities, axis=1)
    return PredictionResult(
        molecule_uids=tuple(f"molecule-{index}" for index in range(len(truth))),
        class_ids=class_ids,
        scores=np.log(np.clip(probabilities, 1e-6, 1.0)),
        probabilities=probabilities,
        class_order=("inactive", "active"),
        split=split,
        experiment_uids=("experiment-a",) * 4 + ("experiment-b",) * 4,
        modalities=("deaminase",) * 4 + ("conversion",) * 4,
        groups=("sample-a",) * 4 + ("sample-b",) * 4,
        truth_class_ids=truth,
        positive_class="active",
        cohort=f"{split}-natural",
        model_id="model-1",
    )


def test_backend_result_names_alias_one_concrete_prediction_contract() -> None:
    assert SklearnPredictionResult is PredictionResult
    assert TorchPredictionResult is PredictionResult

    predictions = _binary_predictions()
    assert not predictions.probabilities.flags.writeable
    assert predictions.positive_class == "active"
    assert predictions.n_observations == 8


def test_prediction_rows_can_include_omit_or_hash_sensitive_identity() -> None:
    predictions = _binary_predictions()

    included = predictions.to_rows()[0]
    omitted = predictions.to_rows(identity_policy="omit")[0]
    hashed = predictions.to_rows(identity_policy="hash", hash_salt="export-1")[0]

    assert included["molecule_uid"] == "molecule-0"
    assert "molecule_uid" not in omitted
    assert hashed["molecule_uid"] != "molecule-0"
    assert hashed["modality"] == "deaminase"
    assert included["truth_class_id"] == 0
    with pytest.raises(EvaluationContractError, match="hash_salt"):
        predictions.to_rows(identity_policy="hash")

    restored = PredictionResult.from_rows(
        predictions.to_rows(),
        class_order=predictions.class_order,
        positive_class=predictions.positive_class,
    )
    np.testing.assert_array_equal(restored.truth_class_ids, predictions.truth_class_ids)
    np.testing.assert_allclose(restored.probabilities, predictions.probabilities)
    assert restored.modalities == predictions.modalities
    with pytest.raises(EvaluationContractError, match="required columns"):
        PredictionResult.from_rows(
            predictions.to_rows(identity_policy="omit"),
            class_order=predictions.class_order,
        )


def test_validation_threshold_is_reused_on_test_without_refitting() -> None:
    validation = _binary_predictions()
    threshold = fit_binary_threshold(validation, method="f1")
    test = _binary_predictions(split="test")

    result = evaluate_predictions(test, threshold=threshold)

    assert result.threshold == threshold
    assert threshold.fitted_split == "validation"
    assert threshold.model_id == "model-1"
    assert threshold.class_order == test.class_order
    assert {record.scope for record in result.metrics} == {"pooled", "modality"}
    assert {record.modality for record in result.class_balance} == {
        None,
        "conversion",
        "deaminase",
    }
    pooled_support = {
        record.class_name: record.count
        for record in result.class_balance
        if record.scope == "pooled"
    }
    assert pooled_support == {"inactive": 4, "active": 4}
    assert {curve.kind for curve in result.curves} == {
        "roc",
        "precision_recall",
        "calibration",
    }

    with pytest.raises(EvaluationContractError, match="locked test"):
        ThresholdProvenance(
            value=0.5,
            positive_class="active",
            method="f1",
            fitted_split="test",
            fitted_cohort="test",
        )
    with pytest.raises(EvaluationContractError, match="train or validation"):
        fit_binary_threshold(test)
    with pytest.raises(EvaluationContractError, match="model_id"):
        evaluate_predictions(
            PredictionResult(
                **{
                    **test.__dict__,
                    "model_id": "model-2",
                }
            ),
            threshold=threshold,
        )


def test_multiclass_metrics_use_persisted_class_order() -> None:
    truth = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    predictions = PredictionResult(
        molecule_uids=tuple(f"multi-{index}" for index in range(len(truth))),
        class_ids=np.argmax(probabilities, axis=1),
        scores=np.log(probabilities),
        probabilities=probabilities,
        class_order=("closed", "open", "intermediate"),
        split="test",
        experiment_uids=("experiment",) * len(truth),
        modalities=("direct",) * len(truth),
        truth_class_ids=truth,
        cohort="three-state-test",
        model_id="model-multi",
    )

    result = evaluate_predictions(predictions)
    per_class_auc = {
        metric.class_name: metric.value
        for metric in result.metrics
        if metric.scope == "pooled" and metric.name == "roc_auc"
    }

    assert tuple(per_class_auc) == predictions.class_order
    assert all(value == pytest.approx(1.0) for value in per_class_auc.values())
    assert result.confusion[0].class_order == predictions.class_order


def test_fold_aggregation_uses_equal_fold_mean_and_sample_sd() -> None:
    first = evaluate_predictions(_binary_predictions(split="test"), by_modality=False)
    changed = _binary_predictions(
        split="test",
        probabilities=np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        ),
    )
    second = evaluate_predictions(changed, by_modality=False)

    summaries = aggregate_fold_metrics({"fold-a": first, "fold-b": second})
    accuracy = next(
        summary for summary in summaries if summary.name == "accuracy" and summary.scope == "pooled"
    )

    assert len(accuracy.values) == 2
    assert accuracy.mean == pytest.approx(np.mean(accuracy.values))
    assert accuracy.standard_deviation == pytest.approx(np.std(accuracy.values, ddof=1))
    assert accuracy.uncertainty == "sample_standard_deviation"


def test_training_history_does_not_require_fabricated_epochs() -> None:
    sklearn_history = TrainingHistory(backend="sklearn")
    torch_history = TrainingHistory(
        backend="torch",
        events=(
            TrainingEvent(
                event_index=0,
                event_type="epoch_completed",
                epoch=1,
                metrics={"train_loss": 0.8, "validation_loss": 0.9},
            ),
        ),
    )

    assert sklearn_history.events == ()
    assert torch_history.events[0].epoch == 1

    one_shot = sklearn_training_history(SimpleNamespace(fit_mode="fit"))
    iterative = torch_training_history(
        SimpleNamespace(
            history=(
                SimpleNamespace(epoch=1, train_loss=0.8, validation_loss=0.9),
                SimpleNamespace(epoch=2, train_loss=0.6, validation_loss=0.7),
            )
        )
    )
    assert one_shot.events[0].event_type == "fit_completed"
    assert one_shot.events[0].epoch is None
    assert [event.epoch for event in iterative.events] == [1, 2]
