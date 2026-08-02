"""Classification evaluation derived only from backend-neutral prediction rows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from smftools.optional_imports import require

from .contracts import (
    CalibrationProvenance,
    ClassBalanceRecord,
    ConfusionRecord,
    CurveRecord,
    EvaluationContractError,
    EvaluationResult,
    FoldMetricSummary,
    MetricRecord,
    PredictionResult,
    ThresholdProvenance,
)


def fit_binary_threshold(
    predictions: PredictionResult,
    *,
    method: str = "f1",
) -> ThresholdProvenance:
    """Select a binary threshold from labeled non-test predictions.

    Supported methods are ``f1`` and ``youden_j``. The returned provenance is
    subsequently reusable on test predictions and cannot identify test as its
    fitting split.
    """
    if predictions.split in {"test", "inference"}:
        raise EvaluationContractError("threshold fitting requires labeled train or validation data")
    if predictions.truth_class_ids is None:
        raise EvaluationContractError("threshold fitting requires truth_class_ids")
    if len(predictions.class_order) != 2 or predictions.positive_class is None:
        raise EvaluationContractError("threshold fitting currently requires binary predictions")
    method = str(method).strip().lower()
    if method not in {"f1", "youden_j"}:
        raise EvaluationContractError("threshold method must be 'f1' or 'youden_j'")
    metrics = require("sklearn.metrics", extra="ml-base", purpose="threshold selection")
    positive_id = predictions.class_order.index(predictions.positive_class)
    truth = (predictions.truth_class_ids == positive_id).astype(np.int64)
    if np.unique(truth).size != 2:
        raise EvaluationContractError("threshold fitting requires both binary classes")
    probability = predictions.probabilities[:, positive_id]
    if method == "f1":
        precision, recall, thresholds = metrics.precision_recall_curve(truth, probability)
        if thresholds.size == 0:
            raise EvaluationContractError("threshold selection produced no candidates")
        denominator = precision[:-1] + recall[:-1]
        objective = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
    else:
        false_positive, true_positive, thresholds = metrics.roc_curve(truth, probability)
        finite = np.isfinite(thresholds)
        thresholds = thresholds[finite]
        objective = (true_positive - false_positive)[finite]
    best = int(np.argmax(objective))
    return ThresholdProvenance(
        value=float(np.clip(thresholds[best], 0.0, 1.0)),
        positive_class=predictions.positive_class,
        method=method,
        fitted_split=predictions.split,
        fitted_cohort=predictions.cohort,
        model_id=predictions.model_id,
        class_order=predictions.class_order,
    )


def _decision_ids(
    predictions: PredictionResult,
    threshold: ThresholdProvenance | None,
) -> np.ndarray:
    if threshold is None:
        return predictions.class_ids
    if len(predictions.class_order) != 2:
        raise EvaluationContractError("decision thresholds currently apply only to binary results")
    if threshold.positive_class != predictions.positive_class:
        raise EvaluationContractError("threshold positive_class differs from prediction schema")
    if threshold.class_order and threshold.class_order != predictions.class_order:
        raise EvaluationContractError("threshold class_order differs from prediction schema")
    if (
        threshold.model_id is not None
        and predictions.model_id is not None
        and threshold.model_id != predictions.model_id
    ):
        raise EvaluationContractError("threshold model_id differs from predictions")
    positive_id = predictions.class_order.index(threshold.positive_class)
    negative_id = 1 - positive_id
    return np.where(
        predictions.probabilities[:, positive_id] >= threshold.value,
        positive_id,
        negative_id,
    ).astype(np.int64)


def _metric(
    predictions: PredictionResult,
    *,
    name: str,
    value: float | None,
    scope: str,
    modality: str | None,
    class_name: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        name=name,
        value=value,
        n_observations=predictions.n_observations,
        split=predictions.split,
        cohort=predictions.cohort or predictions.split,
        scope=scope,
        modality=modality,
        class_name=class_name,
        model_id=predictions.model_id,
    )


def _curve(
    predictions: PredictionResult,
    *,
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    scope: str,
    modality: str | None,
    class_name: str,
    thresholds: np.ndarray | None = None,
) -> CurveRecord:
    return CurveRecord(
        kind=kind,
        x=x,
        y=y,
        thresholds=thresholds,
        split=predictions.split,
        cohort=predictions.cohort or predictions.split,
        scope=scope,
        modality=modality,
        class_name=class_name,
    )


def _evaluate_slice(
    predictions: PredictionResult,
    *,
    threshold: ThresholdProvenance | None,
    scope: str,
    modality: str | None,
) -> tuple[
    tuple[MetricRecord, ...],
    tuple[CurveRecord, ...],
    ConfusionRecord,
    tuple[ClassBalanceRecord, ...],
]:
    if predictions.truth_class_ids is None:
        raise EvaluationContractError("evaluation requires truth_class_ids")
    metrics = require("sklearn.metrics", extra="ml-base", purpose="classification evaluation")
    calibration = require(
        "sklearn.calibration",
        extra="ml-base",
        purpose="classification calibration evaluation",
    )
    truth = predictions.truth_class_ids
    decision = _decision_ids(predictions, threshold)
    labels = np.arange(len(predictions.class_order), dtype=np.int64)
    records = [
        _metric(
            predictions,
            name="accuracy",
            value=metrics.accuracy_score(truth, decision),
            scope=scope,
            modality=modality,
        ),
        _metric(
            predictions,
            name="balanced_accuracy",
            value=metrics.balanced_accuracy_score(truth, decision),
            scope=scope,
            modality=modality,
        ),
        _metric(
            predictions,
            name="f1_macro",
            value=metrics.f1_score(
                truth,
                decision,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            scope=scope,
            modality=modality,
        ),
        _metric(
            predictions,
            name="f1_weighted",
            value=metrics.f1_score(
                truth,
                decision,
                labels=labels,
                average="weighted",
                zero_division=0,
            ),
            scope=scope,
            modality=modality,
        ),
        _metric(
            predictions,
            name="log_loss",
            value=metrics.log_loss(truth, predictions.probabilities, labels=labels),
            scope=scope,
            modality=modality,
        ),
    ]
    curves: list[CurveRecord] = []
    balances: list[ClassBalanceRecord] = []
    n_rows = predictions.n_observations
    for class_id, class_name in enumerate(predictions.class_order):
        binary_truth = (truth == class_id).astype(np.int64)
        binary_decision = (decision == class_id).astype(np.int64)
        count = int(binary_truth.sum())
        balances.append(
            ClassBalanceRecord(
                class_name=class_name,
                count=count,
                fraction=count / n_rows,
                split=predictions.split,
                cohort=predictions.cohort or predictions.split,
                scope=scope,
                modality=modality,
            )
        )
        records.extend(
            (
                _metric(
                    predictions,
                    name="precision",
                    value=metrics.precision_score(binary_truth, binary_decision, zero_division=0),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _metric(
                    predictions,
                    name="recall",
                    value=metrics.recall_score(binary_truth, binary_decision, zero_division=0),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _metric(
                    predictions,
                    name="f1",
                    value=metrics.f1_score(binary_truth, binary_decision, zero_division=0),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
            )
        )
        if np.unique(binary_truth).size < 2:
            records.extend(
                (
                    _metric(
                        predictions,
                        name="roc_auc",
                        value=None,
                        scope=scope,
                        modality=modality,
                        class_name=class_name,
                    ),
                    _metric(
                        predictions,
                        name="average_precision",
                        value=None,
                        scope=scope,
                        modality=modality,
                        class_name=class_name,
                    ),
                )
            )
            continue
        probability = predictions.probabilities[:, class_id]
        false_positive, true_positive, roc_thresholds = metrics.roc_curve(
            binary_truth,
            probability,
        )
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            binary_truth,
            probability,
        )
        observed_fraction, mean_probability = calibration.calibration_curve(
            binary_truth,
            probability,
            n_bins=min(10, n_rows),
            strategy="quantile",
        )
        records.extend(
            (
                _metric(
                    predictions,
                    name="roc_auc",
                    value=metrics.roc_auc_score(binary_truth, probability),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _metric(
                    predictions,
                    name="average_precision",
                    value=metrics.average_precision_score(binary_truth, probability),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _metric(
                    predictions,
                    name="brier_score",
                    value=metrics.brier_score_loss(binary_truth, probability),
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
            )
        )
        curves.extend(
            (
                _curve(
                    predictions,
                    kind="roc",
                    x=false_positive,
                    y=true_positive,
                    thresholds=roc_thresholds,
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _curve(
                    predictions,
                    kind="precision_recall",
                    x=recall,
                    y=precision,
                    thresholds=pr_thresholds,
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
                _curve(
                    predictions,
                    kind="calibration",
                    x=mean_probability,
                    y=observed_fraction,
                    scope=scope,
                    modality=modality,
                    class_name=class_name,
                ),
            )
        )
    confusion = ConfusionRecord(
        matrix=metrics.confusion_matrix(truth, decision, labels=labels),
        class_order=predictions.class_order,
        split=predictions.split,
        cohort=predictions.cohort or predictions.split,
        scope=scope,
        modality=modality,
    )
    return tuple(records), tuple(curves), confusion, tuple(balances)


def evaluate_predictions(
    predictions: PredictionResult,
    *,
    threshold: ThresholdProvenance | None = None,
    calibration: CalibrationProvenance | None = None,
    by_modality: bool = True,
) -> EvaluationResult:
    """Evaluate stored predictions at natural prevalence, pooled and by modality."""
    if predictions.truth_class_ids is None:
        raise EvaluationContractError("evaluation requires stored truth_class_ids")
    if predictions.split == "inference":
        raise EvaluationContractError("unlabeled inference cohorts cannot be evaluated")
    if threshold is not None and threshold.fitted_split == "test":
        raise EvaluationContractError("test-fitted thresholds cannot be used for evaluation")
    if calibration is not None:
        if calibration.class_order and calibration.class_order != predictions.class_order:
            raise EvaluationContractError("calibration class_order differs from prediction schema")
        if (
            calibration.model_id is not None
            and predictions.model_id is not None
            and calibration.model_id != predictions.model_id
        ):
            raise EvaluationContractError("calibration model_id differs from predictions")
    all_metrics: list[MetricRecord] = []
    all_curves: list[CurveRecord] = []
    all_confusion: list[ConfusionRecord] = []
    all_balance: list[ClassBalanceRecord] = []
    slices: list[tuple[PredictionResult, str, str | None]] = [(predictions, "pooled", None)]
    if by_modality:
        for modality in sorted(set(predictions.modalities)):
            selected = np.asarray(predictions.modalities) == modality
            slices.append(
                (
                    predictions.select(selected),
                    "modality",
                    modality,
                )
            )
    for selected, scope, modality in slices:
        metrics, curves, confusion, balance = _evaluate_slice(
            selected,
            threshold=threshold,
            scope=scope,
            modality=modality,
        )
        all_metrics.extend(metrics)
        all_curves.extend(curves)
        all_confusion.append(confusion)
        all_balance.extend(balance)
    return EvaluationResult(
        predictions=predictions,
        metrics=tuple(all_metrics),
        curves=tuple(all_curves),
        confusion=tuple(all_confusion),
        class_balance=tuple(all_balance),
        threshold=threshold,
        calibration=calibration,
    )


def aggregate_fold_metrics(
    folds: Mapping[str, EvaluationResult],
) -> tuple[FoldMetricSummary, ...]:
    """Macro-average matched finite metrics with sample-SD uncertainty."""
    if not folds:
        raise EvaluationContractError("fold aggregation requires at least one fold")
    first = next(iter(folds.values())).predictions
    grouped: dict[tuple[str, str, str | None, str | None], list[tuple[str, float]]] = {}
    for fold_name, result in folds.items():
        if not isinstance(fold_name, str) or not fold_name.strip():
            raise EvaluationContractError("fold names must be non-empty")
        if result.predictions.class_order != first.class_order:
            raise EvaluationContractError("fold predictions must use the same class_order")
        if result.predictions.split != first.split:
            raise EvaluationContractError("fold predictions must use the same split role")
        seen: set[tuple[str, str, str | None, str | None]] = set()
        for metric in result.metrics:
            key = (metric.name, metric.scope, metric.modality, metric.class_name)
            if key in seen:
                raise EvaluationContractError(
                    f"fold {fold_name!r} contains duplicate metric identity {key}"
                )
            seen.add(key)
            if metric.value is not None:
                grouped.setdefault(key, []).append((fold_name, metric.value))
    summaries = []
    for (name, scope, modality, class_name), fold_values in sorted(
        grouped.items(),
        key=lambda item: tuple("" if value is None else value for value in item[0]),
    ):
        fold_names = tuple(fold_name for fold_name, _value in fold_values)
        array = np.asarray([value for _fold_name, value in fold_values], dtype=np.float64)
        summaries.append(
            FoldMetricSummary(
                name=name,
                fold_names=fold_names,
                values=tuple(float(value) for value in array),
                mean=float(array.mean()),
                standard_deviation=(None if len(array) < 2 else float(array.std(ddof=1))),
                uncertainty="sample_standard_deviation",
                scope=scope,
                modality=modality,
                class_name=class_name,
            )
        )
    return tuple(summaries)
