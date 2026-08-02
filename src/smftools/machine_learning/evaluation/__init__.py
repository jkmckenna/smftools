"""Backend-neutral evaluation contracts plus lazy legacy evaluators."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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
    TrainingEvent,
    TrainingHistory,
)
from .history import sklearn_training_history, torch_training_history
from .metrics import aggregate_fold_metrics, evaluate_predictions, fit_binary_threshold

_LAZY_EXPORTS = {
    "flatten_sliding_window_results": (".eval_utils", "flatten_sliding_window_results"),
    "ModelEvaluator": (".evaluators", "ModelEvaluator"),
    "PostInferenceModelEvaluator": (".evaluators", "PostInferenceModelEvaluator"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose plotting-dependent legacy evaluators."""
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CalibrationProvenance",
    "ClassBalanceRecord",
    "ConfusionRecord",
    "CurveRecord",
    "EvaluationContractError",
    "EvaluationResult",
    "FoldMetricSummary",
    "MetricRecord",
    "ModelEvaluator",
    "PostInferenceModelEvaluator",
    "PredictionResult",
    "ThresholdProvenance",
    "TrainingEvent",
    "TrainingHistory",
    "aggregate_fold_metrics",
    "evaluate_predictions",
    "fit_binary_threshold",
    "flatten_sliding_window_results",
    "sklearn_training_history",
    "torch_training_history",
]
