"""Convert canonical ML result records into tidy, persistence-ready tables.

All functions are side-effect free: they accept immutable machine-learning
records and return pandas DataFrames without reading or writing artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from smftools.machine_learning.evaluation.contracts import (
        EvaluationResult,
        FoldMetricSummary,
        TrainingHistory,
    )
    from smftools.machine_learning.interpretability.contracts import AttributionResult

HISTORY_COLUMNS = (
    "model_id",
    "backend",
    "event_index",
    "event_type",
    "epoch",
    "step",
    "metric_name",
    "value",
)
METRIC_COLUMNS = (
    "result_id",
    "model_id",
    "name",
    "value",
    "n_observations",
    "split",
    "cohort",
    "scope",
    "modality",
    "class_name",
)
CURVE_COLUMNS = (
    "result_id",
    "model_id",
    "kind",
    "point_index",
    "x",
    "y",
    "threshold",
    "split",
    "cohort",
    "scope",
    "modality",
    "class_name",
)
CONFUSION_COLUMNS = (
    "result_id",
    "model_id",
    "split",
    "cohort",
    "scope",
    "modality",
    "actual_class",
    "actual_class_index",
    "predicted_class",
    "predicted_class_index",
    "count",
)
BALANCE_COLUMNS = (
    "result_id",
    "model_id",
    "class_name",
    "count",
    "fraction",
    "split",
    "cohort",
    "scope",
    "modality",
)
FOLD_COLUMNS = (
    "name",
    "scope",
    "modality",
    "class_name",
    "fold_name",
    "value",
    "mean",
    "standard_deviation",
    "uncertainty",
)
ATTRIBUTION_COLUMNS = (
    "result_id",
    "model_id",
    "method",
    "split",
    "cohort",
    "target_class",
    "aggregation_level",
    "feature_name",
    "feature_kind",
    "coordinate",
    "channel",
    "biological_role",
    "mean_attribution",
    "mean_absolute_attribution",
    "standard_deviation",
    "n_observations",
)


def _named_records(values: Mapping[str, Any], name: str) -> tuple[tuple[str, Any], ...]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")
    records = []
    for key, value in values.items():
        identity = str(key).strip()
        if not identity:
            raise ValueError(f"{name} keys must be non-empty identifiers")
        records.append((identity, value))
    return tuple(records)


def _record(value: Any, required: Sequence[str], name: str) -> Any:
    missing = [field for field in required if not hasattr(value, field)]
    if missing:
        raise TypeError(f"{name} lacks required record fields: {missing}")
    return value


def training_history_table(histories: Mapping[str, TrainingHistory]) -> pd.DataFrame:
    """Return one row per model, training event, and reported metric.

    Events without numeric metrics are retained with null ``metric_name`` and
    ``value`` fields, so one-shot sklearn fits remain visible without invented
    epochs or losses.
    """
    rows = []
    for model_id, history in _named_records(histories, "histories"):
        history = _record(history, ("backend", "events"), "history")
        for event in history.events:
            metrics = event.metrics.items() or ((None, None),)
            for metric_name, value in metrics:
                rows.append(
                    {
                        "model_id": model_id,
                        "backend": history.backend,
                        "event_index": event.event_index,
                        "event_type": event.event_type,
                        "epoch": event.epoch,
                        "step": event.step,
                        "metric_name": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def evaluation_metric_table(results: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Return scalar evaluation metrics with complete slice semantics."""
    rows = []
    for result_id, result in _named_records(results, "results"):
        result = _record(result, ("predictions", "metrics"), "evaluation result")
        fallback_model = result.predictions.model_id or result_id
        rows.extend(
            {
                "result_id": result_id,
                "model_id": metric.model_id or fallback_model,
                "name": metric.name,
                "value": metric.value,
                "n_observations": metric.n_observations,
                "split": metric.split,
                "cohort": metric.cohort,
                "scope": metric.scope,
                "modality": metric.modality,
                "class_name": metric.class_name,
            }
            for metric in result.metrics
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def evaluation_curve_table(results: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Return one row per stored ROC, PR, or calibration curve point."""
    rows = []
    for result_id, result in _named_records(results, "results"):
        result = _record(result, ("predictions", "curves"), "evaluation result")
        model_id = result.predictions.model_id or result_id
        for curve in result.curves:
            thresholds = () if curve.thresholds is None else tuple(curve.thresholds)
            for point_index, (x_value, y_value) in enumerate(zip(curve.x, curve.y, strict=True)):
                rows.append(
                    {
                        "result_id": result_id,
                        "model_id": model_id,
                        "kind": curve.kind,
                        "point_index": point_index,
                        "x": float(x_value),
                        "y": float(y_value),
                        "threshold": (
                            float(thresholds[point_index])
                            if point_index < len(thresholds)
                            else None
                        ),
                        "split": curve.split,
                        "cohort": curve.cohort,
                        "scope": curve.scope,
                        "modality": curve.modality,
                        "class_name": curve.class_name,
                    }
                )
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def confusion_table(results: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Return confusion matrices in long form with persisted class order."""
    rows = []
    for result_id, result in _named_records(results, "results"):
        result = _record(result, ("predictions", "confusion"), "evaluation result")
        model_id = result.predictions.model_id or result_id
        for record in result.confusion:
            for actual_index, actual_class in enumerate(record.class_order):
                for predicted_index, predicted_class in enumerate(record.class_order):
                    rows.append(
                        {
                            "result_id": result_id,
                            "model_id": model_id,
                            "split": record.split,
                            "cohort": record.cohort,
                            "scope": record.scope,
                            "modality": record.modality,
                            "actual_class": actual_class,
                            "actual_class_index": actual_index,
                            "predicted_class": predicted_class,
                            "predicted_class_index": predicted_index,
                            "count": int(record.matrix[actual_index, predicted_index]),
                        }
                    )
    return pd.DataFrame(rows, columns=CONFUSION_COLUMNS)


def class_balance_table(results: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Return natural class support for every evaluated slice."""
    rows = []
    for result_id, result in _named_records(results, "results"):
        result = _record(result, ("predictions", "class_balance"), "evaluation result")
        model_id = result.predictions.model_id or result_id
        rows.extend(
            {
                "result_id": result_id,
                "model_id": model_id,
                "class_name": record.class_name,
                "count": record.count,
                "fraction": record.fraction,
                "split": record.split,
                "cohort": record.cohort,
                "scope": record.scope,
                "modality": record.modality,
            }
            for record in result.class_balance
        )
    return pd.DataFrame(rows, columns=BALANCE_COLUMNS)


def fold_metric_table(summaries: Sequence[FoldMetricSummary]) -> pd.DataFrame:
    """Expand fold summaries into tidy values with repeated aggregate context."""
    rows = []
    for summary in summaries:
        summary = _record(
            summary,
            (
                "name",
                "scope",
                "modality",
                "class_name",
                "fold_names",
                "values",
                "mean",
                "standard_deviation",
                "uncertainty",
            ),
            "fold metric summary",
        )
        for fold_name, value in zip(summary.fold_names, summary.values, strict=True):
            rows.append(
                {
                    "name": summary.name,
                    "scope": summary.scope,
                    "modality": summary.modality,
                    "class_name": summary.class_name,
                    "fold_name": fold_name,
                    "value": value,
                    "mean": summary.mean,
                    "standard_deviation": summary.standard_deviation,
                    "uncertainty": summary.uncertainty,
                }
            )
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def _attribution_axis_metadata(
    result: AttributionResult,
    remaining_axes: tuple[str, ...],
    index: tuple[int, ...],
) -> dict[str, Any]:
    positions = dict(zip(remaining_axes, index, strict=True))
    if "feature" in positions:
        feature = result.features[positions["feature"]]
        return {
            "feature_name": feature.name,
            "feature_kind": feature.kind,
            "coordinate": feature.coordinate,
            "channel": feature.channel.name,
            "biological_role": feature.channel.biological_role,
        }
    coordinate = int(result.coordinates[positions["position"]])
    if "channel" in positions:
        channel = result.channels[positions["channel"]]
        return {
            "feature_name": f"signal:{channel.name}@{coordinate}",
            "feature_kind": "signal",
            "coordinate": coordinate,
            "channel": channel.name,
            "biological_role": channel.biological_role,
        }
    return {
        "feature_name": f"position@{coordinate}",
        "feature_kind": "position",
        "coordinate": coordinate,
        "channel": None,
        "biological_role": None,
    }


def attribution_summary_table(results: Mapping[str, AttributionResult]) -> pd.DataFrame:
    """Summarize aligned attribution tensors without retaining molecule identities.

    Observation-level tensors are reduced to signed mean, mean absolute value,
    and population standard deviation. Already-global tensors retain their
    stored signed and absolute values and report null standard deviation.
    """
    rows = []
    for _supplied_id, result in _named_records(results, "results"):
        result = _record(
            result,
            (
                "result_id",
                "request",
                "axes",
                "values",
                "coordinates",
                "channels",
                "features",
            ),
            "attribution result",
        )
        axes = tuple(result.axes)
        values = np.asarray(result.values, dtype=np.float64)
        if "observation" in axes:
            observation_axis = axes.index("observation")
            signed = values.mean(axis=observation_axis)
            absolute = np.abs(values).mean(axis=observation_axis)
            deviation: np.ndarray | None = values.std(axis=observation_axis, ddof=0)
            remaining_axes = tuple(axis for axis in axes if axis != "observation")
            aggregation_level = "observation_summary"
        else:
            signed = values
            absolute = np.abs(values)
            deviation = None
            remaining_axes = axes
            aggregation_level = "global"
        for index in np.ndindex(signed.shape):
            metadata = _attribution_axis_metadata(result, remaining_axes, index)
            rows.append(
                {
                    "result_id": result.result_id,
                    "model_id": result.request.model_id,
                    "method": result.request.method,
                    "split": result.request.split_role,
                    "cohort": result.request.cohort,
                    "target_class": result.request.target.class_name,
                    "aggregation_level": aggregation_level,
                    **metadata,
                    "mean_attribution": float(signed[index]),
                    "mean_absolute_attribution": float(absolute[index]),
                    "standard_deviation": (None if deviation is None else float(deviation[index])),
                    "n_observations": len(result.request.observation_uids),
                }
            )
    return pd.DataFrame(rows, columns=ATTRIBUTION_COLUMNS)
