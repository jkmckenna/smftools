"""Render canonical ML tidy result tables to explicit output paths.

The functions in this module do not discover artifacts, load models, select
models, or compute scientific results. Every figure can be rebuilt from its
supplied DataFrame alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _table(frame: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} cannot be empty")
    return frame.copy(deep=True)


def _path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.name or not path.suffix:
        raise ValueError("output_path must identify a file with an extension")
    if not path.parent.is_dir():
        raise ValueError(f"output_path parent does not exist: {path.parent}")
    return path


def _display(value: Any, *, missing: str = "all") -> str:
    return missing if pd.isna(value) else str(value)


def _semantic_label(row: pd.Series, *, include_class: bool = True) -> str:
    parts = [
        f"model={_display(row['model_id'])}",
        f"split={_display(row['split'])}",
        f"cohort={_display(row['cohort'])}",
        f"scope={_display(row['scope'])}",
        f"modality={_display(row['modality'])}",
    ]
    if include_class:
        parts.append(f"class={_display(row['class_name'])}")
    return " | ".join(parts)


def _panel_grid(n_panels: int, *, panel_width: float, panel_height: float):
    n_columns = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(panel_width * n_columns, panel_height * n_rows),
        squeeze=False,
    )
    return figure, axes.ravel()


def _save(figure, output_path: str | Path) -> None:
    try:
        path = _path(output_path)
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_training_history(
    history: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Training history",
) -> None:
    """Plot stored numeric training metrics for one or more models."""
    frame = _table(
        history,
        {"model_id", "event_index", "epoch", "metric_name", "value"},
        "history",
    )
    frame = frame.loc[frame["metric_name"].notna() & frame["value"].notna()].copy()
    if frame.empty:
        raise ValueError("history contains no numeric training metrics to plot")
    metric_names = tuple(sorted(frame["metric_name"].astype(str).unique()))
    figure, axes = _panel_grid(len(metric_names), panel_width=4.2, panel_height=3.2)
    for axis, metric_name in zip(axes, metric_names, strict=False):
        selected = frame.loc[frame["metric_name"].astype(str) == metric_name]
        for model_id, group in selected.groupby("model_id", sort=True, dropna=False):
            x = group["epoch"].where(group["epoch"].notna(), group["event_index"])
            ordered = group.assign(_x=x).sort_values("_x", kind="stable")
            axis.plot(
                ordered["_x"].to_numpy(dtype=float),
                ordered["value"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.2,
                label=f"model={_display(model_id)}",
            )
        axis.set_title(metric_name)
        axis.set_xlabel("Epoch (event index when epoch is unavailable)")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes[len(metric_names) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)


def plot_evaluation_curves(
    curves: pd.DataFrame,
    output_path: str | Path,
    *,
    kinds: Sequence[str] = ("roc", "precision_recall"),
    title: str = "Evaluation curves",
) -> None:
    """Plot ROC, precision-recall, or calibration rows with full semantics."""
    frame = _table(
        curves,
        {
            "model_id",
            "kind",
            "point_index",
            "x",
            "y",
            "split",
            "cohort",
            "scope",
            "modality",
            "class_name",
        },
        "curves",
    )
    requested = tuple(dict.fromkeys(str(kind).strip() for kind in kinds))
    if not requested or any(not kind for kind in requested):
        raise ValueError("kinds must contain at least one non-empty curve kind")
    unknown = sorted(set(requested).difference(frame["kind"].astype(str)))
    if unknown:
        raise ValueError(f"curves do not contain requested kinds: {unknown}")
    figure, axes = _panel_grid(len(requested), panel_width=4.8, panel_height=3.8)
    identity = ["model_id", "split", "cohort", "scope", "modality", "class_name"]
    for axis, kind in zip(axes, requested, strict=True):
        selected = frame.loc[frame["kind"].astype(str) == kind]
        for _key, group in selected.groupby(identity, sort=True, dropna=False):
            ordered = group.sort_values("point_index", kind="stable")
            axis.plot(
                ordered["x"].to_numpy(dtype=float),
                ordered["y"].to_numpy(dtype=float),
                linewidth=1.2,
                label=_semantic_label(ordered.iloc[0]),
            )
        if kind == "roc":
            axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            panel_title = "ROC"
        elif kind == "precision_recall":
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            panel_title = "Precision-recall"
        elif kind == "calibration":
            axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
            axis.set_xlabel("Mean predicted probability")
            axis.set_ylabel("Observed fraction")
            panel_title = "Calibration"
        else:
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            panel_title = kind
        axis.set_title(panel_title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    for axis in axes[len(requested) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)


def plot_calibration_curves(
    curves: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Calibration curves",
) -> None:
    """Plot calibration rows using the generic evaluation-curve renderer."""
    plot_evaluation_curves(curves, output_path, kinds=("calibration",), title=title)


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    names: Sequence[str] | None = None,
    title: str = "Metric comparison",
) -> None:
    """Compare finite scalar metrics across explicit model and cohort slices."""
    frame = _table(
        metrics,
        {
            "model_id",
            "name",
            "value",
            "split",
            "cohort",
            "scope",
            "modality",
            "class_name",
        },
        "metrics",
    )
    frame = frame.loc[frame["value"].notna()].copy()
    requested = (
        tuple(sorted(frame["name"].astype(str).unique()))
        if names is None
        else tuple(dict.fromkeys(str(name).strip() for name in names))
    )
    if not requested or any(not name for name in requested):
        raise ValueError("names must identify at least one finite metric")
    unknown = sorted(set(requested).difference(frame["name"].astype(str)))
    if unknown:
        raise ValueError(f"metrics do not contain requested names: {unknown}")
    figure, axes = _panel_grid(len(requested), panel_width=5.0, panel_height=3.8)
    for axis, metric_name in zip(axes, requested, strict=True):
        selected = frame.loc[frame["name"].astype(str) == metric_name].reset_index(drop=True)
        labels = [_semantic_label(row) for _index, row in selected.iterrows()]
        positions = np.arange(len(selected))
        axis.bar(
            positions,
            selected["value"].to_numpy(dtype=float),
            color="#c9c9c9",
            edgecolor="#555555",
            linewidth=0.8,
        )
        axis.set_xticks(positions, labels, rotation=35, ha="right", fontsize=6)
        axis.set_ylabel("Value")
        axis.set_title(metric_name)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[len(requested) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)


def plot_confusion_matrices(
    confusion: pd.DataFrame,
    output_path: str | Path,
    *,
    normalize: bool = False,
    title: str = "Confusion matrices",
) -> None:
    """Plot long-form confusion counts, faceted by their complete slice identity."""
    frame = _table(
        confusion,
        {
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
        },
        "confusion",
    )
    identities = ["model_id", "split", "cohort", "scope", "modality"]
    groups = tuple(frame.groupby(identities, sort=True, dropna=False))
    figure, axes = _panel_grid(len(groups), panel_width=4.2, panel_height=3.8)
    for axis, (_key, group) in zip(axes, groups, strict=True):
        actual = (
            group[["actual_class_index", "actual_class"]]
            .drop_duplicates()
            .sort_values("actual_class_index")
        )
        predicted = (
            group[["predicted_class_index", "predicted_class"]]
            .drop_duplicates()
            .sort_values("predicted_class_index")
        )
        matrix = np.zeros((len(actual), len(predicted)), dtype=float)
        for row in group.itertuples(index=False):
            matrix[int(row.actual_class_index), int(row.predicted_class_index)] = float(row.count)
        if normalize:
            totals = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        for row_index, column_index in np.ndindex(matrix.shape):
            label = (
                f"{matrix[row_index, column_index]:.2f}"
                if normalize
                else str(int(matrix[row_index, column_index]))
            )
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
        axis.set_xticks(np.arange(len(predicted)), predicted["predicted_class"], rotation=30)
        axis.set_yticks(np.arange(len(actual)), actual["actual_class"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_title(_semantic_label(group.iloc[0], include_class=False), fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)


def plot_feature_importance(
    attributions: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 20,
    title: str = "Feature importance",
) -> None:
    """Plot top mean-absolute attribution features for each result and method."""
    frame = _table(
        attributions,
        {
            "result_id",
            "model_id",
            "method",
            "split",
            "cohort",
            "target_class",
            "feature_name",
            "mean_attribution",
            "mean_absolute_attribution",
        },
        "attributions",
    )
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    identities = ["result_id", "model_id", "method", "split", "cohort", "target_class"]
    groups = tuple(frame.groupby(identities, sort=True, dropna=False))
    figure, axes = _panel_grid(len(groups), panel_width=5.2, panel_height=4.2)
    for axis, (_key, group) in zip(axes, groups, strict=True):
        ranked = group.nlargest(top_n, "mean_absolute_attribution").sort_values(
            "mean_absolute_attribution",
            kind="stable",
        )
        colors = np.where(
            ranked["mean_attribution"].to_numpy(dtype=float) >= 0, "#d62728", "#1f77b4"
        )
        axis.barh(
            ranked["feature_name"].astype(str),
            ranked["mean_absolute_attribution"].to_numpy(dtype=float),
            color=colors,
        )
        first = ranked.iloc[0]
        axis.set_xlabel("Mean absolute attribution")
        axis.set_title(
            f"model={first['model_id']} | method={first['method']} | split={first['split']} | "
            f"cohort={first['cohort']} | class={first['target_class']}",
            fontsize=8,
        )
        axis.grid(axis="x", alpha=0.2)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)


def plot_attribution_summary(
    attributions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Attribution summary",
) -> None:
    """Plot signed mean attribution across genomic coordinates by channel."""
    frame = _table(
        attributions,
        {
            "result_id",
            "model_id",
            "method",
            "split",
            "cohort",
            "target_class",
            "feature_kind",
            "coordinate",
            "channel",
            "mean_attribution",
        },
        "attributions",
    )
    frame = frame.loc[frame["coordinate"].notna()].copy()
    if frame.empty:
        raise ValueError("attributions contain no genomic coordinates to plot")
    identities = ["result_id", "model_id", "method", "split", "cohort", "target_class"]
    groups = tuple(frame.groupby(identities, sort=True, dropna=False))
    figure, axes = _panel_grid(len(groups), panel_width=5.4, panel_height=3.8)
    for axis, (_key, group) in zip(axes, groups, strict=True):
        series_fields = ["channel", "feature_kind"]
        for (channel, feature_kind), series in group.groupby(
            series_fields,
            sort=True,
            dropna=False,
        ):
            collapsed = (
                series.groupby("coordinate", as_index=False)["mean_attribution"]
                .mean()
                .sort_values(
                    "coordinate",
                    kind="stable",
                )
            )
            axis.plot(
                collapsed["coordinate"].to_numpy(dtype=float),
                collapsed["mean_attribution"].to_numpy(dtype=float),
                linewidth=1.1,
                label=f"channel={_display(channel)} | kind={_display(feature_kind)}",
            )
        first = group.iloc[0]
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Coordinate")
        axis.set_ylabel("Mean attribution")
        axis.set_title(
            f"model={first['model_id']} | method={first['method']} | split={first['split']} | "
            f"cohort={first['cohort']} | class={first['target_class']}",
            fontsize=8,
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    _save(figure, output_path)
