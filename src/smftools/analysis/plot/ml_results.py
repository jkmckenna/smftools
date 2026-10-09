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


ATTRIBUTION_ORDERS = ("label", "score", "bins")
_POSITIVE_COLOR = "#D84315"
_OTHER_CLASS_COLORS = ("#455A64", "#90A4AE", "#263238", "#B0BEC5")


def _class_colors(classes: Sequence[str], positive_class: str | None) -> dict[str, str]:
    """True-class colours that cannot be confused with the fold palette."""
    others = [value for value in sorted(set(classes)) if value != positive_class]
    colors = {value: _OTHER_CLASS_COLORS[i % 4] for i, value in enumerate(others)}
    if positive_class is not None:
        colors[positive_class] = _POSITIVE_COLOR
    return colors


# Input colours by a channel's biological role (zero: the latent figures' cream).
_INPUT_ZERO = "#F1ECE2"
_ROLE_COLORS = {"accessib": "#2E7D32", "methyl": "#C62828"}
_UNOBSERVED = "#D9D9D9"


def _input_cmap(role: str | None):
    """Accessibility green, methylation red (from cream), else viridis;
    unobserved positions light grey."""
    from matplotlib.colors import LinearSegmentedColormap

    colour = next(
        (value for key, value in _ROLE_COLORS.items() if role and key in role.lower()), None
    )
    cmap = (
        LinearSegmentedColormap.from_list(f"input_{role}", [_INPUT_ZERO, colour])
        if colour
        else plt.get_cmap("viridis").copy()
    )
    cmap.set_bad(_UNOBSERVED)
    return cmap


def attribution_row_layout(
    molecules: pd.DataFrame,
    attributions: np.ndarray,
    *,
    order: str = "label",
    positive_class: str | None = None,
    bins: Sequence[Any] | None = None,
    bin_order: Sequence[Any] | None = None,
    within: str = "hierarchical",
    cluster_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[str, int, int]], np.ndarray]:
    """Row order, blocks and per-row block labels for an attribution clustermap.

    ``within`` orders rows inside each block: ``"hierarchical"`` (clustered on
    ``cluster_values`` -- default the attributions -- e.g. the inputs) or
    ``"score"`` (out-of-fold score, highest first).

    ``order``: ``"label"`` -- blocks by true class (``positive_class`` first),
    rows clustered within each block on their attributions; ``"score"`` -- one
    block, rows by out-of-fold score, highest first; ``"bins"`` -- blocks by
    ``bins`` (one value per molecule) in ``bin_order``, clustered within.
    """
    from smftools.tools.latent_ordering import hierarchical_block_order

    if order not in ATTRIBUTION_ORDERS:
        raise ValueError(f"order must be one of {ATTRIBUTION_ORDERS}")
    if within not in ("hierarchical", "score"):
        raise ValueError("within must be 'hierarchical' or 'score'")
    n_rows = len(molecules)
    source = attributions if cluster_values is None else cluster_values
    # Unobserved inputs sit halfway, so they neither join nor split clusters.
    points = np.nan_to_num(np.asarray(source, dtype=float).reshape(n_rows, -1), nan=0.5)
    scores = np.nan_to_num(molecules["score"].to_numpy(dtype=float), nan=-np.inf)
    if order == "score":
        scores = molecules["score"].to_numpy(dtype=float)
        row_order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
        labels = molecules["truth"].astype(str).to_numpy()
        return row_order, [("", 0, n_rows)], labels
    if order == "bins":
        if bins is None or len(bins) != n_rows:
            raise ValueError("order='bins' needs one bin value per molecule")
        labels = np.asarray(bins, dtype=object).astype(str)
        present = list(dict.fromkeys(labels))
        wanted = [str(value) for value in (bin_order or sorted(present))]
        block_order = [value for value in wanted if value in present]
        block_order += [value for value in present if value not in block_order]
    else:
        labels = molecules["truth"].astype(str).to_numpy()
        present = sorted(set(labels))
        block_order = (
            [positive_class, *[value for value in present if value != positive_class]]
            if positive_class in present
            else present
        )
    parts, blocks, cursor = [], [], 0
    for value in block_order:
        members = np.flatnonzero(labels == value)
        if members.size == 0:
            continue
        if within == "score":
            parts.append(members[np.argsort(-scores[members], kind="stable")])
        else:
            parts.append(members[hierarchical_block_order(points[members])])
        blocks.append((value, cursor, cursor + members.size))
        cursor += members.size
    return np.concatenate(parts).astype(int), blocks, labels


def plot_attribution_clustermap(
    molecules: pd.DataFrame,
    attributions: np.ndarray,
    *,
    channels: Sequence[str],
    coordinates: Sequence[int],
    inputs: np.ndarray | None = None,
    order: str = "label",
    positive_class: str | None = None,
    bins: Sequence[Any] | None = None,
    bin_order: Sequence[Any] | None = None,
    bin_name: str = "bin",
    bin_colors: dict | None = None,
    coordinate_labels: Sequence[Any] | None = None,
    extra_panels: Sequence[dict] = (),
    extra_strips: Sequence[dict] = (),
    max_rows: int | None = 2000,
    seed: int = 0,
    attribution_limit: float | None = None,
    columns: str = "observed",
    within: str = "hierarchical",
    cluster_on: str = "attributions",
    channel_roles: Sequence[str] | None = None,
    class_colors: dict[str, str] | None = None,
    balance_classes: bool = False,
    title: str = "",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inputs beside per-position attributions, one row per molecule.

    ``columns``: ``"observed"`` (default) keeps only positions observed in at
    least one drawn molecule (for site channels, the sites -- other positions
    are empty and carry no attribution), labelled with their real coordinates
    and with separators only between mask windows; ``"all"`` draws every
    position. Each panel's top trace is split by true class.

    Rows within each class (or bin) block: ``within="hierarchical"``
    (clustered on ``cluster_on``: ``"attributions"`` or ``"inputs"``) or
    ``"score"`` (highest out-of-fold score first). ``channel_roles`` (one per
    channel) colour the inputs: accessibility green, methylation red, others
    viridis; unobserved positions light grey.

    ``molecules`` has one row per molecule (``molecule_uid``, ``fold``,
    ``truth``, ``score``), aligned with ``attributions`` (molecules x channels
    x positions) and ``inputs`` (the same shape; NaN where unobserved). For
    each channel the input panel (when given) sits beside its attribution
    panel; ``extra_panels`` (``name``, ``matrix`` aligned with ``molecules``,
    optional ``positions``, ``cmap``, ``vmin``, ``vmax``) -- e.g. HMM layers --
    follow (or, with ``before`` naming a channel, sit just before that
    channel's input), and ``extra_strips`` (``name``, ``values`` aligned with
    ``molecules``, as `plot_latent_ordered_clustermap`) join the true-class,
    fold and score strips. Every panel and strip uses one row order
    (`attribution_row_layout`). Attributions use a diverging scale symmetric
    about zero (``attribution_limit``, default the 99th percentile of
    absolute values). ``coordinate_labels`` relabels the position axis (e.g.
    TSS-relative); numeric labels also order the columns (ascending), and a
    vertical separator marks every jump in the coordinates (between a position
    mask's windows). Extra panels are drawn as given, unless they carry
    ``coordinates`` equal to the inputs': then they share the inputs' columns. At most ``max_rows`` molecules are drawn (a seeded,
    class-stratified sample); ``balance_classes`` draws the same number of
    each true class (the smallest class's, within ``max_rows``).
    ``class_colors`` (true class -> colour) overrides the class colours.

    Returns the plot summary plus ``row_uids`` (drawn order) and
    ``attribution_limit``.
    """
    from smftools.plotting.latent_plotting import plot_latent_ordered_clustermap

    molecules = molecules.reset_index(drop=True)
    attributions = np.asarray(attributions, dtype=float)
    if attributions.shape[:2] != (len(molecules), len(channels)):
        raise ValueError("attributions must be molecules x channels x positions")
    if inputs is not None and np.shape(inputs) != attributions.shape:
        raise ValueError("inputs must have the attributions' shape")
    keep = np.arange(len(molecules))
    rng = np.random.default_rng(seed)
    if balance_classes:
        groups = list(molecules.groupby("truth", sort=True, dropna=False))
        per_class = min(len(group) for _truth, group in groups)
        if max_rows is not None:
            per_class = min(per_class, max(1, max_rows // len(groups)))
        chosen = [
            rng.choice(group.index.to_numpy(), per_class, replace=False) for _truth, group in groups
        ]
        keep = np.sort(np.concatenate(chosen).astype(int))
    elif max_rows is not None and len(molecules) > max_rows:
        chosen = []
        for _truth, group in molecules.groupby("truth", sort=True, dropna=False):
            take = max(1, round(max_rows * len(group) / len(molecules)))
            chosen.extend(rng.choice(group.index.to_numpy(), min(take, len(group)), replace=False))
        keep = np.sort(np.asarray(chosen, dtype=int))
    subset = molecules.iloc[keep].reset_index(drop=True)
    attribution = attributions[keep]
    if cluster_on not in ("attributions", "inputs"):
        raise ValueError("cluster_on must be 'attributions' or 'inputs'")
    if cluster_on == "inputs" and inputs is None:
        raise ValueError("cluster_on='inputs' needs inputs")
    row_order, blocks, labels = attribution_row_layout(
        subset,
        attribution,
        order=order,
        positive_class=positive_class,
        bins=None if bins is None else np.asarray(bins, dtype=object)[keep],
        bin_order=bin_order,
        within=within,
        cluster_values=None
        if cluster_on == "attributions"
        else np.asarray(inputs, dtype=float)[keep],
    )
    limit = attribution_limit
    if limit is None:
        finite = np.abs(attribution[np.isfinite(attribution)])
        limit = float(np.percentile(finite, 99)) if finite.size else 1.0
    limit = limit or 1.0
    # Columns in label order (e.g. TSS-relative, upstream to downstream) when
    # the labels are numeric, else in coordinate order; a separator wherever
    # the underlying coordinates jump (between windows of a position mask).
    coordinates = np.asarray(list(coordinates))
    labels_given = None if coordinate_labels is None else np.asarray(list(coordinate_labels))
    sort_key = (
        labels_given.astype(float)
        if labels_given is not None and np.issubdtype(labels_given.dtype, np.number)
        else coordinates.astype(float)
    )
    if columns not in ("observed", "all"):
        raise ValueError("columns must be 'observed' or 'all'")
    column_order = np.argsort(sort_key, kind="stable")
    ordered = coordinates[column_order].astype(float)
    # Window breaks are found on the full coordinate list, before any column
    # is dropped, so sparse sites never read as breaks.
    breaks = np.flatnonzero(np.abs(np.diff(ordered)) > 1) + 1
    if columns == "observed":
        # Positions observed in at least one drawn molecule (any channel); with
        # no inputs, positions with any non-zero attribution.
        source = (
            np.isfinite(np.asarray(inputs, dtype=float)[keep])
            if inputs is not None
            else attribution != 0
        )
        observed = source.any(axis=(0, 1))[column_order]
        if not observed.any():
            observed[:] = True
        column_order = column_order[observed]
        dense_index = np.cumsum(observed) - 1  # full position -> dense column
        separators = sorted(
            {int(dense_index[b - 1] + 1) for b in breaks if observed[b:].any()}
            - {0, len(column_order)}
        )
    else:
        separators = [int(b) for b in breaks]
    positions = list((labels_given if labels_given is not None else coordinates)[column_order])
    inputs = None if inputs is None else np.asarray(inputs, dtype=float)[:, :, column_order]
    attribution = attribution[:, :, column_order]
    extras, placed = [], {}
    for panel in extra_panels:
        panel = dict(panel)
        before = panel.pop("before", None)
        if before is not None and before not in channels:
            raise ValueError(f"extra panel {panel.get('name')!r}: no channel {before!r}")
        matrix = np.asarray(panel["matrix"], dtype=float)[keep]
        extra_coordinates = panel.pop("coordinates", None)
        if extra_coordinates is not None:
            # On the input's positions: drawn on the same (ordered, dense)
            # columns, labels and window breaks.
            if [int(value) for value in extra_coordinates] != [int(v) for v in coordinates]:
                raise ValueError(
                    f"extra panel {panel.get('name')!r} coordinates differ from the inputs'"
                )
            matrix = matrix[:, column_order]
            panel.update(positions=positions, column_separators=separators)
        (extras if before is None else placed.setdefault(before, [])).append(
            {**panel, "matrix": matrix}
        )
    panels = []
    for index, channel in enumerate(channels):
        panels.extend(placed.get(channel, []))
        if inputs is not None:
            panels.append(
                {
                    "name": f"{channel} (input)",
                    "matrix": inputs[keep][:, index],
                    "positions": positions,
                    "column_separators": separators,
                    "cmap": _input_cmap(None if channel_roles is None else channel_roles[index]),
                    "vmin": 0.0,
                    "vmax": 1.0,
                }
            )
        panels.append(
            {
                "name": f"{channel} attribution",
                "matrix": attribution[:, index],
                "positions": positions,
                "column_separators": separators,
                "cmap": "RdBu_r",
                "vmin": -limit,
                "vmax": limit,
            }
        )
    panels.extend(extras)
    truth = subset["truth"].astype(str).to_numpy()
    class_colors = {**_class_colors(truth, positive_class), **(class_colors or {})}
    strips = []
    if order == "bins":
        strips.append({"name": "true class", "values": truth, "colors": class_colors})
    # Folds by their held-out group ("holdout=exp_a" -> "exp_a"): readable in place.
    held_out = subset["fold"].astype(str).str.split("=", n=1).str[-1].to_numpy()
    strips.append({"name": "held out", "values": held_out})
    strips.append(
        {
            "name": "score",
            "kind": "continuous",
            "values": subset["score"].to_numpy(dtype=float),
            "vmin": 0.0,
            "vmax": 1.0,
        }
    )
    for strip in extra_strips:
        strips.append({**strip, "values": np.asarray(strip["values"], dtype=object)[keep]})
    result = (
        plot_latent_ordered_clustermap(
            panels,
            row_order=row_order,
            blocks=blocks,
            labels=labels,
            cluster_colors=bin_colors if order == "bins" else class_colors,
            cluster_name=bin_name if order == "bins" else "true class",
            cluster_legend=order == "bins",
            extra_strips=strips,
            trace_groups={
                "values": truth,
                "colors": class_colors,
                "order": list(
                    dict.fromkeys(
                        c for c in (positive_class, *sorted(set(truth))) if c in set(truth)
                    )
                ),
            },
            title=title,
            save_path=output_path,
        )
        or {}
    )
    return {
        **result,
        "row_uids": subset["molecule_uid"].to_numpy()[row_order].tolist(),
        "column_coordinates": coordinates[column_order].tolist(),
        "columns": columns,
        "column_separators": separators,
        "blocks": blocks,
        "attribution_limit": limit,
    }


def plot_run_comparison(comparison: Any, output_path: str | Path, *, metric: str) -> None:
    """Entries on one metric: per-fold points with the fold-mean and its
    bootstrap interval (left), and paired differences against the reference
    with per-fold points (right). ``comparison`` is a `RunComparison`."""
    summary = comparison.summary[comparison.summary["metric"] == metric]
    if summary.empty:
        raise ValueError(f"comparison has no metric {metric!r}")
    entries = list(summary["entry"])
    folds = sorted(comparison.fold_metrics["fold"].unique())
    palette = plt.get_cmap("tab10")
    fold_color = {fold: palette(i % 10) for i, fold in enumerate(folds)}
    differences = comparison.differences[comparison.differences["metric"] == metric]
    fold_differences = comparison.fold_differences[comparison.fold_differences["metric"] == metric]
    height = max(2.5, 0.45 * len(entries) + 1.2)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11, height), sharey=True, gridspec_kw={"width_ratios": (1, 1)}
    )
    y = {entry: index for index, entry in enumerate(entries)}
    values = comparison.fold_metrics[comparison.fold_metrics["metric"] == metric]
    offsets = np.linspace(-0.2, 0.2, max(len(folds), 1))
    for offset, fold in zip(offsets, folds):
        rows = values[values["fold"] == fold]
        left.scatter(
            rows["value"],
            [y[e] + offset for e in rows["entry"]],
            s=14,
            color=fold_color[fold],
            label=str(fold),
            zorder=3,
        )
    for _index, row in summary.iterrows():
        if pd.notna(row.get("fold_ci_low")):
            # Between experiments (folds resampled): grey, under the molecule interval.
            left.hlines(
                y[row["entry"]],
                row["fold_ci_low"],
                row["fold_ci_high"],
                color="#9E9E9E",
                linewidth=4,
                alpha=0.6,
                zorder=2,
            )
        left.errorbar(
            row["mean"],
            y[row["entry"]],
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="D",
            color="black",
            markersize=5,
            capsize=3,
            zorder=4,
        )
    left.set_yticks(range(len(entries)), entries, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel(metric)
    left.set_title(
        "per fold (points); mean with molecule (black) and experiment (grey) intervals",
        fontsize=9,
    )
    left.legend(title="held out", fontsize=7, title_fontsize=7, loc="best", frameon=False)
    reference = comparison.settings.get("reference")
    for offset, fold in zip(offsets, folds):
        rows = fold_differences[fold_differences["fold"] == fold]
        right.scatter(
            rows["difference"],
            [y[e] + offset for e in rows["entry"]],
            s=14,
            color=fold_color[fold],
            zorder=3,
        )
    for _index, row in differences.iterrows():
        if pd.notna(row.get("fold_ci_low")):
            right.hlines(
                y[row["entry"]],
                row["fold_ci_low"],
                row["fold_ci_high"],
                color="#9E9E9E",
                linewidth=4,
                alpha=0.6,
                zorder=2,
            )
        right.errorbar(
            row["mean"],
            y[row["entry"]],
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="D",
            color="black",
            markersize=5,
            capsize=3,
            zorder=4,
        )
        right.annotate(
            f"{row['folds_better']}/{row['n_folds']}",
            (row["ci_high"], y[row["entry"]]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    right.axvline(0, color="grey", linewidth=0.8)
    right.set_xlabel(f"{metric} minus {reference}")
    right.set_title("paired difference (folds better / folds)", fontsize=9)
    figure.suptitle(f"{metric}: {len(entries)} models, {len(folds)} held-out folds", fontsize=11)
    figure.tight_layout()
    _save(figure, output_path)


def plot_detector_catalogue(
    detectors: pd.DataFrame,
    patterns: np.ndarray,
    *,
    channels: Sequence[str],
    top: int = 16,
    title: str = "",
    output_path: str | Path,
) -> list[int]:
    """One fold's most predictive detectors (by distance of AUROC from 0.5), one row each:
    the mean input pattern over its top windows (channels x offsets from the
    detector's position; white where never observed), its AUROC, top-window
    enrichment and where its windows sit on the locus. ``patterns``: detectors
    x channels x window, as `detector_catalogue_run` stores it. Returns the
    detectors drawn, in order."""
    ranked = (
        detectors.assign(strength=(detectors["auroc"] - 0.5).abs())
        .sort_values("strength", ascending=False)
        .head(top)
    )
    chosen = ranked["detector"].astype(int).tolist()
    if not chosen:
        raise ValueError("no detectors to plot")
    width = patterns.shape[-1]
    half = width // 2
    figure, axes = plt.subplots(
        len(chosen),
        2,
        figsize=(10, 0.55 * len(chosen) * max(1, len(channels)) + 1.2),
        gridspec_kw={"width_ratios": (4, 1.4)},
        squeeze=False,
    )
    for row, (_index, item) in enumerate(ranked.iterrows()):
        detector = int(item["detector"])
        pattern_axis, text_axis = axes[row]
        pattern_axis.imshow(
            patterns[detector],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=(-half - 0.5, half + 0.5, len(channels) - 0.5, -0.5),
        )
        pattern_axis.axvline(0, color="white", linewidth=0.6)
        pattern_axis.set_yticks(range(len(channels)), channels, fontsize=6)
        pattern_axis.set_ylabel(
            f"d{detector} (g{int(item['group'])})", fontsize=7, rotation=0, labelpad=22
        )
        if row < len(chosen) - 1:
            pattern_axis.tick_params(labelbottom=False)
        enrichment = item.get("log2_enrichment")
        text_axis.axis("off")
        text_axis.text(
            0,
            0.5,
            f"AUROC {item['auroc']:.2f}"
            + (f" | log2 enr {enrichment:+.2f}" if pd.notna(enrichment) else "")
            + (
                f"\ncentre {item['centre_mean']:.0f} +/- {item['centre_sd']:.0f}"
                if pd.notna(item.get("centre_mean"))
                else ""
            ),
            fontsize=7,
            va="center",
        )
    axes[-1][0].set_xlabel("offset from the detector's position")
    figure.suptitle(title or "detector catalogue", fontsize=10)
    figure.tight_layout()
    _save(figure, output_path)
    return chosen
