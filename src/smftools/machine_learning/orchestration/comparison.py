"""Compare published train runs on the same held-out folds (`MLR-05`).

`select_runs` finds runs in the workspace run index by their tags.
`compare_runs` compares models of those runs from their records alone (held-out
predictions and split records): folds are matched by held-out group, and
within a fold every metric is recomputed on the molecules all entries
predicted, so differences are paired molecule for molecule. Uncertainty is
reported at both levels:

- across folds: each entry's per-fold values, their mean and SD, and for
  each paired difference against the reference the per-fold differences and
  how many folds favour the entry;
- within folds: a seeded bootstrap over molecules (resampled within each
  fold, stratified by class, the same resamples for every entry), giving
  percentile intervals for each entry's fold-mean and each paired difference
  (``ci_low`` / ``ci_high``) -- uncertainty from molecule sampling only;
- between experiments: a bootstrap over held-out folds (``fold_ci_low`` /
  ``fold_ci_high``) and, for paired differences, an exact sign-flip test over
  the per-fold differences (``sign_flip_p``, two-sided) -- what a new batch
  would see. With n folds the smallest attainable p is 2 / 2**n (5 folds:
  0.0625).

Metrics (positive class): ``roc_auc``, ``average_precision``,
``normalized_average_precision`` (over the fold's positive fraction) and
``normalized_average_precision_at_prevalence`` (reweighted to
``prevalence``, see `average_precision_at_prevalence`).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.readwrite import atomic_write_json

from ..workspace import MLWorkspace, resolve_ml_workspace
from .contracts import MLJobServiceError
from .runs import MODELS, PREDICTIONS, SPLITS, TAGS

DEFAULT_METRICS = (
    "roc_auc",
    "average_precision",
    "normalized_average_precision",
    "normalized_average_precision_at_prevalence",
)


# --- metrics on (truth, score), fast enough to bootstrap ------------------------


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged)."""
    truth = np.asarray(truth, dtype=bool)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(
    truth: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Average precision (step-wise, as scikit-learn), optionally weighted."""
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    weights = np.ones(truth.size) if weights is None else np.asarray(weights, dtype=float)
    total = weights[truth].sum()
    if total == 0 or truth.all():
        return float("nan") if total == 0 else 1.0
    order = np.argsort(-score, kind="mergesort")
    score, truth, weights = score[order], truth[order], weights[order]
    tp = np.cumsum(weights * truth)
    fp = np.cumsum(weights * ~truth)
    # One threshold per distinct score: the last row of each tie run.
    last = np.r_[np.flatnonzero(np.diff(score)), score.size - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall_step = np.diff(np.r_[0.0, tp]) / total
    return float(np.sum(recall_step * precision))


def _prevalence_weights(truth: np.ndarray, prevalence: float) -> np.ndarray:
    n_pos, n_neg = truth.sum(), (~truth).sum()
    odds = prevalence / (1 - prevalence)
    return np.where(truth, odds * n_neg / max(n_pos, 1), 1.0)


def _metric_functions(prevalence: float) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "normalized_average_precision": lambda t, s: (
            average_precision(t, s) / np.mean(t) if np.any(t) else float("nan")
        ),
        "normalized_average_precision_at_prevalence": lambda t, s: (
            average_precision(t, s, _prevalence_weights(t, prevalence)) / prevalence
        ),
    }


def _fold_interval(
    values: pd.Series, alpha: float, rng: np.random.Generator, draws: int = 2000
) -> dict[str, float | None]:
    """Percentile interval of the mean over held-out folds resampled with
    replacement: between-experiment uncertainty."""
    finite = values.dropna().to_numpy(dtype=float)
    if finite.size < 2:
        return {"fold_ci_low": None, "fold_ci_high": None}
    means = rng.choice(finite, size=(draws, finite.size), replace=True).mean(axis=1)
    return {
        "fold_ci_low": float(np.quantile(means, alpha)),
        "fold_ci_high": float(np.quantile(means, 1 - alpha)),
    }


def sign_flip_p(
    differences: np.ndarray, *, max_exact: int = 16, draws: int = 20000
) -> float | None:
    """Two-sided paired sign-flip test of a mean difference across folds:
    the share of sign patterns whose absolute mean is at least the observed one --
    every pattern up to ``max_exact`` folds, a seeded sample beyond."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n == 0:
        return None
    observed = abs(differences.mean())
    if n <= max_exact:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(0).choice([-1, 1], size=(draws, n))
    means = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


# --- selecting runs ---------------------------------------------------------------


def _workspace(workspace: MLWorkspace | None, project_dir) -> MLWorkspace:
    if (workspace is None) == (project_dir is None):
        raise MLJobServiceError("pass exactly one of workspace or project_dir")
    return workspace or resolve_ml_workspace(project_dir=project_dir)


def select_runs(
    *,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    tags: Mapping[str, Any] | None = None,
    action: str = "train",
) -> pd.DataFrame:
    """Completed runs whose tags include every given ``tags`` item, from the
    workspace run index: one row per run (``run_id``, ``created_at``,
    ``model_keys``, and a ``tag:<name>`` column per tag)."""
    workspace = _workspace(workspace, project_dir)
    path = workspace.index_root / "runs.json"
    if not path.is_file():
        raise MLJobServiceError(f"no run index at {path}; publish a run or rebuild the index")
    wanted = {key: str(value) for key, value in (tags or {}).items()}
    rows = []
    for record in json.loads(path.read_text())["records"]:
        run_tags = record.get("tags") or {}
        if record["state"] != "completed" or record["action"] != action:
            continue
        if any(run_tags.get(key) != value for key, value in wanted.items()):
            continue
        rows.append(
            {
                "run_id": record["run_id"],
                "created_at": record["created_at"],
                "model_keys": list(record["model_keys"]),
                **{f"tag:{key}": value for key, value in run_tags.items()},
            }
        )
    return (
        pd.DataFrame(rows).sort_values("created_at", ignore_index=True) if rows else pd.DataFrame()
    )


# --- comparing --------------------------------------------------------------------


@dataclass(frozen=True)
class RunComparison:
    """Per-fold metrics, summaries and paired differences of compared entries."""

    entries: pd.DataFrame  # entry, run_id, model, model_class, tags
    folds: pd.DataFrame  # fold, n, n_positive, dropped per entry
    fold_metrics: pd.DataFrame  # entry, fold, metric, value
    summary: pd.DataFrame  # entry, metric, mean, sd, n_folds, ci_low, ci_high
    differences: pd.DataFrame  # entry, reference, metric, mean, sd, ci, folds_better
    fold_differences: pd.DataFrame  # entry, reference, fold, metric, difference
    settings: Mapping[str, Any] = field(default_factory=dict)

    def write(self, directory: str | Path, *, figure_metric: str | None = None) -> Path:
        """Tables (CSV), settings with source run ids (JSON) and a figure."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "entries",
            "folds",
            "fold_metrics",
            "summary",
            "differences",
            "fold_differences",
        ):
            frame = getattr(self, name)
            if name == "entries":
                frame = frame.assign(tags=frame["tags"].map(json.dumps))
            frame.to_csv(directory / f"{name}.csv", index=False)
        atomic_write_json(directory / "comparison.json", dict(self.settings))
        from smftools.analysis.plot.ml_results import plot_run_comparison

        metric = figure_metric or self.settings["metrics"][0]
        plot_run_comparison(self, directory / f"comparison_{metric}.png", metric=metric)
        return directory


def _labels(entries: list[dict], label_tags: Sequence[str] | None) -> list[str]:
    if label_tags is None:
        keys = sorted({key for entry in entries for key in entry["tags"]})
        label_tags = [key for key in keys if len({entry["tags"].get(key) for entry in entries}) > 1]
    labels = []
    for entry in entries:
        parts = [str(entry["tags"].get(key, "")) for key in label_tags]
        labels.append("/".join([*filter(None, parts), entry["model"]]))
    if len(set(labels)) != len(labels):
        labels = [f"{label} ({entry['run_id'][:8]})" for label, entry in zip(labels, entries)]
    return labels


def compare_runs(
    run_ids: Sequence[str],
    *,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    models: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    reference: str | None = None,
    label_tags: Sequence[str] | None = None,
    prevalence: float = 0.10,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> RunComparison:
    """Compare the models of published train runs on their shared folds.

    Args:
        run_ids: Train runs (e.g. ``select_runs(...)["run_id"]``).
        workspace / project_dir: The workspace holding them.
        models: Model keys to include (default: every model of every run).
        metrics: Names from `DEFAULT_METRICS`.
        reference: Entry label the paired differences are taken against
            (default: the first entry).
        label_tags: Tags that name entries (default: the tags that differ
            among the runs), followed by the model key.
        prevalence: For ``normalized_average_precision_at_prevalence``.
        n_bootstrap / confidence / seed: Within-fold molecule bootstrap.
    """
    workspace = _workspace(workspace, project_dir)
    functions = _metric_functions(prevalence)
    unknown = sorted(set(metrics) - set(functions))
    if unknown:
        raise MLJobServiceError(f"unknown comparison metrics {unknown}; use {sorted(functions)}")
    entries, predictions = [], []
    for run_id in run_ids:
        path = workspace.runs_root / run_id
        manifest = json.loads((path / "run_manifest.json").read_text())
        if manifest["action"] != "train" or manifest["state"] != "completed":
            raise MLJobServiceError(f"run {run_id} is not a completed train run")
        tags = json.loads((path / TAGS).read_text())
        classes = {
            item["model"]: item.get("model_class")
            for item in json.loads((path / MODELS).read_text())
        }
        splits = {item["fold"]: item for item in json.loads((path / SPLITS).read_text())}
        table = pd.read_parquet(path / PREDICTIONS)
        for model in manifest["model_keys"]:
            if models is not None and model not in models:
                continue
            rows = table[table["model"] == model]
            entries.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "model_class": classes.get(model),
                    "tags": tags,
                    "splits": splits,
                }
            )
            predictions.append(rows)
    if len(entries) < 1:
        raise MLJobServiceError("no runs / models to compare")
    labels = _labels(entries, label_tags)
    reference = reference or labels[0]
    if reference not in labels:
        raise MLJobServiceError(f"reference {reference!r} is not one of {labels}")
    positive_columns = []
    for rows in predictions:
        columns = [c for c in rows.columns if c.startswith("p_")]
        positive_columns.append(columns)
    # Positive class: the one every run scores, named by the runs' metrics.
    class_names = [{c.removeprefix("p_") for c in columns} for columns in positive_columns]
    shared_classes = set.intersection(*class_names)
    if len(shared_classes) != 2:
        raise MLJobServiceError("comparison needs binary runs over the same two classes")
    truths = set(pd.concat(predictions)["truth"].dropna().unique())
    positive = _positive_class(workspace, entries[0]["run_id"], entries[0]["model"], shared_classes)

    # Folds: matched by held-out group, kept when every entry has them.
    held = [set(rows["held_out"]) for rows in predictions]
    shared = sorted(set.intersection(*held))
    dropped = sorted(set.union(*held) - set(shared))
    if not shared:
        raise MLJobServiceError("the runs share no held-out fold")
    if dropped:
        warnings.warn(f"comparing on shared folds only; dropped {dropped}", stacklevel=2)
    rng = np.random.default_rng(seed)
    fold_rows, metric_rows, boot_rows = [], [], []
    for fold in shared:
        frames = []
        for label, rows in zip(labels, predictions, strict=True):
            part = rows[rows["held_out"] == fold].set_index("molecule_uid")
            frames.append(part[["truth", f"p_{positive}"]].rename(columns={f"p_{positive}": label}))
        common = frames[0].index
        for frame in frames[1:]:
            common = common.intersection(frame.index)
        common = common.sort_values()
        if len(common) == 0:
            raise MLJobServiceError(f"fold {fold}: no molecule is predicted by every entry")
        truth_columns = pd.concat([frame.loc[common, "truth"] for frame in frames], axis=1)
        if not (truth_columns.nunique(axis=1) == 1).all():
            raise MLJobServiceError(f"fold {fold}: entries disagree on molecules' true class")
        truth = truth_columns.iloc[:, 0].to_numpy() == positive
        scores = np.column_stack([frame.loc[common, label] for frame, label in zip(frames, labels)])
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(common)),
                "n_positive": int(truth.sum()),
                **{
                    f"dropped:{label}": int(len(frame) - len(common))
                    for frame, label in zip(frames, labels)
                },
            }
        )
        for metric in metrics:
            for e, label in enumerate(labels):
                metric_rows.append(
                    {
                        "entry": label,
                        "fold": fold,
                        "metric": metric,
                        "value": functions[metric](truth, scores[:, e]),
                    }
                )
        # Class-stratified resamples of this fold's molecules, shared by all entries.
        positives, negatives = np.flatnonzero(truth), np.flatnonzero(~truth)
        for b in range(n_bootstrap):
            rows = np.concatenate(
                [
                    rng.choice(positives, positives.size, replace=True),
                    rng.choice(negatives, negatives.size, replace=True),
                ]
            )
            for metric in metrics:
                for e, label in enumerate(labels):
                    boot_rows.append(
                        (b, fold, metric, label, functions[metric](truth[rows], scores[rows, e]))
                    )
    fold_metrics = pd.DataFrame(metric_rows)
    boot = pd.DataFrame(boot_rows, columns=["draw", "fold", "metric", "entry", "value"])
    # A draw's fold-mean per entry; then the paired difference to the reference.
    draw_means = boot.groupby(["draw", "metric", "entry"])["value"].mean().unstack("entry")
    alpha = (1 - confidence) / 2
    # Its own stream: the molecule bootstrap's draws stay as they were.
    fold_rng = np.random.default_rng([seed, 1])
    summary_rows, difference_rows, fold_difference_rows = [], [], []
    for metric in metrics:
        values = fold_metrics[fold_metrics["metric"] == metric].pivot(
            index="fold", columns="entry", values="value"
        )
        means = draw_means.xs(metric, level="metric")
        for label in labels:
            summary_rows.append(
                {
                    "entry": label,
                    "metric": metric,
                    "mean": float(values[label].mean()),
                    "sd": float(values[label].std(ddof=1)) if len(values) > 1 else None,
                    "n_folds": int(values[label].notna().sum()),
                    "ci_low": float(means[label].quantile(alpha)),
                    "ci_high": float(means[label].quantile(1 - alpha)),
                    **_fold_interval(values[label], alpha, fold_rng),
                }
            )
            if label == reference:
                continue
            per_fold = values[label] - values[reference]
            for fold, value in per_fold.items():
                fold_difference_rows.append(
                    {
                        "entry": label,
                        "reference": reference,
                        "fold": fold,
                        "metric": metric,
                        "difference": float(value),
                    }
                )
            boot_difference = means[label] - means[reference]
            difference_rows.append(
                {
                    "entry": label,
                    "reference": reference,
                    "metric": metric,
                    "mean": float(per_fold.mean()),
                    "sd": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else None,
                    "ci_low": float(boot_difference.quantile(alpha)),
                    "ci_high": float(boot_difference.quantile(1 - alpha)),
                    "folds_better": int((per_fold > 0).sum()),
                    "n_folds": int(per_fold.notna().sum()),
                    **_fold_interval(per_fold, alpha, fold_rng),
                    "sign_flip_p": sign_flip_p(per_fold.to_numpy(dtype=float)),
                }
            )
    entry_frame = pd.DataFrame(
        {
            "entry": labels,
            "run_id": [entry["run_id"] for entry in entries],
            "model": [entry["model"] for entry in entries],
            "model_class": [entry["model_class"] for entry in entries],
            "tags": [entry["tags"] for entry in entries],
        }
    )
    return RunComparison(
        entries=entry_frame,
        folds=pd.DataFrame(fold_rows),
        fold_metrics=fold_metrics,
        summary=pd.DataFrame(summary_rows),
        differences=pd.DataFrame(difference_rows),
        fold_differences=pd.DataFrame(fold_difference_rows),
        settings={
            "run_ids": list(dict.fromkeys(entry["run_id"] for entry in entries)),
            "entries": labels,
            "reference": reference,
            "positive_class": positive,
            "metrics": list(metrics),
            "prevalence": prevalence,
            "folds": shared,
            "dropped_folds": dropped,
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "seed": seed,
            "truth_classes": sorted(truths),
        },
    )


def _positive_class(workspace: MLWorkspace, run_id: str, model: str, classes: set[str]) -> str:
    """The positive class of the run's labels (its plan's label schema)."""
    from ..contracts import LabelSchema
    from ..plan import parse_ml_plan

    path = workspace.runs_root / run_id
    plan = parse_ml_plan(json.loads((path / "resolved_plan.json").read_text()))
    job = json.loads((path / "run_manifest.json").read_text())["job_name"]
    labels = plan.datasets[plan.jobs[job].dataset].labels
    positive = None if labels is None else LabelSchema.from_plan_label(labels).positive_class
    if positive not in classes:
        raise MLJobServiceError(f"run {run_id}: no positive class among {sorted(classes)}")
    return positive
