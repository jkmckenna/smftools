"""
Matrix-level helpers for binary classifier fitting and evaluation.

Inputs are feature matrices, labels, and metadata-derived parameters. No AnnData
access or file I/O occurs here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from smftools.machine_learning.compatibility._warnings import deprecated_ml_alias
from smftools.machine_learning.compatibility.classical_models import (
    build_binary_classifier as _build_binary_classifier,
)
from smftools.machine_learning.compatibility.classical_models import (
    fit_classifier as _fit_classifier,
)
from smftools.machine_learning.compatibility.classical_models import (
    predict_binary_scores as _predict_binary_scores,
)

_CANONICAL_TRAINING = "smftools.machine_learning.orchestration.train_partition_model"
_CANONICAL_INFERENCE = "smftools.machine_learning.orchestration.apply_partition_model"

build_binary_classifier = deprecated_ml_alias(
    "smftools.analysis.compute.ml_metrics.build_binary_classifier",
    "smftools.machine_learning.models.BUILTIN_MODEL_REGISTRY",
)(_build_binary_classifier)
fit_classifier = deprecated_ml_alias(
    "smftools.analysis.compute.ml_metrics.fit_classifier",
    _CANONICAL_TRAINING,
)(_fit_classifier)
predict_binary_scores = deprecated_ml_alias(
    "smftools.analysis.compute.ml_metrics.predict_binary_scores",
    _CANONICAL_INFERENCE,
)(_predict_binary_scores)


def logit_from_probability(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert positive-class probabilities to log-odds (logit) scores.

    Probabilities are clipped to ``[eps, 1 - eps]`` before the transform so
    that scores of exactly 0 or 1 (common for tree ensembles and CNNs) don't
    map to +/-inf.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def normalize_pr_auc(pr_auc: float, pos_freq: float) -> float:
    """
    Express PR AUC as fold improvement over the baseline positive frequency.
    """
    if not np.isfinite(pr_auc) or not np.isfinite(pos_freq) or pos_freq <= 0:
        return float("nan")
    return float(pr_auc / pos_freq)


def _rebalance_eval_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    target_eval_freq: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample an evaluation set to a requested positive-class frequency.

    The resampled evaluation retains all negatives and subsamples positives
    without replacement to the largest feasible count at the requested class
    balance. This mirrors the older project logic where alternate evaluation
    priors were implemented by class-stratified resampling rather than by
    changing model training.
    """
    if not np.isfinite(target_eval_freq) or target_eval_freq <= 0 or target_eval_freq >= 1:
        raise ValueError("target_eval_freq must be between 0 and 1")

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return y_true, y_score, y_pred

    max_pos = int(target_eval_freq * neg_idx.size / (1 - target_eval_freq))
    n_pos = min(pos_idx.size, max_pos)
    if n_pos <= 0:
        return y_true, y_score, y_pred

    rng = np.random.default_rng(random_state)
    if n_pos < pos_idx.size:
        pos_keep = rng.choice(pos_idx, size=n_pos, replace=False)
    else:
        pos_keep = pos_idx
    keep_idx = np.sort(np.concatenate([neg_idx, pos_keep]))
    return y_true[keep_idx], y_score[keep_idx], y_pred[keep_idx]


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    prefix: str = "test",
    target_eval_freq: float | None = None,
    random_state: int = 42,
) -> dict:
    """
    Evaluate a binary classifier from labels and positive-class scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    if target_eval_freq is not None:
        y_true, y_score, y_pred = _rebalance_eval_arrays(
            y_true,
            y_score,
            y_pred,
            target_eval_freq=target_eval_freq,
            random_state=random_state,
        )

    pos_freq = float(np.mean(y_true == 1))

    if np.unique(y_true).size < 2:
        raise ValueError("y_true must contain both classes for ROC/PR evaluation")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(auc(recall, precision))

    return {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred)),
        f"{prefix}_auc": roc_auc,
        f"{prefix}_pr_auc": pr_auc,
        f"{prefix}_pr_auc_norm": normalize_pr_auc(pr_auc, pos_freq),
        f"{prefix}_pos_freq": pos_freq,
        f"{prefix}_num_pos": int(np.sum(y_true == 1)),
        f"{prefix}_roc_curve": (fpr, tpr),
        f"{prefix}_pr_curve": (recall, precision),
    }


def make_metrics_row(**kwargs) -> dict:
    """
    Build a summary row dictionary, dropping keys with value ``None``.
    """
    return {key: value for key, value in kwargs.items() if value is not None}
