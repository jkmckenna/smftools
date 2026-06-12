"""
Generic plotting helpers for binary classifier evaluation.

Inputs are metric arrays and summary tables plus an explicit output path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: str | Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=1.2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
    baseline: float,
    output_path: str | Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.plot(recall, precision, color="#d62728", linewidth=1.2, label=f"PR AUC={pr_auc:.3f}")
    ax.axhline(baseline, linestyle="--", color="#777777", linewidth=0.8)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_pair(
    metrics_dict: dict,
    output_path: str | Path,
    title: str = "",
    prefix: str = "test",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75))

    fpr, tpr = metrics_dict[f"{prefix}_roc_curve"]
    recall, precision = metrics_dict[f"{prefix}_pr_curve"]
    roc_auc = metrics_dict[f"{prefix}_auc"]
    pr_auc = metrics_dict[f"{prefix}_pr_auc"]
    baseline = metrics_dict[f"{prefix}_pos_freq"]

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=1.2, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate", fontsize=9)
    axes[0].set_ylabel("True Positive Rate", fontsize=9)
    axes[0].set_title("ROC", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)

    axes[1].plot(recall, precision, color="#d62728", linewidth=1.2, label=f"PR AUC={pr_auc:.3f}")
    axes[1].axhline(baseline, linestyle="--", color="#777777", linewidth=0.8)
    axes[1].set_xlabel("Recall", fontsize=9)
    axes[1].set_ylabel("Precision", fontsize=9)
    axes[1].set_title("Precision-Recall", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)

    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_barplot(
    summary_df: pd.DataFrame,
    metric_col: str,
    group_cols: list[str],
    output_path: str | Path,
    title: str = "",
) -> None:
    """
    Plot a simple mean ± SEM barplot after grouping rows by ``group_cols``.
    """
    grouped = (
        summary_df.groupby(group_cols, dropna=False)[metric_col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    labels = grouped[group_cols].astype(str).agg(" | ".join, axis=1)
    x = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=(max(3.2, 0.55 * len(grouped)), 3.0))
    ax.bar(
        x,
        grouped["mean"].to_numpy(dtype=float),
        yerr=grouped["sem"].fillna(0.0).to_numpy(dtype=float),
        color="#c9c9c9",
        edgecolor="#555555",
        linewidth=0.8,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_col, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
