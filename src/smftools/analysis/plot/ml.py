"""
Generic plotting helpers for binary classifier evaluation.

Inputs are metric arrays and summary tables plus an explicit output path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
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


def plot_score_distributions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_logit: np.ndarray,
    output_path: str | Path,
    title: str = "",
    bins: int = 30,
) -> None:
    """
    Plot probability and logit score distributions, split by true label.

    ``y_score`` is the positive-class probability in ``[0, 1]``; ``y_logit``
    is its log-odds transform (see ``ml_metrics.logit_from_probability``).

    Top row is raw read counts; bottom row is each class's histogram
    normalized to a probability density (area under each class's curve
    integrates to 1), so distribution *shape* is comparable across
    folds/models regardless of class imbalance or eval-set size.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_logit = np.asarray(y_logit, dtype=float)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    logit_range = (float(np.min(y_logit)), float(np.max(y_logit))) if y_logit.size else (0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2))

    panels = [
        (
            axes[0, 0],
            y_score,
            (0.0, 1.0),
            "Predicted probability",
            "Probability distribution",
            False,
        ),
        (axes[0, 1], y_logit, logit_range, "Logit (log-odds)", "Logit distribution", False),
        (axes[1, 0], y_score, (0.0, 1.0), "Predicted probability", "Probability density", True),
        (axes[1, 1], y_logit, logit_range, "Logit (log-odds)", "Logit density", True),
    ]
    for ax, values, x_range, xlabel, panel_title, density in panels:
        ax.hist(
            values[neg_mask],
            bins=bins,
            range=x_range,
            density=density,
            alpha=0.55,
            color="#1f77b4",
            label="label=0",
        )
        ax.hist(
            values[pos_mask],
            bins=bins,
            range=x_range,
            density=density,
            alpha=0.55,
            color="#d62728",
            label="label=1",
        )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density" if density else "Read count", fontsize=9)
        ax.set_title(panel_title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_read_score_heatmap(
    matrix: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_logit: np.ndarray,
    output_path: str | Path,
    coords: np.ndarray | None = None,
    title: str = "",
    input_cmap=None,
    value_range: tuple[float, float] | None = (0.0, 1.0),
    label_names: dict[int, str] | None = None,
    label_colors: dict[int, str] | None = None,
    x_label: str = "Position",
    dpi: int = 300,
    position_scores: np.ndarray | None = None,
    position_score_label: str = "Contribution",
    position_score_cmap: str = "coolwarm",
    position_score_range: tuple[float, float] | None = None,
    sort_by: str = "logit",
    mean_smooth_window: int | None = None,
    cluster_labels: np.ndarray | None = None,
    cluster_label_name: str = "Cluster",
) -> None:
    """
    Per-read input-feature heatmap, with adjacent single-column strips for
    predicted probability, logit, and true class label.

    Companion to :func:`plot_score_distributions`: where that plot summarizes
    the eval-set score distribution in aggregate, this plot shows which
    specific reads — and which specific input features — drive each read's
    score, in the same molecule x position layout used by the project's
    dimensionality-reduction clustermaps.

    Parameters
    ----------
    matrix : (n_reads, n_positions) input feature matrix (e.g. C_site_binary),
        aligned row-for-row with ``y_true``/``y_score``/``y_logit``.
    coords : optional 1-D array of x-axis tick coordinates, length ``n_positions``.
    position_scores : optional (n_reads, n_positions) per-read, per-position
        attribution/contribution matrix (e.g. SHAP values, NB log-odds
        contributions, or CNN integrated-gradients attributions), aligned
        row-for-row and column-for-column with ``matrix``. When given, a
        second heatmap panel is drawn to the right of the True-label strip,
        with its own diverging colorbar (centered at 0).
    position_score_range : optional explicit (vmin, vmax) for the diverging
        colorbar; default is symmetric around the 99th percentile absolute
        value of ``position_scores``.
    sort_by : ``"logit"`` (default) sorts rows by ascending classifier logit —
        unaffected by ``cluster_labels``. ``"cluster"`` instead
        hierarchically clusters rows on ``position_scores``: with
        ``cluster_labels`` given, rows are first binned by Leiden cluster,
        then by true label within each cluster, then hierarchically
        clustered within each (cluster, label) leaf (via
        ``smftools.analysis.compute.clustering.cluster_row_order_by_labels``);
        without ``cluster_labels``, plain hierarchical clustering with no
        binning (``cluster_block_order``). Requires ``position_scores``.
    mean_smooth_window : optional rolling-average window (in positions) applied
        to the per-label mean lines above the heatmaps. ``None`` (default)
        plots the raw per-position mean; pass e.g. ``3`` for a 3-position
        centered rolling average — a reasonable default window when
        smoothing is wanted. NaN positions (no observed reads) are skipped
        rather than propagating NaN through the window.
    cluster_labels : optional per-read cluster assignment (e.g. Leiden cluster
        id from clustering the attribution matrix), aligned row-for-row with
        ``matrix``/``y_true``. Only meaningful alongside ``position_scores``;
        when given, a categorical strip is drawn to the right of the
        position-score heatmap, colored with the same tab10 cycling used by
        ``smftools.analysis.plot.embeddings.plot_embedding_scatter``.

    Above both the input heatmap and (when given) the position-score
    heatmap, a line-plot panel shows each panel's per-position mean, one
    line per true-class label (using ``label_colors``) — the same summary
    already drawn atop this project's other clustermaps, split by label.
    """
    matrix = np.asarray(matrix, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_logit = np.asarray(y_logit, dtype=float)

    if sort_by == "logit":
        order = np.argsort(y_logit)
    elif sort_by == "cluster":
        if position_scores is None:
            raise ValueError("sort_by='cluster' requires position_scores")
        if cluster_labels is not None:
            from smftools.analysis.compute.clustering import cluster_row_order_by_labels

            order = cluster_row_order_by_labels(
                np.asarray(position_scores, dtype=float), [np.asarray(cluster_labels), y_true]
            )
        else:
            from smftools.analysis.compute.clustering import cluster_block_order

            order = cluster_block_order(np.asarray(position_scores, dtype=float))
    else:
        raise ValueError(f"Unsupported sort_by {sort_by!r}; expected 'logit' or 'cluster'")

    matrix_ord = matrix[order]
    y_true_ord = y_true[order]
    y_score_ord = y_score[order]
    y_logit_ord = y_logit[order]
    scores_ord = None
    if position_scores is not None:
        scores_ord = np.asarray(position_scores, dtype=float)[order]
    cluster_labels_ord = None
    if cluster_labels is not None:
        cluster_labels_ord = np.asarray(cluster_labels)[order]

    n_rows, n_pos = matrix_ord.shape
    label_names = label_names or {0: "label=0", 1: "label=1"}
    label_colors = label_colors or {0: "#1f77b4", 1: "#d62728"}
    if sort_by == "logit":
        row_axis_label = "sorted by logit"
    elif cluster_labels is not None:
        row_axis_label = "clustered by attribution, binned by Leiden cluster then true label"
    else:
        row_axis_label = "clustered by attribution"

    def _column_nanmean_by_label(mat: np.ndarray) -> dict:
        out = {}
        for key in label_keys:
            sub = mat[y_true_ord == key]
            if sub.size == 0:
                continue
            valid = np.isfinite(sub)
            counts = valid.sum(axis=0)
            sums = np.where(valid, sub, 0.0).sum(axis=0)
            means = np.full(mat.shape[1], np.nan, dtype=float)
            nz = counts > 0
            means[nz] = sums[nz] / counts[nz]
            out[key] = means
        return out

    label_keys = sorted(label_names)

    has_scores = scores_ord is not None
    has_clusters = has_scores and cluster_labels_ord is not None
    # column layout: heat, prob, logit, label, cbar(prob+logit) [, scores [, cluster], cbar(scores)]
    width_ratios = (
        [7.0, 0.3, 0.3, 0.3, 0.22]
        + ([7.0] if has_scores else [])
        + ([0.3] if has_clusters else [])
        + ([0.22] if has_scores else [])
    )
    extra_width = (5.7 if has_scores else 0.0) + (0.35 if has_clusters else 0.0)
    fig_h = max(3.5, min(14.0, 0.015 * n_rows + 2.0))
    fig = plt.figure(figsize=(8.9 + extra_width, fig_h + 1.4), dpi=dpi)
    gs = fig.add_gridspec(
        2,
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[1.3, fig_h],
        wspace=0.12,
        hspace=0.15,
    )
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_input_mean = fig.add_subplot(gs[0, 0], sharex=ax_heat)
    ax_prob = fig.add_subplot(gs[1, 1], sharey=ax_heat)
    ax_logit = fig.add_subplot(gs[1, 2], sharey=ax_heat)
    ax_label = fig.add_subplot(gs[1, 3], sharey=ax_heat)
    ax_cbar_col = fig.add_subplot(gs[1, 4])
    ax_scores = fig.add_subplot(gs[1, 5], sharey=ax_heat) if has_scores else None
    ax_scores_mean = fig.add_subplot(gs[0, 5], sharex=ax_scores) if has_scores else None
    cbar_scores_col = 6 + (1 if has_clusters else 0)
    ax_cluster = fig.add_subplot(gs[1, 6], sharey=ax_heat) if has_clusters else None
    ax_cbar_scores = fig.add_subplot(gs[1, cbar_scores_col]) if has_scores else None

    _cmap = input_cmap if input_cmap is not None else matplotlib.colormaps["Greens"].copy()
    _cmap.set_bad(color="#d0d0d0")
    masked = np.ma.masked_invalid(matrix_ord)
    vmin, vmax = value_range if value_range is not None else (None, None)
    ax_heat.imshow(masked, aspect="auto", cmap=_cmap, vmin=vmin, vmax=vmax, interpolation="none")
    ax_heat.set_ylabel(f"Molecules (n={n_rows}), {row_axis_label}", fontsize=9)
    ax_heat.tick_params(axis="y", left=False, labelleft=False)

    def _apply_position_ticks(ax):
        if coords is not None and n_pos > 0:
            n_ticks = min(10, n_pos)
            tick_idx = np.linspace(0, n_pos - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(
                [f"{coords[i]:.0f}" for i in tick_idx], rotation=45, ha="right", fontsize=7
            )
        else:
            ax.set_xticks([])
        ax.set_xlabel(x_label, fontsize=9)

    _apply_position_ticks(ax_heat)

    def _plot_mean_by_label(ax, mat: np.ndarray, title: str, zero_line: bool) -> None:
        means_by_label = _column_nanmean_by_label(mat)
        x = np.arange(mat.shape[1], dtype=float)
        all_vals = []
        if mean_smooth_window is not None and mean_smooth_window > 1:
            title = f"{title}, smoothed w={mean_smooth_window}"
        for key in label_keys:
            means = means_by_label.get(key)
            if means is None:
                continue
            if mean_smooth_window is not None and mean_smooth_window > 1:
                means = (
                    pd.Series(means)
                    .rolling(window=mean_smooth_window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
            ax.plot(
                x,
                means,
                color=label_colors[key],
                linewidth=1.0,
                label=label_names.get(key, str(key)),
            )
            all_vals.append(means)
        if zero_line:
            ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        if all_vals:
            finite = np.concatenate(all_vals)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                if zero_line:
                    max_abs = float(np.max(np.abs(finite)))
                    pad = max(max_abs * 0.1, 1e-6)
                    ax.set_ylim(-max_abs - pad, max_abs + pad)
                else:
                    lo, hi = float(np.min(finite)), float(np.max(finite))
                    pad = max((hi - lo) * 0.1, 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        # Don't touch x tick *positions* here: this axes shares its x-axis
        # (and thus its tick Locator) with the heatmap below it, so calling
        # set_xticks would clobber that axes' position labels too. Just hide
        # this axes' own tick marks/labels.
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.set_ylabel("mean", fontsize=7)
        ax.set_title(title, fontsize=8, pad=2)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.15)

    _plot_mean_by_label(ax_input_mean, matrix_ord, "Input mean (by label)", zero_line=False)

    prob_im = ax_prob.imshow(
        y_score_ord[:, np.newaxis],
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="none",
    )
    ax_prob.set_xticks([])
    ax_prob.tick_params(axis="y", left=False, labelleft=False)
    ax_prob.text(
        0.5,
        1.01,
        "Prob.",
        transform=ax_prob.transAxes,
        fontsize=7,
        rotation=45,
        ha="left",
        va="bottom",
    )

    logit_abs = float(np.nanmax(np.abs(y_logit_ord))) if y_logit_ord.size else 1.0
    logit_abs = logit_abs if logit_abs > 0 else 1.0
    logit_im = ax_logit.imshow(
        y_logit_ord[:, np.newaxis],
        aspect="auto",
        cmap="coolwarm",
        vmin=-logit_abs,
        vmax=logit_abs,
        interpolation="none",
    )
    ax_logit.set_xticks([])
    ax_logit.tick_params(axis="y", left=False, labelleft=False)
    ax_logit.text(
        0.5,
        1.01,
        "Logit",
        transform=ax_logit.transAxes,
        fontsize=7,
        rotation=45,
        ha="left",
        va="bottom",
    )

    label_cmap = mcolors.ListedColormap([label_colors[k] for k in label_keys])
    label_code = np.asarray([label_keys.index(v) for v in y_true_ord], dtype=float)
    ax_label.imshow(
        label_code[:, np.newaxis],
        aspect="auto",
        cmap=label_cmap,
        vmin=0,
        vmax=max(len(label_keys) - 1, 1),
        interpolation="none",
    )
    ax_label.set_xticks([])
    ax_label.tick_params(axis="y", left=False, labelleft=False)
    ax_label.text(
        0.5,
        1.01,
        "True",
        transform=ax_label.transAxes,
        fontsize=7,
        rotation=45,
        ha="left",
        va="bottom",
    )

    scores_im = None
    if has_scores:
        if position_score_range is not None:
            score_vmin, score_vmax = position_score_range
        else:
            score_abs = (
                float(np.nanpercentile(np.abs(scores_ord), 99))
                if np.isfinite(scores_ord).any()
                else 1.0
            )
            score_abs = score_abs if score_abs > 0 else 1.0
            score_vmin, score_vmax = -score_abs, score_abs
        scores_im = ax_scores.imshow(
            np.ma.masked_invalid(scores_ord),
            aspect="auto",
            cmap=position_score_cmap,
            vmin=score_vmin,
            vmax=score_vmax,
            interpolation="none",
        )
        ax_scores.tick_params(axis="y", left=False, labelleft=False)
        _apply_position_ticks(ax_scores)
        _plot_mean_by_label(
            ax_scores_mean, scores_ord, f"{position_score_label} mean (by label)", zero_line=True
        )

    if has_clusters:
        from smftools.analysis.plot.embeddings import _TAB10

        cluster_keys = sorted({str(c) for c in cluster_labels_ord})
        cluster_cmap = mcolors.ListedColormap(
            [_TAB10[i % len(_TAB10)] for i in range(len(cluster_keys))]
        )
        cluster_index = {name: i for i, name in enumerate(cluster_keys)}
        cluster_code = np.asarray([cluster_index[str(c)] for c in cluster_labels_ord], dtype=float)
        ax_cluster.imshow(
            cluster_code[:, np.newaxis],
            aspect="auto",
            cmap=cluster_cmap,
            vmin=0,
            vmax=max(len(cluster_keys) - 1, 1),
            interpolation="none",
        )
        ax_cluster.set_xticks([])
        ax_cluster.tick_params(axis="y", left=False, labelleft=False)
        ax_cluster.text(
            0.5,
            1.01,
            cluster_label_name,
            transform=ax_cluster.transAxes,
            fontsize=7,
            rotation=45,
            ha="left",
            va="bottom",
        )

    # ax_cbar_col is a gridspec cell reserved purely to host the prob/logit
    # colorbars as inset axes — inset_axes is positioned in the parent axes'
    # own coordinate frame, so it stays correctly anchored regardless of how
    # tight_layout subsequently adjusts subplot spacing.
    ax_cbar_col.axis("off")
    cax_prob = ax_cbar_col.inset_axes((0.15, 0.56, 0.5, 0.4))
    cbar_prob = fig.colorbar(prob_im, cax=cax_prob)
    cbar_prob.ax.tick_params(labelsize=6)
    cbar_prob.set_label("probability", fontsize=7)

    cax_logit = ax_cbar_col.inset_axes((0.15, 0.02, 0.5, 0.4))
    cbar_logit = fig.colorbar(logit_im, cax=cax_logit)
    cbar_logit.ax.tick_params(labelsize=6)
    cbar_logit.set_label("logit", fontsize=7)

    if has_scores and ax_cbar_scores is not None:
        ax_cbar_scores.axis("off")
        cax_scores = ax_cbar_scores.inset_axes((0.15, 0.0, 0.5, 1.0))
        cbar_scores = fig.colorbar(scores_im, cax=cax_scores)
        cbar_scores.ax.tick_params(labelsize=6)
        cbar_scores.set_label(position_score_label, fontsize=7)

    handles = [
        matplotlib.patches.Patch(
            facecolor=label_colors[k], edgecolor=label_colors[k], label=label_names.get(k, str(k))
        )
        for k in label_keys
    ]
    fig.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        bbox_transform=ax_cbar_col.transAxes,
    )

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
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
        summary_df.groupby(group_cols, dropna=False)[metric_col].agg(["mean", "sem"]).reset_index()
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
