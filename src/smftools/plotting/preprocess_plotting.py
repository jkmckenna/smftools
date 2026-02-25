from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for ProcessPoolExecutor workers)
# ---------------------------------------------------------------------------


def _resolve_n_jobs(n_jobs: int) -> int:
    import os

    if n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _fill_nan_with_col_means(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    col_means = np.nanmean(filled, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(filled))
    filled[nan_rows, nan_cols] = col_means[nan_cols]
    return filled


def _pca_sort_order(
    matrix: np.ndarray,
    pca_n_components: Optional[int],
    pca_sort_component: int,
) -> np.ndarray:
    n_reads, n_positions = matrix.shape
    if pca_n_components is None:
        n_components = min(n_reads, n_positions)
    else:
        n_components = min(int(pca_n_components), n_reads, n_positions)
    if n_components < 1 or n_reads < 2:
        return np.arange(n_reads)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    if pca_sort_component < 0:
        raise ValueError("pca_sort_component must be >= 0.")
    if pca_sort_component >= scores.shape[1]:
        raise ValueError(
            f"pca_sort_component={pca_sort_component} exceeds available components "
            f"({scores.shape[1]})."
        )
    return np.argsort(scores[:, pca_sort_component], kind="mergesort")


def _plot_one_group(args: dict) -> dict:
    """Render one (ref, sample) read-span/quality clustermap.

    Module-level so it is picklable by ProcessPoolExecutor workers.
    Each worker imports matplotlib/seaborn independently.
    """
    import matplotlib
    import numpy as np
    import scipy.cluster.hierarchy as sch
    from pathlib import Path

    from smftools.optional_imports import require

    # Use the headless backend in worker processes — no display connection needed.
    matplotlib.use("Agg")
    _colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
    _plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
    _sns = require("seaborn", extra="plotting", purpose="plot styling")

    quality_matrix: np.ndarray = args["quality_matrix"]
    read_span_matrix: np.ndarray = args["read_span_matrix"]
    var_names: np.ndarray = args["var_names"]
    ref: str = args["ref"]
    sample: str = args["sample"]
    quality_layer: str = args["quality_layer"]
    read_span_layer: str = args["read_span_layer"]
    quality_cmap: str = args["quality_cmap"]
    read_span_color: str = args["read_span_color"]
    resolved_sort_method: str = args["resolved_sort_method"]
    pca_n_components: Optional[int] = args["pca_n_components"]
    pca_sort_component: int = args["pca_sort_component"]
    resolved_step: Optional[int] = args["resolved_step"]
    xtick_rotation: int = args["xtick_rotation"]
    xtick_fontsize: int = args["xtick_fontsize"]
    show_position_axis: bool = args["show_position_axis"]
    xtick_step: Optional[int] = args["xtick_step"]
    save_path: Optional[str] = args["save_path"]

    # --- row ordering ---
    if quality_matrix.shape[0] < 2:
        order = np.arange(quality_matrix.shape[0])
    else:
        quality_filled = _fill_nan_with_col_means(quality_matrix)
        if resolved_sort_method == "hierarchical":
            linkage = sch.linkage(quality_filled, method="ward")
            order = sch.leaves_list(linkage)
        else:
            order = _pca_sort_order(quality_filled, pca_n_components, pca_sort_component)

    quality_matrix = quality_matrix[order]
    read_span_matrix = read_span_matrix[order]

    # --- figure ---
    fig, axes = _plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 6),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 4], "width_ratios": [1, 1, 0.05]},
    )
    span_bar_ax, quality_bar_ax, bar_spacer_ax = axes[0]
    span_ax, quality_ax, cbar_ax = axes[1]
    bar_spacer_ax.set_axis_off()

    span_mean = np.nanmean(read_span_matrix, axis=0)
    quality_mean = np.nanmean(quality_matrix, axis=0)
    bar_positions = np.arange(read_span_matrix.shape[1]) + 0.5
    span_bar_ax.bar(bar_positions, span_mean, color=read_span_color, width=1.0)
    span_bar_ax.set_title(f"{read_span_layer} mean")
    span_bar_ax.set_xlim(0, read_span_matrix.shape[1])
    span_bar_ax.tick_params(axis="x", labelbottom=False)

    quality_bar_ax.bar(bar_positions, quality_mean, color="#4c72b0", width=1.0)
    quality_bar_ax.set_title(f"{quality_layer} mean")
    quality_bar_ax.set_xlim(0, quality_matrix.shape[1])
    quality_bar_ax.tick_params(axis="x", labelbottom=False)

    span_cmap = _colors.ListedColormap(["white", read_span_color])
    span_norm = _colors.BoundaryNorm([-0.5, 0.5, 1.5], span_cmap.N)
    _sns.heatmap(
        read_span_matrix,
        cmap=span_cmap,
        norm=span_norm,
        ax=span_ax,
        yticklabels=False,
        cbar=False,
    )
    span_ax.set_title(read_span_layer)

    _sns.heatmap(
        quality_matrix,
        cmap=quality_cmap,
        ax=quality_ax,
        yticklabels=False,
        cbar=True,
        cbar_ax=cbar_ax,
    )
    quality_ax.set_title(quality_layer)

    for axis in (span_ax, quality_ax):
        if resolved_step is not None and resolved_step > 0:
            sites = np.arange(0, quality_matrix.shape[1], resolved_step)
            axis.set_xticks(sites)
            axis.set_xticklabels(
                var_names[sites].astype(str),
                rotation=xtick_rotation,
                fontsize=xtick_fontsize,
            )
        else:
            axis.set_xticks([])
        if show_position_axis or xtick_step is not None:
            axis.set_xlabel("Position")

    n_reads = quality_matrix.shape[0]
    fig.suptitle(f"{sample} - {ref} - {n_reads} reads")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_file = None
    if save_path is not None:
        safe_name = f"{ref}__{sample}__read_span_quality".replace("=", "").replace(",", "_")
        out_file = Path(save_path) / f"{safe_name}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        _plt.close(fig)
    else:
        _plt.show()

    return {
        "reference": ref,
        "sample": sample,
        "quality_layer": quality_layer,
        "read_span_layer": read_span_layer,
        "n_positions": int(quality_matrix.shape[1]),
        "output_path": str(out_file) if out_file is not None else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_read_span_quality_clustermaps(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    quality_layer: str = "base_quality_scores",
    read_span_layer: str = "read_span_mask",
    quality_cmap: str = "viridis",
    read_span_color: str = "#2ca25f",
    sort_method: str = "hierarchical",
    pca_n_components: int | None = 20,
    pca_sort_component: int = 0,
    max_nan_fraction: float | None = None,
    min_quality: float | None = None,
    min_length: int | None = None,
    min_mapped_length_to_reference_length_ratio: float | None = None,
    demux_types: Sequence[str] = ("single", "double", "already"),
    max_reads: int | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    show_position_axis: bool = False,
    position_axis_tick_target: int = 25,
    save_path: str | Path | None = None,
    n_jobs: int = 1,
) -> List[Dict[str, Any]]:
    """Plot read-span mask and base quality clustermaps side by side.

    Clustering is performed using the base-quality layer ordering, which is then
    applied to the read-span mask to keep the two panels aligned.

    Args:
        adata: AnnData with read-span and base-quality layers.
        sample_col: Column in ``adata.obs`` that identifies samples.
        reference_col: Column in ``adata.obs`` that identifies references.
        quality_layer: Layer name containing base-quality scores.
        read_span_layer: Layer name containing read-span masks.
        quality_cmap: Colormap for base-quality scores.
        read_span_color: Color for read-span mask (1-values); 0-values are white.
        sort_method: Row ordering strategy ("pca" or "hierarchical").
        pca_n_components: Number of PCA components to compute for ordering. If ``None``,
            uses ``min(n_reads, n_positions)``.
        pca_sort_component: Zero-based PCA component index to sort by (ascending).
        max_nan_fraction: Optional maximum fraction of NaNs allowed per position; positions
            above this threshold are excluded.
        min_quality: Optional minimum read quality filter.
        min_length: Optional minimum mapped length filter.
        min_mapped_length_to_reference_length_ratio: Optional min length ratio filter.
        demux_types: Allowed ``demux_type`` values, if present in ``adata.obs``.
        max_reads: Optional maximum number of reads to plot per sample/reference.
        xtick_step: Spacing between x-axis tick labels (None = no labels).
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        show_position_axis: Whether to draw a position axis with tick labels.
        position_axis_tick_target: Approximate number of ticks to show when auto-sizing.
        save_path: Optional output directory for saving plots.
        n_jobs: Number of parallel worker processes. ``-1`` uses all available CPUs.
            Parallelism is only applied when ``save_path`` is set (interactive display
            is always serial).

    Returns:
        List of dictionaries with per-plot metadata and output paths.
    """
    logger.info("Plotting read span and quality clustermaps.")

    if quality_layer not in adata.layers:
        raise KeyError(f"Layer '{quality_layer}' not found in adata.layers")
    if read_span_layer not in adata.layers:
        raise KeyError(f"Layer '{read_span_layer}' not found in adata.layers")
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")
    if position_axis_tick_target < 1:
        raise ValueError("position_axis_tick_target must be at least 1.")

    sort_method_str = str(sort_method).strip().lower()
    if sort_method_str in ("hierarchical", "hclust", "ward"):
        resolved_sort_method = "hierarchical"
    elif sort_method_str in ("pca", "pc"):
        resolved_sort_method = "pca"
    else:
        raise ValueError("sort_method must be 'pca' or 'hierarchical'.")

    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=s.index)

    def _resolve_xtick_step(n_positions: int) -> int | None:
        if xtick_step is not None:
            return xtick_step
        if not show_position_axis:
            return None
        return max(1, int(np.ceil(n_positions / position_axis_tick_target)))

    # ------------------------------------------------------------------
    # Lazy group generator — yields one group's arrays at a time so that
    # only one group's numpy data is live in the main process at once.
    # For the parallel case, executor.map pickles each yielded dict into
    # the worker queue; the main process can then release that dict before
    # the next one is yielded.
    # ------------------------------------------------------------------
    def _iter_group_args():
        for ref in adata.obs[reference_col].cat.categories:
            for sample in adata.obs[sample_col].cat.categories:
                qmask = _mask_or_true(
                    "read_quality",
                    (lambda s: s >= float(min_quality))
                    if (min_quality is not None)
                    else (lambda s: pd.Series(True, index=s.index)),
                )
                lm_mask = _mask_or_true(
                    "mapped_length",
                    (lambda s: s >= float(min_length))
                    if (min_length is not None)
                    else (lambda s: pd.Series(True, index=s.index)),
                )
                lrr_mask = _mask_or_true(
                    "mapped_length_to_reference_length_ratio",
                    (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                    if (min_mapped_length_to_reference_length_ratio is not None)
                    else (lambda s: pd.Series(True, index=s.index)),
                )
                demux_mask = _mask_or_true(
                    "demux_type",
                    (lambda s: s.astype("string").isin(list(demux_types)))
                    if (demux_types is not None)
                    else (lambda s: pd.Series(True, index=s.index)),
                )

                row_mask = (
                    (adata.obs[reference_col] == ref)
                    & (adata.obs[sample_col] == sample)
                    & qmask
                    & lm_mask
                    & lrr_mask
                    & demux_mask
                )
                if not bool(row_mask.any()):
                    continue

                subset = adata[row_mask, :]
                quality_matrix = np.asarray(subset.layers[quality_layer]).astype(float)
                quality_matrix[quality_matrix < 0] = np.nan
                read_span_matrix = np.asarray(subset.layers[read_span_layer]).astype(float)
                var_names = np.asarray(subset.var_names)

                if max_nan_fraction is not None:
                    nan_mask = np.isnan(quality_matrix) | np.isnan(read_span_matrix)
                    nan_fraction = nan_mask.mean(axis=0)
                    keep_columns = nan_fraction <= max_nan_fraction
                    if not np.any(keep_columns):
                        continue
                    quality_matrix = quality_matrix[:, keep_columns]
                    read_span_matrix = read_span_matrix[:, keep_columns]
                    var_names = var_names[keep_columns]

                if max_reads is not None and quality_matrix.shape[0] > max_reads:
                    quality_matrix = quality_matrix[:max_reads]
                    read_span_matrix = read_span_matrix[:max_reads]

                if quality_matrix.size == 0:
                    continue

                if quality_matrix.shape[0] < 2:
                    logger.debug(
                        "Skipping row ordering for %s/%s: only %d read(s).",
                        sample,
                        ref,
                        quality_matrix.shape[0],
                    )

                yield {
                    "quality_matrix": quality_matrix,
                    "read_span_matrix": read_span_matrix,
                    "var_names": var_names,
                    "ref": str(ref),
                    "sample": str(sample),
                    "quality_layer": quality_layer,
                    "read_span_layer": read_span_layer,
                    "quality_cmap": quality_cmap,
                    "read_span_color": read_span_color,
                    "resolved_sort_method": resolved_sort_method,
                    "pca_n_components": pca_n_components,
                    "pca_sort_component": pca_sort_component,
                    "resolved_step": _resolve_xtick_step(quality_matrix.shape[1]),
                    "xtick_rotation": xtick_rotation,
                    "xtick_fontsize": xtick_fontsize,
                    "show_position_axis": show_position_axis,
                    "xtick_step": xtick_step,
                    "save_path": str(save_path) if save_path is not None else None,
                }

    # ------------------------------------------------------------------
    # Dispatch — parallel only when saving to disk
    # ------------------------------------------------------------------
    n_workers = _resolve_n_jobs(n_jobs) if save_path is not None else 1

    if n_workers <= 1:
        return [_plot_one_group(a) for a in _iter_group_args()]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_plot_one_group, _iter_group_args()))
