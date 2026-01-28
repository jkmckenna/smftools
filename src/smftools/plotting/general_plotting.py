from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
gridspec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
patches = require("matplotlib.patches", extra="plotting", purpose="plot rendering")
plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
sns = require("seaborn", extra="plotting", purpose="plot styling")

logger = get_logger(__name__)

DNA_5COLOR_PALETTE = {
    "A": "#00A000",  # green
    "C": "#0000FF",  # blue
    "G": "#FF7F00",  # orange
    "T": "#FF0000",  # red
    "OTHER": "#808080",  # gray (N, PAD, unknown)
}

if TYPE_CHECKING:
    import anndata as ad


def _fixed_tick_positions(n_positions: int, n_ticks: int) -> np.ndarray:
    """
    Return indices for ~n_ticks evenly spaced labels across [0, n_positions-1].
    Always includes 0 and n_positions-1 when possible.
    """
    n_ticks = int(max(2, n_ticks))
    if n_positions <= n_ticks:
        return np.arange(n_positions)

    # linspace gives fixed count
    pos = np.linspace(0, n_positions - 1, n_ticks)
    return np.unique(np.round(pos).astype(int))


def _select_labels(subset, sites: np.ndarray, reference: str, index_col_suffix: str | None):
    """
    Select tick labels for the heatmap axis.

    Parameters
    ----------
    subset : AnnData view
        The per-bin subset of the AnnData.
    sites : np.ndarray[int]
        Indices of the subset.var positions to annotate.
    reference : str
        Reference name (e.g., '6B6_top').
    index_col_suffix : None or str
        If None → use subset.var_names
        Else     → use subset.var[f"{reference}_{index_col_suffix}"]

    Returns
    -------
    np.ndarray[str]
        The labels to use for tick positions.
    """
    if sites.size == 0:
        return np.array([])

    # Default behavior: use var_names
    if index_col_suffix is None:
        return subset.var_names[sites].astype(str)

    # Otherwise: use a computed column adata.var[f"{reference}_{suffix}"]
    colname = f"{reference}_{index_col_suffix}"

    if colname not in subset.var:
        raise KeyError(
            f"index_col_suffix='{index_col_suffix}' requires var column '{colname}', "
            f"but it is not present in adata.var."
        )

    labels = subset.var[colname].astype(str).values
    return labels[sites]


def normalized_mean(matrix: np.ndarray, *, ignore_nan: bool = True) -> np.ndarray:
    """Compute normalized column means for a matrix.

    Args:
        matrix: Input matrix.

    Returns:
        1D array of normalized means.
    """
    mean = np.nanmean(matrix, axis=0) if ignore_nan else np.mean(matrix, axis=0)
    denom = (mean.max() - mean.min()) + 1e-9
    return (mean - mean.min()) / denom


def plot_nmf_components(
    adata: "ad.AnnData",
    *,
    output_dir: Path | str,
    components_key: str = "H_nmf",
    suffix: str | None = None,
    heatmap_name: str = "heatmap.png",
    lineplot_name: str = "lineplot.png",
    max_features: int = 2000,
) -> Dict[str, Path]:
    """Plot NMF component weights as a heatmap and per-component scatter plot.

    Args:
        adata: AnnData object containing NMF results.
        output_dir: Directory to write plots into.
        components_key: Key in ``adata.varm`` storing the H matrix.
        heatmap_name: Filename for the heatmap plot.
        lineplot_name: Filename for the scatter plot.
        max_features: Maximum number of features to plot (top-weighted by component).

    Returns:
        Dict[str, Path]: Paths to created plots (keys: ``heatmap`` and ``lineplot``).
    """
    if suffix:
        components_key = f"{components_key}_{suffix}"

    heatmap_name = f"{components_key}_{heatmap_name}"
    lineplot_name = f"{components_key}_{lineplot_name}"

    if components_key not in adata.varm:
        logger.warning("NMF components key '%s' not found in adata.varm.", components_key)
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    components = np.asarray(adata.varm[components_key])
    if components.ndim != 2:
        raise ValueError(f"NMF components must be 2D; got shape {components.shape}.")

    all_positions = np.arange(components.shape[0])
    feature_labels = all_positions.astype(str)

    nonzero_mask = np.any(components != 0, axis=1)
    if not np.any(nonzero_mask):
        logger.warning("NMF components are all zeros; skipping plot generation.")
        return {}

    components = components[nonzero_mask]
    feature_positions = all_positions[nonzero_mask]

    if max_features and components.shape[0] > max_features:
        scores = np.nanmax(components, axis=1)
        top_idx = np.argsort(scores)[-max_features:]
        top_idx = np.sort(top_idx)
        components = components[top_idx]
        feature_positions = feature_positions[top_idx]
        logger.info(
            "Downsampled NMF features from %s to %s for plotting.",
            nonzero_mask.sum(),
            components.shape[0],
        )

    n_features, n_components = components.shape
    feature_labels = feature_positions.astype(str)
    component_labels = [f"C{i + 1}" for i in range(n_components)]

    heatmap_width = max(8, min(20, n_features / 60))
    heatmap_height = max(2.5, 0.6 * n_components + 1.5)
    fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
    sns.heatmap(
        components.T,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Component weight"},
        xticklabels=feature_labels if n_features <= 60 else False,
        yticklabels=component_labels,
    )
    ax.set_xlabel("Position index")
    ax.set_ylabel("NMF component")
    if n_features > 60:
        tick_positions = _fixed_tick_positions(n_features, min(20, n_features))
        ax.set_xticks(tick_positions + 0.5)
        ax.set_xticklabels(feature_positions[tick_positions].astype(str), rotation=90, fontsize=8)
    fig.tight_layout()
    heatmap_path = output_path / heatmap_name
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, min(20, n_features / 50)), 3.5))
    x = feature_positions
    for idx, label in enumerate(component_labels):
        ax.scatter(x, components[:, idx], label=label, s=14, alpha=0.75)
    ax.set_xlabel("Position index")
    ax.set_ylabel("Component weight")
    if n_features <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    lineplot_path = output_path / lineplot_name
    fig.savefig(lineplot_path, dpi=200)
    plt.close(fig)

    return {"heatmap": heatmap_path, "lineplot": lineplot_path}


def plot_pca_components(
    adata: "ad.AnnData",
    *,
    output_dir: Path | str,
    components_key: str = "PCs",
    suffix: str | None = None,
    heatmap_name: str = "heatmap.png",
    lineplot_name: str = "lineplot.png",
    max_features: int = 2000,
) -> Dict[str, Path]:
    """Plot PCA loadings as a heatmap and per-component scatter plot.

    Args:
        adata: AnnData object containing PCA results.
        output_dir: Directory to write plots into.
        components_key: Key in ``adata.varm`` storing the PCA loadings.
        heatmap_name: Filename for the heatmap plot.
        lineplot_name: Filename for the scatter plot.
        max_features: Maximum number of features to plot (top-weighted by component).

    Returns:
        Dict[str, Path]: Paths to created plots (keys: ``heatmap`` and ``lineplot``).
    """
    if suffix:
        components_key = f"{components_key}_{suffix}"

    heatmap_name = f"{components_key}_{heatmap_name}"
    lineplot_name = f"{components_key}_{lineplot_name}"

    if components_key not in adata.varm:
        logger.warning("PCA components key '%s' not found in adata.varm.", components_key)
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    components = np.asarray(adata.varm[components_key])
    if components.ndim != 2:
        raise ValueError(f"PCA components must be 2D; got shape {components.shape}.")

    all_positions = np.arange(components.shape[0])
    feature_labels = all_positions.astype(str)

    nonzero_mask = np.any(components != 0, axis=1)
    if not np.any(nonzero_mask):
        logger.warning("PCA components are all zeros; skipping plot generation.")
        return {}

    components = components[nonzero_mask]
    feature_positions = all_positions[nonzero_mask]

    if max_features and components.shape[0] > max_features:
        scores = np.nanmax(np.abs(components), axis=1)
        top_idx = np.argsort(scores)[-max_features:]
        top_idx = np.sort(top_idx)
        components = components[top_idx]
        feature_positions = feature_positions[top_idx]
        logger.info(
            "Downsampled PCA features from %s to %s for plotting.",
            nonzero_mask.sum(),
            components.shape[0],
        )

    n_features, n_components = components.shape
    feature_labels = feature_positions.astype(str)
    component_labels = [f"PC{i + 1}" for i in range(n_components)]

    heatmap_width = max(8, min(20, n_features / 60))
    heatmap_height = max(2.5, 0.6 * n_components + 1.5)
    fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
    sns.heatmap(
        components.T,
        ax=ax,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Loading"},
        xticklabels=feature_labels if n_features <= 60 else False,
        yticklabels=component_labels,
    )
    ax.set_xlabel("Position index")
    ax.set_ylabel("PCA component")
    if n_features > 60:
        tick_positions = _fixed_tick_positions(n_features, min(20, n_features))
        ax.set_xticks(tick_positions + 0.5)
        ax.set_xticklabels(feature_positions[tick_positions].astype(str), rotation=90, fontsize=8)
    fig.tight_layout()
    heatmap_path = output_path / heatmap_name
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, min(20, n_features / 50)), 3.5))
    x = feature_positions
    for idx, label in enumerate(component_labels):
        ax.scatter(x, components[:, idx], label=label, s=14, alpha=0.75)
    ax.set_xlabel("Position index")
    ax.set_ylabel("Loading")
    if n_features <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    lineplot_path = output_path / lineplot_name
    fig.savefig(lineplot_path, dpi=200)
    plt.close(fig)

    return {"heatmap": heatmap_path, "lineplot": lineplot_path}


def plot_cp_sequence_components(
    adata: "ad.AnnData",
    *,
    output_dir: Path | str,
    components_key: str = "H_cp_sequence",
    uns_key: str = "cp_sequence",
    heatmap_name: str = "cp_sequence_position_heatmap.png",
    lineplot_name: str = "cp_sequence_position_lineplot.png",
    base_name: str = "cp_sequence_base_weights.png",
    max_positions: int = 2000,
) -> Dict[str, Path]:
    """Plot CP decomposition position and base factors.

    Args:
        adata: AnnData object containing CP decomposition results.
        output_dir: Directory to write plots into.
        components_key: Key in ``adata.varm`` storing position factors.
        uns_key: Key in ``adata.uns`` storing base factors.
        heatmap_name: Filename for position heatmap.
        lineplot_name: Filename for position line plot.
        base_name: Filename for base factor bar plot.
        max_positions: Maximum number of positions to plot.

    Returns:
        Dict[str, Path]: Paths to created plots.
    """
    if components_key not in adata.varm:
        logger.warning("CP components key '%s' not found in adata.varm.", components_key)
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    components = np.asarray(adata.varm[components_key])
    if components.ndim != 2:
        raise ValueError(f"CP position factors must be 2D; got shape {components.shape}.")

    position_indices = np.arange(components.shape[0])
    valid_mask = np.isfinite(components).any(axis=1)
    if not np.all(valid_mask):
        dropped = int(np.sum(~valid_mask))
        logger.info(
            "Dropping %s CP positions with no finite weights before plotting.",
            dropped,
        )
        components = components[valid_mask]
        position_indices = position_indices[valid_mask]

    if max_positions and components.shape[0] > max_positions:
        original_count = components.shape[0]
        scores = np.nanmax(np.abs(components), axis=1)
        top_idx = np.argsort(scores)[-max_positions:]
        top_idx = np.sort(top_idx)
        components = components[top_idx]
        position_indices = position_indices[top_idx]
        logger.info(
            "Downsampled CP positions from %s to %s for plotting.",
            original_count,
            max_positions,
        )

    outputs = {}
    if components.size == 0:
        logger.warning("No finite CP position factors available; skipping position plots.")
    else:
        n_positions, n_components = components.shape
        component_labels = [f"C{i + 1}" for i in range(n_components)]

        heatmap_width = max(8, min(20, n_positions / 60))
        heatmap_height = max(2.5, 0.6 * n_components + 1.5)
        fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
        sns.heatmap(
            components.T,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Component weight"},
            xticklabels=position_indices if n_positions <= 60 else False,
            yticklabels=component_labels,
        )
        ax.set_xlabel("Position index")
        ax.set_ylabel("CP component")
        fig.tight_layout()
        heatmap_path = output_path / heatmap_name
        fig.savefig(heatmap_path, dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, min(20, n_positions / 50)), 3.5))
        x = position_indices
        for idx, label in enumerate(component_labels):
            ax.scatter(x, components[:, idx], label=label, s=20, alpha=0.8)
        ax.set_xlabel("Position index")
        ax.set_ylabel("Component weight")
        if n_positions <= 60:
            ax.set_xticks(x)
            ax.set_xticklabels([str(pos) for pos in x], rotation=90, fontsize=8)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        lineplot_path = output_path / lineplot_name
        fig.savefig(lineplot_path, dpi=200)
        plt.close(fig)

        outputs["heatmap"] = heatmap_path
        outputs["lineplot"] = lineplot_path
    if uns_key in adata.uns:
        base_factors = adata.uns[uns_key].get("base_factors")
        base_labels = adata.uns[uns_key].get("base_labels")
        if base_factors is not None:
            base_factors = np.asarray(base_factors)
            if base_factors.ndim != 2 or base_factors.size == 0:
                logger.warning(
                    "CP base factors must be 2D and non-empty; got shape %s.",
                    base_factors.shape,
                )
            else:
                base_labels = base_labels or [f"B{i + 1}" for i in range(base_factors.shape[0])]
                fig, ax = plt.subplots(figsize=(4.5, 3))
                width = 0.8 / base_factors.shape[1]
                x = np.arange(base_factors.shape[0])
                for idx in range(base_factors.shape[1]):
                    ax.bar(
                        x + idx * width,
                        base_factors[:, idx],
                        width=width,
                        label=f"C{idx + 1}",
                    )
                ax.set_xticks(x + width * (base_factors.shape[1] - 1) / 2)
                ax.set_xticklabels(base_labels)
                ax.set_ylabel("Base factor weight")
                ax.legend(loc="upper right", frameon=False)
                fig.tight_layout()
                base_path = output_path / base_name
                fig.savefig(base_path, dpi=200)
                plt.close(fig)
                outputs["base_factors"] = base_path

    return outputs


def _resolve_feature_color(cmap: Any) -> Tuple[float, float, float, float]:
    """Resolve a representative feature color from a colormap or color spec."""
    if isinstance(cmap, str):
        try:
            cmap_obj = plt.get_cmap(cmap)
            return colors.to_rgba(cmap_obj(1.0))
        except Exception:
            return colors.to_rgba(cmap)

    if isinstance(cmap, colors.Colormap):
        if hasattr(cmap, "colors") and cmap.colors:
            return colors.to_rgba(cmap.colors[-1])
        return colors.to_rgba(cmap(1.0))

    return colors.to_rgba("black")


def _build_hmm_feature_cmap(
    cmap: Any,
    *,
    zero_color: str = "#f5f1e8",
    nan_color: str = "#E6E6E6",
) -> colors.Colormap:
    """Build a two-color HMM colormap with explicit NaN/under handling."""
    feature_color = _resolve_feature_color(cmap)
    hmm_cmap = colors.LinearSegmentedColormap.from_list(
        "hmm_feature_cmap",
        [zero_color, feature_color],
    )
    hmm_cmap.set_bad(nan_color)
    hmm_cmap.set_under(nan_color)
    return hmm_cmap


def _map_length_matrix_to_subclasses(
    length_matrix: np.ndarray,
    feature_ranges: Sequence[Tuple[int, int, Any]],
) -> np.ndarray:
    """Map length values into subclass integer codes based on feature ranges."""
    mapped = np.zeros_like(length_matrix, dtype=float)
    finite_mask = np.isfinite(length_matrix)
    for idx, (min_len, max_len, _color) in enumerate(feature_ranges, start=1):
        mask = finite_mask & (length_matrix >= min_len) & (length_matrix <= max_len)
        mapped[mask] = float(idx)
    mapped[~finite_mask] = np.nan
    return mapped


def _build_length_feature_cmap(
    feature_ranges: Sequence[Tuple[int, int, Any]],
    *,
    zero_color: str = "#f5f1e8",
    nan_color: str = "#E6E6E6",
) -> Tuple[colors.Colormap, colors.BoundaryNorm]:
    """Build a discrete colormap and norm for length-based subclasses."""
    color_list = [zero_color] + [color for _, _, color in feature_ranges]
    cmap = colors.ListedColormap(color_list, name="hmm_length_feature_cmap")
    cmap.set_bad(nan_color)
    bounds = np.arange(-0.5, len(color_list) + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _layer_to_numpy(
    subset,
    layer_name: str,
    sites: np.ndarray | None = None,
    *,
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
) -> np.ndarray:
    """Return a (copied) numpy array for a layer with optional NaN filling."""
    if sites is not None:
        layer_data = subset[:, sites].layers[layer_name]
    else:
        layer_data = subset.layers[layer_name]

    if hasattr(layer_data, "toarray"):
        arr = layer_data.toarray()
    else:
        arr = np.asarray(layer_data)

    arr = np.array(arr, copy=True)

    if fill_nan_strategy == "none":
        return arr

    if fill_nan_strategy not in {"value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")

    arr = arr.astype(float, copy=False)

    if fill_nan_strategy == "value":
        return np.where(np.isnan(arr), fill_nan_value, arr)

    col_mean = np.nanmean(arr, axis=0)
    if np.any(np.isnan(col_mean)):
        col_mean = np.where(np.isnan(col_mean), fill_nan_value, col_mean)
    return np.where(np.isnan(arr), col_mean, arr)


def _infer_zero_is_valid(layer_name: str | None, matrix: np.ndarray) -> bool:
    """Infer whether zeros should count as valid (unmethylated) values."""
    if layer_name and "nan0_0minus1" in layer_name:
        return False
    if np.isnan(matrix).any():
        return True
    if np.any(matrix < 0):
        return False
    return True


def methylation_fraction(
    matrix: np.ndarray, *, ignore_nan: bool = True, zero_is_valid: bool = False
) -> np.ndarray:
    """
    Fraction methylated per column.
    Methylated = 1
    Valid = finite AND not 0 (unless zero_is_valid=True)
    """
    matrix = np.asarray(matrix)
    if not ignore_nan:
        matrix = np.where(np.isnan(matrix), 0, matrix)
    finite_mask = np.isfinite(matrix)
    valid_mask = finite_mask if zero_is_valid else (finite_mask & (matrix != 0))
    methyl_mask = (matrix == 1) & np.isfinite(matrix)

    methylated = methyl_mask.sum(axis=0)
    valid = valid_mask.sum(axis=0)

    return np.divide(
        methylated, valid, out=np.zeros_like(methylated, dtype=float), where=valid != 0
    )


def _methylation_fraction_for_layer(
    matrix: np.ndarray,
    layer_name: str | None,
    *,
    ignore_nan: bool = True,
    zero_is_valid: bool | None = None,
) -> np.ndarray:
    """Compute methylation fractions with layer-aware zero handling."""
    matrix = np.asarray(matrix)
    if zero_is_valid is None:
        zero_is_valid = _infer_zero_is_valid(layer_name, matrix)
    return methylation_fraction(matrix, ignore_nan=ignore_nan, zero_is_valid=zero_is_valid)


def clean_barplot(
    ax,
    mean_values,
    title,
    *,
    y_max: float | None = 1.0,
    y_label: str = "Mean",
    y_ticks: list[float] | None = None,
):
    """Format a barplot with consistent axes and labels.

    Args:
        ax: Matplotlib axes.
        mean_values: Values to plot.
        title: Plot title.
        y_max: Optional y-axis max; inferred from data if not provided.
        y_label: Y-axis label.
        y_ticks: Optional y-axis ticks.
    """
    x = np.arange(len(mean_values))
    ax.bar(x, mean_values, color="gray", width=1.0, align="edge")
    ax.set_xlim(0, len(mean_values))
    if y_ticks is None and y_max == 1.0:
        y_ticks = [0.0, 0.5, 1.0]
    if y_max is None:
        y_max = np.nanmax(mean_values) if len(mean_values) else 1.0
        if not np.isfinite(y_max) or y_max <= 0:
            y_max = 1.0
        y_max *= 1.05
    ax.set_ylim(0, y_max)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, pad=2)

    # Hide all spines except left
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "left")

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)


def combined_hmm_raw_clustermap(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    hmm_feature_layer: str = "hmm_combined",
    layer_gpc: str = "nan0_0minus1",
    layer_cpg: str = "nan0_0minus1",
    layer_c: str = "nan0_0minus1",
    layer_a: str = "nan0_0minus1",
    cmap_hmm: str = "tab10",
    cmap_gpc: str = "coolwarm",
    cmap_cpg: str = "viridis",
    cmap_c: str = "coolwarm",
    cmap_a: str = "coolwarm",
    min_quality: int = 20,
    min_length: int = 200,
    min_mapped_length_to_reference_length_ratio: float = 0.8,
    min_position_valid_fraction: float = 0.5,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sample_mapping: Optional[Mapping[str, str]] = None,
    save_path: str | Path | None = None,
    normalize_hmm: bool = False,
    sort_by: str = "gpc",
    bins: Optional[Dict[str, Any]] = None,
    deaminase: bool = False,
    min_signal: float = 0.0,
    # ---- fixed tick label controls (counts, not spacing)
    n_xticks_hmm: int = 10,
    n_xticks_any_c: int = 8,
    n_xticks_gpc: int = 8,
    n_xticks_cpg: int = 8,
    n_xticks_a: int = 8,
    index_col_suffix: str | None = None,
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
):
    """
    Makes a multi-panel clustermap per (sample, reference):
      HMM panel (always) + optional raw panels for C, GpC, CpG, and A sites.

    Panels are added only if the corresponding site mask exists AND has >0 sites.

    sort_by options:
      'gpc', 'cpg', 'c', 'a', 'gpc_cpg', 'none', 'hmm', or 'obs:<col>'

    NaN fill strategy is applied in-memory for clustering/plotting only.
    """
    if fill_nan_strategy not in {"none", "value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")

    def pick_xticks(labels: np.ndarray, n_ticks: int):
        """Pick tick indices/labels from an array."""
        if labels.size == 0:
            return [], []
        idx = np.linspace(0, len(labels) - 1, n_ticks).round().astype(int)
        idx = np.unique(idx)
        return idx.tolist(), labels[idx].tolist()

    # Helper: build a True mask if filter is inactive or column missing
    def _mask_or_true(series_name: str, predicate):
        """Return a mask from predicate or an all-True mask."""
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            # Fallback: all True if bad dtype / predicate failure
            return pd.Series(True, index=adata.obs.index)

    results = []
    signal_type = "deamination" if deaminase else "methylation"

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            # Optionally remap sample label for display
            display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample
            # Row-level masks (obs)
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

            ref_mask = adata.obs[reference_col] == ref
            sample_mask = adata.obs[sample_col] == sample

            row_mask = ref_mask & sample_mask & qmask & lm_mask & lrr_mask & demux_mask

            if not bool(row_mask.any()):
                print(
                    f"No reads for {display_sample} - {ref} after read quality and length filtering"
                )
                continue

            try:
                # ---- subset reads ----
                subset = adata[row_mask, :].copy()

                # Column-level mask (var)
                if min_position_valid_fraction is not None:
                    valid_key = f"{ref}_valid_fraction"
                    if valid_key in subset.var:
                        v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                        col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                        if col_mask.any():
                            subset = subset[:, col_mask].copy()
                        else:
                            print(
                                f"No positions left after valid_fraction filter for {display_sample} - {ref}"
                            )
                            continue

                if subset.shape[0] == 0:
                    print(f"No reads left after filtering for {display_sample} - {ref}")
                    continue

                # ---- bins ----
                if bins is None:
                    bins_temp = {"All": np.ones(subset.n_obs, dtype=bool)}
                else:
                    bins_temp = bins

                # ---- site masks (robust) ----
                def _sites(*keys):
                    """Return indices for the first matching site key."""
                    for k in keys:
                        if k in subset.var:
                            return np.where(subset.var[k].values)[0]
                    return np.array([], dtype=int)

                gpc_sites = _sites(f"{ref}_GpC_site")
                cpg_sites = _sites(f"{ref}_CpG_site")
                any_c_sites = _sites(f"{ref}_any_C_site", f"{ref}_C_site")
                any_a_sites = _sites(f"{ref}_A_site", f"{ref}_any_A_site")

                # ---- labels via _select_labels ----
                # HMM uses *all* columns
                hmm_sites = np.arange(subset.n_vars, dtype=int)
                hmm_labels = _select_labels(subset, hmm_sites, ref, index_col_suffix)
                gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)
                any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                # storage
                stacked_hmm = []
                stacked_hmm_raw = []
                stacked_any_c = []
                stacked_any_c_raw = []
                stacked_gpc = []
                stacked_gpc_raw = []
                stacked_cpg = []
                stacked_cpg_raw = []
                stacked_any_a = []
                stacked_any_a_raw = []

                row_labels, bin_labels, bin_boundaries = [], [], []
                total_reads = subset.n_obs
                percentages = {}
                last_idx = 0

                # ---------------- process bins ----------------
                for bin_label, bin_filter in bins_temp.items():
                    sb = subset[bin_filter].copy()
                    n = sb.n_obs
                    if n == 0:
                        continue

                    pct = (n / total_reads) * 100 if total_reads else 0
                    percentages[bin_label] = pct

                    # ---- sorting ----
                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        order = np.argsort(sb.obs[colname].values)

                    elif sort_by == "gpc" and gpc_sites.size:
                        gpc_matrix = _layer_to_numpy(
                            sb,
                            layer_gpc,
                            gpc_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "cpg" and cpg_sites.size:
                        cpg_matrix = _layer_to_numpy(
                            sb,
                            layer_cpg,
                            cpg_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(cpg_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "c" and any_c_sites.size:
                        any_c_matrix = _layer_to_numpy(
                            sb,
                            layer_c,
                            any_c_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_c_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "a" and any_a_sites.size:
                        any_a_matrix = _layer_to_numpy(
                            sb,
                            layer_a,
                            any_a_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_a_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "gpc_cpg" and gpc_sites.size and cpg_sites.size:
                        gpc_matrix = _layer_to_numpy(
                            sb,
                            layer_gpc,
                            None,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "hmm" and hmm_sites.size:
                        hmm_matrix = _layer_to_numpy(
                            sb,
                            hmm_feature_layer,
                            hmm_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(hmm_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    else:
                        order = np.arange(n)

                    sb = sb[order]

                    # ---- collect matrices ----
                    stacked_hmm.append(
                        _layer_to_numpy(
                            sb,
                            hmm_feature_layer,
                            None,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                    )
                    stacked_hmm_raw.append(
                        _layer_to_numpy(
                            sb,
                            hmm_feature_layer,
                            None,
                            fill_nan_strategy="none",
                            fill_nan_value=fill_nan_value,
                        )
                    )
                    if any_c_sites.size:
                        stacked_any_c.append(
                            _layer_to_numpy(
                                sb,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_c_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if gpc_sites.size:
                        stacked_gpc.append(
                            _layer_to_numpy(
                                sb,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_gpc_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if cpg_sites.size:
                        stacked_cpg.append(
                            _layer_to_numpy(
                                sb,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_cpg_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if any_a_sites.size:
                        stacked_any_a.append(
                            _layer_to_numpy(
                                sb,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_a_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )

                    row_labels.extend([bin_label] * n)
                    bin_labels.append(f"{bin_label}: {n} reads ({pct:.1f}%)")
                    last_idx += n
                    bin_boundaries.append(last_idx)

                # ---------------- stack ----------------
                hmm_matrix = np.vstack(stacked_hmm)
                hmm_matrix_raw = np.vstack(stacked_hmm_raw)
                mean_hmm = (
                    normalized_mean(hmm_matrix_raw)
                    if normalize_hmm
                    else np.nanmean(hmm_matrix_raw, axis=0)
                )
                hmm_plot_matrix = hmm_matrix_raw
                hmm_plot_cmap = _build_hmm_feature_cmap(cmap_hmm)

                panels = [
                    (
                        f"HMM - {hmm_feature_layer}",
                        hmm_plot_matrix,
                        hmm_labels,
                        hmm_plot_cmap,
                        mean_hmm,
                        n_xticks_hmm,
                    ),
                ]

                if stacked_any_c:
                    m = np.vstack(stacked_any_c)
                    m_raw = np.vstack(stacked_any_c_raw)
                    panels.append(
                        (
                            "C",
                            m,
                            any_c_labels,
                            cmap_c,
                            _methylation_fraction_for_layer(m_raw, layer_c),
                            n_xticks_any_c,
                        )
                    )

                if stacked_gpc:
                    m = np.vstack(stacked_gpc)
                    m_raw = np.vstack(stacked_gpc_raw)
                    panels.append(
                        (
                            "GpC",
                            m,
                            gpc_labels,
                            cmap_gpc,
                            _methylation_fraction_for_layer(m_raw, layer_gpc),
                            n_xticks_gpc,
                        )
                    )

                if stacked_cpg:
                    m = np.vstack(stacked_cpg)
                    m_raw = np.vstack(stacked_cpg_raw)
                    panels.append(
                        (
                            "CpG",
                            m,
                            cpg_labels,
                            cmap_cpg,
                            _methylation_fraction_for_layer(m_raw, layer_cpg),
                            n_xticks_cpg,
                        )
                    )

                if stacked_any_a:
                    m = np.vstack(stacked_any_a)
                    m_raw = np.vstack(stacked_any_a_raw)
                    panels.append(
                        (
                            "A",
                            m,
                            any_a_labels,
                            cmap_a,
                            _methylation_fraction_for_layer(m_raw, layer_a),
                            n_xticks_a,
                        )
                    )

                # ---------------- plotting ----------------
                n_panels = len(panels)
                fig = plt.figure(figsize=(4.5 * n_panels, 10))
                gs = gridspec.GridSpec(2, n_panels, height_ratios=[1, 6], hspace=0.01)
                fig.suptitle(
                    f"{sample} — {ref} — {total_reads} reads ({signal_type})", fontsize=14, y=0.98
                )

                axes_heat = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]
                axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(n_panels)]

                for i, (name, matrix, labels, cmap, mean_vec, n_ticks) in enumerate(panels):
                    # ---- your clean barplot ----
                    clean_barplot(axes_bar[i], mean_vec, name)

                    # ---- heatmap ----
                    heatmap_kwargs = dict(
                        cmap=cmap,
                        ax=axes_heat[i],
                        yticklabels=False,
                        cbar=False,
                    )
                    if name.startswith("HMM -"):
                        heatmap_kwargs.update(vmin=0.0, vmax=1.0)
                    sns.heatmap(matrix, **heatmap_kwargs)

                    # ---- xticks ----
                    xtick_pos, xtick_labels = pick_xticks(np.asarray(labels), n_ticks)
                    axes_heat[i].set_xticks(xtick_pos)
                    axes_heat[i].set_xticklabels(xtick_labels, rotation=90, fontsize=8)

                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=1.2)

                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    safe_name = f"{ref}__{sample}".replace("/", "_")
                    out_file = save_path / f"{safe_name}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close(fig)
                else:
                    plt.show()

            except Exception:
                import traceback

                traceback.print_exc()
                continue


def combined_raw_clustermap(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    mod_target_bases: Sequence[str] = ("GpC", "CpG"),
    layer_c: str = "nan0_0minus1",
    layer_gpc: str = "nan0_0minus1",
    layer_cpg: str = "nan0_0minus1",
    layer_a: str = "nan0_0minus1",
    cmap_c: str = "coolwarm",
    cmap_gpc: str = "coolwarm",
    cmap_cpg: str = "viridis",
    cmap_a: str = "coolwarm",
    min_quality: float | None = 20,
    min_length: int | None = 200,
    min_mapped_length_to_reference_length_ratio: float | None = 0,
    min_position_valid_fraction: float | None = 0,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sample_mapping: Optional[Mapping[str, str]] = None,
    save_path: str | Path | None = None,
    sort_by: str = "gpc",  # 'gpc','cpg','c','gpc_cpg','a','none','obs:<col>'
    bins: Optional[Dict[str, Any]] = None,
    deaminase: bool = False,
    min_signal: float = 0,
    n_xticks_any_c: int = 10,
    n_xticks_gpc: int = 10,
    n_xticks_cpg: int = 10,
    n_xticks_any_a: int = 10,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    index_col_suffix: str | None = None,
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
):
    """
    Plot stacked heatmaps + per-position mean barplots for C, GpC, CpG, and optional A.

    Key fixes vs old version:
      - order computed ONCE per bin, applied to all matrices
      - no hard-coded axes indices
      - NaNs excluded from methylation denominators
      - var_names not forced to int
      - fixed count of x tick labels per block (controllable)
      - optional NaN fill strategy for clustering/plotting (in-memory only)
      - adata.uns updated once at end

    Returns
    -------
    results : list[dict]
        One entry per (sample, ref) plot with matrices + bin metadata.
    """
    if fill_nan_strategy not in {"none", "value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")

    # Helper: build a True mask if filter is inactive or column missing
    def _mask_or_true(series_name: str, predicate):
        """Return a mask from predicate or an all-True mask."""
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            # Fallback: all True if bad dtype / predicate failure
            return pd.Series(True, index=adata.obs.index)

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    # Ensure categorical
    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype("category")

    base_set = set(mod_target_bases)
    include_any_c = any(b in {"C", "CpG", "GpC"} for b in base_set)
    include_any_a = "A" in base_set

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            # Optionally remap sample label for display
            display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample

            # Row-level masks (obs)
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

            ref_mask = adata.obs[reference_col] == ref
            sample_mask = adata.obs[sample_col] == sample

            row_mask = ref_mask & sample_mask & qmask & lm_mask & lrr_mask & demux_mask

            if not bool(row_mask.any()):
                print(
                    f"No reads for {display_sample} - {ref} after read quality and length filtering"
                )
                continue

            try:
                subset = adata[row_mask, :].copy()

                # Column-level mask (var)
                if min_position_valid_fraction is not None:
                    valid_key = f"{ref}_valid_fraction"
                    if valid_key in subset.var:
                        v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                        col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                        if col_mask.any():
                            subset = subset[:, col_mask].copy()
                        else:
                            print(
                                f"No positions left after valid_fraction filter for {display_sample} - {ref}"
                            )
                            continue

                if subset.shape[0] == 0:
                    print(f"No reads left after filtering for {display_sample} - {ref}")
                    continue

                # bins mode
                if bins is None:
                    bins_temp = {"All": (subset.obs[reference_col] == ref)}
                else:
                    bins_temp = bins

                # find sites (positions)
                any_c_sites = gpc_sites = cpg_sites = np.array([], dtype=int)
                any_a_sites = np.array([], dtype=int)

                num_any_c = num_gpc = num_cpg = num_any_a = 0

                if include_any_c:
                    any_c_sites = np.where(subset.var.get(f"{ref}_C_site", False).values)[0]
                    gpc_sites = np.where(subset.var.get(f"{ref}_GpC_site", False).values)[0]
                    cpg_sites = np.where(subset.var.get(f"{ref}_CpG_site", False).values)[0]

                    num_any_c, num_gpc, num_cpg = len(any_c_sites), len(gpc_sites), len(cpg_sites)

                    any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                    gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                    cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)

                if include_any_a:
                    any_a_sites = np.where(subset.var.get(f"{ref}_A_site", False).values)[0]
                    num_any_a = len(any_a_sites)
                    any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                stacked_any_c, stacked_gpc, stacked_cpg, stacked_any_a = [], [], [], []
                stacked_any_c_raw, stacked_gpc_raw, stacked_cpg_raw, stacked_any_a_raw = (
                    [],
                    [],
                    [],
                    [],
                )
                row_labels, bin_labels, bin_boundaries = [], [], []
                percentages = {}
                last_idx = 0
                total_reads = subset.shape[0]

                # ----------------------------
                # per-bin stacking
                # ----------------------------
                for bin_label, bin_filter in bins_temp.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    if num_reads == 0:
                        percentages[bin_label] = 0.0
                        continue

                    percent_reads = (num_reads / total_reads) * 100
                    percentages[bin_label] = percent_reads

                    # compute order ONCE
                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        order = np.argsort(subset_bin.obs[colname].values)

                    elif sort_by == "gpc" and num_gpc > 0:
                        gpc_matrix = _layer_to_numpy(
                            subset_bin,
                            layer_gpc,
                            gpc_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "cpg" and num_cpg > 0:
                        cpg_matrix = _layer_to_numpy(
                            subset_bin,
                            layer_cpg,
                            cpg_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(cpg_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "c" and num_any_c > 0:
                        any_c_matrix = _layer_to_numpy(
                            subset_bin,
                            layer_c,
                            any_c_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_c_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "gpc_cpg":
                        gpc_matrix = _layer_to_numpy(
                            subset_bin,
                            layer_gpc,
                            None,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "a" and num_any_a > 0:
                        any_a_matrix = _layer_to_numpy(
                            subset_bin,
                            layer_a,
                            any_a_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_a_matrix, method="ward")
                        order = sch.leaves_list(linkage)

                    elif sort_by == "none":
                        order = np.arange(num_reads)

                    else:
                        order = np.arange(num_reads)

                    subset_bin = subset_bin[order]

                    # stack consistently
                    if include_any_c and num_any_c > 0:
                        stacked_any_c.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_c_raw.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if include_any_c and num_gpc > 0:
                        stacked_gpc.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_gpc_raw.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if include_any_c and num_cpg > 0:
                        stacked_cpg.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_cpg_raw.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if include_any_a and num_any_a > 0:
                        stacked_any_a.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_a_raw.append(
                            _layer_to_numpy(
                                subset_bin,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )

                    row_labels.extend([bin_label] * num_reads)
                    bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
                    last_idx += num_reads
                    bin_boundaries.append(last_idx)

                # ----------------------------
                # build matrices + means
                # ----------------------------
                blocks = []  # list of dicts describing what to plot in order

                if include_any_c and stacked_any_c:
                    any_c_matrix = np.vstack(stacked_any_c)
                    any_c_matrix_raw = np.vstack(stacked_any_c_raw)
                    gpc_matrix = np.vstack(stacked_gpc) if stacked_gpc else np.empty((0, 0))
                    gpc_matrix_raw = (
                        np.vstack(stacked_gpc_raw) if stacked_gpc_raw else np.empty((0, 0))
                    )
                    cpg_matrix = np.vstack(stacked_cpg) if stacked_cpg else np.empty((0, 0))
                    cpg_matrix_raw = (
                        np.vstack(stacked_cpg_raw) if stacked_cpg_raw else np.empty((0, 0))
                    )

                    mean_any_c = (
                        _methylation_fraction_for_layer(any_c_matrix_raw, layer_c)
                        if any_c_matrix_raw.size
                        else None
                    )
                    mean_gpc = (
                        _methylation_fraction_for_layer(gpc_matrix_raw, layer_gpc)
                        if gpc_matrix_raw.size
                        else None
                    )
                    mean_cpg = (
                        _methylation_fraction_for_layer(cpg_matrix_raw, layer_cpg)
                        if cpg_matrix_raw.size
                        else None
                    )

                    if any_c_matrix.size:
                        blocks.append(
                            dict(
                                name="c",
                                matrix=any_c_matrix,
                                mean=mean_any_c,
                                labels=any_c_labels,
                                cmap=cmap_c,
                                n_xticks=n_xticks_any_c,
                                title="any C site Modification Signal",
                            )
                        )
                    if gpc_matrix.size:
                        blocks.append(
                            dict(
                                name="gpc",
                                matrix=gpc_matrix,
                                mean=mean_gpc,
                                labels=gpc_labels,
                                cmap=cmap_gpc,
                                n_xticks=n_xticks_gpc,
                                title="GpC Modification Signal",
                            )
                        )
                    if cpg_matrix.size:
                        blocks.append(
                            dict(
                                name="cpg",
                                matrix=cpg_matrix,
                                mean=mean_cpg,
                                labels=cpg_labels,
                                cmap=cmap_cpg,
                                n_xticks=n_xticks_cpg,
                                title="CpG Modification Signal",
                            )
                        )

                if include_any_a and stacked_any_a:
                    any_a_matrix = np.vstack(stacked_any_a)
                    any_a_matrix_raw = np.vstack(stacked_any_a_raw)
                    mean_any_a = (
                        _methylation_fraction_for_layer(any_a_matrix_raw, layer_a)
                        if any_a_matrix_raw.size
                        else None
                    )
                    if any_a_matrix.size:
                        blocks.append(
                            dict(
                                name="a",
                                matrix=any_a_matrix,
                                mean=mean_any_a,
                                labels=any_a_labels,
                                cmap=cmap_a,
                                n_xticks=n_xticks_any_a,
                                title="any A site Modification Signal",
                            )
                        )

                if not blocks:
                    print(f"No matrices to plot for {display_sample} - {ref}")
                    continue

                gs_dim = len(blocks)
                fig = plt.figure(figsize=(5.5 * gs_dim, 11))
                gs = gridspec.GridSpec(2, gs_dim, height_ratios=[1, 6], hspace=0.02)
                fig.suptitle(f"{display_sample} - {ref} - {total_reads} reads", fontsize=14, y=0.97)

                axes_heat = [fig.add_subplot(gs[1, i]) for i in range(gs_dim)]
                axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(gs_dim)]

                # ----------------------------
                # plot blocks
                # ----------------------------
                for i, blk in enumerate(blocks):
                    mat = blk["matrix"]
                    mean = blk["mean"]
                    labels = np.asarray(blk["labels"], dtype=str)
                    n_xticks = blk["n_xticks"]

                    # barplot
                    clean_barplot(axes_bar[i], mean, blk["title"])

                    # heatmap
                    sns.heatmap(
                        mat, cmap=blk["cmap"], ax=axes_heat[i], yticklabels=False, cbar=False
                    )

                    # fixed tick labels
                    tick_pos = _fixed_tick_positions(len(labels), n_xticks)
                    axes_heat[i].set_xticks(tick_pos)
                    axes_heat[i].set_xticklabels(
                        labels[tick_pos], rotation=xtick_rotation, fontsize=xtick_fontsize
                    )

                    # bin separators
                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=2)

                    axes_heat[i].set_xlabel("Position", fontsize=9)

                plt.tight_layout()

                # save or show
                if save_path is not None:
                    safe_name = (
                        f"{ref}__{display_sample}".replace("=", "")
                        .replace("__", "_")
                        .replace(",", "_")
                        .replace(" ", "_")
                    )
                    out_file = save_path / f"{safe_name}.png"
                    fig.savefig(out_file, dpi=300)
                    plt.close(fig)
                    print(f"Saved: {out_file}")
                else:
                    plt.show()

                # record results
                rec = {
                    "sample": str(sample),
                    "ref": str(ref),
                    "row_labels": row_labels,
                    "bin_labels": bin_labels,
                    "bin_boundaries": bin_boundaries,
                    "percentages": percentages,
                }
                for blk in blocks:
                    rec[f"{blk['name']}_matrix"] = blk["matrix"]
                    rec[f"{blk['name']}_labels"] = list(map(str, blk["labels"]))
                results.append(rec)

                print(f"Summary for {display_sample} - {ref}:")
                for bin_label, percent in percentages.items():
                    print(f"  - {bin_label}: {percent:.1f}%")

            except Exception:
                import traceback

                traceback.print_exc()
                continue

    return results


def combined_hmm_length_clustermap(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    length_layer: str = "hmm_combined_lengths",
    layer_gpc: str = "nan0_0minus1",
    layer_cpg: str = "nan0_0minus1",
    layer_c: str = "nan0_0minus1",
    layer_a: str = "nan0_0minus1",
    cmap_lengths: Any = "Greens",
    cmap_gpc: str = "coolwarm",
    cmap_cpg: str = "viridis",
    cmap_c: str = "coolwarm",
    cmap_a: str = "coolwarm",
    min_quality: int = 20,
    min_length: int = 200,
    min_mapped_length_to_reference_length_ratio: float = 0.8,
    min_position_valid_fraction: float = 0.5,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sample_mapping: Optional[Mapping[str, str]] = None,
    save_path: str | Path | None = None,
    sort_by: str = "gpc",
    bins: Optional[Dict[str, Any]] = None,
    deaminase: bool = False,
    min_signal: float = 0.0,
    n_xticks_lengths: int = 10,
    n_xticks_any_c: int = 8,
    n_xticks_gpc: int = 8,
    n_xticks_cpg: int = 8,
    n_xticks_a: int = 8,
    index_col_suffix: str | None = None,
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
    length_feature_ranges: Optional[Sequence[Tuple[int, int, Any]]] = None,
):
    """
    Plot clustermaps for length-encoded HMM feature layers with optional subclass colors.

    Length-based feature ranges map integer lengths into subclass colors for accessible
    and footprint layers. Raw methylation panels are included when available.
    """
    if fill_nan_strategy not in {"none", "value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")

    def pick_xticks(labels: np.ndarray, n_ticks: int):
        """Pick tick indices/labels from an array."""
        if labels.size == 0:
            return [], []
        idx = np.linspace(0, len(labels) - 1, n_ticks).round().astype(int)
        idx = np.unique(idx)
        return idx.tolist(), labels[idx].tolist()

    def _mask_or_true(series_name: str, predicate):
        """Return a mask from predicate or an all-True mask."""
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    results = []
    signal_type = "deamination" if deaminase else "methylation"
    feature_ranges = tuple(length_feature_ranges or ())

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            display_sample = sample_mapping.get(sample, sample) if sample_mapping else sample
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

            ref_mask = adata.obs[reference_col] == ref
            sample_mask = adata.obs[sample_col] == sample

            row_mask = ref_mask & sample_mask & qmask & lm_mask & lrr_mask & demux_mask

            if not bool(row_mask.any()):
                print(
                    f"No reads for {display_sample} - {ref} after read quality and length filtering"
                )
                continue

            try:
                subset = adata[row_mask, :].copy()

                if min_position_valid_fraction is not None:
                    valid_key = f"{ref}_valid_fraction"
                    if valid_key in subset.var:
                        v = pd.to_numeric(subset.var[valid_key], errors="coerce").to_numpy()
                        col_mask = np.asarray(v > float(min_position_valid_fraction), dtype=bool)
                        if col_mask.any():
                            subset = subset[:, col_mask].copy()
                        else:
                            print(
                                f"No positions left after valid_fraction filter for {display_sample} - {ref}"
                            )
                            continue

                if subset.shape[0] == 0:
                    print(f"No reads left after filtering for {display_sample} - {ref}")
                    continue

                if bins is None:
                    bins_temp = {"All": np.ones(subset.n_obs, dtype=bool)}
                else:
                    bins_temp = bins

                def _sites(*keys):
                    """Return indices for the first matching site key."""
                    for k in keys:
                        if k in subset.var:
                            return np.where(subset.var[k].values)[0]
                    return np.array([], dtype=int)

                gpc_sites = _sites(f"{ref}_GpC_site")
                cpg_sites = _sites(f"{ref}_CpG_site")
                any_c_sites = _sites(f"{ref}_any_C_site", f"{ref}_C_site")
                any_a_sites = _sites(f"{ref}_A_site", f"{ref}_any_A_site")

                length_sites = np.arange(subset.n_vars, dtype=int)
                length_labels = _select_labels(subset, length_sites, ref, index_col_suffix)
                gpc_labels = _select_labels(subset, gpc_sites, ref, index_col_suffix)
                cpg_labels = _select_labels(subset, cpg_sites, ref, index_col_suffix)
                any_c_labels = _select_labels(subset, any_c_sites, ref, index_col_suffix)
                any_a_labels = _select_labels(subset, any_a_sites, ref, index_col_suffix)

                stacked_lengths = []
                stacked_lengths_raw = []
                stacked_any_c = []
                stacked_any_c_raw = []
                stacked_gpc = []
                stacked_gpc_raw = []
                stacked_cpg = []
                stacked_cpg_raw = []
                stacked_any_a = []
                stacked_any_a_raw = []

                row_labels, bin_labels, bin_boundaries = [], [], []
                total_reads = subset.n_obs
                percentages = {}
                last_idx = 0

                for bin_label, bin_filter in bins_temp.items():
                    sb = subset[bin_filter].copy()
                    n = sb.n_obs
                    if n == 0:
                        continue

                    pct = (n / total_reads) * 100 if total_reads else 0
                    percentages[bin_label] = pct

                    if sort_by.startswith("obs:"):
                        colname = sort_by.split("obs:")[1]
                        order = np.argsort(sb.obs[colname].values)
                    elif sort_by == "gpc" and gpc_sites.size:
                        gpc_matrix = _layer_to_numpy(
                            sb,
                            layer_gpc,
                            gpc_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    elif sort_by == "cpg" and cpg_sites.size:
                        cpg_matrix = _layer_to_numpy(
                            sb,
                            layer_cpg,
                            cpg_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(cpg_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    elif sort_by == "c" and any_c_sites.size:
                        any_c_matrix = _layer_to_numpy(
                            sb,
                            layer_c,
                            any_c_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_c_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    elif sort_by == "a" and any_a_sites.size:
                        any_a_matrix = _layer_to_numpy(
                            sb,
                            layer_a,
                            any_a_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(any_a_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    elif sort_by == "gpc_cpg" and gpc_sites.size and cpg_sites.size:
                        gpc_matrix = _layer_to_numpy(
                            sb,
                            layer_gpc,
                            None,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(gpc_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    elif sort_by == "hmm" and length_sites.size:
                        length_matrix = _layer_to_numpy(
                            sb,
                            length_layer,
                            length_sites,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                        linkage = sch.linkage(length_matrix, method="ward")
                        order = sch.leaves_list(linkage)
                    else:
                        order = np.arange(n)

                    sb = sb[order]

                    stacked_lengths.append(
                        _layer_to_numpy(
                            sb,
                            length_layer,
                            None,
                            fill_nan_strategy=fill_nan_strategy,
                            fill_nan_value=fill_nan_value,
                        )
                    )
                    stacked_lengths_raw.append(
                        _layer_to_numpy(
                            sb,
                            length_layer,
                            None,
                            fill_nan_strategy="none",
                            fill_nan_value=fill_nan_value,
                        )
                    )
                    if any_c_sites.size:
                        stacked_any_c.append(
                            _layer_to_numpy(
                                sb,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_c_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_c,
                                any_c_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if gpc_sites.size:
                        stacked_gpc.append(
                            _layer_to_numpy(
                                sb,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_gpc_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_gpc,
                                gpc_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if cpg_sites.size:
                        stacked_cpg.append(
                            _layer_to_numpy(
                                sb,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_cpg_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_cpg,
                                cpg_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )
                    if any_a_sites.size:
                        stacked_any_a.append(
                            _layer_to_numpy(
                                sb,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy=fill_nan_strategy,
                                fill_nan_value=fill_nan_value,
                            )
                        )
                        stacked_any_a_raw.append(
                            _layer_to_numpy(
                                sb,
                                layer_a,
                                any_a_sites,
                                fill_nan_strategy="none",
                                fill_nan_value=fill_nan_value,
                            )
                        )

                    row_labels.extend([bin_label] * n)
                    bin_labels.append(f"{bin_label}: {n} reads ({pct:.1f}%)")
                    last_idx += n
                    bin_boundaries.append(last_idx)

                length_matrix = np.vstack(stacked_lengths)
                length_matrix_raw = np.vstack(stacked_lengths_raw)
                capped_lengths = np.where(length_matrix_raw > 1, 1.0, length_matrix_raw)
                mean_lengths = np.nanmean(capped_lengths, axis=0)
                length_plot_matrix = length_matrix_raw
                length_plot_cmap = cmap_lengths
                length_plot_norm = None

                if feature_ranges:
                    length_plot_matrix = _map_length_matrix_to_subclasses(
                        length_matrix_raw, feature_ranges
                    )
                    length_plot_cmap, length_plot_norm = _build_length_feature_cmap(feature_ranges)

                panels = [
                    (
                        f"HMM lengths - {length_layer}",
                        length_plot_matrix,
                        length_labels,
                        length_plot_cmap,
                        mean_lengths,
                        n_xticks_lengths,
                        length_plot_norm,
                    ),
                ]

                if stacked_any_c:
                    m = np.vstack(stacked_any_c)
                    m_raw = np.vstack(stacked_any_c_raw)
                    panels.append(
                        (
                            "C",
                            m,
                            any_c_labels,
                            cmap_c,
                            _methylation_fraction_for_layer(m_raw, layer_c),
                            n_xticks_any_c,
                            None,
                        )
                    )

                if stacked_gpc:
                    m = np.vstack(stacked_gpc)
                    m_raw = np.vstack(stacked_gpc_raw)
                    panels.append(
                        (
                            "GpC",
                            m,
                            gpc_labels,
                            cmap_gpc,
                            _methylation_fraction_for_layer(m_raw, layer_gpc),
                            n_xticks_gpc,
                            None,
                        )
                    )

                if stacked_cpg:
                    m = np.vstack(stacked_cpg)
                    m_raw = np.vstack(stacked_cpg_raw)
                    panels.append(
                        (
                            "CpG",
                            m,
                            cpg_labels,
                            cmap_cpg,
                            _methylation_fraction_for_layer(m_raw, layer_cpg),
                            n_xticks_cpg,
                            None,
                        )
                    )

                if stacked_any_a:
                    m = np.vstack(stacked_any_a)
                    m_raw = np.vstack(stacked_any_a_raw)
                    panels.append(
                        (
                            "A",
                            m,
                            any_a_labels,
                            cmap_a,
                            _methylation_fraction_for_layer(m_raw, layer_a),
                            n_xticks_a,
                            None,
                        )
                    )

                n_panels = len(panels)
                fig = plt.figure(figsize=(4.5 * n_panels, 10))
                gs = gridspec.GridSpec(2, n_panels, height_ratios=[1, 6], hspace=0.01)
                fig.suptitle(
                    f"{sample} — {ref} — {total_reads} reads ({signal_type})", fontsize=14, y=0.98
                )

                axes_heat = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]
                axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(n_panels)]

                for i, (name, matrix, labels, cmap, mean_vec, n_ticks, norm) in enumerate(panels):
                    clean_barplot(axes_bar[i], mean_vec, name)

                    heatmap_kwargs = dict(
                        cmap=cmap,
                        ax=axes_heat[i],
                        yticklabels=False,
                        cbar=False,
                    )
                    if norm is not None:
                        heatmap_kwargs["norm"] = norm
                    sns.heatmap(matrix, **heatmap_kwargs)

                    xtick_pos, xtick_labels = pick_xticks(np.asarray(labels), n_ticks)
                    axes_heat[i].set_xticks(xtick_pos)
                    axes_heat[i].set_xticklabels(xtick_labels, rotation=90, fontsize=8)

                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=1.2)

                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    safe_name = f"{ref}__{sample}".replace("/", "_")
                    out_file = save_path / f"{safe_name}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close(fig)
                else:
                    plt.show()

                results.append((sample, ref))

            except Exception:
                import traceback

                traceback.print_exc()
                print(f"Failed {sample} - {ref} - {length_layer}")

    return results


def make_row_colors(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert metadata columns to RGB colors without invoking pandas Categorical.map
    (MultiIndex-safe, category-safe).
    """
    row_colors = pd.DataFrame(index=meta.index)

    for col in meta.columns:
        # Force plain python objects to avoid ExtensionArray/Categorical behavior
        s = meta[col].astype("object")

        def _to_label(x):
            if x is None:
                return "NA"
            if isinstance(x, float) and np.isnan(x):
                return "NA"
            # If a MultiIndex object is stored in a cell (rare), bucket it
            if isinstance(x, pd.MultiIndex):
                return "MultiIndex"
            # Tuples are common when MultiIndex-ish things get stored as values
            if isinstance(x, tuple):
                return "|".join(map(str, x))
            return str(x)

        labels = np.array([_to_label(x) for x in s.to_numpy()], dtype=object)
        uniq = pd.unique(labels)
        palette = dict(zip(uniq, sns.color_palette(n_colors=len(uniq))))

        # Map via python loop -> no pandas map machinery
        colors = [palette.get(lbl, (0.7, 0.7, 0.7)) for lbl in labels]
        row_colors[col] = colors

    return row_colors


def plot_rolling_nn_and_layer(
    subset,
    obsm_key: str = "rolling_nn_dist",
    layer_key: str = "nan0_0minus1",
    meta_cols=("Reference_strand", "Sample"),
    col_cluster: bool = False,
    fill_nn_with_colmax: bool = True,
    fill_layer_value: float = 0.0,
    drop_all_nan_windows: bool = True,
    max_nan_fraction: float | None = None,
    var_valid_fraction_col: str | None = None,
    var_nan_fraction_col: str | None = None,
    figsize=(14, 10),
    right_panel_var_mask=None,  # optional boolean mask over subset.var to reduce width
    robust=True,
    title: str | None = None,
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 8,
    save_name=None,
):
    """
    1) Cluster rows by subset.obsm[obsm_key] (rolling NN distances)
    2) Plot two heatmaps side-by-side in the SAME row order, with mean barplots above:
         - left: rolling NN distance matrix
         - right: subset.layers[layer_key] matrix

    Handles categorical/MultiIndex issues in metadata coloring.

    Args:
        subset: AnnData subset with rolling NN distances stored in ``obsm``.
        obsm_key: Key in ``subset.obsm`` containing rolling NN distances.
        layer_key: Layer name to plot alongside rolling NN distances.
        meta_cols: Obs columns used for row color annotations.
        col_cluster: Whether to cluster columns in the rolling NN clustermap.
        fill_nn_with_colmax: Fill NaNs in rolling NN distances with per-column max values.
        fill_layer_value: Fill NaNs in the layer heatmap with this value.
        drop_all_nan_windows: Drop rolling windows that are all NaN.
        max_nan_fraction: Maximum allowed NaN fraction per position (filtering columns).
        var_valid_fraction_col: ``subset.var`` column with valid fractions (1 - NaN fraction).
        var_nan_fraction_col: ``subset.var`` column with NaN fractions.
        figsize: Figure size for the combined plot.
        right_panel_var_mask: Optional boolean mask over ``subset.var`` for the right panel.
        robust: Use robust color scaling in seaborn.
        title: Optional figure title (suptitle).
        xtick_step: Spacing between x-axis tick labels.
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        save_name: Optional output path for saving the plot.
    """
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")

    def _apply_xticks(ax, labels, step):
        if labels is None or len(labels) == 0:
            ax.set_xticks([])
            return
        if step is None or step <= 0:
            step = max(1, len(labels) // 10)
        ticks = np.arange(0, len(labels), step)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(
            [labels[i] for i in ticks],
            rotation=xtick_rotation,
            fontsize=xtick_fontsize,
        )

    # --- rolling NN distances
    X = subset.obsm[obsm_key]
    valid = ~np.all(np.isnan(X), axis=1)

    X_df = pd.DataFrame(X[valid], index=subset.obs_names[valid])

    if drop_all_nan_windows:
        X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]

    X_df_filled = X_df.copy()
    if fill_nn_with_colmax:
        col_max = X_df_filled.max(axis=0, skipna=True)
        X_df_filled = X_df_filled.fillna(col_max)

    # Ensure non-MultiIndex index for seaborn
    X_df_filled.index = X_df_filled.index.astype(str)

    # --- row colors from metadata (MultiIndex-safe)
    meta = subset.obs.loc[X_df.index, list(meta_cols)].copy()
    meta.index = meta.index.astype(str)
    row_colors = make_row_colors(meta)

    # --- get row order via clustermap
    g = sns.clustermap(
        X_df_filled,
        cmap="viridis",
        col_cluster=col_cluster,
        row_cluster=True,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        robust=robust,
    )
    row_order = g.dendrogram_row.reordered_ind
    ordered_index = X_df_filled.index[row_order]
    plt.close(g.fig)

    # reorder rolling NN matrix
    X_ord = X_df_filled.loc[ordered_index]

    # --- layer matrix
    L = subset.layers[layer_key]
    L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)

    L_df = pd.DataFrame(L[valid], index=subset.obs_names[valid], columns=subset.var_names)
    L_df.index = L_df.index.astype(str)

    if right_panel_var_mask is not None:
        # right_panel_var_mask must be boolean array/Series aligned to subset.var_names
        if hasattr(right_panel_var_mask, "values"):
            right_panel_var_mask = right_panel_var_mask.values
        right_panel_var_mask = np.asarray(right_panel_var_mask, dtype=bool)

    if max_nan_fraction is not None:
        nan_fraction = None
        if var_nan_fraction_col and var_nan_fraction_col in subset.var:
            nan_fraction = pd.to_numeric(
                subset.var[var_nan_fraction_col], errors="coerce"
            ).to_numpy()
        elif var_valid_fraction_col and var_valid_fraction_col in subset.var:
            valid_fraction = pd.to_numeric(
                subset.var[var_valid_fraction_col], errors="coerce"
            ).to_numpy()
            nan_fraction = 1 - valid_fraction
        if nan_fraction is not None:
            nan_mask = nan_fraction <= max_nan_fraction
            if right_panel_var_mask is None:
                right_panel_var_mask = nan_mask
            else:
                right_panel_var_mask = right_panel_var_mask & nan_mask

    if right_panel_var_mask is not None:
        if right_panel_var_mask.size != L_df.shape[1]:
            raise ValueError("right_panel_var_mask must align with subset.var_names.")
        L_df = L_df.loc[:, right_panel_var_mask]

    L_ord = L_df.loc[ordered_index]
    L_plot = L_ord.fillna(fill_layer_value)

    # --- plot side-by-side with barplots above
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 0.05, 1, 0.05],
        height_ratios=[1, 6],
        wspace=0.2,
        hspace=0.05,
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax1_cbar = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_cbar = fig.add_subplot(gs[1, 3])
    ax1_bar = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2_bar = fig.add_subplot(gs[0, 2], sharex=ax2)
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 3]).axis("off")

    mean_nn = np.nanmean(X_ord.to_numpy(), axis=0)
    clean_barplot(
        ax1_bar,
        mean_nn,
        obsm_key,
        y_max=None,
        y_label="Mean distance",
        y_ticks=None,
    )

    sns.heatmap(
        X_ord,
        ax=ax1,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax1_cbar,
    )
    starts = subset.uns.get(f"{obsm_key}_starts")
    if starts is not None:
        starts = np.asarray(starts)
        window_labels = [str(s) for s in starts]
        try:
            col_idx = X_ord.columns.to_numpy()
            if np.issubdtype(col_idx.dtype, np.number):
                col_idx = col_idx.astype(int)
                if col_idx.size and col_idx.max() < len(starts):
                    window_labels = [str(s) for s in starts[col_idx]]
        except Exception:
            window_labels = [str(s) for s in starts]
        _apply_xticks(ax1, window_labels, xtick_step)

    methylation_fraction = _methylation_fraction_for_layer(L_ord.to_numpy(), layer_key)
    clean_barplot(
        ax2_bar,
        methylation_fraction,
        layer_key,
        y_max=1.0,
        y_label="Methylation fraction",
        y_ticks=[0.0, 0.5, 1.0],
    )

    sns.heatmap(
        L_plot,
        ax=ax2,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False,
        robust=robust,
        cbar_ax=ax2_cbar,
    )
    _apply_xticks(ax2, [str(x) for x in L_plot.columns], xtick_step)

    if title:
        fig.suptitle(title)

    if save_name is not None:
        fname = os.path.join(save_name)
        plt.savefig(fname, dpi=200, bbox_inches="tight")

    else:
        plt.show()

    return ordered_index


def plot_sequence_integer_encoding_clustermaps(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    layer: str = "sequence_integer_encoding",
    mismatch_layer: str = "mismatch_integer_encoding",
    min_quality: float | None = 20,
    min_length: int | None = 200,
    min_mapped_length_to_reference_length_ratio: float | None = 0,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sort_by: str = "none",  # "none", "hierarchical", "obs:<col>"
    cmap: str = "viridis",
    max_unknown_fraction: float | None = None,
    unknown_values: Sequence[int] = (4, 5),
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    max_reads: int | None = None,
    save_path: str | Path | None = None,
    use_dna_5color_palette: bool = True,
    show_numeric_colorbar: bool = False,
    show_position_axis: bool = False,
    position_axis_tick_target: int = 25,
):
    """Plot integer-encoded sequence clustermaps per sample/reference.

    Args:
        adata: AnnData with a ``sequence_integer_encoding`` layer.
        sample_col: Column in ``adata.obs`` that identifies samples.
        reference_col: Column in ``adata.obs`` that identifies references.
        layer: Layer name containing integer-encoded sequences.
        mismatch_layer: Optional layer name containing mismatch integer encodings.
        min_quality: Optional minimum read quality filter.
        min_length: Optional minimum mapped length filter.
        min_mapped_length_to_reference_length_ratio: Optional min length ratio filter.
        demux_types: Allowed ``demux_type`` values, if present in ``adata.obs``.
        sort_by: Row sorting strategy: ``none``, ``hierarchical``, or ``obs:<col>``.
        cmap: Matplotlib colormap for the heatmap when ``use_dna_5color_palette`` is False.
        max_unknown_fraction: Optional maximum fraction of ``unknown_values`` allowed per
            position; positions above this threshold are excluded.
        unknown_values: Integer values to treat as unknown/padding.
        xtick_step: Spacing between x-axis tick labels (None = no labels).
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        max_reads: Optional maximum number of reads to plot per sample/reference.
        save_path: Optional output directory for saving plots.
        use_dna_5color_palette: Whether to use a fixed A/C/G/T/Other palette.
        show_numeric_colorbar: If False, use a legend instead of a numeric colorbar.
        show_position_axis: Whether to draw a position axis with tick labels.
        position_axis_tick_target: Approximate number of ticks to show when auto-sizing.

    Returns:
        List of dictionaries with per-plot metadata and output paths.
    """

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    if max_unknown_fraction is not None and not (0 <= max_unknown_fraction <= 1):
        raise ValueError("max_unknown_fraction must be between 0 and 1.")

    if position_axis_tick_target < 1:
        raise ValueError("position_axis_tick_target must be at least 1.")

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    int_to_base = adata.uns.get("sequence_integer_decoding_map", {}) or {}
    if not int_to_base:
        encoding_map = adata.uns.get("sequence_integer_encoding_map", {}) or {}
        int_to_base = {int(v): str(k) for k, v in encoding_map.items()} if encoding_map else {}

    coerced_int_to_base = {}
    for key, value in int_to_base.items():
        try:
            coerced_key = int(key)
        except Exception:
            continue
        coerced_int_to_base[coerced_key] = str(value)
    int_to_base = coerced_int_to_base

    def normalize_base(base: str) -> str:
        return base if base in {"A", "C", "G", "T"} else "OTHER"

    mismatch_int_to_base = {}
    if mismatch_layer in adata.layers:
        mismatch_encoding_map = adata.uns.get("mismatch_integer_encoding_map", {}) or {}
        mismatch_int_to_base = {
            int(v): str(k)
            for k, v in mismatch_encoding_map.items()
            if isinstance(v, (int, np.integer))
        }

    def _resolve_xtick_step(n_positions: int) -> int | None:
        if xtick_step is not None:
            return xtick_step
        if not show_position_axis:
            return None
        return max(1, int(np.ceil(n_positions / position_axis_tick_target)))

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

            subset = adata[row_mask, :].copy()
            matrix = np.asarray(subset.layers[layer])
            mismatch_matrix = None
            if mismatch_layer in subset.layers:
                mismatch_matrix = np.asarray(subset.layers[mismatch_layer])

            if max_unknown_fraction is not None:
                unknown_mask = np.isin(matrix, np.asarray(unknown_values))
                unknown_fraction = unknown_mask.mean(axis=0)
                keep_columns = unknown_fraction <= max_unknown_fraction
                if not np.any(keep_columns):
                    continue
                matrix = matrix[:, keep_columns]
                subset = subset[:, keep_columns].copy()
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[:, keep_columns]

            if max_reads is not None and matrix.shape[0] > max_reads:
                matrix = matrix[:max_reads]
                subset = subset[:max_reads, :].copy()
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[:max_reads]

            if matrix.size == 0:
                continue

            if use_dna_5color_palette and not int_to_base:
                uniq_vals = np.unique(matrix[~pd.isna(matrix)])
                guess = {}
                for val in uniq_vals:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    guess[int_val] = {0: "A", 1: "C", 2: "G", 3: "T"}.get(int_val, "OTHER")
                int_to_base_local = guess
            else:
                int_to_base_local = int_to_base

            order = None
            if sort_by.startswith("obs:"):
                colname = sort_by.split("obs:")[1]
                order = np.argsort(subset.obs[colname].values)
            elif sort_by == "hierarchical":
                linkage = sch.linkage(np.nan_to_num(matrix), method="ward")
                order = sch.leaves_list(linkage)
            elif sort_by != "none":
                raise ValueError("sort_by must be 'none', 'hierarchical', or 'obs:<col>'")

            if order is not None:
                matrix = matrix[order]
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[order]

            has_mismatch = mismatch_matrix is not None
            fig, axes = plt.subplots(
                ncols=2 if has_mismatch else 1,
                figsize=(18, 6) if has_mismatch else (12, 6),
                sharey=has_mismatch,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            ax = axes[0]

            if use_dna_5color_palette and int_to_base_local:
                int_to_color = {
                    int(int_val): DNA_5COLOR_PALETTE[normalize_base(str(base))]
                    for int_val, base in int_to_base_local.items()
                }
                uniq_matrix = np.unique(matrix[~pd.isna(matrix)])
                for val in uniq_matrix:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    if int_val not in int_to_color:
                        int_to_color[int_val] = DNA_5COLOR_PALETTE["OTHER"]

                ordered = sorted(int_to_color.items(), key=lambda x: x[0])
                colors_list = [color for _, color in ordered]
                bounds = [int_val - 0.5 for int_val, _ in ordered]
                bounds.append(ordered[-1][0] + 0.5)

                cmap_obj = colors.ListedColormap(colors_list)
                norm = colors.BoundaryNorm(bounds, cmap_obj.N)

                sns.heatmap(
                    matrix,
                    cmap=cmap_obj,
                    norm=norm,
                    ax=ax,
                    yticklabels=False,
                    cbar=show_numeric_colorbar,
                )

                legend_handles = [
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["A"], label="A"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["C"], label="C"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["G"], label="G"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["T"], label="T"),
                    patches.Patch(
                        facecolor=DNA_5COLOR_PALETTE["OTHER"],
                        label="Other (N / PAD / unknown)",
                    ),
                ]
                ax.legend(
                    handles=legend_handles,
                    title="Base",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )
            else:
                sns.heatmap(matrix, cmap=cmap, ax=ax, yticklabels=False, cbar=True)

            ax.set_title(layer)

            resolved_step = _resolve_xtick_step(matrix.shape[1])
            if resolved_step is not None and resolved_step > 0:
                sites = np.arange(0, matrix.shape[1], resolved_step)
                ax.set_xticks(sites)
                ax.set_xticklabels(
                    subset.var_names[sites].astype(str),
                    rotation=xtick_rotation,
                    fontsize=xtick_fontsize,
                )
            else:
                ax.set_xticks([])
            if show_position_axis or xtick_step is not None:
                ax.set_xlabel("Position")

            if has_mismatch:
                mismatch_ax = axes[1]
                mismatch_int_to_base_local = mismatch_int_to_base or int_to_base_local
                if use_dna_5color_palette and mismatch_int_to_base_local:
                    mismatch_int_to_color = {}
                    for int_val, base in mismatch_int_to_base_local.items():
                        base_upper = str(base).upper()
                        if base_upper == "PAD":
                            mismatch_int_to_color[int(int_val)] = "#D3D3D3"
                        elif base_upper == "N":
                            mismatch_int_to_color[int(int_val)] = "#808080"
                        else:
                            mismatch_int_to_color[int(int_val)] = DNA_5COLOR_PALETTE[
                                normalize_base(base_upper)
                            ]

                    uniq_mismatch = np.unique(mismatch_matrix[~pd.isna(mismatch_matrix)])
                    for val in uniq_mismatch:
                        try:
                            int_val = int(val)
                        except Exception:
                            continue
                        if int_val not in mismatch_int_to_color:
                            mismatch_int_to_color[int_val] = DNA_5COLOR_PALETTE["OTHER"]

                    ordered_mismatch = sorted(mismatch_int_to_color.items(), key=lambda x: x[0])
                    mismatch_colors = [color for _, color in ordered_mismatch]
                    mismatch_bounds = [int_val - 0.5 for int_val, _ in ordered_mismatch]
                    mismatch_bounds.append(ordered_mismatch[-1][0] + 0.5)

                    mismatch_cmap = colors.ListedColormap(mismatch_colors)
                    mismatch_norm = colors.BoundaryNorm(mismatch_bounds, mismatch_cmap.N)

                    sns.heatmap(
                        mismatch_matrix,
                        cmap=mismatch_cmap,
                        norm=mismatch_norm,
                        ax=mismatch_ax,
                        yticklabels=False,
                        cbar=show_numeric_colorbar,
                    )

                    mismatch_legend_handles = [
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["A"], label="A"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["C"], label="C"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["G"], label="G"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["T"], label="T"),
                        patches.Patch(facecolor="#808080", label="Match/N"),
                        patches.Patch(facecolor="#D3D3D3", label="PAD"),
                    ]
                    mismatch_ax.legend(
                        handles=mismatch_legend_handles,
                        title="Mismatch base",
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        frameon=False,
                    )
                else:
                    sns.heatmap(
                        mismatch_matrix,
                        cmap=cmap,
                        ax=mismatch_ax,
                        yticklabels=False,
                        cbar=True,
                    )

                mismatch_ax.set_title(mismatch_layer)
                if resolved_step is not None and resolved_step > 0:
                    sites = np.arange(0, mismatch_matrix.shape[1], resolved_step)
                    mismatch_ax.set_xticks(sites)
                    mismatch_ax.set_xticklabels(
                        subset.var_names[sites].astype(str),
                        rotation=xtick_rotation,
                        fontsize=xtick_fontsize,
                    )
                else:
                    mismatch_ax.set_xticks([])
                if show_position_axis or xtick_step is not None:
                    mismatch_ax.set_xlabel("Position")

            fig.suptitle(f"{sample} - {ref}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            out_file = None
            if save_path is not None:
                safe_name = f"{ref}__{sample}__{layer}".replace("=", "").replace(",", "_")
                out_file = save_path / f"{safe_name}.png"
                fig.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            results.append(
                {
                    "reference": str(ref),
                    "sample": str(sample),
                    "layer": layer,
                    "n_positions": int(matrix.shape[1]),
                    "mismatch_layer": mismatch_layer if has_mismatch else None,
                    "mismatch_layer_present": bool(has_mismatch),
                    "output_path": str(out_file) if out_file is not None else None,
                }
            )

    return results


def plot_read_span_quality_clustermaps(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    quality_layer: str = "base_quality_scores",
    read_span_layer: str = "read_span_mask",
    quality_cmap: str = "viridis",
    read_span_color: str = "#2ca25f",
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

    Returns:
        List of dictionaries with per-plot metadata and output paths.
    """

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    def _resolve_xtick_step(n_positions: int) -> int | None:
        if xtick_step is not None:
            return xtick_step
        if not show_position_axis:
            return None
        return max(1, int(np.ceil(n_positions / position_axis_tick_target)))

    def _fill_nan_with_col_means(matrix: np.ndarray) -> np.ndarray:
        filled = matrix.copy()
        col_means = np.nanmean(filled, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_rows, nan_cols = np.where(np.isnan(filled))
        filled[nan_rows, nan_cols] = col_means[nan_cols]
        return filled

    if quality_layer not in adata.layers:
        raise KeyError(f"Layer '{quality_layer}' not found in adata.layers")
    if read_span_layer not in adata.layers:
        raise KeyError(f"Layer '{read_span_layer}' not found in adata.layers")
    if max_nan_fraction is not None and not (0 <= max_nan_fraction <= 1):
        raise ValueError("max_nan_fraction must be between 0 and 1.")
    if position_axis_tick_target < 1:
        raise ValueError("position_axis_tick_target must be at least 1.")

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

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

            subset = adata[row_mask, :].copy()
            quality_matrix = np.asarray(subset.layers[quality_layer]).astype(float)
            quality_matrix[quality_matrix < 0] = np.nan
            read_span_matrix = np.asarray(subset.layers[read_span_layer]).astype(float)

            if max_nan_fraction is not None:
                nan_mask = np.isnan(quality_matrix) | np.isnan(read_span_matrix)
                nan_fraction = nan_mask.mean(axis=0)
                keep_columns = nan_fraction <= max_nan_fraction
                if not np.any(keep_columns):
                    continue
                quality_matrix = quality_matrix[:, keep_columns]
                read_span_matrix = read_span_matrix[:, keep_columns]
                subset = subset[:, keep_columns].copy()

            if max_reads is not None and quality_matrix.shape[0] > max_reads:
                quality_matrix = quality_matrix[:max_reads]
                read_span_matrix = read_span_matrix[:max_reads]
                subset = subset[:max_reads, :].copy()

            if quality_matrix.size == 0:
                continue

            quality_filled = _fill_nan_with_col_means(quality_matrix)
            linkage = sch.linkage(quality_filled, method="ward")
            order = sch.leaves_list(linkage)

            quality_matrix = quality_matrix[order]
            read_span_matrix = read_span_matrix[order]

            fig, axes = plt.subplots(
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
            span_bar_ax.bar(
                bar_positions,
                span_mean,
                color=read_span_color,
                width=1.0,
            )
            span_bar_ax.set_title(f"{read_span_layer} mean")
            span_bar_ax.set_xlim(0, read_span_matrix.shape[1])
            span_bar_ax.tick_params(axis="x", labelbottom=False)

            quality_bar_ax.bar(
                bar_positions,
                quality_mean,
                color="#4c72b0",
                width=1.0,
            )
            quality_bar_ax.set_title(f"{quality_layer} mean")
            quality_bar_ax.set_xlim(0, quality_matrix.shape[1])
            quality_bar_ax.tick_params(axis="x", labelbottom=False)

            span_cmap = colors.ListedColormap(["white", read_span_color])
            span_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], span_cmap.N)
            sns.heatmap(
                read_span_matrix,
                cmap=span_cmap,
                norm=span_norm,
                ax=span_ax,
                yticklabels=False,
                cbar=False,
            )
            span_ax.set_title(read_span_layer)

            sns.heatmap(
                quality_matrix,
                cmap=quality_cmap,
                ax=quality_ax,
                yticklabels=False,
                cbar=True,
                cbar_ax=cbar_ax,
            )
            quality_ax.set_title(quality_layer)

            resolved_step = _resolve_xtick_step(quality_matrix.shape[1])
            for axis in (span_ax, quality_ax):
                if resolved_step is not None and resolved_step > 0:
                    sites = np.arange(0, quality_matrix.shape[1], resolved_step)
                    axis.set_xticks(sites)
                    axis.set_xticklabels(
                        subset.var_names[sites].astype(str),
                        rotation=xtick_rotation,
                        fontsize=xtick_fontsize,
                    )
                else:
                    axis.set_xticks([])
                if show_position_axis or xtick_step is not None:
                    axis.set_xlabel("Position")

            fig.suptitle(f"{sample} - {ref}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            out_file = None
            if save_path is not None:
                safe_name = f"{ref}__{sample}__read_span_quality".replace("=", "").replace(",", "_")
                out_file = save_path / f"{safe_name}.png"
                fig.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            results.append(
                {
                    "reference": str(ref),
                    "sample": str(sample),
                    "quality_layer": quality_layer,
                    "read_span_layer": read_span_layer,
                    "n_positions": int(quality_matrix.shape[1]),
                    "output_path": str(out_file) if out_file is not None else None,
                }
            )

    return results


def plot_hmm_layers_rolling_by_sample_ref(
    adata,
    layers: Optional[Sequence[str]] = None,
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    samples: Optional[Sequence[str]] = None,
    references: Optional[Sequence[str]] = None,
    window: int = 51,
    min_periods: int = 1,
    center: bool = True,
    rows_per_page: int = 6,
    figsize_per_cell: Tuple[float, float] = (4.0, 2.5),
    dpi: int = 160,
    output_dir: Optional[str] = None,
    save: bool = True,
    show_raw: bool = False,
    cmap: str = "tab20",
    layer_colors: Optional[Mapping[str, Any]] = None,
    use_var_coords: bool = True,
    reindexed_var_suffix: str = "reindexed",
):
    """
    For each sample (row) and reference (col) plot the rolling average of the
    positional mean (mean across reads) for each layer listed.

    Parameters
    ----------
    adata : AnnData
        Input annotated data (expects obs columns sample_col and ref_col).
    layers : list[str] | None
        Which adata.layers to plot. If None, attempts to autodetect layers whose
        matrices look like "HMM" outputs (else will error). If None and layers
        cannot be found, user must pass a list.
    sample_col, ref_col : str
        obs columns used to group rows.
    samples, references : optional lists
        explicit ordering of samples / references. If None, categories in adata.obs are used.
    window : int
        rolling window size (odd recommended). If window <= 1, no smoothing applied.
    min_periods : int
        min periods param for pd.Series.rolling.
    center : bool
        center the rolling window.
    rows_per_page : int
        paginate rows per page into multiple figures if needed.
    figsize_per_cell : (w,h)
        per-subplot size in inches.
    dpi : int
        figure dpi when saving.
    output_dir : str | None
        directory to save pages; created if necessary. If None and save=True, uses cwd.
    save : bool
        whether to save PNG files.
    show_raw : bool
        draw unsmoothed mean as faint line under smoothed curve.
    cmap : str
        matplotlib colormap for layer lines.
    layer_colors : dict[str, Any] | None
        Optional mapping of layer name to explicit line colors.
    use_var_coords : bool
        if True, tries to use adata.var_names (coerced to int) as x-axis coordinates; otherwise uses 0..n-1.
    reindexed_var_suffix : str
        Suffix for per-reference reindexed var columns (e.g., ``Reference_reindexed``) used when available.

    Returns
    -------
    saved_files : list[str]
        list of saved filenames (may be empty if save=False).
    """

    # --- basic checks / defaults ---
    if sample_col not in adata.obs.columns or ref_col not in adata.obs.columns:
        raise ValueError(
            f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs"
        )

    # canonicalize samples / refs
    if samples is None:
        sseries = adata.obs[sample_col]
        if not pd.api.types.is_categorical_dtype(sseries):
            sseries = sseries.astype("category")
        samples_all = list(sseries.cat.categories)
    else:
        samples_all = list(samples)

    if references is None:
        rseries = adata.obs[ref_col]
        if not pd.api.types.is_categorical_dtype(rseries):
            rseries = rseries.astype("category")
        refs_all = list(rseries.cat.categories)
    else:
        refs_all = list(references)

    # choose layers: if not provided, try a sensible default: all layers
    if layers is None:
        layers = list(adata.layers.keys())
        if len(layers) == 0:
            raise ValueError(
                "No adata.layers found. Please pass `layers=[...]` of the HMM layers to plot."
            )
    layers = list(layers)

    # x coordinates (positions) + optional labels
    x_labels = None
    try:
        if use_var_coords:
            x_coords = np.array([int(v) for v in adata.var_names])
        else:
            raise Exception("user disabled var coords")
    except Exception:
        # fallback to 0..n_vars-1, but keep var_names as labels
        x_coords = np.arange(adata.shape[1], dtype=int)
        x_labels = adata.var_names.astype(str).tolist()

    ref_reindexed_cols = {
        ref: f"{ref}_{reindexed_var_suffix}"
        for ref in refs_all
        if f"{ref}_{reindexed_var_suffix}" in adata.var
    }

    # make output dir
    if save:
        outdir = output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None

    n_samples = len(samples_all)
    n_refs = len(refs_all)
    total_pages = math.ceil(n_samples / rows_per_page)
    saved_files = []

    # color cycle for layers
    cmap_obj = plt.get_cmap(cmap)
    n_layers = max(1, len(layers))
    fallback_colors = [cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)]
    layer_colors = layer_colors or {}
    colors = [layer_colors.get(layer, fallback_colors[idx]) for idx, layer in enumerate(layers)]

    for page in range(total_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n_samples)
        chunk = samples_all[start:end]
        nrows = len(chunk)
        ncols = n_refs

        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
        )

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(refs_all):
                ax = axes[r_idx][c_idx]

                # subset adata
                mask = (adata.obs[sample_col].values == sample_name) & (
                    adata.obs[ref_col].values == ref_name
                )
                sub = adata[mask]
                if sub.n_obs == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No reads",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="gray",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r_idx == 0:
                        ax.set_title(str(ref_name), fontsize=9)
                    if c_idx == 0:
                        total_reads = int((adata.obs[sample_col] == sample_name).sum())
                        ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                    continue

                # for each layer, compute positional mean across reads (ignore NaNs)
                plotted_any = False
                reindexed_col = ref_reindexed_cols.get(ref_name)
                if reindexed_col is not None:
                    try:
                        ref_coords = np.asarray(adata.var[reindexed_col], dtype=int)
                    except Exception:
                        ref_coords = x_coords
                else:
                    ref_coords = x_coords
                for li, layer in enumerate(layers):
                    if layer in sub.layers:
                        mat = sub.layers[layer]
                    else:
                        # fallback: try .X only for the first layer if layer not present
                        if layer == layers[0] and getattr(sub, "X", None) is not None:
                            mat = sub.X
                        else:
                            # layer not present for this subset
                            continue

                    # convert matrix to numpy 2D
                    if hasattr(mat, "toarray"):
                        try:
                            arr = mat.toarray()
                        except Exception:
                            arr = np.asarray(mat)
                    else:
                        arr = np.asarray(mat)

                    if arr.size == 0 or arr.shape[1] == 0:
                        continue

                    # compute column-wise mean ignoring NaNs
                    # if arr is boolean or int, convert to float to support NaN
                    arr = arr.astype(float)
                    with np.errstate(all="ignore"):
                        col_mean = np.nanmean(arr, axis=0)

                    # If all-NaN, skip
                    if np.all(np.isnan(col_mean)):
                        continue

                    valid_mask = np.isfinite(col_mean)

                    # smooth via pandas rolling (centered)
                    if (window is None) or (window <= 1):
                        smoothed = col_mean
                    else:
                        ser = pd.Series(col_mean)
                        smoothed = (
                            ser.rolling(window=window, min_periods=min_periods, center=center)
                            .mean()
                            .to_numpy()
                        )
                        smoothed = np.where(valid_mask, smoothed, np.nan)

                    # x axis: x_coords (trim/pad to match length)
                    L = len(col_mean)
                    x = ref_coords[:L]

                    # optionally plot raw faint line first
                    if show_raw:
                        ax.plot(x, col_mean[:L], linewidth=0.7, alpha=0.25, zorder=1)

                    ax.plot(
                        x,
                        smoothed[:L],
                        label=layer,
                        color=colors[li],
                        linewidth=1.2,
                        alpha=0.95,
                        zorder=2,
                    )
                    plotted_any = True

                # labels / titles
                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=9)
                if c_idx == 0:
                    total_reads = int((adata.obs[sample_col] == sample_name).sum())
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                if r_idx == nrows - 1:
                    ax.set_xlabel("position", fontsize=8)
                    if x_labels is not None and reindexed_col is None:
                        max_ticks = 8
                        tick_step = max(1, int(math.ceil(len(x_labels) / max_ticks)))
                        tick_positions = x_coords[::tick_step]
                        tick_labels = x_labels[::tick_step]
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")

                # legend (only show in top-left plot to reduce clutter)
                if (r_idx == 0 and c_idx == 0) and plotted_any:
                    ax.legend(fontsize=7, loc="upper right")

                ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"Rolling mean of layer positional means (window={window}) — page {page + 1}/{total_pages}",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            fname = os.path.join(outdir, f"hmm_layers_rolling_page{page + 1}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=dpi)
            saved_files.append(fname)
        else:
            plt.show()
        plt.close(fig)

    return saved_files


def _resolve_embedding(adata: "ad.AnnData", basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    embedding = np.asarray(adata.obsm[key])
    if embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have at least two dimensions.")
    return embedding[:, :2]


def plot_embedding(
    adata: "ad.AnnData",
    *,
    basis: str,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Dict[str, Path]:
    """Plot a 2D embedding with scanpy-style color options.

    Args:
        adata: AnnData object with ``obsm['X_<basis>']``.
        basis: Embedding basis name (e.g., ``'umap'``, ``'pca'``).
        color: Obs column name or list of names to color by.
        output_dir: Directory to save plots.
        prefix: Optional filename prefix.
        point_size: Marker size for scatter plots.
        alpha: Marker transparency.

    Returns:
        Dict[str, Path]: Mapping of color keys to saved plot paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    embedding = _resolve_embedding(adata, basis)
    colors = [color] if isinstance(color, str) else list(color)
    saved: Dict[str, Path] = {}

    for color_key in colors:
        if color_key not in adata.obs:
            logger.warning("Color key '%s' not found in adata.obs; skipping.", color_key)
            continue
        values = adata.obs[color_key]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        if pd.api.types.is_categorical_dtype(values) or values.dtype == object:
            categories = pd.Categorical(values)
            label_strings = categories.categories.astype(str)
            palette = sns.color_palette("tab20", n_colors=len(label_strings))
            color_map = dict(zip(label_strings, palette))
            codes = categories.codes
            mapped = np.empty(len(codes), dtype=object)
            valid = codes >= 0
            if np.any(valid):
                valid_codes = codes[valid]
                mapped_values = np.empty(len(valid_codes), dtype=object)
                for i, idx in enumerate(valid_codes):
                    mapped_values[i] = palette[idx]
                mapped[valid] = mapped_values
            mapped[~valid] = "#bdbdbd"
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=list(mapped),
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            handles = [
                patches.Patch(color=color_map[label], label=str(label)) for label in label_strings
            ]
            ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
        else:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=values.astype(float),
                cmap="viridis",
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            fig.colorbar(scatter, ax=ax, label=color_key)

        ax.set_xlabel(f"{basis.upper()} 1")
        ax.set_ylabel(f"{basis.upper()} 2")
        ax.set_title(f"{basis.upper()} colored by {color_key}")
        fig.tight_layout()

        filename_prefix = prefix or basis
        safe_key = str(color_key).replace(" ", "_")
        output_file = output_path / f"{filename_prefix}_{safe_key}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        saved[color_key] = output_file

    return saved


def _grid_dimensions(n_items: int, ncols: int | None) -> tuple[int, int]:
    if n_items < 1:
        return 0, 0
    if ncols is None:
        ncols = 2 if n_items > 1 else 1
    ncols = max(1, min(ncols, n_items))
    nrows = int(math.ceil(n_items / ncols))
    return nrows, ncols


def plot_embedding_grid(
    adata: "ad.AnnData",
    *,
    basis: str,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    ncols: int | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Path | None:
    """Plot a 2D embedding grid with legends to the right of each subplot.

    Args:
        adata: AnnData object with ``obsm['X_<basis>']``.
        basis: Embedding basis name (e.g., ``'umap'``, ``'pca'``).
        color: Obs column name or list of names to color by.
        output_dir: Directory to save plots.
        prefix: Optional filename prefix.
        ncols: Number of columns in the grid.
        point_size: Marker size for scatter plots.
        alpha: Marker transparency.

    Returns:
        Path to the saved grid image, or None if no valid color keys exist.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    embedding = _resolve_embedding(adata, basis)
    colors = [color] if isinstance(color, str) else list(color)

    valid_colors = []
    for color_key in colors:
        if color_key not in adata.obs:
            logger.warning("Color key '%s' not found in adata.obs; skipping.", color_key)
            continue
        valid_colors.append(color_key)

    if not valid_colors:
        return None

    nrows, ncols = _grid_dimensions(len(valid_colors), ncols)
    plot_width = 4.8
    legend_width = 2.4
    plot_height = 4.2
    fig = plt.figure(
        figsize=(ncols * (plot_width + legend_width), nrows * plot_height),
    )
    width_ratios = [plot_width, legend_width] * ncols
    grid = gridspec.GridSpec(
        nrows,
        ncols * 2,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.35,
    )

    for idx, color_key in enumerate(valid_colors):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(grid[row, col * 2])
        legend_ax = fig.add_subplot(grid[row, col * 2 + 1])
        legend_ax.axis("off")

        values = adata.obs[color_key]
        if pd.api.types.is_categorical_dtype(values) or values.dtype == object:
            categories = pd.Categorical(values)
            label_strings = categories.categories.astype(str)
            palette = sns.color_palette("tab20", n_colors=len(label_strings))
            color_map = dict(zip(label_strings, palette))
            codes = categories.codes
            mapped = np.empty(len(codes), dtype=object)
            valid = codes >= 0
            if np.any(valid):
                valid_codes = codes[valid]
                mapped_values = np.empty(len(valid_codes), dtype=object)
                for i, idx_code in enumerate(valid_codes):
                    mapped_values[i] = palette[idx_code]
                mapped[valid] = mapped_values
            mapped[~valid] = "#bdbdbd"
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=list(mapped),
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            handles = [
                patches.Patch(color=color_map[label], label=str(label)) for label in label_strings
            ]
            legend_ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(0.0, 0.5),
                fontsize=8,
                frameon=False,
                title=str(color_key),
            )
        else:
            values_float = values.astype(float)

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=values_float,
                cmap="viridis",
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )

            colorbar = fig.colorbar(
                scatter,
                ax=ax,
                fraction=0.04,   # ← thickness (default ~0.15)
                pad=0.02,        # gap from plot
                shrink=0.9,      # slightly shorter
                aspect=30        # larger = thinner
            )

            # Ticks (5 evenly spaced)
            vmin, vmax = np.nanmin(values_float), np.nanmax(values_float)
            ticks = np.linspace(vmin, vmax, 5)
            colorbar.set_ticks(ticks)
            colorbar.ax.tick_params(labelsize=8, length=2)
            colorbar.set_label(str(color_key), fontsize=9)

        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"{color_key}")

    filename_prefix = prefix or basis
    output_file = output_path / f"{filename_prefix}_grid.png"
    fig.suptitle(f"{basis}")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_umap(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
) -> Dict[str, Path]:
    """Plot UMAP embedding with scanpy-style color options."""
    if subset:
        basis = f"umap_{subset}"
    else:
        basis = "umap"
    return plot_embedding(adata, basis=basis, color=color, output_dir=output_dir, prefix=basis)


def plot_umap_grid(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    ncols: int | None = None,
) -> Path | None:
    """Plot UMAP embedding grid with scanpy-style color options."""
    if subset:
        basis = f"umap_{subset}"
    else:
        basis = "umap"
    return plot_embedding_grid(
        adata,
        basis=basis,
        color=color,
        output_dir=output_dir,
        prefix=basis,
        ncols=ncols,
    )


def plot_pca(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
) -> Dict[str, Path]:
    """Plot PCA embedding with scanpy-style color options."""
    if subset:
        basis = f"pca_{subset}"
    else:
        basis = "pca"
    return plot_embedding(adata, basis=basis, color=color, output_dir=output_dir, prefix=basis)


def plot_pca_grid(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    ncols: int | None = None,
) -> Path | None:
    """Plot PCA embedding grid with scanpy-style color options."""
    if subset:
        basis = f"pca_{subset}"
    else:
        basis = "pca"
    return plot_embedding_grid(
        adata,
        basis=basis,
        color=color,
        output_dir=output_dir,
        prefix=basis,
        ncols=ncols,
    )


def plot_pca_explained_variance(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    output_dir: Path | str,
    filename: str | None = None,
    max_pcs: int | None = None,
) -> Path | None:
    """Plot per-PC explained variance ratios and cumulative variance.

    Args:
        adata: AnnData object containing PCA results.
        subset: Optional PCA subset suffix used by ``calculate_pca``.
        output_dir: Directory to write the plot into.
        filename: Optional output filename. Uses a default if not provided.
        max_pcs: Optional cap on number of PCs to plot.

    Returns:
        Path to the saved plot, or None if explained variance is unavailable.
    """
    if subset:
        pca_key = f"X_pca_{subset}"
        default_filename = f"pca_{subset}_explained_variance.png"
    else:
        pca_key = "X_pca"
        default_filename = "pca_explained_variance.png"

    explained_variance_ratio = adata.uns.get(pca_key, {}).get("explained_variance_ratio")
    if explained_variance_ratio is None:
        logger.warning("Explained variance ratio not found in adata.uns[%s].", pca_key)
        return None

    variance = np.asarray(explained_variance_ratio, dtype=float)
    if variance.size == 0:
        logger.warning("Explained variance ratio for %s is empty; skipping plot.", pca_key)
        return None

    if max_pcs is not None:
        variance = variance[: int(max_pcs)]

    pcs = np.arange(1, variance.size + 1)
    cumulative = np.cumsum(variance)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (filename or default_filename)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pcs, variance, color="#4C72B0", alpha=0.8, label="Explained variance")
    ax.plot(pcs, cumulative, color="#DD8452", marker="o", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_xticks(pcs)
    ax.set_ylim(0, max(1.0, cumulative.max() * 1.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
