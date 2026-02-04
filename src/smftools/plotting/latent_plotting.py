from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require
from smftools.plotting.plotting_utils import _fixed_tick_positions

patches = require("matplotlib.patches", extra="plotting", purpose="plot rendering")
plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")

sns = require("seaborn", extra="plotting", purpose="plot styling")

grid_spec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")

logger = get_logger(__name__)

if TYPE_CHECKING:
    import anndata as ad


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
    logger.info("Plotting NMF components to %s.", output_dir)
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
    logger.info("Saved NMF heatmap to %s.", heatmap_path)

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
    logger.info("Saved NMF line plot to %s.", lineplot_path)

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
    """Plot PCA component loadings as a heatmap and per-component scatter plot.

    Args:
        adata: AnnData object containing PCA results.
        output_dir: Directory to write plots into.
        components_key: Key in ``adata.varm`` storing the components.
        heatmap_name: Filename for the heatmap plot.
        lineplot_name: Filename for the scatter plot.
        max_features: Maximum number of features to plot (top-weighted by component).

    Returns:
        Dict[str, Path]: Paths to created plots (keys: ``heatmap`` and ``lineplot``).
    """
    logger.info("Plotting PCA components to %s.", output_dir)
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
        scores = np.nanmax(components, axis=1)
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
        cbar_kws={"label": "Component loading"},
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
    logger.info("Saved PCA heatmap to %s.", heatmap_path)

    fig, ax = plt.subplots(figsize=(max(8, min(20, n_features / 50)), 3.5))
    x = feature_positions
    for idx, label in enumerate(component_labels):
        ax.scatter(x, components[:, idx], label=label, s=14, alpha=0.75)
    ax.set_xlabel("Position index")
    ax.set_ylabel("Component loading")
    if n_features <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    lineplot_path = output_path / lineplot_name
    fig.savefig(lineplot_path, dpi=200)
    plt.close(fig)
    logger.info("Saved PCA line plot to %s.", lineplot_path)

    return {"heatmap": heatmap_path, "lineplot": lineplot_path}


def plot_cp_sequence_components(
    adata: "ad.AnnData",
    *,
    output_dir: Path | str,
    components_key: str = "H_cp_sequence",
    uns_key: str = "cp_sequence",
    base_factors_key: str | None = None,
    suffix: str | None = None,
    heatmap_name: str = "cp_sequence_position_heatmap.png",
    lineplot_name: str = "cp_sequence_position_lineplot.png",
    base_factors_name: str = "cp_sequence_base_weights.png",
    max_positions: int = 2000,
) -> Dict[str, Path]:
    """Plot CP sequence components as heatmaps and line plots.

    Args:
        adata: AnnData object with CP decomposition in ``varm`` and ``uns``.
        output_dir: Directory to write plots into.
        components_key: Key in ``adata.varm`` for position factors.
        uns_key: Key in ``adata.uns`` for CP metadata (base factors/labels).
        base_factors_key: Optional key in ``adata.uns`` for base factors.
        suffix: Optional suffix appended to the component keys.
        heatmap_name: Filename for the heatmap plot.
        lineplot_name: Filename for the line plot.
        base_factors_name: Filename for the base factors plot.
        max_positions: Maximum number of positions to plot.

    Returns:
        Dict[str, Path]: Paths to generated plots.
    """
    logger.info("Plotting CP sequence components to %s.", output_dir)
    if suffix:
        components_key = f"{components_key}_{suffix}"
        if base_factors_key is not None:
            base_factors_key = f"{base_factors_key}_{suffix}"
        uns_key = f"{uns_key}_{suffix}"

    heatmap_name = f"{components_key}_{heatmap_name}"
    lineplot_name = f"{components_key}_{lineplot_name}"
    base_name = f"{components_key}_{base_factors_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if components_key not in adata.varm:
        logger.warning("CP components key '%s' not found in adata.varm.", components_key)
        return {}

    components = np.asarray(adata.varm[components_key])
    if components.ndim != 2:
        raise ValueError(f"CP position factors must be 2D; got shape {components.shape}.")

    position_indices = np.arange(components.shape[0])
    valid_mask = np.isfinite(components).any(axis=1)
    if not np.all(valid_mask):
        dropped = int(np.sum(~valid_mask))
        logger.info("Dropping %s CP positions with no finite weights before plotting.", dropped)
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

    outputs: Dict[str, Path] = {}
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
        logger.info("Saved CP sequence heatmap to %s.", heatmap_path)

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
        logger.info("Saved CP sequence line plot to %s.", lineplot_path)

        outputs["heatmap"] = heatmap_path
        outputs["lineplot"] = lineplot_path

    base_factors = None
    base_labels = None
    if uns_key in adata.uns and isinstance(adata.uns[uns_key], dict):
        base_factors = adata.uns[uns_key].get("base_factors")
        base_labels = adata.uns[uns_key].get("base_labels")
    if base_factors is None and base_factors_key:
        base_factors = adata.uns.get(base_factors_key)
        base_labels = adata.uns.get("cp_base_labels")

    if base_factors is not None:
        base_factors = np.asarray(base_factors)
        if base_factors.ndim != 2 or base_factors.shape[0] == 0:
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
            logger.info("Saved CP base factors plot to %s.", base_path)

    return outputs


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
    logger.info("Plotting %s embedding to %s.", basis, output_dir)
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

        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
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

        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"{color_key}")
        fig.tight_layout()

        filename_prefix = prefix or basis
        safe_key = str(color_key).replace(" ", "_")
        output_file = output_path / f"{filename_prefix}_{safe_key}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        logger.info("Saved %s embedding plot to %s.", basis, output_file)
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
    logger.info("Plotting %s embedding grid to %s.", basis, output_dir)
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
    grid = grid_spec.GridSpec(
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
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
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
                for i, idx2 in enumerate(valid_codes):
                    mapped_values[i] = palette[idx2]
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
            legend_ax.legend(handles=handles, loc="center left", fontsize=8, frameon=False)
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
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)

        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"{color_key}")

    fig.tight_layout()

    filename_prefix = prefix or basis
    output_file = output_path / f"{filename_prefix}_grid.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    logger.info("Saved %s embedding grid to %s.", basis, output_file)
    return output_file


def plot_umap(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Dict[str, Path]:
    logger.info("Plotting UMAP embedding to %s.", output_dir)

    if subset:
        umap_key = f"umap_{subset}"
    else:
        umap_key = "umap"

    return plot_embedding(adata, basis=umap_key, color=color, output_dir=output_dir, prefix=prefix)


def plot_umap_grid(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    ncols: int | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Path | None:
    logger.info("Plotting UMAP embedding grid to %s.", output_dir)

    if subset:
        umap_key = f"umap_{subset}"
    else:
        umap_key = "umap"

    return plot_embedding_grid(
        adata,
        basis=umap_key,
        color=color,
        output_dir=output_dir,
        prefix=prefix,
        ncols=ncols,
        point_size=point_size,
        alpha=alpha,
    )


def plot_pca(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Dict[str, Path]:
    logger.info("Plotting PCA embedding to %s.", output_dir)
    if subset:
        pca_key = f"pca_{subset}"
    else:
        pca_key = "pca"
    return plot_embedding(adata, basis=pca_key, color=color, output_dir=output_dir, prefix=prefix)


def plot_pca_grid(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    color: str | Sequence[str],
    output_dir: Path | str,
    prefix: str | None = None,
    ncols: int | None = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Path | None:
    logger.info("Plotting PCA embedding grid to %s.", output_dir)

    if subset:
        pca_key = f"pca_{subset}"
    else:
        pca_key = "pca"

    return plot_embedding_grid(
        adata,
        basis=pca_key,
        color=color,
        output_dir=output_dir,
        prefix=prefix,
        ncols=ncols,
        point_size=point_size,
        alpha=alpha,
    )


def plot_pca_explained_variance(
    adata: "ad.AnnData",
    *,
    subset: str | None = None,
    output_dir: Path | str,
    pca_key: str = "pca",
    suffix: str | None = None,
    max_pcs: int | None = None,
) -> Path | None:
    """Plot cumulative explained variance for PCA results.

    Args:
        adata: AnnData object containing PCA results in ``uns``.
        subset: Optional subset suffix used in key naming.
        output_dir: Directory to write the plot into.
        pca_key: Base key in ``adata.uns`` storing PCA results.
        suffix: Optional suffix to append to the key.
        max_pcs: Optional cap on number of PCs to plot.

    Returns:
        Path to the saved plot, or None if explained variance is unavailable.
    """
    logger.info("Plotting PCA explained variance to %s.", output_dir)

    if subset:
        pca_key = f"{pca_key}_{subset}"
    if suffix:
        pca_key = f"{pca_key}_{suffix}"

    if pca_key not in adata.uns:
        logger.warning("Explained variance ratio not found in adata.uns[%s].", pca_key)
        return None

    pca_data = adata.uns[pca_key]
    if not isinstance(pca_data, Mapping) or "variance_ratio" not in pca_data:
        logger.warning("Explained variance ratio not found in adata.uns[%s].", pca_key)
        return None

    variance_ratio = np.asarray(pca_data.get("variance_ratio", []), dtype=float)
    if variance_ratio.size == 0:
        logger.warning("Explained variance ratio for %s is empty; skipping plot.", pca_key)
        return None

    if max_pcs is not None:
        variance_ratio = variance_ratio[:max_pcs]

    cumulative = np.cumsum(variance_ratio)
    pcs = np.arange(1, len(variance_ratio) + 1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pcs, cumulative, color="#DD8452", marker="o", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_file = output_path / f"{pca_key}_explained_variance.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    logger.info("Saved PCA explained variance plot to %s.", out_file)

    return out_file
