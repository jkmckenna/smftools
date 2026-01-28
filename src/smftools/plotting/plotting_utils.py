from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from smftools.optional_imports import require

sns = require("seaborn", extra="plotting", purpose="plot styling")

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

    pos = np.linspace(0, n_positions - 1, n_ticks)
    return np.unique(np.round(pos).astype(int))


def _select_labels(
    subset: "ad.AnnData", sites: np.ndarray, reference: str, index_col_suffix: str | None
) -> np.ndarray:
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

    if index_col_suffix is None:
        return subset.var_names[sites].astype(str)

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


def _layer_to_numpy(
    subset: "ad.AnnData",
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

    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "left")

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)


def make_row_colors(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert metadata columns to RGB colors without invoking pandas Categorical.map
    (MultiIndex-safe, category-safe).
    """
    row_colors = pd.DataFrame(index=meta.index)

    for col in meta.columns:
        s = meta[col].astype("object")

        def _to_label(x: Any) -> str:
            if x is None:
                return "NA"
            if isinstance(x, float) and np.isnan(x):
                return "NA"
            if isinstance(x, pd.MultiIndex):
                return "MultiIndex"
            if isinstance(x, tuple):
                return "|".join(map(str, x))
            return str(x)

        labels = np.array([_to_label(x) for x in s.to_numpy()], dtype=object)
        uniq = pd.unique(labels)
        palette = dict(zip(uniq, sns.color_palette(n_colors=len(uniq))))

        colors = [palette.get(lbl, (0.7, 0.7, 0.7)) for lbl in labels]
        row_colors[col] = colors

    return row_colors
