from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

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


def _values_to_str_labels(values) -> np.ndarray:
    """Convert values to string labels, formatting whole-number floats as ints.

    Handles the case where integer var columns have been promoted to float64
    (e.g. by ``_harmonize_var_schema`` for HDF5 NaN support) so that tick
    labels read ``"123"`` rather than ``"123.0"``.
    """
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        # Check if all finite values are whole numbers
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and np.all(finite == np.floor(finite)):
            return np.array([str(int(v)) if np.isfinite(v) else str(v) for v in arr])
    return np.asarray(arr, dtype=str)


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
        return _values_to_str_labels(subset.var_names[sites])

    colname = f"{reference}_{index_col_suffix}"

    if colname not in subset.var:
        raise KeyError(
            f"index_col_suffix='{index_col_suffix}' requires var column '{colname}', "
            f"but it is not present in adata.var."
        )

    return _values_to_str_labels(subset.var[colname].values[sites])


def _ordered_columns(
    subset: "ad.AnnData", sites: np.ndarray, reference: str, index_col_suffix: str | None
) -> "tuple[np.ndarray, np.ndarray]":
    """Return (sites, labels) with sites reordered so the reindexed coordinate is ascending.

    Superset of ``_select_labels``: same parameters and label computation, plus
    an ordering step. When ``index_col_suffix`` is ``None``, ``sites``/labels
    are returned unchanged (identical to calling ``_select_labels`` alone --
    the existing, physical column order). When a per-reference-signed
    reindexed column is present (see ``reindex_references_adata``'s ``invert``
    parameter), sorting ``sites`` ascending by that column's value naturally
    reverses the rendered column order for inverted references, without ever
    touching the underlying AnnData array order -- an inverted reference's
    reindexed values are already negated, so ascending-value order is
    descending-``var_names`` order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(sites, labels)`` -- use the returned ``sites`` (not the original
        argument) to index the data matrix so plotted columns match the
        returned ``labels``.
    """
    labels = _select_labels(subset, sites, reference, index_col_suffix)
    if sites.size == 0 or index_col_suffix is None:
        return sites, labels

    colname = f"{reference}_{index_col_suffix}"
    values = np.asarray(subset.var[colname].values, dtype=float)[sites]
    order = np.argsort(values, kind="stable")
    return sites[order], labels[order]


def subsample_reads_for_plot(subset, max_reads: int | None, *, seed: int = 0):
    """Randomly subsample an AnnData's reads to at most ``max_reads`` rows.

    A clustermap of a (reference, sample) group with hundreds of thousands of
    reads is both illegible (rows sub-pixel) and memory-heavy to render -- and
    the per-reference materialize + per-worker pickle that feeds it scales with
    the read count. Capping reads per plot bounds all three. Returns ``subset``
    unchanged when ``max_reads`` is falsy/non-positive or the group already fits;
    otherwise returns a copy of a random subset. The draw is reproducible (fixed
    ``seed``) and the kept rows are returned in their original order, so the plot
    is stable across re-runs and downstream row ordering/clustering is unaffected
    by selection order.
    """
    if max_reads is None or max_reads <= 0 or subset.n_obs <= int(max_reads):
        return subset
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(subset.n_obs, size=int(max_reads), replace=False))
    return subset[chosen, :].copy()


def subsample_read_ids(read_ids: Sequence[str], max_reads: int | None, *, seed: int = 0) -> list:
    """Randomly subsample a list of read IDs to at most ``max_reads``, list-level.

    Companion to ``subsample_reads_for_plot`` for callers that build up a read
    selection *before* materializing an AnnData -- a reduce-phase plot that
    materializes a whole reference's/region's reads (every barcode combined)
    just to subsample per-(reference, sample) group afterward defeats the
    point: capping each group's read IDs up front bounds the materialize
    itself instead of only the final plot (see dev/pipeline_scaling_audit.md,
    finding E). Same reproducible, order-preserving semantics as
    ``subsample_reads_for_plot``; call once per group (e.g. per barcode) so
    each group is independently capped rather than the combined pool.
    """
    read_ids = list(read_ids)
    if max_reads is None or max_reads <= 0 or len(read_ids) <= int(max_reads):
        return read_ids
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(read_ids), size=int(max_reads), replace=False))
    return [read_ids[i] for i in chosen]


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


def _layer_to_numpy_np(
    arr: np.ndarray,
    sites: np.ndarray | None = None,
    *,
    fill_nan_strategy: str = "value",
    fill_nan_value: float = -1,
) -> np.ndarray:
    """Numpy-only equivalent of ``_layer_to_numpy`` for use in worker processes.

    Operates directly on a pre-extracted numpy array rather than an AnnData object,
    so it is safe to call in ``ProcessPoolExecutor`` workers.
    """
    if sites is not None:
        arr = arr[:, sites]
    arr = np.array(arr, copy=True).astype(float)

    if fill_nan_strategy == "none":
        return arr
    if fill_nan_strategy not in {"value", "col_mean"}:
        raise ValueError("fill_nan_strategy must be 'none', 'value', or 'col_mean'.")
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
    logger.debug("Formatting barplot '%s' with %s values.", title, len(mean_values))
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
    from smftools.optional_imports import require

    sns = require("seaborn", extra="plotting", purpose="plot styling")
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
