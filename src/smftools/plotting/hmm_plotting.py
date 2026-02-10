from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger
from smftools.optional_imports import require
from smftools.plotting.plotting_utils import (
    _layer_to_numpy,
    _methylation_fraction_for_layer,
    _select_labels,
    clean_barplot,
    normalized_mean,
)

gridspec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
patches = require("matplotlib.patches", extra="plotting", purpose="plot rendering")
sns = require("seaborn", extra="plotting", purpose="plot styling")

plt = require("matplotlib.pyplot", extra="plotting", purpose="HMM plots")
mpl_colors = require("matplotlib.colors", extra="plotting", purpose="HMM plots")
pdf_backend = require(
    "matplotlib.backends.backend_pdf",
    extra="plotting",
    purpose="PDF output",
)
PdfPages = pdf_backend.PdfPages

logger = get_logger(__name__)


def _local_sites_to_global_indices(
    adata,
    subset,
    local_sites: np.ndarray,
) -> np.ndarray:
    """Translate subset-local column indices into global ``adata.var`` indices."""
    local_sites = np.asarray(local_sites, dtype=int)
    if local_sites.size == 0:
        return local_sites
    subset_to_global = adata.var_names.get_indexer(subset.var_names)
    global_sites = subset_to_global[local_sites]
    if np.any(global_sites < 0):
        missing = int(np.sum(global_sites < 0))
        logger.warning(
            "Could not map %d plotted positions back to full var index; skipping those points.",
            missing,
        )
        global_sites = global_sites[global_sites >= 0]
    return global_sites


def _overlay_variant_calls_on_panels(
    adata,
    reference: str,
    ordered_obs_names: list,
    panels_with_indices: list,
    seq1_color: str = "white",
    seq2_color: str = "black",
    marker_size: float = 4.0,
) -> bool:
    """
    Overlay variant call circles on heatmap panels using nearest-neighbor mapping.

    This function maps variant call column indices to the nearest displayed column
    in each panel, using var index space for mapping. This handles both regular
    var_names and reindexed label coordinates.

    Parameters
    ----------
    adata : AnnData
        The AnnData containing variant call layers (should be full adata, not subset).
    reference : str
        Reference name used to auto-detect the variant call layer.
    ordered_obs_names : list
        Obs names in display order (rows of the heatmap).
    panels_with_indices : list of (ax, site_indices)
        Each entry is a matplotlib axes and the var indices for that panel's columns.
    seq1_color, seq2_color : str
        Colors for seq1 (value 1) and seq2 (value 2) variant calls.
    marker_size : float
        Size of the circle markers.

    Returns
    -------
    bool
        True if overlay was applied to at least one panel.
    """
    # Auto-detect variant call layer - find any layer ending with _variant_call
    vc_layer_key = None
    for key in adata.layers:
        if key.endswith("_variant_call"):
            vc_layer_key = key
            break

    if vc_layer_key is None:
        return False

    # Build row index mapping
    obs_name_to_idx = {str(name): i for i, name in enumerate(adata.obs_names)}
    common_obs = [str(name) for name in ordered_obs_names if str(name) in obs_name_to_idx]
    if not common_obs:
        return False

    obs_idx = [obs_name_to_idx[name] for name in common_obs]
    row_index_map = {str(name): i for i, name in enumerate(ordered_obs_names)}
    heatmap_row_indices = np.array([row_index_map[name] for name in common_obs])

    # Get variant call matrix for the ordered obs
    vc_data = adata.layers[vc_layer_key]
    if hasattr(vc_data, "toarray"):
        vc_data = vc_data.toarray()
    vc_matrix = np.asarray(vc_data)[obs_idx, :]

    # Find columns with actual variant calls (value 1 or 2)
    has_calls = np.isin(vc_matrix, [1, 2]).any(axis=0)
    call_col_indices = np.where(has_calls)[0]

    if len(call_col_indices) == 0:
        return False

    call_sub = vc_matrix[:, call_col_indices]

    applied = False
    for ax, site_indices in panels_with_indices:
        site_indices = np.asarray(site_indices)
        if site_indices.size == 0:
            continue

        # Use nearest-neighbor mapping in var index space
        # site_indices are the var indices displayed in this panel (sorted by position)
        # call_col_indices are var indices where calls exist
        # Map each call to the nearest displayed site index

        # Sort site indices for searchsorted
        sorted_order = np.argsort(site_indices)
        sorted_sites = site_indices[sorted_order]

        # Find nearest site for each call
        insert_idx = np.searchsorted(sorted_sites, call_col_indices)
        insert_idx = np.clip(insert_idx, 0, len(sorted_sites) - 1)
        left_idx = np.clip(insert_idx - 1, 0, len(sorted_sites) - 1)

        dist_right = np.abs(sorted_sites[insert_idx].astype(float) - call_col_indices.astype(float))
        dist_left = np.abs(sorted_sites[left_idx].astype(float) - call_col_indices.astype(float))
        nearest_sorted = np.where(dist_left < dist_right, left_idx, insert_idx)

        # Map back to original (unsorted) heatmap column positions
        nearest_heatmap_col = sorted_order[nearest_sorted]

        # Plot circles for each variant value
        for call_val, color in [(1, seq1_color), (2, seq2_color)]:
            local_rows, local_cols = np.where(call_sub == call_val)
            if len(local_rows) == 0:
                continue

            plot_y = heatmap_row_indices[local_rows]
            plot_x = nearest_heatmap_col[local_cols]

            ax.scatter(
                plot_x + 0.5,
                plot_y + 0.5,
                c=color,
                s=marker_size,
                marker="o",
                edgecolors="gray",
                linewidths=0.3,
                zorder=3,
            )
        applied = True

    if applied:
        logger.info("Overlaid variant calls from layer '%s'.", vc_layer_key)
    return applied


def plot_hmm_size_contours(
    adata,
    length_layer: str,
    sample_col: str,
    ref_obs_col: str,
    rows_per_page: int = 4,
    max_length_cap: Optional[int] = 1000,
    figsize_per_cell: Tuple[float, float] = (4.0, 2.5),
    cmap: str = "viridis",
    log_scale_z: bool = False,
    save_path: Optional[str] = None,
    save_pdf: bool = True,
    save_each_page: bool = False,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    feature_ranges: Optional[Sequence[Tuple[int, int, str]]] = None,
    zero_color: str = "#eee6d9",
    nan_color: str = "#D0D0D0",
    # ---------------- smoothing params ----------------
    smoothing_sigma: Optional[Union[float, Tuple[float, float]]] = None,
    normalize_after_smoothing: bool = True,
    use_scipy_if_available: bool = True,
):
    """
    Create contour/pcolormesh plots of P(length | position) using a length-encoded HMM layer.
    Optional Gaussian smoothing applied to the 2D probability grid before plotting.
    When feature_ranges is provided, each length row is assigned a base color based
    on the matching (min_len, max_len) range and the probability value modulates
    the color intensity.

    smoothing_sigma: None or 0 -> no smoothing.
        float -> same sigma applied to (length_axis, position_axis)
        (sigma_len, sigma_pos) -> separate sigmas.
    normalize_after_smoothing: if True, renormalize each position-column to sum to 1 after smoothing.

    Other args are the same as prior function.
    """
    feature_ranges = tuple(feature_ranges or ())
    logger.info("Plotting HMM size contours%s.", f" -> {save_path}" if save_path else "")

    def _resolve_length_color(length: int, fallback: str) -> Tuple[float, float, float, float]:
        for min_len, max_len, color in feature_ranges:
            if min_len <= length <= max_len:
                return mpl_colors.to_rgba(color)
        return mpl_colors.to_rgba(fallback)

    def _build_length_facecolors(
        Z_values: np.ndarray,
        lengths: np.ndarray,
        fallback_color: str,
        *,
        vmin_local: Optional[float],
        vmax_local: Optional[float],
    ) -> np.ndarray:
        zero_rgba = np.array(mpl_colors.to_rgba(zero_color))
        nan_rgba = np.array(mpl_colors.to_rgba(nan_color))
        base_colors = np.array(
            [_resolve_length_color(int(length), fallback_color) for length in lengths],
            dtype=float,
        )
        base_colors[:, 3] = 1.0

        scale = np.array(Z_values, copy=True, dtype=float)
        finite_mask = np.isfinite(scale)
        if not finite_mask.any():
            facecolors = np.zeros(scale.shape + (4,), dtype=float)
            facecolors[:] = nan_rgba
            return facecolors.reshape(-1, 4)

        vmin_use = np.nanmin(scale) if vmin_local is None else vmin_local
        vmax_use = np.nanmax(scale) if vmax_local is None else vmax_local
        denom = vmax_use - vmin_use
        if denom <= 0:
            norm = np.zeros_like(scale)
        else:
            norm = (scale - vmin_use) / denom
        norm = np.clip(norm, 0, 1)

        row_colors = base_colors[:, None, :]
        facecolors = zero_rgba + norm[..., None] * (row_colors - zero_rgba)
        facecolors[..., 3] = 1.0
        facecolors[~finite_mask] = nan_rgba
        return facecolors.reshape(-1, 4)

    # --- helper: gaussian smoothing (scipy fallback -> numpy separable conv) ---
    def _gaussian_1d_kernel(sigma: float, eps: float = 1e-12):
        """Build a normalized 1D Gaussian kernel."""
        if sigma <= 0 or sigma is None:
            return np.array([1.0], dtype=float)
        # choose kernel size = odd ~ 6*sigma (covers +/-3 sigma)
        radius = max(1, int(math.ceil(3.0 * float(sigma))))
        xs = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-(xs**2) / (2.0 * sigma**2))
        k_sum = k.sum()
        if k_sum <= eps:
            k = np.array([1.0], dtype=float)
            k_sum = 1.0
        return k / k_sum

    def _smooth_with_numpy_separable(
        Z: np.ndarray, sigma_len: float, sigma_pos: float
    ) -> np.ndarray:
        """Apply separable Gaussian smoothing with NumPy."""
        # Z shape: (n_lengths, n_positions)
        out = Z.copy()
        # smooth along length axis (axis=0)
        if sigma_len and sigma_len > 0:
            k_len = _gaussian_1d_kernel(sigma_len)
            # convolve each column
            out = np.apply_along_axis(
                lambda col: np.convolve(col, k_len, mode="same"), axis=0, arr=out
            )
        # smooth along position axis (axis=1)
        if sigma_pos and sigma_pos > 0:
            k_pos = _gaussian_1d_kernel(sigma_pos)
            out = np.apply_along_axis(
                lambda row: np.convolve(row, k_pos, mode="same"), axis=1, arr=out
            )
        return out

    # prefer scipy.ndimage if available (faster and better boundary handling)
    _have_scipy = False
    if use_scipy_if_available:
        try:
            from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter

            _have_scipy = True
        except Exception:
            _have_scipy = False

    def _smooth_Z(Z: np.ndarray, sigma_len: float, sigma_pos: float) -> np.ndarray:
        """Smooth a matrix using scipy if available or NumPy fallback."""
        if (sigma_len is None or sigma_len == 0) and (sigma_pos is None or sigma_pos == 0):
            return Z
        if _have_scipy:
            # scipy expects sigma sequence in axis order (axis=0 length, axis=1 pos)
            sigma_seq = (float(sigma_len or 0.0), float(sigma_pos or 0.0))
            return _scipy_gaussian_filter(Z, sigma=sigma_seq, mode="reflect")
        else:
            return _smooth_with_numpy_separable(Z, float(sigma_len or 0.0), float(sigma_pos or 0.0))

    # --- gather unique ordered labels ---
    samples = (
        list(adata.obs[sample_col].cat.categories)
        if getattr(adata.obs[sample_col], "dtype", None) == "category"
        else list(pd.Categorical(adata.obs[sample_col]).categories)
    )
    refs = (
        list(adata.obs[ref_obs_col].cat.categories)
        if getattr(adata.obs[ref_obs_col], "dtype", None) == "category"
        else list(pd.Categorical(adata.obs[ref_obs_col]).categories)
    )

    n_samples = len(samples)
    n_refs = len(refs)
    if n_samples == 0 or n_refs == 0:
        raise ValueError("No samples or references found for plotting.")

    # Try to get numeric coordinates for x axis; fallback to range indices
    try:
        coords = np.asarray(adata.var_names, dtype=int)
        x_ticks_is_positions = True
    except Exception:
        coords = np.arange(adata.shape[1], dtype=int)
        x_ticks_is_positions = False

    # helper to get dense layer array for subset
    def _get_layer_array(layer):
        """Convert a layer to a dense NumPy array."""
        arr = layer
        # sparse -> toarray
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return np.asarray(arr)

    # fetch the whole layer once (not necessary but helps)
    if length_layer not in adata.layers:
        raise KeyError(f"Layer {length_layer} not found in adata.layers")
    full_layer = _get_layer_array(adata.layers[length_layer])  # shape (n_obs, n_vars)

    # Precompute pages
    pages = math.ceil(n_samples / rows_per_page)
    figs = []

    # decide global max length to allocate y axis (cap to avoid huge memory)
    finite_lengths = full_layer[np.isfinite(full_layer) & (full_layer > 0)]
    observed_max_len = int(np.nanmax(finite_lengths)) if finite_lengths.size > 0 else 0
    if max_length_cap is None:
        max_len = observed_max_len
    else:
        max_len = min(int(max_length_cap), max(1, observed_max_len))
    if max_len < 1:
        max_len = 1

    # parse smoothing_sigma
    if smoothing_sigma is None or smoothing_sigma == 0:
        sigma_len, sigma_pos = 0.0, 0.0
    elif isinstance(smoothing_sigma, (int, float)):
        sigma_len = float(smoothing_sigma)
        sigma_pos = float(smoothing_sigma)
    else:
        sigma_len = float(smoothing_sigma[0])
        sigma_pos = float(smoothing_sigma[1])

    # iterate pages
    for p in range(pages):
        start_sample = p * rows_per_page
        end_sample = min(n_samples, (p + 1) * rows_per_page)
        page_samples = samples[start_sample:end_sample]
        rows_on_page = len(page_samples)

        fig_w = n_refs * figsize_per_cell[0]
        fig_h = rows_on_page * figsize_per_cell[1]
        fig, axes = plt.subplots(rows_on_page, n_refs, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f"HMM size contours (page {p + 1}/{pages})", fontsize=12)

        # for each panel compute p(length | position)
        for i_row, sample in enumerate(page_samples):
            for j_col, ref in enumerate(refs):
                ax = axes[i_row][j_col]
                panel_mask = (adata.obs[sample_col] == sample) & (adata.obs[ref_obs_col] == ref)
                if not panel_mask.any():
                    ax.text(0.5, 0.5, "no reads", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{sample} / {ref}")
                    continue

                row_idx = np.nonzero(
                    panel_mask.values if hasattr(panel_mask, "values") else np.asarray(panel_mask)
                )[0]
                if row_idx.size == 0:
                    ax.text(0.5, 0.5, "no reads", ha="center", va="center")
                    ax.set_title(f"{sample} / {ref}")
                    continue

                sub = full_layer[row_idx, :]  # (n_reads, n_positions)
                if sub.size == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    ax.set_title(f"{sample} / {ref}")
                    continue
                valid_lengths = sub[np.isfinite(sub) & (sub > 0)]
                if valid_lengths.size == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    ax.set_title(f"{sample} / {ref}")
                    continue

                # compute counts per length per position
                n_positions = sub.shape[1]
                max_len_local = int(valid_lengths.max()) if valid_lengths.size > 0 else 0
                max_len_here = min(max_len, max_len_local)

                lengths_range = np.arange(1, max_len_here + 1, dtype=int)
                Z = np.zeros(
                    (len(lengths_range), n_positions), dtype=float
                )  # rows=length, cols=pos

                # fill Z by efficient bincount across columns
                for j in range(n_positions):
                    col_vals = sub[:, j]
                    pos_vals = col_vals[np.isfinite(col_vals) & (col_vals > 0)].astype(int)
                    if pos_vals.size == 0:
                        continue
                    clipped = np.clip(pos_vals, 1, max_len_here)
                    counts = np.bincount(clipped, minlength=max_len_here + 1)[1:]
                    s = counts.sum()
                    if s > 0:
                        Z[:, j] = counts.astype(float)  # keep counts for smoothing

                # normalize per-column -> p(length | pos) BEFORE smoothing OR AFTER
                # We'll smooth counts and then optionally renormalize (normalize_after_smoothing controls)
                # Apply smoothing to Z (counts)
                if sigma_len > 0 or sigma_pos > 0:
                    Z = _smooth_Z(Z, sigma_len, sigma_pos)

                # normalize to conditional probability per column
                if normalize_after_smoothing:
                    col_sums = Z.sum(axis=0, keepdims=True)
                    # avoid divide-by-zero
                    col_sums[col_sums == 0] = 1.0
                    Z = Z / col_sums

                if log_scale_z:
                    Z_plot = np.log1p(Z)
                else:
                    Z_plot = Z

                # Build x and y grids for pcolormesh: x = coords (positions)
                x = coords[:n_positions]
                if n_positions >= 2:
                    dx = np.diff(x).mean()
                    x_edges = np.concatenate([x - dx / 2.0, [x[-1] + dx / 2.0]])
                else:
                    x_edges = np.array([x[0] - 0.5, x[0] + 0.5])

                y = lengths_range
                dy = 1.0
                y_edges = np.concatenate([y - 0.5, [y[-1] + 0.5]])

                if feature_ranges:
                    fallback_color = mpl_colors.to_rgba(plt.get_cmap(cmap)(1.0))
                    facecolors = _build_length_facecolors(
                        Z_plot,
                        lengths_range,
                        fallback_color,
                        vmin_local=vmin,
                        vmax_local=vmax,
                    )
                    pcm = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        Z_plot,
                        shading="auto",
                        vmin=vmin,
                        vmax=vmax,
                        facecolors=facecolors,
                    )
                else:
                    pcm = ax.pcolormesh(
                        x_edges, y_edges, Z_plot, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
                    )
                ax.set_title(f"{sample} / {ref}")
                ax.set_ylabel("length")
                if i_row == rows_on_page - 1:
                    ax.set_xlabel("position")
                else:
                    ax.set_xticklabels([])

        # colorbar
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        try:
            fig.colorbar(pcm, cax=cax)
        except Exception:
            pass

        figs.append(fig)

        # saving per page if requested
        if save_path is not None:
            import os

            os.makedirs(save_path, exist_ok=True)
            if save_each_page:
                fname = f"hmm_size_page_{p + 1:03d}.png"
                out = os.path.join(save_path, fname)
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
                logger.info("Saved HMM size contour page to %s.", out)

    # multipage PDF if requested
    if save_path is not None and save_pdf:
        pdf_file = os.path.join(save_path, "hmm_size_contours_pages.pdf")
        with PdfPages(pdf_file) as pp:
            for fig in figs:
                pp.savefig(fig, bbox_inches="tight")
        logger.info("Saved HMM size contour PDF to %s.", pdf_file)

    return figs


def _resolve_feature_color(cmap: Any) -> Tuple[float, float, float, float]:
    """Resolve a representative feature color from a colormap or color spec."""
    if isinstance(cmap, str):
        try:
            cmap_obj = plt.get_cmap(cmap)
            return mpl_colors.to_rgba(cmap_obj(1.0))
        except Exception:
            return mpl_colors.to_rgba(cmap)

    if isinstance(cmap, mpl_colors.Colormap):
        if hasattr(cmap, "colors") and cmap.colors:
            return mpl_colors.to_rgba(cmap.colors[-1])
        return mpl_colors.to_rgba(cmap(1.0))

    return mpl_colors.to_rgba("black")


def _build_hmm_feature_cmap(
    cmap: Any,
    *,
    zero_color: str = "#eee6d9",
    nan_color: str = "#D0D0D0",
) -> mpl_colors.Colormap:
    """Build a two-color HMM colormap with explicit NaN/under handling."""
    feature_color = _resolve_feature_color(cmap)
    hmm_cmap = mpl_colors.LinearSegmentedColormap.from_list(
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
    zero_color: str = "#eee6d9",
    nan_color: str = "#D0D0D0",
) -> Tuple[mpl_colors.Colormap, mpl_colors.BoundaryNorm]:
    """Build a discrete colormap and norm for length-based subclasses."""
    color_list = [zero_color] + [color for _, _, color in feature_ranges]
    cmap = mpl_colors.ListedColormap(color_list, name="hmm_length_feature_cmap")
    cmap.set_bad(nan_color)
    bounds = np.arange(-0.5, len(color_list) + 0.5, 1)
    norm = mpl_colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


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
    overlay_variant_calls: bool = False,
    variant_overlay_seq1_color: str = "white",
    variant_overlay_seq2_color: str = "black",
    variant_overlay_marker_size: float = 4.0,
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
                ordered_obs_names = []

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
                    ordered_obs_names.extend(sb.obs_names.tolist())

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
                    f"{sample} — {ref} — {total_reads} reads ({signal_type})", fontsize=18, y=0.98
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
                    axes_heat[i].set_xticklabels(xtick_labels, rotation=90, fontsize=10)

                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=1.2)

                if overlay_variant_calls and ordered_obs_names:
                    try:
                        # Map panel sites from subset-local coordinates to full adata indices
                        hmm_sites_global = _local_sites_to_global_indices(adata, subset, hmm_sites)
                        any_c_sites_global = _local_sites_to_global_indices(
                            adata, subset, any_c_sites
                        )
                        gpc_sites_global = _local_sites_to_global_indices(adata, subset, gpc_sites)
                        cpg_sites_global = _local_sites_to_global_indices(adata, subset, cpg_sites)
                        any_a_sites_global = _local_sites_to_global_indices(
                            adata, subset, any_a_sites
                        )

                        # Build panels_with_indices using site indices for each panel
                        # Map panel names to their site index arrays
                        name_to_sites = {
                            f"HMM - {hmm_feature_layer}": hmm_sites_global,
                            "C": any_c_sites_global,
                            "GpC": gpc_sites_global,
                            "CpG": cpg_sites_global,
                            "A": any_a_sites_global,
                        }
                        panels_with_indices = []
                        for idx, (name, *_rest) in enumerate(panels):
                            sites = name_to_sites.get(name)
                            if sites is not None and len(sites) > 0:
                                panels_with_indices.append((axes_heat[idx], sites))
                        if panels_with_indices:
                            _overlay_variant_calls_on_panels(
                                adata,
                                ref,
                                ordered_obs_names,
                                panels_with_indices,
                                seq1_color=variant_overlay_seq1_color,
                                seq2_color=variant_overlay_seq2_color,
                                marker_size=variant_overlay_marker_size,
                            )
                    except Exception as overlay_err:
                        logger.warning(
                            "Variant overlay failed for %s - %s: %s", sample, ref, overlay_err
                        )

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
    overlay_variant_calls: bool = False,
    variant_overlay_seq1_color: str = "white",
    variant_overlay_seq2_color: str = "black",
    variant_overlay_marker_size: float = 4.0,
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
                ordered_obs_names = []

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
                    ordered_obs_names.extend(sb.obs_names.tolist())

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

                feature_class_label = None
                lower_layer = str(length_layer).lower()
                if "accessible" in lower_layer:
                    feature_class_label = "Accessible Patch Sizes"
                    hmm_title = "HMM Accessible Patches"
                elif "footprint" in lower_layer:
                    feature_class_label = "Footprint Sizes"
                    hmm_title = "HMM Footprints"

                if feature_ranges:
                    length_plot_matrix = _map_length_matrix_to_subclasses(
                        length_matrix_raw, feature_ranges
                    )
                    length_plot_cmap, length_plot_norm = _build_length_feature_cmap(feature_ranges)
                    legend_handles = []
                    legend_labels = []
                    for min_len, max_len, color in feature_ranges:
                        if max_len >= int(1e9):
                            label = f">= {min_len} bp"
                        else:
                            label = f"{min_len}-{max_len} bp"
                        legend_handles.append(patches.Patch(facecolor=color, edgecolor="none"))
                        legend_labels.append(label)

                panels = [
                    (
                        f"HMM - {hmm_title}",
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
                    f"{sample} — {ref} — {total_reads} reads ({signal_type})", fontsize=18, y=0.98
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
                    axes_heat[i].set_xticklabels(xtick_labels, rotation=90, fontsize=10)

                    for boundary in bin_boundaries[:-1]:
                        axes_heat[i].axhline(y=boundary, color="black", linewidth=1.2)

                    if feature_ranges and name.startswith("HMM -") and legend_handles:
                        axes_heat[i].legend(
                            legend_handles,
                            legend_labels,
                            fontsize=12,
                            loc="center left",
                            bbox_to_anchor=(-0.85, 0.65),
                            borderaxespad=0.0,
                            frameon=False,
                            title=feature_class_label,
                            title_fontsize=14,
                        )

                if overlay_variant_calls and ordered_obs_names:
                    try:
                        # Map panel sites from subset-local coordinates to full adata indices
                        length_sites_global = _local_sites_to_global_indices(
                            adata, subset, length_sites
                        )
                        any_c_sites_global = _local_sites_to_global_indices(
                            adata, subset, any_c_sites
                        )
                        gpc_sites_global = _local_sites_to_global_indices(adata, subset, gpc_sites)
                        cpg_sites_global = _local_sites_to_global_indices(adata, subset, cpg_sites)
                        any_a_sites_global = _local_sites_to_global_indices(
                            adata, subset, any_a_sites
                        )

                        # Build panels_with_indices using site indices for each panel
                        name_to_sites = {
                            f"HMM - {length_layer}": length_sites_global,
                            "C": any_c_sites_global,
                            "GpC": gpc_sites_global,
                            "CpG": cpg_sites_global,
                            "A": any_a_sites_global,
                        }
                        panels_with_indices = []
                        for idx, (name, *_rest) in enumerate(panels):
                            sites = name_to_sites.get(name)
                            if sites is not None and len(sites) > 0:
                                panels_with_indices.append((axes_heat[idx], sites))
                        if panels_with_indices:
                            _overlay_variant_calls_on_panels(
                                adata,
                                ref,
                                ordered_obs_names,
                                panels_with_indices,
                                seq1_color=variant_overlay_seq1_color,
                                seq2_color=variant_overlay_seq2_color,
                                marker_size=variant_overlay_marker_size,
                            )
                    except Exception as overlay_err:
                        logger.warning(
                            "Variant overlay failed for %s - %s: %s", sample, ref, overlay_err
                        )

                if feature_ranges:
                    fig.subplots_adjust(left=0.18)
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
    logger.info("Plotting rolling HMM layers by sample/ref.")

    if sample_col not in adata.obs.columns or ref_col not in adata.obs.columns:
        raise ValueError(
            f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs"
        )

    if samples is None:
        sseries = adata.obs[sample_col]
        if not isinstance(sseries.dtype, pd.CategoricalDtype):
            sseries = sseries.astype("category")
        samples_all = list(sseries.cat.categories)
    else:
        samples_all = list(samples)

    if references is None:
        rseries = adata.obs[ref_col]
        if not isinstance(rseries.dtype, pd.CategoricalDtype):
            rseries = rseries.astype("category")
        refs_all = list(rseries.cat.categories)
    else:
        refs_all = list(references)

    if layers is None:
        layers = list(adata.layers.keys())
        if len(layers) == 0:
            raise ValueError(
                "No adata.layers found. Please pass `layers=[...]` of the HMM layers to plot."
            )
    layers = list(layers)

    x_labels = None
    try:
        if use_var_coords:
            x_coords = np.array([int(v) for v in adata.var_names])
        else:
            raise Exception("user disabled var coords")
    except Exception:
        x_coords = np.arange(adata.shape[1], dtype=int)
        x_labels = adata.var_names.astype(str).tolist()

    ref_reindexed_cols = {
        ref: f"{ref}_{reindexed_var_suffix}"
        for ref in refs_all
        if f"{ref}_{reindexed_var_suffix}" in adata.var
    }

    if save:
        outdir = output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None

    n_samples = len(samples_all)
    n_refs = len(refs_all)
    total_pages = math.ceil(n_samples / rows_per_page)
    saved_files = []

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
                        if layer == layers[0] and getattr(sub, "X", None) is not None:
                            mat = sub.X
                        else:
                            continue

                    if hasattr(mat, "toarray"):
                        try:
                            arr = mat.toarray()
                        except Exception:
                            arr = np.asarray(mat)
                    else:
                        arr = np.asarray(mat)

                    if arr.size == 0 or arr.shape[1] == 0:
                        continue

                    arr = arr.astype(float)
                    with np.errstate(all="ignore"):
                        col_mean = np.nanmean(arr, axis=0)

                    if np.all(np.isnan(col_mean)):
                        continue

                    valid_mask = np.isfinite(col_mean)

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

                    L = len(col_mean)
                    x = ref_coords[:L]

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
            logger.info("Saved HMM layers rolling plot to %s.", fname)
        else:
            plt.show()
        plt.close(fig)

    return saved_files
