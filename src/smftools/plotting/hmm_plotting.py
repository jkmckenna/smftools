import math
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    # ---------------- smoothing params ----------------
    smoothing_sigma: Optional[Union[float, Tuple[float, float]]] = None,
    normalize_after_smoothing: bool = True,
    use_scipy_if_available: bool = True,
):
    """
    Create contour/pcolormesh plots of P(length | position) using a length-encoded HMM layer.
    Optional Gaussian smoothing applied to the 2D probability grid before plotting.

    smoothing_sigma: None or 0 -> no smoothing.
        float -> same sigma applied to (length_axis, position_axis)
        (sigma_len, sigma_pos) -> separate sigmas.
    normalize_after_smoothing: if True, renormalize each position-column to sum to 1 after smoothing.

    Other args are the same as prior function.
    """
    # --- helper: gaussian smoothing (scipy fallback -> numpy separable conv) ---
    def _gaussian_1d_kernel(sigma: float, eps: float = 1e-12):
        if sigma <= 0 or sigma is None:
            return np.array([1.0], dtype=float)
        # choose kernel size = odd ~ 6*sigma (covers +/-3 sigma)
        radius = max(1, int(math.ceil(3.0 * float(sigma))))
        xs = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
        k_sum = k.sum()
        if k_sum <= eps:
            k = np.array([1.0], dtype=float)
            k_sum = 1.0
        return k / k_sum

    def _smooth_with_numpy_separable(Z: np.ndarray, sigma_len: float, sigma_pos: float) -> np.ndarray:
        # Z shape: (n_lengths, n_positions)
        out = Z.copy()
        # smooth along length axis (axis=0)
        if sigma_len and sigma_len > 0:
            k_len = _gaussian_1d_kernel(sigma_len)
            # convolve each column
            out = np.apply_along_axis(lambda col: np.convolve(col, k_len, mode="same"), axis=0, arr=out)
        # smooth along position axis (axis=1)
        if sigma_pos and sigma_pos > 0:
            k_pos = _gaussian_1d_kernel(sigma_pos)
            out = np.apply_along_axis(lambda row: np.convolve(row, k_pos, mode="same"), axis=1, arr=out)
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
        if (sigma_len is None or sigma_len == 0) and (sigma_pos is None or sigma_pos == 0):
            return Z
        if _have_scipy:
            # scipy expects sigma sequence in axis order (axis=0 length, axis=1 pos)
            sigma_seq = (float(sigma_len or 0.0), float(sigma_pos or 0.0))
            return _scipy_gaussian_filter(Z, sigma=sigma_seq, mode="reflect")
        else:
            return _smooth_with_numpy_separable(Z, float(sigma_len or 0.0), float(sigma_pos or 0.0))

    # --- gather unique ordered labels ---
    samples = list(adata.obs[sample_col].cat.categories) if getattr(adata.obs[sample_col], "dtype", None) == "category" else list(pd.Categorical(adata.obs[sample_col]).categories)
    refs = list(adata.obs[ref_obs_col].cat.categories) if getattr(adata.obs[ref_obs_col], "dtype", None) == "category" else list(pd.Categorical(adata.obs[ref_obs_col]).categories)

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
    observed_max_len = int(np.max(full_layer)) if full_layer.size > 0 else 0
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
        fig.suptitle(f"HMM size contours (page {p+1}/{pages})", fontsize=12)

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

                row_idx = np.nonzero(panel_mask.values if hasattr(panel_mask, "values") else np.asarray(panel_mask))[0]
                if row_idx.size == 0:
                    ax.text(0.5, 0.5, "no reads", ha="center", va="center")
                    ax.set_title(f"{sample} / {ref}")
                    continue

                sub = full_layer[row_idx, :]  # (n_reads, n_positions)
                if sub.size == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    ax.set_title(f"{sample} / {ref}")
                    continue

                # compute counts per length per position
                n_positions = sub.shape[1]
                max_len_local = int(sub.max()) if sub.size > 0 else 0
                max_len_here = min(max_len, max_len_local)

                lengths_range = np.arange(1, max_len_here + 1, dtype=int)
                Z = np.zeros((len(lengths_range), n_positions), dtype=float)  # rows=length, cols=pos

                # fill Z by efficient bincount across columns
                for j in range(n_positions):
                    col_vals = sub[:, j]
                    pos_vals = col_vals[col_vals > 0].astype(int)
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

                pcm = ax.pcolormesh(x_edges, y_edges, Z_plot, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
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
                fname = f"hmm_size_page_{p+1:03d}.png"
                out = os.path.join(save_path, fname)
                fig.savefig(out, dpi=dpi, bbox_inches="tight")

    # multipage PDF if requested
    if save_path is not None and save_pdf:
        pdf_file = os.path.join(save_path, "hmm_size_contours_pages.pdf")
        with PdfPages(pdf_file) as pp:
            for fig in figs:
                pp.savefig(fig, bbox_inches="tight")
        print(f"Saved multipage PDF: {pdf_file}")

    return figs
