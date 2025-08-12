import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_spatial_autocorr_grid(
    adata,
    out_dir,
    site_types=("GpC", "CpG", "any_C"),
    sample_col="Sample",
    # plotting
    window=25,
    rows_per_fig=6,
    dpi=160,
    filename_prefix="autocorr_grid",
):
    """
    Plot spatial autocorrelation (rolling mean ± std) in a grid:
      rows = samples, columns = site types.

    Optionally precomputes autocorrelation matrices into:
      - adata.obsm[f"{site_type}_spatial_autocorr"]
      - adata.uns[f"{site_type}_spatial_autocorr_lags"]

    Parameters
    ----------
    adata : AnnData
        AnnData with layers like '{site}_site_binary' (0/1/NaN per read×position),
        and obs[sample_col] labeling samples.
    out_dir : str
        Root directory; figures will go under f"{out_dir}".
    site_types : tuple[str]
        Which site types to include as columns.
    sample_col : str
        obs column with sample labels (categorical recommended).
    window : int
        Rolling window over lags for smoothing (set 1 to disable).
    rows_per_fig : int
        Number of samples per page (pagination).
    dpi : int
        Figure resolution.
    filename_prefix : str
        Prefix for output PNGs.

    Returns
    -------
    list[str]
        Paths to the saved PNGs.
    """

    # --- helpers ---
    def _rolling_1d(arr, win):
        if win <= 1:
            return arr
        valid = np.isfinite(arr).astype(float)
        arr_z = np.nan_to_num(arr, nan=0.0)
        k = np.ones(win, dtype=float)
        num = np.convolve(arr_z, k, mode="same")
        den = np.convolve(valid, k, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[den == 0] = np.nan
        return out

    def _compute_group_summary(adata, site_type, mask):
        mat = adata.obsm[f"{site_type}_spatial_autocorr"][mask]
        if mat.size == 0:
            return None, None, None
        mean_per_lag = np.nanmean(mat, axis=0)
        std_per_lag  = np.nanstd(mat, axis=0, ddof=1)
        mean_s = _rolling_1d(mean_per_lag, window)
        std_s  = _rolling_1d(std_per_lag, window)
        lags = adata.uns[f"{site_type}_spatial_autocorr_lags"]
        return lags, mean_s, std_s

    # --- samples (categorical preferred) ---
    samples = adata.obs[sample_col]
    if not pd.api.types.is_categorical_dtype(samples):
        samples = samples.astype("category")
    sample_levels = list(samples.cat.categories)

    # --- plotting (paginated) ---
    saved = []
    for start in range(0, len(sample_levels), rows_per_fig):
        chunk = sample_levels[start:start + rows_per_fig]
        nrows, ncols = len(chunk), len(site_types)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(4.8 * ncols, 2.8 * nrows),
            dpi=dpi, sharex="col", sharey="col"
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for r, sample_name in enumerate(chunk):
            mask = (adata.obs[sample_col].values == sample_name)
            n_reads_grp = int(mask.sum())
            for c, site in enumerate(site_types):
                ax = axes[r, c]
                lags, mean_curve, std_curve = _compute_group_summary(adata, site, mask)
                if lags is None:
                    ax.text(0.5, 0.5, "No reads", ha="center", va="center")
                else:
                    ax.plot(lags, mean_curve, lw=1.5)
                    upper = mean_curve + std_curve
                    lower = mean_curve - std_curve
                    ax.fill_between(lags, lower, upper, alpha=0.25)
                if r == 0:
                    ax.set_title(site)
                if c == 0:
                    ax.set_ylabel(f"{sample_name}\n(n={n_reads_grp})")
                if r == nrows - 1:
                    ax.set_xlabel("Lag (bp)")
                ax.grid(True, alpha=0.25)

        fig.suptitle("Spatial autocorrelation (rolling mean ± std) by sample × site type",
                     y=0.995, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        page_idx = start // rows_per_fig + 1
        out_png = os.path.join(out_dir, f"{filename_prefix}_page{page_idx}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_png)

    return saved
