from typing import Optional


def plot_spatial_autocorr_grid(
    adata,
    out_dir: str,
    site_types=("GpC", "CpG", "any_C"),
    sample_col: str = "Sample",
    reference_col: str = "Reference_strand",
    window: int = 25,
    rows_per_fig: int = 6,
    dpi: int = 160,
    filename_prefix: str = "autocorr_grid",
    include_combined_column: bool = True,
    references: Optional[list] = None,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    site_types = list(site_types)

    def _rolling_1d(arr: np.ndarray, win: int) -> np.ndarray:
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

    def _compute_group_summary_for_mask(site: str, mask: np.ndarray):
        obsm_key = f"{site}_spatial_autocorr"
        lags_key = f"{site}_spatial_autocorr_lags"
        if obsm_key not in adata.obsm or lags_key not in adata.uns:
            return None, None, None
        mat = np.asarray(adata.obsm[obsm_key])
        if mat.size == 0:
            return None, None, None
        sel = mat[mask, :]
        if sel.size == 0:
            return None, None, None
        mean_per_lag = np.nanmean(sel, axis=0)
        std_per_lag = np.nanstd(sel, axis=0, ddof=1)
        return np.asarray(adata.uns[lags_key]), _rolling_1d(mean_per_lag, window), _rolling_1d(std_per_lag, window)

    # samples
    if sample_col not in adata.obs:
        raise KeyError(f"sample_col '{sample_col}' not present in adata.obs")
    samples = adata.obs[sample_col]
    if not pd.api.types.is_categorical_dtype(samples):
        samples = samples.astype("category")
    sample_levels = list(samples.cat.categories)

    # references
    if reference_col not in adata.obs:
        raise KeyError(f"reference_col '{reference_col}' not present in adata.obs")
    if references is None:
        refs_series = adata.obs[reference_col]
        if not pd.api.types.is_categorical_dtype(refs_series):
            refs_series = refs_series.astype("category")
        references = list(refs_series.cat.categories)
    references = list(references)

    # Build column meta
    group_column_meta = []
    for site in site_types:
        cols = []
        if include_combined_column:
            cols.append(("all", None))
        for r in references:
            cols.append(("ref", r))
        group_column_meta.append((site, cols))

    ncols = sum(len(cols) for _, cols in group_column_meta)

    saved_pages = []
    for start_idx in range(0, len(sample_levels), rows_per_fig):
        chunk = sample_levels[start_idx : start_idx + rows_per_fig]
        nrows = len(chunk)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(4.2 * ncols, 2.4 * nrows),
            dpi=dpi,
            squeeze=False,
        )

        col_idx = 0
        for site, cols in group_column_meta:
            for col_kind, col_val in cols:
                for r, sample_name in enumerate(chunk):
                    ax = axes[r, col_idx]

                    # compute masks and read count
                    sample_mask = (adata.obs[sample_col].values == sample_name)
                    if col_kind == "ref":
                        ref_mask = (adata.obs[reference_col].values == col_val)
                        mask = sample_mask & ref_mask
                    else:
                        mask = sample_mask
                    n_reads_grp = int(mask.sum())

                    lags, mean_curve, std_curve = _compute_group_summary_for_mask(site, mask)

                    # Title for top row
                    if r == 0:
                        title = f"{site} (all refs)" if col_kind == "all" else f"{site} [{col_val}]"
                        ax.set_title(title, fontsize=9)

                    if lags is None:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                        ax.set_xlim(0, 1)
                    else:
                        ax.plot(lags, mean_curve, lw=1.1)
                        upper = mean_curve + std_curve
                        lower = mean_curve - std_curve
                        ax.fill_between(lags, lower, upper, alpha=0.2)
                        finite_mask = np.isfinite(lags)
                        if finite_mask.any():
                            ax.set_xlim(float(np.nanmin(lags[finite_mask])), float(np.nanmax(lags[finite_mask])))

                    ax.set_xlabel("Lag (bp)", fontsize=7)
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.grid(True, alpha=0.22)

                col_idx += 1

        # tighten layout then place one centered vertical label per row
        fig.tight_layout(rect=[0.06, 0, 1, 0.97])  # left margin leaves space for the row labels
        # compute and add row labels using axes positions (centers)
        for r, sample_name in enumerate(chunk):
            # use the center y of the first axis in the row as anchor
            first_ax = axes[r, 0]
            pos = first_ax.get_position()  # Bbox in figure coords
            ycenter = pos.y0 + pos.height / 2.0
            # optionally include read count
            n_reads_grp = int((adata.obs[sample_col].values == sample_name).sum())
            label = f"{sample_name}\n(n={n_reads_grp})"
            fig.text(0.02, ycenter, label, va='center', ha='left', rotation='vertical', fontsize=9)

        fig.suptitle("Spatial autocorrelation by sample × (site_type × reference)", y=0.995, fontsize=11)

        page_idx = start_idx // rows_per_fig + 1
        out_png = os.path.join(out_dir, f"{filename_prefix}_page{page_idx}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        saved_pages.append(out_png)

    return saved_pages

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_spatial_autocorr_grid(
#     adata,
#     out_dir,
#     site_types=("GpC", "CpG", "any_C"),
#     sample_col="Sample",
#     # plotting
#     window=25,
#     rows_per_fig=6,
#     dpi=160,
#     filename_prefix="autocorr_grid",
# ):
#     """
#     Plot spatial autocorrelation (rolling mean ± std) in a grid:
#       rows = samples, columns = site types.

#     Optionally precomputes autocorrelation matrices into:
#       - adata.obsm[f"{site_type}_spatial_autocorr"]
#       - adata.uns[f"{site_type}_spatial_autocorr_lags"]

#     Parameters
#     ----------
#     adata : AnnData
#         AnnData with layers like '{site}_site_binary' (0/1/NaN per read×position),
#         and obs[sample_col] labeling samples.
#     out_dir : str
#         Root directory; figures will go under f"{out_dir}".
#     site_types : tuple[str]
#         Which site types to include as columns.
#     sample_col : str
#         obs column with sample labels (categorical recommended).
#     window : int
#         Rolling window over lags for smoothing (set 1 to disable).
#     rows_per_fig : int
#         Number of samples per page (pagination).
#     dpi : int
#         Figure resolution.
#     filename_prefix : str
#         Prefix for output PNGs.

#     Returns
#     -------
#     list[str]
#         Paths to the saved PNGs.
#     """

#     # --- helpers ---
#     def _rolling_1d(arr, win):
#         if win <= 1:
#             return arr
#         valid = np.isfinite(arr).astype(float)
#         arr_z = np.nan_to_num(arr, nan=0.0)
#         k = np.ones(win, dtype=float)
#         num = np.convolve(arr_z, k, mode="same")
#         den = np.convolve(valid, k, mode="same")
#         with np.errstate(invalid="ignore", divide="ignore"):
#             out = num / den
#         out[den == 0] = np.nan
#         return out

#     def _compute_group_summary(adata, site_type, mask):
#         mat = adata.obsm[f"{site_type}_spatial_autocorr"][mask]
#         if mat.size == 0:
#             return None, None, None
#         mean_per_lag = np.nanmean(mat, axis=0)
#         std_per_lag  = np.nanstd(mat, axis=0, ddof=1)
#         mean_s = _rolling_1d(mean_per_lag, window)
#         std_s  = _rolling_1d(std_per_lag, window)
#         lags = adata.uns[f"{site_type}_spatial_autocorr_lags"]
#         return lags, mean_s, std_s

#     # --- samples (categorical preferred) ---
#     samples = adata.obs[sample_col]
#     if not pd.api.types.is_categorical_dtype(samples):
#         samples = samples.astype("category")
#     sample_levels = list(samples.cat.categories)

#     # --- plotting (paginated) ---
#     saved = []
#     for start in range(0, len(sample_levels), rows_per_fig):
#         chunk = sample_levels[start:start + rows_per_fig]
#         nrows, ncols = len(chunk), len(site_types)
#         fig, axes = plt.subplots(
#             nrows=nrows, ncols=ncols,
#             figsize=(4.8 * ncols, 2.8 * nrows),
#             dpi=dpi, sharex="col", sharey="col"
#         )
#         if nrows == 1 and ncols == 1:
#             axes = np.array([[axes]])
#         elif nrows == 1:
#             axes = axes[np.newaxis, :]
#         elif ncols == 1:
#             axes = axes[:, np.newaxis]

#         for r, sample_name in enumerate(chunk):
#             mask = (adata.obs[sample_col].values == sample_name)
#             n_reads_grp = int(mask.sum())
#             for c, site in enumerate(site_types):
#                 ax = axes[r, c]
#                 lags, mean_curve, std_curve = _compute_group_summary(adata, site, mask)
#                 if lags is None:
#                     ax.text(0.5, 0.5, "No reads", ha="center", va="center")
#                 else:
#                     ax.plot(lags, mean_curve, lw=1.5)
#                     upper = mean_curve + std_curve
#                     lower = mean_curve - std_curve
#                     ax.fill_between(lags, lower, upper, alpha=0.25)
#                 if r == 0:
#                     ax.set_title(site)
#                 if c == 0:
#                     ax.set_ylabel(f"{sample_name}\n(n={n_reads_grp})")
#                 if r == nrows - 1:
#                     ax.set_xlabel("Lag (bp)")
#                 ax.grid(True, alpha=0.25)

#         fig.suptitle("Spatial autocorrelation (rolling mean ± std) by sample × site type",
#                      y=0.995, fontsize=12)
#         fig.tight_layout(rect=[0, 0, 1, 0.98])

#         page_idx = start // rows_per_fig + 1
#         out_png = os.path.join(out_dir, f"{filename_prefix}_page{page_idx}.png")
#         plt.savefig(out_png, bbox_inches="tight")
#         plt.close(fig)
#         saved.append(out_png)

#     return saved
