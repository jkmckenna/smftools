import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_read_qc_histograms(
    adata,
    outdir,
    obs_keys,
    sample_key,
    bins=60,
    clip_quantiles=(0.0, 0.995),
    min_non_nan=10,
    rows_per_fig=6,
    topn_categories=15,
    figsize_cell=(3.6, 2.6),
    dpi=150,
):
    """
    Plot a grid of QC histograms: rows = samples (from `sample_key`), columns = `obs_keys`.

    Numeric columns -> histogram per sample.
    Categorical columns -> bar chart of top categories per sample.

    Saves paginated PNGs to `outdir`.

    Parameters
    ----------
    adata : AnnData
    outdir : str
    obs_keys : list[str]
    sample_key : str
        Column in adata.obs defining rows (samples/barcodes).
    bins : int
        Histogram bins for numeric metrics.
    clip_quantiles : tuple or None
        Clip numeric data globally per metric for consistent axes, e.g. (0.0, 0.995).
    min_non_nan : int
        Minimum finite values to plot a panel.
    rows_per_fig : int
        Number of samples per page.
    topn_categories : int
        For categorical metrics, show top-N categories (per sample).
    figsize_cell : (float, float)
        Size of each subplot cell (width, height).
    dpi : int
        Figure resolution.
    """
    os.makedirs(outdir, exist_ok=True)

    if sample_key not in adata.obs.columns:
        raise KeyError(f"'{sample_key}' not found in adata.obs")

    # Ensure sample_key is categorical for stable ordering
    samples = adata.obs[sample_key]
    if not pd.api.types.is_categorical_dtype(samples):
        samples = samples.astype("category")
    sample_levels = list(samples.cat.categories)

    # Validate keys, and classify numeric vs categorical
    valid_keys = []
    is_numeric = {}
    for key in obs_keys:
        if key not in adata.obs.columns:
            print(f"[WARN] '{key}' not found in obs; skipping.")
            continue
        s = adata.obs[key]
        num = pd.api.types.is_numeric_dtype(s)
        valid_keys.append(key)
        is_numeric[key] = num
    if not valid_keys:
        print("[plot_read_qc_grid] No valid obs_keys to plot.")
        return

    # Precompute global numeric ranges (after clipping) so rows share x-axis per column
    global_ranges = {}
    for key in valid_keys:
        if not is_numeric[key]:
            continue
        s = pd.to_numeric(adata.obs[key], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.size < min_non_nan:
            # still set something to avoid errors; just use min/max or (0,1)
            lo, hi = (0.0, 1.0) if s.size == 0 else (float(s.min()), float(s.max()))
        else:
            if clip_quantiles:
                qlo = s.quantile(clip_quantiles[0]) if clip_quantiles[0] is not None else s.min()
                qhi = s.quantile(clip_quantiles[1]) if clip_quantiles[1] is not None else s.max()
                lo, hi = float(qlo), float(qhi)
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    lo, hi = float(s.min()), float(s.max())
            else:
                lo, hi = float(s.min()), float(s.max())
        global_ranges[key] = (lo, hi)

    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))

    ncols = len(valid_keys)
    fig_w = figsize_cell[0] * ncols
    # rows per page is rows_per_fig; figure height scales accordingly
    fig_h_unit = figsize_cell[1]

    for start in range(0, len(sample_levels), rows_per_fig):
        chunk = sample_levels[start:start + rows_per_fig]
        nrows = len(chunk)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_w, fig_h_unit * nrows),
            dpi=dpi,
            squeeze=False,
        )

        for r, sample_val in enumerate(chunk):
            row_mask = (adata.obs[sample_key].values == sample_val)
            n_in_row = int(row_mask.sum())

            for c, key in enumerate(valid_keys):
                ax = axes[r, c]
                series = adata.obs.loc[row_mask, key]

                if is_numeric[key]:
                    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if x.size < min_non_nan:
                        ax.text(0.5, 0.5, f"n={x.size} (<{min_non_nan})", ha="center", va="center")
                    else:
                        # clip to global range for consistent axes
                        lo, hi = global_ranges[key]
                        x = x.clip(lo, hi)
                        ax.hist(x.values, bins=bins, range=(lo, hi), edgecolor="black", alpha=0.7)
                        ax.set_xlim(lo, hi)
                    if r == 0:
                        ax.set_title(key)
                    if c == 0:
                        ax.set_ylabel(f"{sample_val}\n(n={n_in_row})")
                    ax.grid(alpha=0.25)
                    ax.set_xlabel("")  # keep uncluttered; x-limit conveys scale
                else:
                    vc = series.astype("category").value_counts(dropna=False)
                    if vc.sum() < min_non_nan:
                        ax.text(0.5, 0.5, f"n={vc.sum()} (<{min_non_nan})", ha="center", va="center")
                    else:
                        vc_top = vc.iloc[:topn_categories][::-1]  # show top-N, reversed for barh
                        ax.barh(vc_top.index.astype(str), vc_top.values)
                        ax.invert_yaxis()
                    if r == 0:
                        ax.set_title(f"{key} (cat)")
                    if c == 0:
                        ax.set_ylabel(f"{sample_val}\n(n={n_in_row})")
                    ax.grid(alpha=0.25)
                    # trim labels to reduce clutter
                    if vc.sum() >= min_non_nan:
                        ax.tick_params(axis="y", labelsize=8)

        plt.tight_layout()
        page = start // rows_per_fig + 1
        out_png = os.path.join(outdir, f"qc_grid_{_sanitize(sample_key)}_page{page}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)


# def plot_read_qc_histograms(
#     adata,
#     outdir,
#     obs_keys,
#     sample_key=None,
#     *,
#     bins=100,
#     clip_quantiles=(0.0, 0.995),
#     min_non_nan=10,
#     figsize=(6, 4),
#     dpi=150
# ):
#     """
#     Plots histograms for given obs_keys, optionally grouped by sample_key.

#     Parameters
#     ----------
#     adata : AnnData
#         AnnData object.
#     outdir : str
#         Output directory for PNG files.
#     obs_keys : list[str]
#         List of obs columns to plot.
#     sample_key : str or None
#         Column in adata.obs to group by (e.g., 'Barcode').
#         If None, plots are for the full dataset only.
#     bins : int
#         Number of histogram bins for numeric data.
#     clip_quantiles : tuple or None
#         (low_q, high_q) to clip extreme values for plotting.
#     min_non_nan : int
#         Minimum number of finite values to plot.
#     figsize : tuple
#         Figure size.
#     dpi : int
#         Figure resolution.
#     """
#     os.makedirs(outdir, exist_ok=True)

#     # Define grouping
#     if sample_key and sample_key in adata.obs.columns:
#         groups = adata.obs.groupby(sample_key)
#     else:
#         groups = [(None, adata.obs)]  # single group

#     for group_name, group_df in groups:
#         # For each metric
#         for key in obs_keys:
#             if key not in group_df.columns:
#                 print(f"[WARN] '{key}' not found in obs; skipping.")
#                 continue

#             series = group_df[key]

#             # Numeric columns
#             if pd.api.types.is_numeric_dtype(series):
#                 x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
#                 if len(x) < min_non_nan:
#                     continue

#                 # Clip for better visualization
#                 if clip_quantiles:
#                     lo = x.quantile(clip_quantiles[0]) if clip_quantiles[0] is not None else x.min()
#                     hi = x.quantile(clip_quantiles[1]) if clip_quantiles[1] is not None else x.max()
#                     if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
#                         x = x.clip(lo, hi)

#                 fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
#                 ax.hist(x, bins=bins, edgecolor="black", alpha=0.7)
#                 ax.set_xlabel(key)
#                 ax.set_ylabel("Count")

#                 title = f"{key}" if group_name is None else f"{key} — {sample_key}={group_name}"
#                 ax.set_title(title)

#                 plt.tight_layout()

#                 # Save PNG
#                 safe_group = "all" if group_name is None else str(group_name)
#                 fname = f"{key}_{sample_key}_{safe_group}.png" if sample_key else f"{key}.png"
#                 fname = fname.replace("/", "_")
#                 fig.savefig(os.path.join(outdir, fname))
#                 plt.close(fig)

#             else:
#                 # Categorical columns
#                 vc = series.astype("category").value_counts(dropna=False)
#                 if vc.sum() < min_non_nan:
#                     continue

#                 fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
#                 vc.plot(kind="barh", ax=ax)
#                 ax.set_xlabel("Count")

#                 title = f"{key} (categorical)" if group_name is None else f"{key} — {sample_key}={group_name}"
#                 ax.set_title(title)

#                 plt.tight_layout()

#                 safe_group = "all" if group_name is None else str(group_name)
#                 fname = f"{key}_{sample_key}_{safe_group}.png" if sample_key else f"{key}.png"
#                 fname = fname.replace("/", "_")
#                 fig.savefig(os.path.join(outdir, fname))
#                 plt.close(fig)