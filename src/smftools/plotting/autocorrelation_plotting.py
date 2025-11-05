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
    annotate_periodicity: bool = True,
    counts_key_suffix: str = "_counts",
    # plotting thresholds
    plot_min_count: int = 10,
):
    """
    Plot a grid of mean spatial autocorrelations per sample × (site_type × reference).
    Expects preprocessing to have created:
      - adata.obsm[f"{site}_spatial_autocorr"]   -> (n_molecules, n_lags) float32
      - adata.obsm[f"{site}_spatial_autocorr_counts"] -> (n_molecules, n_lags) int32  (optional)
      - adata.uns[f"{site}_spatial_autocorr_lags"] -> 1D lags array
      - adata.uns[f"{site}_spatial_periodicity_metrics_by_group"] -> dict keyed by (sample, ref)
    If per-group metrics are missing and `analyze_autocorr_matrix` is importable, the function will
    fall back to running the analyzer for that group (slow) and cache the result into adata.uns.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings

    # Try importing analyzer (used only as fallback)
    try:
        from ..tools.spatial_autocorrelation import analyze_autocorr_matrix  # prefer packaged analyzer
    except Exception:
        analyze_autocorr_matrix = globals().get("analyze_autocorr_matrix", None)

    os.makedirs(out_dir, exist_ok=True)
    site_types = list(site_types)

    # small rolling average helper for smoother visualization
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

    # group summary extractor: returns (lags, mean_curve_smoothed, std_curve_smoothed, counts_block_or_None)
    def _compute_group_summary_for_mask(site: str, mask: np.ndarray):
        obsm_key = f"{site}_spatial_autocorr"
        lags_key = f"{site}_spatial_autocorr_lags"
        counts_key = f"{site}_spatial_autocorr{counts_key_suffix}"
        if obsm_key not in adata.obsm or lags_key not in adata.uns:
            return None, None, None, None
        mat = np.asarray(adata.obsm[obsm_key])
        if mat.size == 0:
            return None, None, None, None
        sel = mat[mask, :]
        if sel.size == 0:
            return None, None, None, None
        mean_per_lag = np.nanmean(sel, axis=0)
        std_per_lag = np.nanstd(sel, axis=0, ddof=1)
        counts = None
        if counts_key in adata.obsm:
            counts_mat = np.asarray(adata.obsm[counts_key])
            counts = counts_mat[mask, :].astype(int)
        return np.asarray(adata.uns[lags_key]), _rolling_1d(mean_per_lag, window), _rolling_1d(std_per_lag, window), counts

    # samples meta
    if sample_col not in adata.obs:
        raise KeyError(f"sample_col '{sample_col}' not present in adata.obs")
    samples = adata.obs[sample_col]
    if not pd.api.types.is_categorical_dtype(samples):
        samples = samples.astype("category")
    sample_levels = list(samples.cat.categories)

    # references meta
    if reference_col not in adata.obs:
        raise KeyError(f"reference_col '{reference_col}' not present in adata.obs")
    if references is None:
        refs_series = adata.obs[reference_col]
        if not pd.api.types.is_categorical_dtype(refs_series):
            refs_series = refs_series.astype("category")
        references = list(refs_series.cat.categories)
    references = list(references)

    # build column metadata
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
    # metrics_cache for fallback-computed entries (persisted at end)
    metrics_cache = {site: {} for site in site_types}

    # Iterate pages
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
        # per-site prefetching (avoid repeated conversion)
        for site, cols in group_column_meta:
            obsm_key = f"{site}_spatial_autocorr"
            counts_key = f"{site}_spatial_autocorr{counts_key_suffix}"
            lags_key = f"{site}_spatial_autocorr_lags"
            ac_full = np.asarray(adata.obsm[obsm_key]) if obsm_key in adata.obsm else None
            counts_full = np.asarray(adata.obsm[counts_key]) if counts_key in adata.obsm else None
            lags = np.asarray(adata.uns[lags_key]) if lags_key in adata.uns else None

            # metrics_by_group may already exist (precomputed)
            metrics_by_group_key = f"{site}_spatial_periodicity_metrics_by_group"
            metrics_by_group_precomp = adata.uns.get(metrics_by_group_key, None)

            for col_kind, col_val in cols:
                for r, sample_name in enumerate(chunk):
                    ax = axes[r, col_idx]

                    # compute mask
                    sample_mask = (adata.obs[sample_col].values == sample_name)
                    if col_kind == "ref":
                        ref_mask = (adata.obs[reference_col].values == col_val)
                        mask = sample_mask & ref_mask
                    else:
                        mask = sample_mask

                    # count molecules
                    n_reads_grp = int(mask.sum())

                    # group summary (mean/std and counts_block)
                    lags_local, mean_curve, std_curve, counts_block = _compute_group_summary_for_mask(site, mask)

                    # plot title for top row
                    if r == 0:
                        title = f"{site} (all refs)" if col_kind == "all" else f"{site} [{col_val}]"
                        ax.set_title(title, fontsize=9)

                    # handle no-data
                    if lags_local is None:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Lag (bp)", fontsize=7)
                        ax.tick_params(axis='both', which='major', labelsize=6)
                        ax.grid(True, alpha=0.22)
                        col_idx += 1
                        continue

                    # mask low-support lags if counts available
                    mean_plot = mean_curve.copy()
                    if counts_block is not None:
                        # counts_block shape: (n_molecules_in_group, n_lags)
                        support = counts_block.sum(axis=0)
                        low_support = support < plot_min_count
                        high_support = ~low_support

                        # smooth the original mean once for plotting context
                        mean_curve_smooth = _rolling_1d(mean_curve, window)

                        # mask the smoothed mean to only show high-support points as the main trace
                        mean_plot = mean_curve_smooth.copy()
                        mean_plot[low_support] = np.nan

                        # plot a faint grey line for the low-support regions (context only)
                        if low_support.any():
                            ax.plot(lags_local[low_support], mean_curve_smooth[low_support], color="0.85", lw=0.6, label="_nolegend_")

                    # plot mean (high-support only) and +/- std (std is computed from all molecules)
                    ax.plot(lags_local, mean_plot, lw=1.1)
                    upper = mean_curve + std_curve
                    lower = mean_curve - std_curve
                    ax.fill_between(lags_local, lower, upper, alpha=0.18)

                    # ---------- use precomputed metrics if present, otherwise fallback ----------
                    group_key = (sample_name, None if col_kind == "all" else col_val)
                    res = None
                    if metrics_by_group_precomp is not None:
                        # metrics_by_group_precomp can be dict-like
                        res = metrics_by_group_precomp.get(group_key, None)

                    if res is None and annotate_periodicity and (analyze_autocorr_matrix is not None) and (ac_full is not None):
                        # fallback: run analyzer on the subset (warn + cache)
                        ac_sel = ac_full[mask, :]
                        cnt_sel = counts_full[mask, :] if counts_full is not None else None
                        if ac_sel.size:
                            warnings.warn(f"Precomputed periodicity metrics for {site} {group_key} not found — running analyzer as fallback (slow).")
                            try:
                                res = analyze_autocorr_matrix(
                                    ac_sel,
                                    cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                                    lags_local,
                                    nrl_search_bp=(120, 260),
                                    pad_factor=4,
                                    min_count=plot_min_count,
                                    max_harmonics=6,
                                )
                            except Exception as e:
                                res = {"error": str(e)}
                            # cache into adata.uns for future plotting runs
                            if metrics_by_group_precomp is None:
                                adata.uns[metrics_by_group_key] = {}
                                metrics_by_group_precomp = adata.uns[metrics_by_group_key]
                            metrics_by_group_precomp[group_key] = res
                            # also record in local metrics_cache for persistence at the end
                            metrics_cache[site][group_key] = res

                    # overlay periodicity annotations if available and valid
                    if annotate_periodicity and (res is not None) and ("error" not in res):
                        # safe array conversion
                        sample_lags = np.asarray(res.get("envelope_sample_lags", np.array([])))
                        envelope_heights = np.asarray(res.get("envelope_heights", np.array([])))
                        nrl = res.get("nrl_bp", None)
                        xi_val = res.get("xi", None)
                        snr = res.get("snr", None)
                        fwhm_bp = res.get("fwhm_bp", None)

                        # vertical NRL line & harmonics (safe check)
                        if (nrl is not None) and np.isfinite(nrl):
                            ax.axvline(float(nrl), color="C3", linestyle="--", linewidth=1.0, alpha=0.9)
                            for m in range(2, 5):
                                ax.axvline(float(nrl) * m, color="C3", linestyle=":", linewidth=0.7, alpha=0.6)

                        # envelope points + fitted exponential
                        if sample_lags.size:
                            ax.scatter(sample_lags, envelope_heights, s=18, color="C2")
                            if (xi_val is not None) and np.isfinite(xi_val) and np.isfinite(res.get("xi_A", np.nan)):
                                A = float(res.get("xi_A", np.nan))
                                xi_val = float(xi_val)
                                env_x = np.linspace(np.min(sample_lags), np.max(sample_lags), 200)
                                env_y = A * np.exp(-env_x / xi_val)
                                ax.plot(env_x, env_y, linestyle="--", color="C2", linewidth=1.0, alpha=0.9)

                        # inset PSD plotted vs NRL (linear x-axis)
                        freqs = res.get("freqs", None)
                        power = res.get("power", None)
                        peak_f = res.get("f0", None)
                        if freqs is not None and power is not None:
                            freqs = np.asarray(freqs)
                            power = np.asarray(power)
                            valid = (freqs > 0) & np.isfinite(freqs) & np.isfinite(power)
                            if valid.any():
                                inset = ax.inset_axes([0.62, 0.58, 0.36, 0.37])
                                nrl_vals = 1.0 / freqs[valid]  # convert freq -> NRL (bp)
                                inset.plot(nrl_vals, power[valid], lw=0.7)
                                if peak_f is not None and peak_f > 0:
                                    inset.axvline(1.0 / float(peak_f), color="C3", linestyle="--", linewidth=0.8)
                                # choose a reasonable linear x-limits (prefer typical NRL range but fallback to data)
                                default_xlim = (60, 400)
                                data_xlim = (float(np.nanmin(nrl_vals)), 600)
                                # pick intersection/covering range
                                left = min(default_xlim[0], data_xlim[0])
                                right = max(default_xlim[1], data_xlim[1])
                                inset.set_xlim(left, right)
                                inset.set_xlabel("NRL (bp)", fontsize=6)
                                inset.set_ylabel("power", fontsize=6)
                                inset.tick_params(labelsize=6)
                                if (snr is not None) and np.isfinite(snr):
                                    inset.text(0.95, 0.95, f"SNR={float(snr):.1f}", transform=inset.transAxes,
                                            ha="right", va="top", fontsize=6, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

                    # set x-limits based on finite lags
                    finite_mask = np.isfinite(lags_local)
                    if finite_mask.any():
                        ax.set_xlim(float(np.nanmin(lags_local[finite_mask])), float(np.nanmax(lags_local[finite_mask])))

                    # small cosmetics
                    ax.set_xlabel("Lag (bp)", fontsize=7)
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.grid(True, alpha=0.22)

                col_idx += 1

        # layout and left-hand sample labels
        fig.tight_layout(rect=[0.06, 0, 1, 0.97])
        for r, sample_name in enumerate(chunk):
            first_ax = axes[r, 0]
            pos = first_ax.get_position()
            ycenter = pos.y0 + pos.height / 2.0
            n_reads_grp = int((adata.obs[sample_col].values == sample_name).sum())
            label = f"{sample_name}\n(n={n_reads_grp})"
            fig.text(0.02, ycenter, label, va='center', ha='left', rotation='vertical', fontsize=9)

        fig.suptitle("Spatial autocorrelation by sample × (site_type × reference)", y=0.995, fontsize=11)

        page_idx = start_idx // rows_per_fig + 1
        out_png = os.path.join(out_dir, f"{filename_prefix}_page{page_idx}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        saved_pages.append(out_png)

    # persist any metrics we computed via fallback into adata.uns
    for site, d in metrics_cache.items():
        if d:
            adata.uns[f"{site}_spatial_periodicity_metrics_by_group"] = d

    # ---------------------------
    # Write combined CSV + per-sample/ref CSVs
    # ---------------------------
    csv_dir = os.path.join(out_dir, "periodicity_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    # include combined ('all') as a reference group for convenience
    ref_values = list(references) + ["all"]

    combined_rows = []

    for sample_name in sample_levels:
        for ref in ref_values:
            rows = []
            for site in site_types:
                key = (sample_name, None) if ref == "all" else (sample_name, ref)
                metrics_by_group_key = f"{site}_spatial_periodicity_metrics_by_group"
                group_dict = adata.uns.get(metrics_by_group_key, None)
                entry = None
                if group_dict is not None:
                    entry = group_dict.get(key, None)

                def to_list(x):
                    """
                    Normalize x to a Python list:
                    - None -> []
                    - list/tuple -> list(x)
                    - numpy array -> arr.tolist()
                    - scalar -> [scalar]
                    - string -> [string]  (preserve)
                    """
                    if x is None:
                        return []
                    if isinstance(x, (list, tuple)):
                        return list(x)
                    # treat strings separately to avoid splitting into characters
                    if isinstance(x, str):
                        return [x]
                    try:
                        arr = np.asarray(x)
                    except Exception:
                        return [x]
                    # numpy scalars -> 0-dim arrays
                    if arr.ndim == 0:
                        return [arr.item()]
                    # convert to python list
                    return arr.tolist()

                def _safe_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return float("nan")

                # --- inside your combined CSV loop, replace the envelope handling with this ---
                env_lags_raw = entry.get("envelope_sample_lags", []) if entry is not None else []
                env_heights_raw = entry.get("envelope_heights", []) if entry is not None else []

                env_lags_list = to_list(env_lags_raw)
                env_heights_list = to_list(env_heights_raw)

                row = {
                    "site": site,
                    "sample": sample_name,
                    "reference": ref,
                    "nrl_bp": _safe_float(entry.get("nrl_bp", float("nan"))) if entry is not None else float("nan"),
                    "snr": _safe_float(entry.get("snr", float("nan"))) if entry is not None else float("nan"),
                    "fwhm_bp": _safe_float(entry.get("fwhm_bp", float("nan"))) if entry is not None else float("nan"),
                    "xi": _safe_float(entry.get("xi", float("nan"))) if entry is not None else float("nan"),
                    "xi_A": _safe_float(entry.get("xi_A", float("nan"))) if entry is not None else float("nan"),
                    "xi_r2": _safe_float(entry.get("xi_r2", float("nan"))) if entry is not None else float("nan"),
                    "envelope_sample_lags": ";".join(map(str, env_lags_list)) if len(env_lags_list) else "",
                    "envelope_heights": ";".join(map(str, env_heights_list)) if len(env_heights_list) else "",
                    "analyzer_error": entry.get("error", entry.get("analyzer_error", None)) if entry is not None else "no_metrics",
                }
                rows.append(row)
                combined_rows.append(row)

            # write per-(sample,ref) CSV
            df_group = pd.DataFrame(rows)
            safe_sample = str(sample_name).replace(os.sep, "_")
            safe_ref = str(ref).replace(os.sep, "_")
            out_csv = os.path.join(csv_dir, f"{safe_sample}__{safe_ref}__periodicity_metrics.csv")
            try:
                df_group.to_csv(out_csv, index=False)
            except Exception as e:
                # don't fail the whole pipeline for a single write error; log and continue
                import warnings
                warnings.warn(f"Failed to write {out_csv}: {e}")

    # write the single combined CSV (one row per sample x ref x site)
    combined_df = pd.DataFrame(combined_rows)
    combined_out = os.path.join(out_dir, "periodicity_metrics_combined.csv")
    try:
        combined_df.to_csv(combined_out, index=False)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to write combined CSV {combined_out}: {e}")

    return saved_pages


def plot_rolling_metrics(df, out_png=None, title=None, figsize=(10, 3.5), dpi=160, show=False):
    """
    Plot NRL and SNR vs window center from the dataframe returned by rolling_autocorr_metrics.
    If out_png is None, returns the matplotlib Figure object; otherwise saves PNG and returns path.
    """
    import matplotlib.pyplot as plt
    # sort by center
    df2 = df.sort_values("center")
    x = df2["center"].values
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi, sharex=True)

    axes[0].plot(x, df2["nrl_bp"].values, marker="o", lw=1)
    axes[0].set_xlabel("Window center (bp)")
    axes[0].set_ylabel("NRL (bp)")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(x, df2["snr"].values, marker="o", lw=1, color="C3")
    axes[1].set_xlabel("Window center (bp)")
    axes[1].set_ylabel("SNR")
    axes[1].grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        if not show:
            import matplotlib
            matplotlib.pyplot.close(fig)
        return out_png
    if not show:
        import matplotlib
        matplotlib.pyplot.close(fig)
    return fig

import numpy as np
import pandas as pd

def plot_rolling_grid(
    rolling_dict,
    out_dir,
    site,
    metrics=("nrl_bp", "snr", "xi"),
    sample_order=None,
    reference_order=None,
    rows_per_page: int = 6,
    cols_per_page: int = None,
    dpi: int = 160,
    figsize_per_panel=(3.5, 2.2),
    per_metric_ylim: dict = None,
    filename_prefix: str = "rolling_grid",
    metric_display_names: dict = None,
):
    """
    Plot rolling metrics in a grid, creating a separate paginated page-set for each metric.

    Parameters
    ----------
    rolling_dict : dict
        mapping (sample, ref) -> DataFrame (must contain 'center' and metric columns).
        Keys may use `None` for combined/"all" reference.
    out_dir : str
    site : str
    metrics : sequence[str]
        list of metric column names to plot. One page-set per metric will be written.
    sample_order, reference_order : optional lists for ordering (values as in keys)
    rows_per_page : int
        number of sample rows per page.
    cols_per_page : int | None
        number of columns per page (defaults to number of unique refs).
    figsize_per_panel : (w,h) for each subplot panel.
    per_metric_ylim : dict or None
        optional mapping metric -> (ymin,ymax) to force consistent y-limits for that metric.
        If absent, y-limits are autoscaled per page.
    filename_prefix : str
    metric_display_names : dict or None
        optional mapping metric -> friendly label for y-axis/title.

    Returns
    -------
    pages_by_metric : dict mapping metric -> [out_png_paths]
    """
    import os
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if per_metric_ylim is None:
        per_metric_ylim = {}
    if metric_display_names is None:
        metric_display_names = {}

    os.makedirs(out_dir, exist_ok=True)

    keys = list(rolling_dict.keys())
    if not keys:
        raise ValueError("rolling_dict is empty")

    # normalize reference labels and keep mapping to original
    label_to_orig = {}
    for (_sample, ref) in keys:
        label = "all" if (ref is None) else str(ref)
        if label not in label_to_orig:
            label_to_orig[label] = ref

    # sample ordering
    all_samples = sorted({k[0] for k in keys}, key=lambda x: str(x))
    sample_list = [s for s in (sample_order or all_samples) if s in all_samples]

    # reference labels ordering
    default_ref_labels = sorted(label_to_orig.keys(), key=lambda s: s)
    if reference_order is not None:
        ref_labels = [("all" if r is None else str(r)) for r in reference_order if (("all" if r is None else str(r)) in label_to_orig)]
    else:
        ref_labels = default_ref_labels

    ncols_total = len(ref_labels)
    if cols_per_page is None:
        cols_per_page = ncols_total

    pages_by_metric = {}

    # for each metric produce pages
    for metric in metrics:
        saved_pages = []
        display_name = metric_display_names.get(metric, metric)

        # paginate samples
        for start in range(0, len(sample_list), rows_per_page):
            page_samples = sample_list[start : start + rows_per_page]
            nrows = len(page_samples)

            fig, axes = plt.subplots(
                nrows=nrows, ncols=cols_per_page,
                figsize=(figsize_per_panel[0] * cols_per_page, figsize_per_panel[1] * nrows),
                dpi=dpi, squeeze=False
            )

            for i, sample in enumerate(page_samples):
                for j in range(cols_per_page):
                    ax = axes[i, j]
                    if j >= len(ref_labels):
                        ax.axis("off")
                        continue

                    label = ref_labels[j]
                    orig_ref = label_to_orig.get(label, None)
                    key = (sample, orig_ref)
                    df = rolling_dict.get(key, None)

                    ax.set_title(f"{sample} | {label}", fontsize=8)

                    if df is None or df.empty or (metric not in df.columns):
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    df2 = df.sort_values("center")
                    x = df2["center"].values
                    y = df2[metric].values

                    ax.plot(x, y, lw=1, marker="o")
                    ax.set_xlabel("center (bp)", fontsize=7)
                    ax.set_ylabel(display_name, fontsize=7)
                    ax.grid(True, alpha=0.18)

                    # apply per-metric y-lim if provided
                    if metric in per_metric_ylim:
                        yl = per_metric_ylim[metric]
                        try:
                            ax.set_ylim(float(yl[0]), float(yl[1]))
                        except Exception:
                            pass

            fig.suptitle(f"{site} — {display_name}", fontsize=10)
            fig.tight_layout(rect=[0.03, 0.03, 1, 0.96])

            page_idx = start // rows_per_page + 1
            out_png = os.path.join(out_dir, f"{filename_prefix}_{site}_{metric}_page{page_idx}.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            saved_pages.append(out_png)

        pages_by_metric[metric] = saved_pages

    return pages_by_metric

