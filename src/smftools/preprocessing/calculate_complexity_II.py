from typing import Optional
def calculate_complexity_II(
    adata,
    output_directory='',
    sample_col='Sample_names',
    ref_col: Optional[str] = 'Reference_strand',
    cluster_col='sequence__merged_cluster_id',
    plot=True,
    save_plot=False,
    n_boot=30,
    n_depths=12,
    random_state=0,
    csv_summary=True,
    uns_flag='complexity_analysis_complete',
    force_redo=False,
    bypass=False
):
    """
    Estimate and plot library complexity.

    If ref_col is None (default), behaves as before: one calculation per sample.
    If ref_col is provided, computes complexity for each (sample, ref) pair.

    Results:
      - adata.uns['Library_complexity_results'] : dict keyed by (sample,) or (sample, ref) -> dict with fields
          C0, n_reads, n_unique, depths, mean_unique, ci_low, ci_high
      - Also stores per-entity record in adata.uns[f'Library_complexity_{sanitized_name}'] (backwards compatible)
      - Optionally saves PNGs and CSVs (curve points + fit summary)
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from datetime import datetime

    # early exits
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo):
        return None
    if bypass:
        return None

    rng = np.random.default_rng(random_state)

    def lw(x, C0):
        return C0 * (1.0 - np.exp(-x / C0))

    def sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))

    # checks
    for col in (sample_col, cluster_col):
        if col not in adata.obs.columns:
            raise KeyError(f"Required column '{col}' not found in adata.obs")
    if ref_col is not None and ref_col not in adata.obs.columns:
        raise KeyError(f"ref_col '{ref_col}' not found in adata.obs")

    if save_plot or csv_summary:
        os.makedirs(output_directory or ".", exist_ok=True)

    # containers to collect CSV rows across all groups
    fit_records = []
    curve_records = []

    # output dict stored centrally
    results = {}

    # build list of groups: either samples only, or (sample, ref) pairs
    sseries = adata.obs[sample_col].astype("category")
    samples = list(sseries.cat.categories)
    if ref_col is None:
        group_keys = [(s,) for s in samples]
    else:
        rseries = adata.obs[ref_col].astype("category")
        references = list(rseries.cat.categories)
        group_keys = []
        # iterate only pairs that exist in data to avoid empty processing
        for s in samples:
            mask_s = (adata.obs[sample_col] == s)
            # find references present for this sample
            ref_present = pd.Categorical(adata.obs.loc[mask_s, ref_col]).categories
            # Use intersection of known reference categories and those present for sample
            for r in ref_present:
                group_keys.append((s, r))

    # iterate groups
    for g in group_keys:
        if ref_col is None:
            sample = g[0]
            # filter mask
            mask = (adata.obs[sample_col] == sample).values
            group_label = f"{sample}"
        else:
            sample, ref = g
            mask = (adata.obs[sample_col] == sample) & (adata.obs[ref_col] == ref)
            group_label = f"{sample}__{ref}"

        n_reads = int(mask.sum())
        if n_reads < 2:
            # store empty placeholders and continue
            results[g] = {
                "C0": np.nan,
                "n_reads": int(n_reads),
                "n_unique": 0,
                "depths": np.array([], dtype=int),
                "mean_unique": np.array([], dtype=float),
                "ci_low": np.array([], dtype=float),
                "ci_high": np.array([], dtype=float),
            }
            # also store back-compat key
            adata.uns[f'Library_complexity_{sanitize(group_label)}'] = results[g]
            continue

        # cluster ids array for this group
        clusters = adata.obs.loc[mask, cluster_col].to_numpy()
        # observed unique molecules at full depth
        observed_unique = int(pd.unique(clusters).size)

        # choose subsampling depths
        if n_depths < 2:
            depths = np.array([n_reads], dtype=int)
        else:
            lo = max(10, int(0.05 * n_reads))
            depths = np.unique(np.linspace(lo, n_reads, n_depths, dtype=int))
            depths = depths[depths > 0]
        depths = depths.astype(int)
        if depths.size == 0:
            depths = np.array([n_reads], dtype=int)

        # bootstrap sampling: for each depth, sample without replacement (if possible)
        idx_all = np.arange(n_reads)
        boot_unique = np.zeros((len(depths), n_boot), dtype=float)
        for di, d in enumerate(depths):
            d_use = int(min(d, n_reads))
            # if d_use == n_reads we can short-circuit and set boot results to full observed uniques
            if d_use == n_reads:
                # bootstraps are deterministic in this special case
                uniq_val = float(observed_unique)
                boot_unique[di, :] = uniq_val
                continue
            # otherwise run bootstraps
            for b in range(n_boot):
                take = rng.choice(idx_all, size=d_use, replace=False)
                boot_unique[di, b] = np.unique(clusters[take]).size

        mean_unique = boot_unique.mean(axis=1)
        lo_ci = np.percentile(boot_unique, 2.5, axis=1)
        hi_ci = np.percentile(boot_unique, 97.5, axis=1)

        # fit Lander-Waterman to the mean curve (safe bounds)
        C0_init = max(observed_unique, mean_unique[-1] if mean_unique.size else observed_unique)
        try:
            popt, _ = curve_fit(
                lw,
                xdata=depths.astype(float),
                ydata=mean_unique.astype(float),
                p0=[C0_init],
                bounds=(1.0, 1e12),
                maxfev=10000,
            )
            C0 = float(popt[0])
        except Exception:
            C0 = float(observed_unique)

        # store results
        results[g] = {
            "C0": C0,
            "n_reads": int(n_reads),
            "n_unique": int(observed_unique),
            "depths": depths,
            "mean_unique": mean_unique,
            "ci_low": lo_ci,
            "ci_high": hi_ci,
        }

        # save per-group in adata.uns for backward compatibility
        adata.uns[f'Library_complexity_{sanitize(group_label)}'] = results[g]

        # prepare curve and fit records for CSV
        fit_records.append({
            "sample": sample,
            "reference": ref if ref_col is not None else "",
            "C0": float(C0),
            "n_reads": int(n_reads),
            "n_unique_observed": int(observed_unique),
        })

        x_fit = np.linspace(0, max(n_reads, int(depths[-1]) if depths.size else n_reads), 200)
        y_fit = lw(x_fit, C0)
        for d, mu, lo, hi in zip(depths, mean_unique, lo_ci, hi_ci):
            curve_records.append({
                "sample": sample,
                "reference": ref if ref_col is not None else "",
                "type": "bootstrap",
                "depth": int(d),
                "mean_unique": float(mu),
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
        for xf, yf in zip(x_fit, y_fit):
            curve_records.append({
                "sample": sample,
                "reference": ref if ref_col is not None else "",
                "type": "fit",
                "depth": float(xf),
                "mean_unique": float(yf),
                "ci_low": np.nan,
                "ci_high": np.nan,
            })

        # plotting for this group
        if plot:
            plt.figure(figsize=(6.5, 4.5))
            plt.fill_between(depths, lo_ci, hi_ci, alpha=0.25, label="Bootstrap 95% CI")
            plt.plot(depths, mean_unique, "o", label="Bootstrap mean")
            plt.plot([n_reads], [observed_unique], "s", label="Observed (full)")
            plt.plot(x_fit, y_fit, "-", label=f"LW fit  C0≈{C0:,.0f}")
            plt.xlabel("Total reads (subsampled depth)")
            plt.ylabel("Unique molecules (clusters)")
            title = f"Library Complexity — {sample}" + (f" / {ref}" if ref_col is not None else "")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            if save_plot:
                fname = f"complexity_{sanitize(group_label)}.png"
                plt.savefig(os.path.join(output_directory or ".", fname), dpi=160, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    # store central results dict
    adata.uns["Library_complexity_results"] = results

    # mark complexity analysis as complete
    adata.uns[uns_flag] = True

    # CSV outputs
    if csv_summary and (fit_records or curve_records):
        fit_df = pd.DataFrame(fit_records)
        curve_df = pd.DataFrame(curve_records)
        base = output_directory or "."
        fit_df.to_csv(os.path.join(base, f"complexity_fit_summary.csv"), index=False)
        curve_df.to_csv(os.path.join(base, f"complexity_curves.csv"), index=False)

    return results
