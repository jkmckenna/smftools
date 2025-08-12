def calculate_complexity_II(
    adata,
    output_directory='',
    sample_col='Sample_names',
    cluster_col='merged_cluster_id',
    plot=True,
    save_plot=False,
    n_boot=30,
    n_depths=12,
    random_state=0,
    csv_summary=True,
):
    """
    Estimate and plot library complexity per sample using duplicate clusters.

    Requires:
        - adata.obs[sample_col]: sample label per read
        - adata.obs[cluster_col]: integer/str cluster id per read where duplicates share the same id

    Produces (per sample):
        - Fit of Lander–Waterman: U(d) = C0 * (1 - exp(-d / C0))
        - Figure with bootstrap mean ± 95% CI, observed point, and fit
        - Adds adata.uns[f'Library_complexity_{sample}'] = {'C0': ..., 'n_reads': ..., 'n_unique': ...}
        - Optionally saves PNGs and a CSV of curve points

    Parameters
    ----------
    output_directory : str
        Where to save plots/CSVs if save_plot=True or csv_summary=True.
    sample_col : str
        Column in obs for sample grouping.
    cluster_col : str
        Column in obs that identifies duplicate clusters (molecules).
    plot : bool
        Whether to generate plots.
    save_plot : bool
        Save PNGs instead of showing.
    n_boot : int
        Number of bootstrap replicates per depth.
    n_depths : int
        Number of subsampling depths (evenly spaced from small to full depth).
    random_state : int
        RNG seed.
    csv_summary : bool
        Write CSV with subsampling mean/CI and fitted curve per sample.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from datetime import datetime

    rng = np.random.default_rng(random_state)

    def lw(x, C0):
        return C0 * (1.0 - np.exp(-x / C0))

    def sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))

    # Checks
    for col in (sample_col, cluster_col):
        if col not in adata.obs.columns:
            raise KeyError(f"Required column '{col}' not found in adata.obs")
    if save_plot or csv_summary:
        os.makedirs(output_directory or ".", exist_ok=True)

    # Prepare outputs
    fit_records = []
    curve_records = []

    samples = adata.obs[sample_col].astype("category").cat.categories
    for sample in samples:
        mask = (adata.obs[sample_col] == sample).values
        if not mask.any():
            continue

        # cluster ids per read for this sample
        clusters = adata.obs.loc[mask, cluster_col].values
        n_reads = clusters.shape[0]
        # observed unique molecules at full depth
        observed_unique = int(pd.unique(clusters).size)

        # choose subsampling depths
        if n_depths < 2:
            depths = np.array([n_reads], dtype=int)
        else:
            # spread from ~5% to 100% (at least 10 reads)
            lo = max(10, int(0.05 * n_reads))
            depths = np.unique(np.linspace(lo, n_reads, n_depths, dtype=int))
        depths = depths[depths > 0]

        # bootstrap expected unique at each depth
        boot_unique = np.zeros((len(depths), n_boot), dtype=float)
        idx_all = np.arange(n_reads)
        for di, d in enumerate(depths):
            if d > n_reads:
                d = n_reads
            for b in range(n_boot):
                take = rng.choice(idx_all, size=d, replace=False)
                boot_unique[di, b] = np.unique(clusters[take]).size

        mean_unique = boot_unique.mean(axis=1)
        lo_ci = np.percentile(boot_unique, 2.5, axis=1)
        hi_ci = np.percentile(boot_unique, 97.5, axis=1)

        # fit LW to bootstrap means (guard with bounds)
        # Initial C0 guess: observed_unique or mean at max depth
        C0_init = max(observed_unique, mean_unique[-1])
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
            # fallback: use observed unique as C0
            C0 = float(observed_unique)

        # Store fit in adata.uns
        adata.uns[f'Library_complexity_{sample}'] = {
            "C0": C0,
            "n_reads": int(n_reads),
            "n_unique": int(observed_unique),
            "depths": depths,
            "mean_unique": mean_unique,
            "ci_low": lo_ci,
            "ci_high": hi_ci,
        }

        # Generate smooth curve for plotting
        x_fit = np.linspace(0, max(n_reads, depths[-1]), 200)
        y_fit = lw(x_fit, C0)

        # Records for optional CSVs
        fit_records.append({
            "sample": sample,
            "C0": C0,
            "n_reads": int(n_reads),
            "n_unique_observed": int(observed_unique)
        })
        for d, mu, lo, hi in zip(depths, mean_unique, lo_ci, hi_ci):
            curve_records.append({
                "sample": sample,
                "type": "bootstrap",
                "depth": int(d),
                "mean_unique": float(mu),
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
        for xf, yf in zip(x_fit, y_fit):
            curve_records.append({
                "sample": sample,
                "type": "fit",
                "depth": float(xf),
                "mean_unique": float(yf),
                "ci_low": np.nan,
                "ci_high": np.nan,
            })

        # Plot
        if plot:
            plt.figure(figsize=(6.5, 4.5))
            # CI band
            plt.fill_between(depths, lo_ci, hi_ci, alpha=0.25, label="Bootstrap 95% CI")
            # mean points
            plt.plot(depths, mean_unique, "o", label="Bootstrap mean")
            # observed full-depth point
            plt.plot([n_reads], [observed_unique], "s", label="Observed (full)")
            # LW fit
            plt.plot(x_fit, y_fit, "-", label=f"LW fit  C0≈{C0:,.0f}")

            plt.xlabel("Total reads (subsampled depth)")
            plt.ylabel("Unique molecules (clusters)")
            plt.title(f"Library Complexity — {sample}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            if save_plot:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{ts}_complexity_{sanitize(sample)}.png"
                plt.savefig(os.path.join(output_directory or ".", fname), dpi=160, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    # Optional CSV outputs
    if csv_summary and (fit_records or curve_records):
        fit_df = pd.DataFrame(fit_records)
        curve_df = pd.DataFrame(curve_records)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = output_directory or "."
        fit_df.to_csv(os.path.join(base, f"{ts}_complexity_fit_summary.csv"), index=False)
        curve_df.to_csv(os.path.join(base, f"{ts}_complexity_curves.csv"), index=False)
