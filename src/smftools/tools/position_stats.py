# ------------------------- Utilities -------------------------
def random_fill_nans(X):
    import numpy as np
    nan_mask = np.isnan(X)
    X[nan_mask] = np.random.rand(*X[nan_mask].shape)
    return X

def calculate_relative_risk_on_activity(adata, sites, alpha=0.05, groupby=None):
    """
    Perform Bayesian-style methylation vs activity analysis independently within each group.

    Parameters:
        adata (AnnData): Annotated data matrix.
        sites (list of str): List of site keys (e.g., ['GpC_site', 'CpG_site']).
        alpha (float): FDR threshold for significance.
        groupby (str or list of str): Column(s) in adata.obs to group by.

    Returns:
        results_dict (dict): Dictionary with structure:
            results_dict[ref][group_label] = (results_df, sig_df)
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    def compute_risk_df(ref, site_subset, positions_list, relative_risks, p_values):
        p_adj = multipletests(p_values, method='fdr_bh')[1] if p_values else []

        genomic_positions = np.array(site_subset.var_names)[positions_list]
        is_gpc_site = site_subset.var[f"{ref}_GpC_site"].values[positions_list]
        is_cpg_site = site_subset.var[f"{ref}_CpG_site"].values[positions_list]

        results_df = pd.DataFrame({
            'Feature_Index': positions_list,
            'Genomic_Position': genomic_positions.astype(int),
            'Relative_Risk': relative_risks,
            'Adjusted_P_Value': p_adj,
            'GpC_Site': is_gpc_site,
            'CpG_Site': is_cpg_site
        })

        results_df['log2_Relative_Risk'] = np.log2(results_df['Relative_Risk'].replace(0, 1e-300))
        results_df['-log10_Adj_P'] = -np.log10(results_df['Adjusted_P_Value'].replace(0, 1e-300))
        sig_df = results_df[results_df['Adjusted_P_Value'] < alpha]
        return results_df, sig_df

    results_dict = {}

    for ref in adata.obs['Reference_strand'].unique():
        ref_subset = adata[adata.obs['Reference_strand'] == ref].copy()
        if ref_subset.shape[0] == 0:
            continue

        # Normalize groupby to list
        if groupby is not None:
            if isinstance(groupby, str):
                groupby = [groupby]
            def format_group_label(row):
                return ",".join([f"{col}={row[col]}" for col in groupby])

            combined_label = '__'.join(groupby)
            ref_subset.obs[combined_label] = ref_subset.obs.apply(format_group_label, axis=1)
            groups = ref_subset.obs[combined_label].unique()
        else:
            combined_label = None
            groups = ['all']

        results_dict[ref] = {}

        for group in groups:
            if group == 'all':
                group_subset = ref_subset
            else:
                group_subset = ref_subset[ref_subset.obs[combined_label] == group]

            if group_subset.shape[0] == 0:
                continue

            # Build site mask
            site_mask = np.zeros(group_subset.shape[1], dtype=bool)
            for site in sites:
                site_mask |= group_subset.var[f"{ref}_{site}"]
            site_subset = group_subset[:, site_mask].copy()

            # Matrix and labels
            X = random_fill_nans(site_subset.X.copy())
            y = site_subset.obs['activity_status'].map({'Active': 1, 'Silent': 0}).values
            P_active = np.mean(y)

            # Analysis
            positions_list, relative_risks, p_values = [], [], []
            for pos in range(X.shape[1]):
                methylation_state = (X[:, pos] > 0).astype(int)
                table = pd.crosstab(methylation_state, y)
                if table.shape != (2, 2):
                    continue

                P_methylated = np.mean(methylation_state)
                P_methylated_given_active = np.mean(methylation_state[y == 1])
                P_methylated_given_inactive = np.mean(methylation_state[y == 0])

                if P_methylated_given_inactive == 0 or P_methylated in [0, 1]:
                    continue

                P_active_given_methylated = (P_methylated_given_active * P_active) / P_methylated
                P_active_given_unmethylated = ((1 - P_methylated_given_active) * P_active) / (1 - P_methylated)
                RR = P_active_given_methylated / P_active_given_unmethylated

                _, p_value = fisher_exact(table)
                positions_list.append(pos)
                relative_risks.append(RR)
                p_values.append(p_value)

            results_df, sig_df = compute_risk_df(ref, site_subset, positions_list, relative_risks, p_values)
            results_dict[ref][group] = (results_df, sig_df)

    return results_dict

import copy
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    from scipy.stats import chi2_contingency
    SCIPY_STATS_AVAILABLE = True
except Exception:
    SCIPY_STATS_AVAILABLE = False

# -----------------------------
# Compute positionwise statistic (multi-method + simple site_types)
# -----------------------------
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Dict, Any, Tuple
from contextlib import contextmanager
from joblib import Parallel, delayed, cpu_count
import joblib
from tqdm import tqdm
from scipy.stats import chi2_contingency
import warnings
import matplotlib.pyplot as plt
from itertools import cycle
import os
import warnings


# ---------------------------
# joblib <-> tqdm integration
# ---------------------------
@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to update a tqdm progress bar."""
    old = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old):  # type: ignore
        def __call__(self, *args, **kwargs):
            try:
                tqdm_object.update(n=self.batch_size)
            except Exception:
                tqdm_object.update(1)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old


# ---------------------------
# row workers (upper-triangle only)
# ---------------------------
def _chi2_row_job(i: int, X_bin: np.ndarray, min_count_for_pairwise: int) -> Tuple[int, np.ndarray]:
    n_pos = X_bin.shape[1]
    row = np.full((n_pos,), np.nan, dtype=float)
    xi = X_bin[:, i]
    for j in range(i, n_pos):
        xj = X_bin[:, j]
        mask = (~np.isnan(xi)) & (~np.isnan(xj))
        if int(mask.sum()) < int(min_count_for_pairwise):
            continue
        try:
            table = pd.crosstab(xi[mask], xj[mask])
            if table.shape != (2, 2):
                continue
            chi2, _, _, _ = chi2_contingency(table, correction=False)
            row[j] = float(chi2)
        except Exception:
            row[j] = np.nan
    return (i, row)


def _relative_risk_row_job(i: int, X_bin: np.ndarray, min_count_for_pairwise: int) -> Tuple[int, np.ndarray]:
    n_pos = X_bin.shape[1]
    row = np.full((n_pos,), np.nan, dtype=float)
    xi = X_bin[:, i]
    for j in range(i, n_pos):
        xj = X_bin[:, j]
        mask = (~np.isnan(xi)) & (~np.isnan(xj))
        if int(mask.sum()) < int(min_count_for_pairwise):
            continue
        a = np.sum((xi[mask] == 1) & (xj[mask] == 1))
        b = np.sum((xi[mask] == 1) & (xj[mask] == 0))
        c = np.sum((xi[mask] == 0) & (xj[mask] == 1))
        d = np.sum((xi[mask] == 0) & (xj[mask] == 0))
        try:
            if (a + b) > 0 and (c + d) > 0 and (c > 0):
                p1 = a / float(a + b)
                p2 = c / float(c + d)
                row[j] = float(p1 / p2) if p2 > 0 else np.nan
            else:
                row[j] = np.nan
        except Exception:
            row[j] = np.nan
    return (i, row)

def compute_positionwise_statistics(
    adata,
    layer: str,
    methods: Sequence[str] = ("pearson",),
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    site_types: Optional[Sequence[str]] = None,
    encoding: str = "signed",
    output_key: str = "positionwise_result",
    min_count_for_pairwise: int = 10,
    max_threads: Optional[int] = None,
    reverse_indices_on_store: bool = False,
):
    """
    Compute per-(sample,ref) positionwise matrices for methods in `methods`.

    Results stored at:
      adata.uns[output_key][method][ (sample, ref) ] = DataFrame
      adata.uns[output_key + "_n"][method][ (sample, ref) ] = int(n_reads)
    """
    if isinstance(methods, str):
        methods = [methods]
    methods = [m.lower() for m in methods]

    # prepare containers
    adata.uns[output_key] = {m: {} for m in methods}
    adata.uns[output_key + "_n"] = {m: {} for m in methods}

    # workers
    if max_threads is None or max_threads <= 0:
        n_jobs = max(1, cpu_count() or 1)
    else:
        n_jobs = max(1, int(max_threads))

    # samples / refs
    sseries = adata.obs[sample_col]
    if not pd.api.types.is_categorical_dtype(sseries):
        sseries = sseries.astype("category")
    samples = list(sseries.cat.categories)

    rseries = adata.obs[ref_col]
    if not pd.api.types.is_categorical_dtype(rseries):
        rseries = rseries.astype("category")
    references = list(rseries.cat.categories)

    total_tasks = len(samples) * len(references)
    pbar_outer = tqdm(total=total_tasks, desc="positionwise (sample x ref)", unit="cell")

    for sample in samples:
        for ref in references:
            label = (sample, ref)
            try:
                mask = (adata.obs[sample_col] == sample) & (adata.obs[ref_col] == ref)
                subset = adata[mask]
                n_reads = subset.shape[0]

                # nothing to do -> store empty placeholders
                if n_reads == 0:
                    for m in methods:
                        adata.uns[output_key][m][label] = pd.DataFrame()
                        adata.uns[output_key + "_n"][m][label] = 0
                    pbar_outer.update(1)
                    continue

                # select var columns based on site_types and reference
                if site_types:
                    col_mask = np.zeros(subset.shape[1], dtype=bool)
                    for st in site_types:
                        colname = f"{ref}_{st}"
                        if colname in subset.var.columns:
                            col_mask |= np.asarray(subset.var[colname].values, dtype=bool)
                        else:
                            # if mask not present, warn once (but keep searching)
                            # user may pass generic site types
                            pass
                    if not col_mask.any():
                        selected_var_idx = np.arange(subset.shape[1])
                    else:
                        selected_var_idx = np.nonzero(col_mask)[0]
                else:
                    selected_var_idx = np.arange(subset.shape[1])

                if selected_var_idx.size == 0:
                    for m in methods:
                        adata.uns[output_key][m][label] = pd.DataFrame()
                        adata.uns[output_key + "_n"][m][label] = int(n_reads)
                    pbar_outer.update(1)
                    continue

                # extract matrix
                if (layer in subset.layers) and (subset.layers[layer] is not None):
                    X = subset.layers[layer]
                else:
                    X = subset.X
                X = np.asarray(X, dtype=float)
                X = X[:, selected_var_idx]  # (n_reads, n_pos)

                # binary encoding
                if encoding == "signed":
                    X_bin = np.where(X == 1, 1.0, np.where(X == -1, 0.0, np.nan))
                else:
                    X_bin = np.where(X == 1, 1.0, np.where(X == 0, 0.0, np.nan))

                n_pos = X_bin.shape[1]
                if n_pos == 0:
                    for m in methods:
                        adata.uns[output_key][m][label] = pd.DataFrame()
                        adata.uns[output_key + "_n"][m][label] = int(n_reads)
                    pbar_outer.update(1)
                    continue

                var_names = list(subset.var_names[selected_var_idx])

                # compute per-method
                for method in methods:
                    m = method.lower()
                    if m == "pearson":
                        # pairwise Pearson with column demean (nan-aware approximation)
                        with np.errstate(invalid="ignore"):
                            col_mean = np.nanmean(X_bin, axis=0)
                            Xc = X_bin - col_mean  # nan preserved
                            Xc0 = np.nan_to_num(Xc, nan=0.0)
                            cov = Xc0.T @ Xc0
                            denom = (np.sqrt((Xc0**2).sum(axis=0))[:, None] * np.sqrt((Xc0**2).sum(axis=0))[None, :])
                            with np.errstate(divide="ignore", invalid="ignore"):
                                mat = np.where(denom != 0.0, cov / denom, np.nan)
                    elif m == "binary_covariance":
                        binary = (X_bin == 1).astype(float)
                        valid = (~np.isnan(X_bin)).astype(float)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            numerator = binary.T @ binary
                            denominator = valid.T @ valid
                            mat = np.true_divide(numerator, denominator)
                            mat[~np.isfinite(mat)] = 0.0
                    elif m in ("chi_squared", "relative_risk"):
                        if m == "chi_squared":
                            worker = _chi2_row_job
                        else:
                            worker = _relative_risk_row_job
                        out = np.full((n_pos, n_pos), np.nan, dtype=float)
                        tasks = (delayed(worker)(i, X_bin, min_count_for_pairwise) for i in range(n_pos))
                        pbar_rows = tqdm(total=n_pos, desc=f"{m}: rows ({sample}__{ref})", leave=False)
                        with tqdm_joblib(pbar_rows):
                            results = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
                        pbar_rows.close()
                        for i, row in results:
                            out[int(i), :] = row
                        iu = np.triu_indices(n_pos, k=1)
                        out[iu[1], iu[0]] = out[iu]
                        mat = out
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    # optionally reverse order at store-time
                    if reverse_indices_on_store:
                        mat_store = np.flip(np.flip(mat, axis=0), axis=1)
                        idx_names = var_names[::-1]
                    else:
                        mat_store = mat
                        idx_names = var_names

                    # make dataframe with labels
                    df = pd.DataFrame(mat_store, index=idx_names, columns=idx_names)

                    adata.uns[output_key][m][label] = df
                    adata.uns[output_key + "_n"][m][label] = int(n_reads)

            except Exception as exc:
                warnings.warn(f"Failed computing positionwise for {sample}__{ref}: {exc}")
            finally:
                pbar_outer.update(1)

    pbar_outer.close()
    return None


# ---------------------------
# Plotting function
# ---------------------------

def plot_positionwise_matrices(
    adata,
    methods: List[str],
    cmaps: Optional[List[str]] = None,
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    output_dir: Optional[str] = None,
    vmin: Optional[Dict[str, float]] = None,
    vmax: Optional[Dict[str, float]] = None,
    figsize_per_cell: Tuple[float, float] = (3.5, 3.5),
    dpi: int = 160,
    cbar_shrink: float = 0.9,
    output_key: str = "positionwise_result",
    show_colorbar: bool = True,
    flip_display_axes: bool = False,
    rows_per_page: int = 6,
    sample_label_rotation: float = 90.0,
):
    """
    Plot grids of matrices for each method with pagination and rotated sample-row labels.

    New args:
      - rows_per_page: how many sample rows per page/figure (pagination)
      - sample_label_rotation: rotation angle (deg) for the sample labels placed in the left margin.
    Returns:
      dict mapping method -> list of saved filenames (empty list if figures were shown).
    """
    if isinstance(methods, str):
        methods = [methods]
    if cmaps is None:
        cmaps = ["viridis"] * len(methods)
    cmap_cycle = cycle(cmaps)

    # canonicalize sample/ref order
    sseries = adata.obs[sample_col]
    if not pd.api.types.is_categorical_dtype(sseries):
        sseries = sseries.astype("category")
    samples = list(sseries.cat.categories)

    rseries = adata.obs[ref_col]
    if not pd.api.types.is_categorical_dtype(rseries):
        rseries = rseries.astype("category")
    references = list(rseries.cat.categories)

    # ensure directories
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    saved_files_by_method = {}

    def _get_df_from_store(store, sample, ref):
        """
        try multiple key formats: (sample, ref) tuple, 'sample__ref' string,
        or str(sample)+'__'+str(ref). Return None if not found.
        """
        if store is None:
            return None
        # try tuple key
        key_t = (sample, ref)
        if key_t in store:
            return store[key_t]
        # try string key
        key_s = f"{sample}__{ref}"
        if key_s in store:
            return store[key_s]
        # try stringified tuple keys (some callers store differently)
        for k in store.keys():
            try:
                if isinstance(k, tuple) and len(k) == 2 and str(k[0]) == str(sample) and str(k[1]) == str(ref):
                    return store[k]
                if isinstance(k, str) and key_s == k:
                    return store[k]
            except Exception:
                continue
        return None

    for method, cmap in zip(methods, cmap_cycle):
        m = method.lower()
        method_store = adata.uns.get(output_key, {}).get(m, {})
        if not method_store:
            warnings.warn(f"No results found for method '{method}' in adata.uns['{output_key}']. Skipping.", stacklevel=2)
            saved_files_by_method[method] = []
            continue

        # gather numeric values to pick sensible vmin/vmax when not provided
        vals = []
        for s in samples:
            for r in references:
                df = _get_df_from_store(method_store, s, r)
                if isinstance(df, pd.DataFrame) and df.size > 0:
                    a = df.values
                    a = a[np.isfinite(a)]
                    if a.size:
                        vals.append(a)
        if vals:
            allvals = np.concatenate(vals)
        else:
            allvals = np.array([])

        # decide per-method defaults
        if m == "pearson":
            vmn = -1.0 if (vmin is None or (isinstance(vmin, dict) and m not in vmin)) else (vmin.get(m) if isinstance(vmin, dict) else vmin)
            vmx = 1.0 if (vmax is None or (isinstance(vmax, dict) and m not in vmax)) else (vmax.get(m) if isinstance(vmax, dict) else vmax)
            vmn = -1.0 if vmn is None else vmn
            vmx = 1.0 if vmx is None else vmx
        elif m == "binary_covariance":
            vmn = 0.0 if (vmin is None or (isinstance(vmin, dict) and m not in vmin)) else (vmin.get(m) if isinstance(vmin, dict) else vmin)
            vmx = 1.0 if (vmax is None or (isinstance(vmax, dict) and m not in vmax)) else (vmax.get(m) if isinstance(vmax, dict) else vmax)
            vmn = 0.0 if vmn is None else vmn
            vmx = 1.0 if vmx is None else vmx
        else:
            vmn = 0.0 if (vmin is None or (isinstance(vmin, dict) and m not in vmin)) else (vmin.get(m) if isinstance(vmin, dict) else vmin)
            if (vmax is None) or (isinstance(vmax, dict) and m not in vmax):
                vmx = float(np.nanpercentile(allvals, 99.0)) if allvals.size else 1.0
            else:
                vmx = (vmax.get(m) if isinstance(vmax, dict) else vmax)
            vmn = 0.0 if vmn is None else vmn
            if vmx is None:
                vmx = 1.0

        # prepare pagination over sample rows
        saved_files = []
        n_pages = max(1, int(np.ceil(len(samples) / float(max(1, rows_per_page)))))
        for page_idx in range(n_pages):
            start = page_idx * rows_per_page
            chunk = samples[start : start + rows_per_page]
            nrows = len(chunk)
            ncols = max(1, len(references))
            fig_w = ncols * figsize_per_cell[0]
            fig_h = nrows * figsize_per_cell[1]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

            # leave margin for rotated sample labels
            plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.05)

            any_plotted = False
            im = None
            for r_idx, sample in enumerate(chunk):
                for c_idx, ref in enumerate(references):
                    ax = axes[r_idx][c_idx]
                    df = _get_df_from_store(method_store, sample, ref)
                    if not isinstance(df, pd.DataFrame) or df.size == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        mat = df.values.astype(float)
                        origin = "upper" if flip_display_axes else "lower"
                        im = ax.imshow(mat, origin=origin, aspect="auto", vmin=vmn, vmax=vmx, cmap=cmap)
                        any_plotted = True
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # top title is reference (only for top-row)
                    if r_idx == 0:
                        ax.set_title(str(ref), fontsize=9)

                # draw rotated sample label into left margin centered on the row
                # compute vertical center of this row's axis in figure coords
                ax0 = axes[r_idx][0]
                ax_y0, ax_y1 = ax0.get_position().y0, ax0.get_position().y1
                y_center = 0.5 * (ax_y0 + ax_y1)
                # place text at x=0.01 (just inside left margin); rotation controls orientation
                fig.text(0.01, y_center, str(chunk[r_idx]), va="center", ha="left", rotation=sample_label_rotation, fontsize=9)

            fig.suptitle(f"{method} — per-sample x per-reference matrices (page {page_idx+1}/{n_pages})", fontsize=12, y=0.99)
            fig.tight_layout(rect=[0.05, 0.02, 0.9, 0.96])

            # colorbar (shared)
            if any_plotted and show_colorbar and (im is not None):
                try:
                    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
                    fig.colorbar(im, cax=cbar_ax, shrink=cbar_shrink)
                except Exception:
                    try:
                        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
                    except Exception:
                        pass

            # save or show
            if output_dir:
                fname = f"positionwise_{method}_page{page_idx+1}.png"
                outpath = os.path.join(output_dir, fname)
                plt.savefig(outpath, bbox_inches="tight")
                saved_files.append(outpath)
                plt.close(fig)
            else:
                plt.show()
                saved_files.append("")  # placeholder to indicate a figure was shown

        saved_files_by_method[method] = saved_files

    return saved_files_by_method
