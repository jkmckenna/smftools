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


# ---------------------------
# Main compute function
# ---------------------------
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
):
    """
    Plot grids of matrices for each method.

    - methods: list of methods computed in compute_positionwise_statistics
    - cmaps: list of colormap names (cycled if shorter)
    - vmin/vmax: optional dicts mapping method->value or single scalar (global). If None, sensible defaults used:
         pearson -> [-1, 1], binary_covariance -> [0,1], chi_squared/relative_risk -> [0, 99th percentile]
    - output_key: where matrices were stored in adata.uns (compute_positionwise_statistics `output_key`)
    - flip_display_axes: if True, display with origin='upper' (reverses diagonal visually)
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

    for method, cmap in zip(methods, cmap_cycle):
        m = method.lower()
        method_store = adata.uns.get(output_key, {}).get(m, {})
        if not method_store:
            warnings.warn(f"No results found for method '{method}' in adata.uns['{output_key}'][{m}]; skipping.")
            continue

        # determine sensible vmin/vmax
        vals = []
        for s in samples:
            for r in references:
                df = method_store.get((s, r))
                if isinstance(df, pd.DataFrame) and df.size > 0:
                    a = df.values
                    a = a[np.isfinite(a)]
                    if a.size:
                        vals.append(a)
        if vals:
            allvals = np.concatenate(vals)
            if m == "pearson":
                vmn, vmx = (-1.0, 1.0) if vmin is None and vmax is None else (vmin.get(m) if isinstance(vmin, dict) and m in vmin else vmin, vmax.get(m) if isinstance(vmax, dict) and m in vmax else vmax)
                # if user passed None for one of them, fill sensible default
                vmn = -1.0 if vmn is None else vmn
                vmx = 1.0 if vmx is None else vmx
            elif m == "binary_covariance":
                vmn = 0.0 if (vmin is None or (isinstance(vmin, dict) and m not in vmin)) else (vmin.get(m) if isinstance(vmin, dict) else vmin)
                vmx = 1.0 if (vmax is None or (isinstance(vmax, dict) and m not in vmax)) else (vmax.get(m) if isinstance(vmax, dict) else vmax)
            else:
                # chi_squared or relative_risk: use 99th percentile as vmax default to avoid huge outliers
                vmn = 0.0 if (vmin is None or (isinstance(vmin, dict) and m not in vmin)) else (vmin.get(m) if isinstance(vmin, dict) else vmin)
                if (vmax is None) or (isinstance(vmax, dict) and m not in vmax):
                    vmx = float(np.nanpercentile(allvals, 99.0)) if allvals.size else 1.0
                else:
                    vmx = (vmax.get(m) if isinstance(vmax, dict) else vmax)
        else:
            # no data -> fallback
            vmn, vmx = (0.0, 1.0)

        # build figure grid
        nrows = max(1, len(samples))
        ncols = max(1, len(references))
        fig_w = ncols * figsize_per_cell[0]
        fig_h = nrows * figsize_per_cell[1]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

        # leave left margin for sample labels
        plt.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.05)

        any_plotted = False
        im = None
        for r_idx, sample in enumerate(samples):
            for c_idx, ref in enumerate(references):
                ax = axes[r_idx][c_idx]
                df = method_store.get((sample, ref))
                if not isinstance(df, pd.DataFrame) or df.size == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    mat = df.values.astype(float)
                    if flip_display_axes:
                        origin = "upper"
                    else:
                        origin = "lower"
                    im = ax.imshow(mat, origin=origin, aspect="auto", vmin=vmn, vmax=vmx, cmap=cmap)
                    any_plotted = True
                    ax.set_xticks([])
                    ax.set_yticks([])

                # top title is reference
                if r_idx == 0:
                    ax.set_title(str(ref), fontsize=9)

                # put sample name visibly on left of each row
                if c_idx == 0:
                    # draw sample label in left margin aligned with row center
                    # compute vertical center of axes in figure coordinates and place text there
                    ax_y0, ax_y1 = ax.get_position().y0, ax.get_position().y1
                    y_center = 0.5 * (ax_y0 + ax_y1)
                    fig.text(0.01, y_center, str(sample), va="center", ha="left", fontsize=9)

        fig.suptitle(f"{method} — per-sample x per-reference matrices", fontsize=12, y=0.99)
        fig.tight_layout(rect=[0.05, 0.02, 0.9, 0.96])

        # colorbar
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
        fname = f"positionwise_{method}.png"
        if output_dir:
            outpath = os.path.join(output_dir, fname)
            plt.savefig(outpath, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    return None

# def compute_positionwise_statistic(
#     adata,
#     layer,
#     method="pearson",
#     groupby=["Reference_strand"],
#     output_key="positionwise_result",
#     site_config=None,
#     encoding="signed",
#     max_threads=None
# ):
#     """
#     Computes a position-by-position matrix (correlation, RR, or Chi-squared) from an adata layer.

#     Parameters:
#         adata (AnnData): Annotated data matrix.
#         layer (str): Name of the adata layer to use.
#         method (str): 'pearson', 'binary_covariance', 'relative_risk', or 'chi_squared'.
#         groupby (str or list): Column(s) in adata.obs to group by.
#         output_key (str): Key in adata.uns to store results.
#         site_config (dict): Optional {ref: [site_types]} to restrict sites per reference.
#         encoding (str): 'signed' (1/-1/0) or 'binary' (1/0/NaN).
#         max_threads (int): Number of parallel threads to use (joblib).
#     """
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import chi2_contingency
#     from joblib import Parallel, delayed
#     from tqdm import tqdm

#     if isinstance(groupby, str):
#         groupby = [groupby]

#     adata.uns[output_key] = {}
#     adata.uns[output_key + "_n"] = {}

#     label_col = "__".join(groupby)
#     adata.obs[label_col] = adata.obs[groupby].astype(str).agg("_".join, axis=1)

#     for group in adata.obs[label_col].unique():
#         subset = adata[adata.obs[label_col] == group].copy()
#         if subset.shape[0] == 0:
#             continue

#         ref = subset.obs["Reference_strand"].unique()[0] if "Reference_strand" in groupby else None

#         if site_config and ref in site_config:
#             site_mask = np.zeros(subset.shape[1], dtype=bool)
#             for site in site_config[ref]:
#                 site_mask |= subset.var[f"{ref}_{site}"]
#             subset = subset[:, site_mask].copy()

#         X = subset.layers[layer].copy()

#         if encoding == "signed":
#             X_bin = np.where(X == 1, 1, np.where(X == -1, 0, np.nan))
#         else:
#             X_bin = np.where(X == 1, 1, np.where(X == 0, 0, np.nan))

#         n_pos = subset.shape[1]
#         mat = np.zeros((n_pos, n_pos))

#         if method == "pearson":
#             with np.errstate(invalid='ignore'):
#                 mat = np.corrcoef(np.nan_to_num(X_bin).T)

#         elif method == "binary_covariance":
#             binary = (X_bin == 1).astype(float)
#             valid = (X_bin == 1) | (X_bin == 0)  # Only consider true binary (ignore NaN)
#             valid = valid.astype(float)

#             numerator = np.dot(binary.T, binary)
#             denominator = np.dot(valid.T, valid)

#             with np.errstate(divide='ignore', invalid='ignore'):
#                 mat = np.true_divide(numerator, denominator)
#                 mat[~np.isfinite(mat)] = 0

#         elif method in ["relative_risk", "chi_squared"]:
#             def compute_row(i):
#                 row = np.zeros(n_pos)
#                 xi = X_bin[:, i]
#                 for j in range(n_pos):
#                     xj = X_bin[:, j]
#                     mask = ~np.isnan(xi) & ~np.isnan(xj)
#                     if np.sum(mask) < 10:
#                         row[j] = np.nan
#                         continue
#                     if method == "relative_risk":
#                         a = np.sum((xi[mask] == 1) & (xj[mask] == 1))
#                         b = np.sum((xi[mask] == 1) & (xj[mask] == 0))
#                         c = np.sum((xi[mask] == 0) & (xj[mask] == 1))
#                         d = np.sum((xi[mask] == 0) & (xj[mask] == 0))
#                         if (a + b) > 0 and (c + d) > 0 and c > 0:
#                             p1 = a / (a + b)
#                             p2 = c / (c + d)
#                             row[j] = p1 / p2 if p2 > 0 else np.nan
#                         else:
#                             row[j] = np.nan
#                     elif method == "chi_squared":
#                         table = pd.crosstab(xi[mask], xj[mask])
#                         if table.shape != (2, 2):
#                             row[j] = np.nan
#                         else:
#                             chi2, _, _, _ = chi2_contingency(table, correction=False)
#                             row[j] = chi2
#                 return row

#             mat = np.array(
#                 Parallel(n_jobs=max_threads)(
#                     delayed(compute_row)(i) for i in tqdm(range(n_pos), desc=f"{method}: {group}")
#                 )
#             )

#         else:
#             raise ValueError(f"Unsupported method: {method}")

#         var_names = subset.var_names.astype(int)
#         mat_df = pd.DataFrame(mat, index=var_names, columns=var_names)
#         adata.uns[output_key][group] = mat_df
#         adata.uns[output_key + "_n"][group] = subset.shape[0]