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

def compute_positionwise_statistic(
    adata,
    layer,
    method="pearson",
    groupby=["Reference_strand"],
    output_key="positionwise_result",
    site_config=None,
    encoding="signed",
    max_threads=None
):
    """
    Computes a position-by-position matrix (correlation, RR, or Chi-squared) from an adata layer.

    Parameters:
        adata (AnnData): Annotated data matrix.
        layer (str): Name of the adata layer to use.
        method (str): 'pearson', 'binary_covariance', 'relative_risk', or 'chi_squared'.
        groupby (str or list): Column(s) in adata.obs to group by.
        output_key (str): Key in adata.uns to store results.
        site_config (dict): Optional {ref: [site_types]} to restrict sites per reference.
        encoding (str): 'signed' (1/-1/0) or 'binary' (1/0/NaN).
        max_threads (int): Number of parallel threads to use (joblib).
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency
    from joblib import Parallel, delayed
    from tqdm import tqdm

    if isinstance(groupby, str):
        groupby = [groupby]

    adata.uns[output_key] = {}
    adata.uns[output_key + "_n"] = {}

    label_col = "__".join(groupby)
    adata.obs[label_col] = adata.obs[groupby].astype(str).agg("_".join, axis=1)

    for group in adata.obs[label_col].unique():
        subset = adata[adata.obs[label_col] == group].copy()
        if subset.shape[0] == 0:
            continue

        ref = subset.obs["Reference_strand"].unique()[0] if "Reference_strand" in groupby else None

        if site_config and ref in site_config:
            site_mask = np.zeros(subset.shape[1], dtype=bool)
            for site in site_config[ref]:
                site_mask |= subset.var[f"{ref}_{site}"]
            subset = subset[:, site_mask].copy()

        X = subset.layers[layer].copy()

        if encoding == "signed":
            X_bin = np.where(X == 1, 1, np.where(X == -1, 0, np.nan))
        else:
            X_bin = np.where(X == 1, 1, np.where(X == 0, 0, np.nan))

        n_pos = subset.shape[1]
        mat = np.zeros((n_pos, n_pos))

        if method == "pearson":
            with np.errstate(invalid='ignore'):
                mat = np.corrcoef(np.nan_to_num(X_bin).T)

        elif method == "binary_covariance":
            binary = (X_bin == 1).astype(float)
            valid = (X_bin == 1) | (X_bin == 0)  # Only consider true binary (ignore NaN)
            valid = valid.astype(float)

            numerator = np.dot(binary.T, binary)
            denominator = np.dot(valid.T, valid)

            with np.errstate(divide='ignore', invalid='ignore'):
                mat = np.true_divide(numerator, denominator)
                mat[~np.isfinite(mat)] = 0

        elif method in ["relative_risk", "chi_squared"]:
            def compute_row(i):
                row = np.zeros(n_pos)
                xi = X_bin[:, i]
                for j in range(n_pos):
                    xj = X_bin[:, j]
                    mask = ~np.isnan(xi) & ~np.isnan(xj)
                    if np.sum(mask) < 10:
                        row[j] = np.nan
                        continue
                    if method == "relative_risk":
                        a = np.sum((xi[mask] == 1) & (xj[mask] == 1))
                        b = np.sum((xi[mask] == 1) & (xj[mask] == 0))
                        c = np.sum((xi[mask] == 0) & (xj[mask] == 1))
                        d = np.sum((xi[mask] == 0) & (xj[mask] == 0))
                        if (a + b) > 0 and (c + d) > 0 and c > 0:
                            p1 = a / (a + b)
                            p2 = c / (c + d)
                            row[j] = p1 / p2 if p2 > 0 else np.nan
                        else:
                            row[j] = np.nan
                    elif method == "chi_squared":
                        table = pd.crosstab(xi[mask], xj[mask])
                        if table.shape != (2, 2):
                            row[j] = np.nan
                        else:
                            chi2, _, _, _ = chi2_contingency(table, correction=False)
                            row[j] = chi2
                return row

            mat = np.array(
                Parallel(n_jobs=max_threads)(
                    delayed(compute_row)(i) for i in tqdm(range(n_pos), desc=f"{method}: {group}")
                )
            )

        else:
            raise ValueError(f"Unsupported method: {method}")

        var_names = subset.var_names.astype(int)
        mat_df = pd.DataFrame(mat, index=var_names, columns=var_names)
        adata.uns[output_key][group] = mat_df
        adata.uns[output_key + "_n"][group] = subset.shape[0]