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
