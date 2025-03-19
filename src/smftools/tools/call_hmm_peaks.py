def call_hmm_peaks(adata, feature_configs, obs_column='Reference_strand', site_types=['GpC_site', 'CpG_site'], save_plot=False, output_dir=None, date_tag=None):
    """
    Calls peaks from HMM feature layers and annotates them into the AnnData object.

    Parameters:
        adata : AnnData object with HMM layers (from apply_hmm)
        feature_configs : dict
        min_distance : minimum distance between peaks
        peak_width : window size around peak centers
        peak_prominence : required peak prominence
        peak_threshold : threshold for labeling a read as "present" at a peak
        site_types : list of var site types to aggregate
        save_plot : whether to save the plot
        output_dir : path to save the figure if save_plot=True
        date_tag : optional tag for filename
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import os
    import numpy as np

    peak_columns = []
        
    for feature_layer, config in feature_configs.items():
        min_distance = config.get('min_distance', 200)
        peak_width = config.get('peak_width', 200)
        peak_prominence = config.get('peak_prominence', 0.2)
        peak_threshold = config.get('peak_threshold', 0.8)

        # 1️⃣ Calculate mean intensity profile
        matrix = adata.layers[feature_layer]
        means = np.mean(matrix, axis=0)
        feature_peak_columns = []

        # 2️⃣ Peak calling
        peak_centers, _ = find_peaks(means, prominence=peak_prominence, distance=min_distance)
        adata.uns[f'{feature_layer} peak_centers'] = peak_centers

        # 3️⃣ Plot
        plt.figure(figsize=(6, 3))
        plt.plot(range(len(means)), means)
        plt.title(f"{feature_layer} density with peak calls")
        plt.xlabel("Genomic position")
        plt.ylabel("Mean feature density")
        y = max(means) / 2
        for i, center in enumerate(peak_centers):
            plus_minus_width = peak_width // 2
            start = center - plus_minus_width
            end = center + plus_minus_width
            plt.axvspan(start, end, color='purple', alpha=0.2)
            plt.axvline(center, color='red', linestyle='--')
            if i%2:
                aligned = [end, 'left']
            else:
                aligned = [start, 'right']
            plt.text(aligned[0], 0, f"Peak {i}\n{center}", color='red', ha=aligned[1])

        if save_plot and output_dir:
            filename = f"{output_dir}/{date_tag or 'output'}_{feature_layer}_peaks.png"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Saved plot to {filename}")
        else:
            plt.show()

        # 4️⃣ Annotate peaks back into adata.obs
        for center in peak_centers:
            half_width = peak_width // 2
            start, end = center - half_width, center + half_width
            colname = f'{feature_layer}_peak_{center}'
            peak_columns.append(colname)
            feature_peak_columns.append(colname)

            adata.var[colname] = (
                (adata.var_names.astype(int) >= start) & 
                (adata.var_names.astype(int) <= end)
            )

            # Feature layer intensity around peak
            mean_values = np.mean(matrix[:, start:end+1], axis=1)
            sum_values = np.sum(matrix[:, start:end+1], axis=1)
            adata.obs[f'mean_{feature_layer}_around_{center}'] = mean_values
            adata.obs[f'sum_{feature_layer}_around_{center}'] = sum_values
            adata.obs[f'{feature_layer}_present_at_{center}'] = mean_values > peak_threshold

            # Site-type based aggregation
            for site_type in site_types:
                adata.obs[f'{site_type}_sum_around_{center}'] = 0
                adata.obs[f'{site_type}_mean_around_{center}'] = np.nan

            references = adata.obs[obs_column].cat.categories
            for ref in adata.obs[obs_column].cat.categories:
                subset = adata[adata.obs[obs_column] == ref]
                for site_type in site_types:
                    mask = subset.var.get(f'{ref}_{site_type}', None)
                    if mask is not None:
                        region_mask = (subset.var_names[mask].astype(int) >= start) & (subset.var_names[mask].astype(int) <= end)
                        region = subset[:, mask].X[:, region_mask]
                        adata.obs.loc[subset.obs.index, f'{site_type}_sum_around_{center}'] = np.nansum(region, axis=1)
                        adata.obs.loc[subset.obs.index, f'{site_type}_mean_around_{center}'] = np.nanmean(region, axis=1)

        adata.var[f'is_in_any_{feature_layer}_peak'] = adata.var[feature_peak_columns].any(axis=1)
        print(f"✅ Peak annotation completed for {feature_layer} with {len(peak_centers)} peaks.")

    # Combine all peaks into a single "is_in_any_peak" column
    adata.var['is_in_any_peak'] = adata.var[peak_columns].any(axis=1)
