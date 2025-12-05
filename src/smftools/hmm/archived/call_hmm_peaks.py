def call_hmm_peaks(
    adata,
    feature_configs,
    obs_column='Reference_strand',
    site_types=['GpC_site', 'CpG_site'],
    save_plot=False,
    output_dir=None,
    date_tag=None,
    inplace=False
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    if not inplace:
        adata = adata.copy()

    # Ensure obs_column is categorical
    if not isinstance(adata.obs[obs_column].dtype, pd.CategoricalDtype):
        adata.obs[obs_column] = pd.Categorical(adata.obs[obs_column])

    coordinates = adata.var_names.astype(int).values
    peak_columns = []

    obs_updates = {}

    for feature_layer, config in feature_configs.items():
        min_distance = config.get('min_distance', 200)
        peak_width = config.get('peak_width', 200)
        peak_prominence = config.get('peak_prominence', 0.2)
        peak_threshold = config.get('peak_threshold', 0.8)

        matrix = adata.layers[feature_layer]
        means = np.mean(matrix, axis=0)
        peak_indices, _ = find_peaks(means, prominence=peak_prominence, distance=min_distance)
        peak_centers = coordinates[peak_indices]
        adata.uns[f'{feature_layer} peak_centers'] = peak_centers.tolist()

        # Plot
        plt.figure(figsize=(6, 3))
        plt.plot(coordinates, means)
        plt.title(f"{feature_layer} with peak calls")
        plt.xlabel("Genomic position")
        plt.ylabel("Mean intensity")
        for i, center in enumerate(peak_centers):
            start, end = center - peak_width // 2, center + peak_width // 2
            plt.axvspan(start, end, color='purple', alpha=0.2)
            plt.axvline(center, color='red', linestyle='--')
            aligned = [end if i % 2 else start, 'left' if i % 2 else 'right']
            plt.text(aligned[0], 0, f"Peak {i}\n{center}", color='red', ha=aligned[1])
        if save_plot and output_dir:
            filename = f"{output_dir}/{date_tag or 'output'}_{feature_layer}_peaks.png"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Saved plot to {filename}")
        else:
            plt.show()

        feature_peak_columns = []
        for center in peak_centers:
            start, end = center - peak_width // 2, center + peak_width // 2
            colname = f'{feature_layer}_peak_{center}'
            peak_columns.append(colname)
            feature_peak_columns.append(colname)

            peak_mask = (coordinates >= start) & (coordinates <= end)
            adata.var[colname] = peak_mask

            region = matrix[:, peak_mask]
            obs_updates[f'mean_{feature_layer}_around_{center}'] = np.mean(region, axis=1)
            obs_updates[f'sum_{feature_layer}_around_{center}'] = np.sum(region, axis=1)
            obs_updates[f'{feature_layer}_present_at_{center}'] = np.mean(region, axis=1) > peak_threshold

            for site_type in site_types:
                adata.obs[f'{site_type}_sum_around_{center}'] = 0
                adata.obs[f'{site_type}_mean_around_{center}'] = np.nan

            for ref in adata.obs[obs_column].cat.categories:
                ref_idx = adata.obs[obs_column] == ref
                mask_key = f"{ref}_{site_type}"
                for site_type in site_types:
                    if mask_key not in adata.var:
                        continue
                    site_mask = adata.var[mask_key].values
                    site_coords = coordinates[site_mask]
                    region_mask = (site_coords >= start) & (site_coords <= end)
                    if not region_mask.any():
                        continue
                    full_mask = site_mask.copy()
                    full_mask[site_mask] = region_mask
                    site_region = adata[ref_idx, full_mask].X
                    if hasattr(site_region, "A"):
                        site_region = site_region.A
                    if site_region.shape[1] > 0:
                        adata.obs.loc[ref_idx, f'{site_type}_sum_around_{center}'] = np.nansum(site_region, axis=1)
                        adata.obs.loc[ref_idx, f'{site_type}_mean_around_{center}'] = np.nanmean(site_region, axis=1)
                    else:
                        pass

        adata.var[f'is_in_any_{feature_layer}_peak'] = adata.var[feature_peak_columns].any(axis=1)
        print(f"Annotated {len(peak_centers)} peaks for {feature_layer}")

    adata.var['is_in_any_peak'] = adata.var[peak_columns].any(axis=1)
    adata.obs = pd.concat([adata.obs, pd.DataFrame(obs_updates, index=adata.obs.index)], axis=1)

    return adata if not inplace else None