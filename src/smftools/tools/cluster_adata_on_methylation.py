# cluster_adata_on_methylation

def cluster_adata_on_methylation(adata, obs_columns, method='hierarchical', n_clusters=3, layer=None, site_types = ['GpC_site', 'CpG_site']):
    """
    Adds cluster groups to the adata object as an observation column

    Parameters:
        adata
        obs_columns
        method
        n_clusters
        layer
        site_types

    Returns:
        None
    """
    import pandas as pd
    import numpy as np
    from . import subset_adata
    from ..readwrite import adata_to_df

    # Ensure obs_columns are categorical
    for col in obs_columns:
        adata.obs[col] = adata.obs[col].astype('category')

    references = adata.obs['Reference'].cat.categories

    # Add subset metadata to the adata
    subset_adata(adata, obs_columns)

    subgroup_name = '_'.join(obs_columns)
    subgroups = adata.obs[subgroup_name].cat.categories

    subgroup_to_reference_map = {}
    for subgroup in subgroups:
        for reference in references:
            if reference in subgroup:
                subgroup_to_reference_map[subgroup] = reference
            else:
                pass

    if method == 'hierarchical':
        for site_type in site_types:
            adata.obs[f'{site_type}_{layer}_hierarchical_clustering_index_within_{subgroup_name}'] = pd.Series(-1, index=adata.obs_names, dtype=int)
    elif method == 'kmeans':
        for site_type in site_types:
            adata.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'] = pd.Series(-1, index=adata.obs_names, dtype=int)
        
    for subgroup in subgroups:
        subgroup_subset = adata[adata.obs[subgroup_name] == subgroup].copy()
        reference = subgroup_to_reference_map[subgroup]
        for site_type in site_types:
            site_subset = subgroup_subset[:, np.array(subgroup_subset.var[f'{reference}_{site_type}'])].copy()
            df = adata_to_df(site_subset, layer=layer)
            df2 = df.reset_index(drop=True)
            if method == 'hierarchical':
                try:
                    from scipy.cluster.hierarchy import linkage, dendrogram
                    # Perform hierarchical clustering on rows using the average linkage method and Euclidean metric
                    row_linkage = linkage(df2.values, method='average', metric='euclidean')

                    # Generate the dendrogram to get the ordered indices
                    dendro = dendrogram(row_linkage, no_plot=True)
                    reordered_row_indices = np.array(dendro['leaves']).astype(int)

                    # Get the reordered observation names
                    reordered_obs_names = [df.index[i] for i in reordered_row_indices]

                    temp_obs_data = pd.DataFrame({f'{site_type}_{layer}_hierarchical_clustering_index_within_{subgroup_name}': np.arange(0, len(reordered_obs_names), 1)}, index=reordered_obs_names, dtype=int)
                    adata.obs.update(temp_obs_data)
                except:
                    print(f'Error found in {subgroup} of {site_type}_{layer}_hierarchical_clustering_index_within_{subgroup_name}')
            elif method == 'kmeans':
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters)
                    kmeans.fit(site_subset.layers[layer])
                    # Get the cluster labels for each data point
                    cluster_labels = kmeans.labels_
                    # Add the kmeans cluster data as an observation to the anndata object
                    site_subset.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'] = cluster_labels.astype(str)
                    # Calculate the mean of each observation categoty of each cluster
                    cluster_means = site_subset.obs.groupby(f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}').mean()
                    # Sort the cluster indices by mean methylation value
                    sorted_clusters = cluster_means.sort_values(by=f'{site_type}_row_methylation_means', ascending=False).index
                    # Create a mapping of the old cluster values to the new cluster values
                    sorted_cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}
                    # Apply the mapping to create a new observation value: kmeans_labels_reordered
                    site_subset.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'] = site_subset.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'].map(sorted_cluster_mapping)
                    temp_obs_data = pd.DataFrame({f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}': site_subset.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}']}, index=site_subset.obs_names, dtype=int)
                    adata.obs.update(temp_obs_data)      
                except:
                        print(f'Error found in {subgroup} of {site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}')   

    if method == 'hierarchical':             
        # Ensure that the observation values are type int
        for site_type in site_types:
            adata.obs[f'{site_type}_{layer}_hierarchical_clustering_index_within_{subgroup_name}'] = adata.obs[f'{site_type}_{layer}_hierarchical_clustering_index_within_{subgroup_name}'].astype(int)
    elif method == 'kmeans':
        # Ensure that the observation values are type int
        for site_type in site_types:
            adata.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'] = adata.obs[f'{site_type}_{layer}_kmeans_clustering_index_within_{subgroup_name}'].astype(int)

    return None   