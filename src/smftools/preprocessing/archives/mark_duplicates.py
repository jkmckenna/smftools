## mark_duplicates

def mark_duplicates(adata, layers, obs_column='Reference', sample_col='Sample_names', method='N_masked_distances', distance_thresholds={}):
    """
    Marks duplicates in the adata object.

    Parameters:
        adata (AnnData): An adata object.
        layers (list): A list of strings representing the layers to use.
        obs_column (str): A string representing the obs column name to first subset on. Default is 'Reference'.
        sample_col (str): A string representing the obs column name to second subset on. Default is 'Sample_names'.
        method (str): method to use for calculating the distance metric
        distance_thresholds (dict): A dictionary keyed by obs_column categories that points to a float corresponding to the distance threshold to apply. Default is an empty dict.
    
    Returns:
        None
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import networkx as nx
    from .binary_layers_to_ohe import binary_layers_to_ohe
    from .calculate_pairwise_differences import calculate_pairwise_differences
    from .min_non_diagonal import min_non_diagonal

    categories = adata.obs[obs_column].cat.categories 
    sample_names = adata.obs[sample_col].cat.categories 

    # Calculate the pairwise Hamming distances within each reference/sample set. Determine distance thresholds for each reference/sample pair
    adata.obs['Nearest_neighbor_Hamming_distance'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    cat_sample_dict = {}
    for cat in categories:
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        for sample in sample_names:
            sample_subset = cat_subset[cat_subset.obs[sample_col] == sample].copy()
            sample_subset = sample_subset[:, sample_subset.var[f'{cat}_any_C_site'] == True].copy() # only uses C sites from the converted strand
            # Encode sequencing reads as a one-hot-encodings
            print(f'One-hot encoding reads from {sample} on {cat}')
            cat_sample_dict[f'{cat}_{sample}_read_OHE_dict'] = binary_layers_to_ohe(sample_subset, layers, stack='hstack')
            # Unpack the read names and one hot encodings into lists
            read_names = []
            ohe_list = []
            for read_name, ohe in cat_sample_dict[f'{cat}_{sample}_read_OHE_dict'].items():
                read_names.append(read_name)
                ohe_list.append(ohe)
            # Calculate the pairwise hamming distances
            if method == 'N_masked_distances':
                print(f'Calculating N_masked_distances for {sample} on {cat} allele')
                distance_matrix = calculate_pairwise_differences(ohe_list)
            else:
                print(f'{method} for calculating differences is not available')
            n_reads = distance_matrix.shape[0]
            # Load the hamming matrix into a dataframe with index and column names as the read_names
            distance_df = pd.DataFrame(distance_matrix, index=read_names, columns=read_names)
            cat_sample_dict[f'Pairwise_Hamming_distance_within_{cat}_{sample}'] = distance_df
            
            if n_reads > 1:
                # Calculate the minimum non-self distance for every read in the reference and sample
                min_distance_values = min_non_diagonal(distance_matrix)
                min_distance_df = pd.DataFrame({'Nearest_neighbor_Hamming_distance': min_distance_values}, index=read_names)
                adata.obs.update(min_distance_df)

                if cat in distance_thresholds: 
                    adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}'] = distance_thresholds[cat]
                else: # eventually this should be written to use known PCR duplicate controls for thresholding.
                    # Generate a histogram of minimum non-self distances for each read
                    if n_reads > 3:
                        n_bins = n_reads // 4
                    else:
                        n_bins = 1
                    min_distance_bins = plt.hist(min_distance_values, bins=n_bins)
                    # Normalize the max value in any histogram bin to 1
                    normalized_min_distance_counts = min_distance_bins[0] / np.max(min_distance_bins[0])
                    # Extract the bin index of peak centers in the histogram
                    try:
                        peak_centers, _ = find_peaks(normalized_min_distance_counts, prominence=0.2, distance=5)
                        first_peak_index = peak_centers[0]
                        offset_index = first_peak_index-1
                        # Use the distance corresponding to the first peak as the threshold distance in graph construction
                        first_peak_distance = min_distance_bins[1][first_peak_index]
                        offset_distance = min_distance_bins[1][offset_index]
                    except:
                        offset_distance = normalized_min_distance_counts[0]
                    adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}'] = offset_distance
            else:
                adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}'] = 0

    ## Detect likely duplicate reads and mark them in the adata object.
    adata.obs['Marked_duplicate'] = pd.Series(False, index=adata.obs_names, dtype=bool)
    adata.obs['Unique_in_final_read_set'] = pd.Series(False, index=adata.obs_names, dtype=bool)
    adata.obs[f'Hamming_distance_cluster_within_{obs_column}_and_sample'] = pd.Series(-1, index=adata.obs_names, dtype=int)

    for cat in categories:
        for sample in sample_names:
            distance_df = cat_sample_dict[f'Pairwise_Hamming_distance_within_{cat}_{sample}']
            read_names = distance_df.index
            distance_matrix = distance_df.values
            n_reads = distance_matrix.shape[0]
            distance_threshold = adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}']
            # Initialize the read distance graph
            G = nx.Graph()
            # Add each read as a node to the graph
            G.add_nodes_from(range(n_reads))
            # Add edges based on the threshold
            for i in range(n_reads):
                for j in range(i + 1, n_reads):
                    if distance_matrix[i, j] <= distance_threshold:
                        G.add_edge(i, j)        
            # Determine distinct clusters using connected components
            clusters = list(nx.connected_components(G))
            clusters = [list(cluster) for cluster in clusters]
            # Get the number of clusters
            cluster_count = len(clusters)
            if n_reads > 0:
                fraction_unique = cluster_count / n_reads
            else:
                fraction_unique = 0
            adata.uns[f'Hamming_distance_cluster_count_within_{cat}_{sample}'] = cluster_count
            adata.uns[f'total_reads_within_{cat}_{sample}'] = n_reads
            # Update the adata object
            read_cluster_map = {}
            read_duplicate_map = {}
            read_keep_map = {}
            for i, cluster in enumerate(clusters):
                for j, read_index in enumerate(cluster):
                    read_name = read_names[read_index]
                    read_cluster_map[read_name] = i
                    if len(cluster) > 1:
                        read_duplicate_map[read_name] = True
                        if j == 0:
                            read_keep_map[read_name] = True
                        else:
                            read_keep_map[read_name] = False
                    elif len(cluster) == 1:
                        read_duplicate_map[read_name] = False
                        read_keep_map[read_name] = True
            cluster_df = pd.DataFrame.from_dict(read_cluster_map, orient='index', columns=[f'Hamming_distance_cluster_within_{obs_column}_and_sample'], dtype=int)
            duplicate_df = pd.DataFrame.from_dict(read_duplicate_map, orient='index', columns=['Marked_duplicate'], dtype=bool)
            keep_df = pd.DataFrame.from_dict(read_keep_map, orient='index', columns=['Unique_in_final_read_set'], dtype=bool)
            df_combined = pd.concat([cluster_df, duplicate_df, keep_df], axis=1)
            adata.obs.update(df_combined)
            adata.obs['Marked_duplicate'] = adata.obs['Marked_duplicate'].astype(bool)
            adata.obs['Unique_in_final_read_set'] = adata.obs['Unique_in_final_read_set'].astype(bool)
            print(f'Hamming clusters for {sample} on {cat}\nThreshold: {distance_threshold}\nNumber clusters: {cluster_count}\nNumber reads: {n_reads}\nFraction unique: {fraction_unique}')   