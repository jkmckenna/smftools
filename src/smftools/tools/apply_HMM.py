# apply_HMM

def apply_HMM(adata, model, obs_column, layer=None, footprints=True, accessible_patches=False, cpg=False, methbases=["Combined","C"]):
    """
    Applies an HMM model to an anndata object

    Parameters:
        adata (Anndata) : Anndata object to apply HMM to
        model (): a trained pomegranate HMM

    Returns:
        None
    """
    import numpy as np
    import pandas as pd
    from . import classify_non_methylated_features, classify_methylated_features, calculate_distances
    from tqdm import tqdm

    footprint_features =  {
        "small_bound_stretch": [0, 80],
        "putative_nucleosome": [80, 200],
        "large_bound_stretch": [200, np.inf]

    }

    accessible_features = {
        "small_accessible_patch": [0, 30],
        "mid_accessible_patch": [30, 80],
        "large_accessible_patch": [80, np.inf]
    }

    cpg_features = {
        "cpg_patch": [0, np.inf]
    }

    all_features = []
    if footprints:
        for methbase in methbases:
            all_features += [f"{methbase}_{feature}" for feature in footprint_features]
            all_features += [f'{methbase}_all_footprint_features']   
    if accessible_patches:
        for methbase in methbases:
            all_features += [f"{methbase}_{feature}" for feature in accessible_features]
            all_features += [f'{methbase}_all_accessible_features']   
    if cpg:
        all_features += [feature for feature in cpg_features]
        all_features += ['all_cpg_features']

    # Initialize columns for classifications
    for feature in all_features:
        adata.obs[feature] = pd.Series([[] for _ in range(adata.shape[0])], index=adata.obs.index, dtype='object')
        adata.obs[f'{feature}_distances'] = pd.Series([None for _ in range(adata.shape[0])], index=adata.obs.index, dtype='object')
        adata.obs[f'n_{feature}'] = pd.Series([-1 for _ in range(adata.shape[0])], index=adata.obs.index, dtype=int)

    # Apply the HMM and add the classifications to the observation metadata.

    references = adata.obs[obs_column].cat.categories

    if footprints:
        for reference in tqdm(references, total=len(references), desc="Processing footprint features"):
            reference_subset = adata[adata.obs[obs_column] == reference]
            site_subset = reference_subset[:, reference_subset.var[f'{reference}_position_passed_QC'] == True]
            for methbase in methbases:
                if methbase.lower() == 'combined':
                    subset_condition = (site_subset.var[f"{reference}_strand_FASTA_base_at_coordinate"] == "A") | (site_subset.var[f"{reference}_GpC_site"] == True)
                elif methbase.lower() == 'a':
                    subset_condition = (site_subset.var[f"{reference}_strand_FASTA_base_at_coordinate"] == "A")
                elif methbase.lower() == 'c':
                    subset_condition = (site_subset.var[f"{reference}_GpC_site"] == True)
                methbase_subset = site_subset[:, subset_condition]
                if layer:
                    matrix = methbase_subset.layers[layer]
                else:
                    matrix = methbase_subset.X
                total_reads = len(matrix)
                for i, read in enumerate(matrix):
                    index = methbase_subset.obs.index[i]
                    classifications = classify_non_methylated_features(read, model, methbase_subset.var_names, footprint_features)
                    for start, length, classification, prob in classifications:
                        try:
                            adata.obs.at[index, f"{methbase}_{classification}"].append([start, length, prob])
                            adata.obs.at[index, f'{methbase}_all_footprint_features'].append([start, length, prob])
                        except:
                            pass

    if accessible_patches:
        for reference in tqdm(references, total=len(references), desc="Processing accessible features"):
            reference_subset = adata[adata.obs[obs_column] == reference]
            site_subset = reference_subset[:, reference_subset.var[f'{reference}_position_passed_QC'] == True]
            for methbase in methbases:
                if methbase.lower() == 'combined':
                    subset_condition = (site_subset.var[f"{reference}_strand_FASTA_base_at_coordinate"] == "A") | (site_subset.var[f"{reference}_GpC_site"] == True)
                elif methbase.lower() == 'a':
                    subset_condition = (site_subset.var[f"{reference}_strand_FASTA_base_at_coordinate"] == "A")
                elif methbase.lower() == 'c':
                    subset_condition = (site_subset.var[f"{reference}_GpC_site"] == True)            
                methbase_subset = site_subset[:, subset_condition]
                if layer:
                    matrix = methbase_subset.layers[layer]
                else:
                    matrix = methbase_subset.X
                total_reads = len(matrix)
                for i, read in enumerate(matrix):
                    index = methbase_subset.obs.index[i]
                    classifications = classify_methylated_features(read, model, methbase_subset.var_names, accessible_features)
                    for start, length, classification, prob in classifications:
                        try:
                            adata.obs.at[index, f"{methbase}_{classification}"].append([start, length, prob])
                            adata.obs.at[index, f'{methbase}_all_accessible_features'].append([start, length, prob])
                        except:
                            pass

    if cpg:
        for reference in tqdm(references, total=len(references), desc="Processing cpg features"):
            reference_subset = adata[adata.obs[obs_column] == reference]
            site_subset = reference_subset[:, reference_subset.var[f'{reference}_position_passed_QC'] == True]
            subset_condition = (adata.var[f"{reference}_CpG_site"] == True)
            site_subset = site_subset[:, subset_condition]
            if layer:
                matrix = site_subset.layers[layer]
            else:
                matrix = site_subset.X
            total_reads = len(matrix)
            for i, read in enumerate(matrix):
                index = site_subset.obs.index[i]
                classifications = classify_methylated_features(read, model, site_subset.var_names, cpg_features)
                for start, length, classification, prob in classifications:
                    try:
                        adata.obs.at[index, classification].append([start, length, prob])
                        adata.obs.at[index, 'all_cpg_features'].append([start, length, prob])
                    except:
                        pass

    # Calculate distances between homotypic feature boundaries
    for feature in all_features:
        adata.obs[f'{feature}_distances'] = adata.obs[feature].apply(calculate_distances)
                
    ## Make a layer keyed by feature type that points to a binarized matrix for the feature
    threshold = 0.8 # Minimum probability threshold for adding feature to matrix
    positions = adata.var_names.astype(int).to_numpy()

    # Make new layers for each feature
    for feature in tqdm(all_features, total=len(all_features), desc="Adding features layers"):
        # Create a new matrix filled with zeros
        new_matrix = np.zeros((adata.shape[0],adata.shape[1]), dtype=int)
        # Init an empty list to hold feature counts
        feature_counts = np.zeros(adata.shape[0], dtype=int)
        # Update the new matrix based on intervals
        for read, intervals in enumerate(adata.obs[feature]):
            n_features = 0
            for start, length, prob in intervals:
                end = start + length
                if prob > threshold:
                    new_matrix[read, start:end] = 1
                    n_features += 1
                feature_counts[read] = n_features
        adata.layers[f'{feature}'] = new_matrix
        adata.obs[f'n_{feature}'] = feature_counts