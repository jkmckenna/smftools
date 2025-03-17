import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def apply_hmm(adata, model, obs_column, layer=None, footprints=True, accessible_patches=False, cpg=False, methbases=["GpC", "CpG", "A"], device="cpu", threshold=0.7):
    """
    Applies an HMM model to an AnnData object using tensor-based sequence inputs.
    If multiple methbases are passed, generates a combined feature set.
    """
    model.to(device)

    # --- Feature Definitions ---
    feature_sets = {}
    if footprints:
        feature_sets["footprint"] = {
            "features": {
                "small_bound_stretch": [0, 30],
                "medium_bound_stretch": [30, 80],
                "putative_nucleosome": [80, 200],
                "large_bound_stretch": [200, np.inf]
            },
            "state": "Non-Methylated"
        }
    if accessible_patches:
        feature_sets["accessible"] = {
            "features": {
                "small_accessible_patch": [0, 30],
                "mid_accessible_patch": [30, 80],
                "large_accessible_patch": [80, np.inf]
            },
            "state": "Methylated"
        }
    if cpg:
        feature_sets["cpg"] = {
            "features": {
                "cpg_patch": [0, np.inf]
            },
            "state": "Methylated"
        }

    # --- Init columns ---
    all_features = []
    combined_prefix = "Combined"
    for key, fs in feature_sets.items():
        if key == 'cpg':
            all_features += [f"CpG_{f}" for f in fs["features"]]
            all_features.append(f"CpG_all_{key}_features")
        else:
            for methbase in methbases:
                all_features += [f"{methbase}_{f}" for f in fs["features"]]
                all_features.append(f"{methbase}_all_{key}_features")
            all_features += [f"{combined_prefix}_{f}" for f in fs["features"]]
            all_features.append(f"{combined_prefix}_all_{key}_features")

    for feature in all_features:
        adata.obs[feature] = pd.Series([[] for _ in range(adata.shape[0])], dtype=object, index=adata.obs.index)
        adata.obs[f"{feature}_distances"] = pd.Series([None] * adata.shape[0])
        adata.obs[f"n_{feature}"] = -1

    # --- Main loop ---
    references = adata.obs[obs_column].cat.categories

    for ref in tqdm(references, desc="Processing References"):
        ref_subset = adata[adata.obs[obs_column] == ref]

        # Create combined mask for methbases
        combined_mask = None
        for methbase in methbases:
            mask = {
                "a": ref_subset.var[f"{ref}_strand_FASTA_base"] == "A",
                "gpc": ref_subset.var[f"{ref}_GpC_site"] == True,
                "cpg": ref_subset.var[f"{ref}_CpG_site"] == True
            }[methbase.lower()]
            combined_mask = mask if combined_mask is None else combined_mask | mask

            methbase_subset = ref_subset[:, mask]
            matrix = methbase_subset.layers[layer] if layer else methbase_subset.X

            for i, raw_read in enumerate(matrix):
                read = [int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in raw_read]
                tensor_read = torch.tensor(read, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)
                coords = methbase_subset.var_names

                for key, fs in feature_sets.items():
                    if key == 'cpg':
                        continue
                    state_target = fs["state"]
                    feature_map = fs["features"]

                    classifications = classify_features(tensor_read, model, coords, feature_map, target_state=state_target)
                    idx = methbase_subset.obs.index[i]

                    for start, length, label, prob in classifications:
                        adata.obs.at[idx, f"{methbase}_{label}"].append([start, length, prob])
                        adata.obs.at[idx, f"{methbase}_all_{key}_features"].append([start, length, prob])

        # Combined methbase subset
        combined_subset = ref_subset[:, combined_mask]
        combined_matrix = combined_subset.layers[layer] if layer else combined_subset.X

        for i, raw_read in enumerate(combined_matrix):
            read = [int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in raw_read]
            tensor_read = torch.tensor(read, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)
            coords = combined_subset.var_names

            for key, fs in feature_sets.items():
                if key == 'cpg':
                    continue
                state_target = fs["state"]
                feature_map = fs["features"]

                classifications = classify_features(tensor_read, model, coords, feature_map, target_state=state_target)
                idx = combined_subset.obs.index[i]

                for start, length, label, prob in classifications:
                    adata.obs.at[idx, f"{combined_prefix}_{label}"].append([start, length, prob])
                    adata.obs.at[idx, f"{combined_prefix}_all_{key}_features"].append([start, length, prob])

    # --- Special handling for CpG ---
    if cpg:
        for ref in tqdm(references, desc="Processing CpG"):
            ref_subset = adata[adata.obs[obs_column] == ref]
            mask = (ref_subset.var[f"{ref}_CpG_site"] == True)
            cpg_subset = ref_subset[:, mask]
            matrix = cpg_subset.layers[layer] if layer else cpg_subset.X

            for i, raw_read in enumerate(matrix):
                read = [int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in raw_read]
                tensor_read = torch.tensor(read, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)
                coords = cpg_subset.var_names
                fs = feature_sets['cpg']
                state_target = fs["state"]
                feature_map = fs["features"]
                classifications = classify_features(tensor_read, model, coords, feature_map, target_state=state_target)
                idx = cpg_subset.obs.index[i]
                for start, length, label, prob in classifications:
                    adata.obs.at[idx, f"CpG_{label}"].append([start, length, prob])
                    adata.obs.at[idx, f"CpG_all_cpg_features"].append([start, length, prob])

    # --- Binarization + Distance ---
    for feature in tqdm(all_features, desc="Finalizing Layers"):
        bin_matrix = np.zeros((adata.shape[0], adata.shape[1]), dtype=int)
        counts = np.zeros(adata.shape[0], dtype=int)
        for row_idx, intervals in enumerate(adata.obs[feature]):
            if not isinstance(intervals, list):
                intervals = []
            for start, length, prob in intervals:
                if prob > threshold:
                    bin_matrix[row_idx, start:start+length] = 1
                    counts[row_idx] += 1
        adata.layers[f"{feature}"] = bin_matrix
        adata.obs[f"n_{feature}"] = counts
        adata.obs[f"{feature}_distances"] = adata.obs[feature].apply(lambda x: calculate_distances(x, threshold))

def calculate_distances(intervals, threshold=0.9):
    """Calculates distances between consecutive features in a read."""
    intervals = sorted([iv for iv in intervals if iv[2] > threshold], key=lambda x: x[0])
    distances = [(intervals[i + 1][0] - (intervals[i][0] + intervals[i][1]))
                 for i in range(len(intervals) - 1)]
    return distances


def classify_features(sequence, model, coordinates, classification_mapping={}, target_state="Methylated"):
    """
    Classifies regions based on HMM state.

    Parameters:
        sequence (torch.Tensor): Tensor of binarized data [batch_size, seq_len, 1]
        model: Trained pomegranate HMM
        coordinates (list): Genomic coordinates for sequence
        classification_mapping (dict): Mapping for feature labeling
        target_state (str): The state to classify ("Methylated" or "Non-Methylated")
    """
    predicted_states = model.predict(sequence).squeeze(-1).squeeze(0).cpu().numpy()
    probabilities = model.predict_proba(sequence).squeeze(0).cpu().numpy()
    state_labels = ["Non-Methylated", "Methylated"]
    
    classifications, current_start, current_length, current_probs = [], None, 0, []

    for i, state_index in enumerate(predicted_states):
        state_name = state_labels[state_index]
        state_prob = probabilities[i][state_index]

        if state_name == target_state:
            if current_start is None:
                current_start = i
            current_length += 1
            current_probs.append(state_prob)
        elif current_start is not None:
            classifications.append((current_start, current_length, avg := np.mean(current_probs)))
            current_start, current_length, current_probs = None, 0, []

    if current_start is not None:
        classifications.append((current_start, current_length, avg := np.mean(current_probs)))

    final = []
    for start, length, prob in classifications:
        feature_length = int(coordinates[start + length - 1]) - int(coordinates[start]) + 1
        label = next((ftype for ftype, rng in classification_mapping.items() if rng[0] <= feature_length < rng[1]), target_state)
        final.append((int(coordinates[start]) + 1, feature_length, label, prob))
    return final