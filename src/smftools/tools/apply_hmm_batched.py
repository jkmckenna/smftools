import numpy as np
import pandas as pd
import torch
from tqdm import tqdm    

def apply_hmm_batched(adata, model, obs_column, layer=None, footprints=True, accessible_patches=False, cpg=False, methbases=["GpC", "CpG", "A"], device="cpu", threshold=0.7):
    """
    Applies an HMM model to an AnnData object using tensor-based sequence inputs.
    If multiple methbases are passed, generates a combined feature set.
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    model.to(device)

    # --- Feature Definitions ---
    feature_sets = {}
    if footprints:
        feature_sets["footprint"] = {
            "features": {
                "small_bound_stretch": [0, 20],
                "medium_bound_stretch": [20, 50],
                "putative_nucleosome": [50, 200],
                "large_bound_stretch": [200, np.inf]
            },
            "state": "Non-Methylated"
        }
    if accessible_patches:
        feature_sets["accessible"] = {
            "features": {
                "small_accessible_patch": [0, 20],
                "mid_accessible_patch": [20, 80],
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
            if len(methbases) > 1:
                all_features += [f"{combined_prefix}_{f}" for f in fs["features"]]
                all_features.append(f"{combined_prefix}_all_{key}_features")

    for feature in all_features:
        adata.obs[feature] = [[] for _ in range(adata.shape[0])]
        adata.obs[f"{feature}_distances"] = [None] * adata.shape[0]
        adata.obs[f"n_{feature}"] = -1

    # --- Main loop ---
    references = adata.obs[obs_column].cat.categories

    for ref in tqdm(references, desc="Processing References"):
        ref_subset = adata[adata.obs[obs_column] == ref]

        # Combined methbase mask
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

            processed_reads = [[int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in read] for read in matrix]
            tensor_batch = torch.tensor(processed_reads, dtype=torch.long, device=device).unsqueeze(-1)

            coords = methbase_subset.var_names
            for key, fs in feature_sets.items():
                if key == 'cpg':
                    continue
                state_target = fs["state"]
                feature_map = fs["features"]

                pred_states = model.predict(tensor_batch)
                probs = model.predict_proba(tensor_batch)
                classifications = classify_batch(pred_states, probs, coords, feature_map, target_state=state_target)

                for i, idx in enumerate(methbase_subset.obs.index):
                    for start, length, label, prob in classifications[i]:
                        adata.obs.at[idx, f"{methbase}_{label}"].append([start, length, prob])
                        adata.obs.at[idx, f"{methbase}_all_{key}_features"].append([start, length, prob])

        # Combined subset
        if len(methbases) > 1:
            combined_subset = ref_subset[:, combined_mask]
            combined_matrix = combined_subset.layers[layer] if layer else combined_subset.X
            processed_combined_reads = [[int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in read] for read in combined_matrix]
            tensor_combined_batch = torch.tensor(processed_combined_reads, dtype=torch.long, device=device).unsqueeze(-1)

            coords = combined_subset.var_names
            for key, fs in feature_sets.items():
                if key == 'cpg':
                    continue
                state_target = fs["state"]
                feature_map = fs["features"]

                pred_states = model.predict(tensor_combined_batch)
                probs = model.predict_proba(tensor_combined_batch)
                classifications = classify_batch(pred_states, probs, coords, feature_map, target_state=state_target)

                for i, idx in enumerate(combined_subset.obs.index):
                    for start, length, label, prob in classifications[i]:
                        adata.obs.at[idx, f"{combined_prefix}_{label}"].append([start, length, prob])
                        adata.obs.at[idx, f"{combined_prefix}_all_{key}_features"].append([start, length, prob])

    # --- Special handling for CpG ---
    if cpg:
        for ref in tqdm(references, desc="Processing CpG"):
            ref_subset = adata[adata.obs[obs_column] == ref]
            mask = (ref_subset.var[f"{ref}_CpG_site"] == True)
            cpg_subset = ref_subset[:, mask]
            matrix = cpg_subset.layers[layer] if layer else cpg_subset.X

            processed_reads = [[int(x) if not np.isnan(x) else np.random.choice([0, 1]) for x in read] for read in matrix]
            tensor_batch = torch.tensor(processed_reads, dtype=torch.long, device=device).unsqueeze(-1)

            coords = cpg_subset.var_names
            fs = feature_sets['cpg']
            state_target = fs["state"]
            feature_map = fs["features"]

            pred_states = model.predict(tensor_batch)
            probs = model.predict_proba(tensor_batch)
            classifications = classify_batch(pred_states, probs, coords, feature_map, target_state=state_target)

            for i, idx in enumerate(cpg_subset.obs.index):
                for start, length, label, prob in classifications[i]:
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
        adata.obs[f"{feature}_distances"] = calculate_batch_distances(adata.obs[feature].tolist(), threshold)

def calculate_batch_distances(intervals_list, threshold=0.9):
    """
    Vectorized calculation of distances across multiple reads.

    Parameters:
        intervals_list (list of list): Outer list = reads, inner list = intervals [start, length, prob]
        threshold (float): Minimum probability threshold for filtering

    Returns:
        List of distance lists per read.
    """
    results = []
    for intervals in intervals_list:
        if not isinstance(intervals, list) or len(intervals) == 0:
            results.append([])
            continue
        valid = [iv for iv in intervals if iv[2] > threshold]
        valid = sorted(valid, key=lambda x: x[0])
        dists = [(valid[i + 1][0] - (valid[i][0] + valid[i][1])) for i in range(len(valid) - 1)]
        results.append(dists)
    return results



def classify_batch(predicted_states_batch, probabilities_batch, coordinates, classification_mapping, target_state="Methylated"):
    """
    Classify batch sequences efficiently.

    Parameters:
        predicted_states_batch: Tensor [batch_size, seq_len]
        probabilities_batch: Tensor [batch_size, seq_len, n_states]
        coordinates: list of genomic coordinates
        classification_mapping: dict of feature bins
        target_state: state name ("Methylated" or "Non-Methylated")

    Returns:
        List of classifications for each sequence.
    """
    import numpy as np

    state_labels = ["Non-Methylated", "Methylated"]
    target_idx = state_labels.index(target_state)
    batch_size = predicted_states_batch.shape[0]

    all_classifications = []

    for b in range(batch_size):
        predicted_states = predicted_states_batch[b].cpu().numpy()
        probabilities = probabilities_batch[b].cpu().numpy()

        regions = []
        current_start, current_length, current_probs = None, 0, []

        for i, state_index in enumerate(predicted_states):
            state_prob = probabilities[i][state_index]
            if state_index == target_idx:
                if current_start is None:
                    current_start = i
                current_length += 1
                current_probs.append(state_prob)
            elif current_start is not None:
                regions.append((current_start, current_length, np.mean(current_probs)))
                current_start, current_length, current_probs = None, 0, []

        if current_start is not None:
            regions.append((current_start, current_length, np.mean(current_probs)))

        final = []
        for start, length, prob in regions:
            feature_length = int(coordinates[start + length - 1]) - int(coordinates[start]) + 1
            label = next((ftype for ftype, rng in classification_mapping.items() if rng[0] <= feature_length < rng[1]), target_state)
            final.append((int(coordinates[start]) + 1, feature_length, label, prob))
        all_classifications.append(final)

    return all_classifications