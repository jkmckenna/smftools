import torch

class UnionFind:
    def __init__(self, size):
        self.parent = torch.arange(size)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def flag_duplicate_reads(adata, var_filters, distance_threshold=0.05):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Step 1: Apply combined column filter (True = OR logic, False = exclusion)
    true_keys = [k for k, v in var_filters.items() if v is True]
    false_keys = [k for k, v in var_filters.items() if v is False]

    mask_true = torch.zeros(len(adata.var), dtype=torch.bool)
    for key in true_keys:
        mask_true |= torch.from_numpy(adata.var[key].values)

    mask_false = torch.zeros(len(adata.var), dtype=torch.bool)
    for key in false_keys:
        mask_false |= torch.from_numpy(adata.var[key].values)

    # Keep columns where true condition is satisfied and none of the false filters match
    final_mask = mask_true & (~mask_false)

    selected_cols = adata.var.index[final_mask.numpy()].to_list()
    col_indices = [adata.var.index.get_loc(col) for col in selected_cols]

    print(f"Selected {len(col_indices)} columns out of {adata.var.shape[0]} based on provided filters (OR logic).")

    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X_subset = X[:, col_indices]
    X_tensor = torch.from_numpy(X_subset.astype(np.float32))
    N = X_tensor.shape[0]
    all_hamming_dists = []

    # These will hold distances for each row
    fwd_hamming_to_next = torch.full((N,), float('nan'))
    rev_hamming_to_prev = torch.full((N,), float('nan'))

    def cluster_pass(X_tensor, mask_to_consider=None, reverse=False, window_size=50, record_distances=False):
        if mask_to_consider is None:
            mask_to_consider = torch.ones(N, dtype=torch.bool)

        X_masked = X_tensor[mask_to_consider]
        indices_masked = mask_to_consider.nonzero(as_tuple=True)[0]

        X_sortable = X_masked.nan_to_num(-1)
        sort_keys = X_sortable.tolist()
        sorted_idx = sorted(range(len(X_masked)), key=lambda i: sort_keys[i], reverse=reverse)
        sorted_X = X_masked[sorted_idx]

        cluster_pairs = []

        for i in range(len(sorted_X)):
            row_i = sorted_X[i]
            j_range = range(i + 1, min(i + 1 + window_size, len(sorted_X)))

            if len(j_range) > 0:
                row_i_exp = row_i.unsqueeze(0)
                block_rows = sorted_X[j_range]
                valid_mask = (~torch.isnan(row_i_exp)) & (~torch.isnan(block_rows))
                valid_counts = valid_mask.sum(dim=1)
                diffs = (row_i_exp != block_rows) & valid_mask
                hamming_dists = diffs.sum(dim=1) / valid_counts.clamp(min=1)
                all_hamming_dists.extend(hamming_dists.cpu().numpy().tolist())

                matches = (hamming_dists < distance_threshold) & (valid_counts > 0)
                for offset_idx, m in zip(j_range, matches):
                    if m:
                        cluster_pairs.append((indices_masked[sorted_idx[i]].item(), indices_masked[sorted_idx[offset_idx]].item()))

                # Save distance to nearest next row (just row i vs i+1)
                if record_distances and i + 1 < len(sorted_X):
                    next_idx = sorted_idx[i + 1]
                    valid_mask_pair = (~torch.isnan(row_i)) & (~torch.isnan(sorted_X[i + 1]))
                    if valid_mask_pair.sum() > 0:
                        d = (row_i[valid_mask_pair] != sorted_X[i + 1][valid_mask_pair]).sum()
                        norm_d = d.item() / valid_mask_pair.sum().item()
                        if reverse:
                            rev_hamming_to_prev[indices_masked[next_idx]] = norm_d
                        else:
                            fwd_hamming_to_next[indices_masked[sorted_idx[i]]] = norm_d

        return cluster_pairs

    # Step 1: Collect edges from both passes + record distances
    pairs_fwd = cluster_pass(X_tensor, reverse=False, record_distances=True)
    mask_for_rev = torch.ones(N, dtype=torch.bool)
    involved_in_fwd = set([p[0] for p in pairs_fwd] + [p[1] for p in pairs_fwd])
    mask_for_rev[list(involved_in_fwd)] = False
    pairs_rev = cluster_pass(X_tensor, mask_to_consider=mask_for_rev, reverse=True, record_distances=True)

    all_pairs = pairs_fwd + pairs_rev

    # Step 2: Union-Find on all edges
    uf = UnionFind(N)
    for i, j in all_pairs:
        uf.union(i, j)

    # Step 3: Assign merged cluster IDs
    merged_cluster = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        merged_cluster[i] = uf.find(i)

    # Step 4: Compute cluster sizes
    cluster_sizes = torch.zeros_like(merged_cluster)
    for cid in merged_cluster.unique():
        members = (merged_cluster == cid).nonzero(as_tuple=True)[0]
        cluster_sizes[members] = len(members)

    # Step 5: Flag duplicates (keep the first molecule in each cluster)
    is_duplicate = torch.zeros(N, dtype=torch.bool)
    for cid in merged_cluster.unique():
        members = (merged_cluster == cid).nonzero(as_tuple=True)[0]
        if len(members) > 1:
            is_duplicate[members[1:]] = True

    # Step 6: Add to obs
    adata.obs['is_duplicate'] = pd.Series(is_duplicate.numpy(), index=adata.obs_names)
    adata.obs['merged_cluster_id'] = pd.Series(merged_cluster.numpy(), index=adata.obs_names)
    adata.obs['cluster_size'] = pd.Series(cluster_sizes.numpy(), index=adata.obs_names)
    adata.obs['fwd_hamming_to_next'] = pd.Series(fwd_hamming_to_next.numpy(), index=adata.obs_names)
    adata.obs['rev_hamming_to_prev'] = pd.Series(rev_hamming_to_prev.numpy(), index=adata.obs_names)

    adata_unique = adata[~adata.obs['is_duplicate']].copy()

    # Step 7: Plot histogram
    plt.figure(figsize=(5, 4))
    plt.hist(all_hamming_dists, bins=50, alpha=0.75)
    plt.axvline(distance_threshold, color="red", linestyle="--", label=f"threshold = {distance_threshold}")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pairwise Hamming Distances")
    plt.legend()
    plt.show()

    return adata_unique, adata
