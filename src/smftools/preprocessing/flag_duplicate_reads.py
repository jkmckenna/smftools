import torch
from tqdm import tqdm

class UnionFind:
    def __init__(self, size):
        self.parent = torch.arange(size)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def flag_duplicate_reads(adata, var_filters_sets, distance_threshold=0.05, obs_reference_col='Reference_strand'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    all_hamming_dists = []
    merged_results = []

    references = adata.obs[obs_reference_col].cat.categories

    for ref in references:
        print(f'🔹 Processing reference: {ref}')

        ref_mask = adata.obs[obs_reference_col] == ref
        adata_subset = adata[ref_mask].copy()
        N = adata_subset.shape[0]

        combined_mask = torch.zeros(len(adata.var), dtype=torch.bool)
        for var_set in var_filters_sets:
            if any(ref in v for v in var_set):
                set_mask = torch.ones(len(adata.var), dtype=torch.bool)
                for key in var_set:
                    set_mask &= torch.from_numpy(adata.var[key].values)
                combined_mask |= set_mask

        selected_cols = adata.var.index[combined_mask.numpy()].to_list()
        col_indices = [adata.var.index.get_loc(col) for col in selected_cols]

        print(f"Selected {len(col_indices)} columns out of {adata.var.shape[0]} for {ref}")

        X = adata_subset.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X_subset = X[:, col_indices]
        X_tensor = torch.from_numpy(X_subset.astype(np.float32))

        fwd_hamming_to_next = torch.full((N,), float('nan'))
        rev_hamming_to_prev = torch.full((N,), float('nan'))

        def cluster_pass(X_tensor, reverse=False, window_size=50, record_distances=False):
            N_local = X_tensor.shape[0]
            X_sortable = X_tensor.nan_to_num(-1)
            sort_keys = X_sortable.tolist()
            sorted_idx = sorted(range(N_local), key=lambda i: sort_keys[i], reverse=reverse)
            sorted_X = X_tensor[sorted_idx]

            cluster_pairs = []

            for i in tqdm(range(len(sorted_X)), desc=f"Pass {'rev' if reverse else 'fwd'} ({ref})"):
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
                            cluster_pairs.append((sorted_idx[i], sorted_idx[offset_idx]))

                    if record_distances and i + 1 < len(sorted_X):
                        next_idx = sorted_idx[i + 1]
                        valid_mask_pair = (~torch.isnan(row_i)) & (~torch.isnan(sorted_X[i + 1]))
                        if valid_mask_pair.sum() > 0:
                            d = (row_i[valid_mask_pair] != sorted_X[i + 1][valid_mask_pair]).sum()
                            norm_d = d.item() / valid_mask_pair.sum().item()
                            if reverse:
                                rev_hamming_to_prev[next_idx] = norm_d
                            else:
                                fwd_hamming_to_next[sorted_idx[i]] = norm_d

            return cluster_pairs

        pairs_fwd = cluster_pass(X_tensor, reverse=False, record_distances=True)
        involved_in_fwd = set([p[0] for p in pairs_fwd] + [p[1] for p in pairs_fwd])
        mask_for_rev = torch.ones(N, dtype=torch.bool)
        mask_for_rev[list(involved_in_fwd)] = False
        pairs_rev = cluster_pass(X_tensor[mask_for_rev], reverse=True, record_distances=True)

        all_pairs = pairs_fwd + [(list(mask_for_rev.nonzero(as_tuple=True)[0])[i], list(mask_for_rev.nonzero(as_tuple=True)[0])[j]) for i, j in pairs_rev]

        uf = UnionFind(N)
        for i, j in all_pairs:
            uf.union(i, j)

        merged_cluster = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            merged_cluster[i] = uf.find(i)

        cluster_sizes = torch.zeros_like(merged_cluster)
        for cid in merged_cluster.unique():
            members = (merged_cluster == cid).nonzero(as_tuple=True)[0]
            cluster_sizes[members] = len(members)

        is_duplicate = torch.zeros(N, dtype=torch.bool)
        for cid in merged_cluster.unique():
            members = (merged_cluster == cid).nonzero(as_tuple=True)[0]
            if len(members) > 1:
                is_duplicate[members[1:]] = True

        adata_subset.obs['is_duplicate'] = is_duplicate.numpy()
        adata_subset.obs['merged_cluster_id'] = merged_cluster.numpy()
        adata_subset.obs['cluster_size'] = cluster_sizes.numpy()
        adata_subset.obs['fwd_hamming_to_next'] = fwd_hamming_to_next.numpy()
        adata_subset.obs['rev_hamming_to_prev'] = rev_hamming_to_prev.numpy()

        merged_results.append(adata_subset.obs)

    merged_obs = pd.concat(merged_results)
    adata.obs = adata.obs.join(merged_obs[['is_duplicate', 'merged_cluster_id', 'cluster_size', 'fwd_hamming_to_next', 'rev_hamming_to_prev']])

    adata_unique = adata[~adata.obs['is_duplicate']].copy()

    plt.figure(figsize=(5, 4))
    plt.hist(all_hamming_dists, bins=50, alpha=0.75)
    plt.axvline(distance_threshold, color="red", linestyle="--", label=f"threshold = {distance_threshold}")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pairwise Hamming Distances")
    plt.legend()
    plt.show()

    return adata_unique, adata