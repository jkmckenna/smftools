import copy
import warnings
import math
import os
from collections import defaultdict
from typing import Dict, Any, Tuple, Union, List, Optional

import torch
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# optional imports
try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None

try:
    import hdbscan
except Exception:
    hdbscan = None

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
except Exception:
    KMeans = DBSCAN = GaussianMixture = silhouette_score = None

def merge_uns_preserve(orig_uns: dict, new_uns: dict, prefer="orig") -> dict:
    """
    Merge two .uns dicts. prefer='orig' will keep orig_uns values on conflict,
    prefer='new' will keep new_uns values on conflict. Conflicts are reported.
    """
    out = copy.deepcopy(new_uns) if new_uns is not None else {}
    for k, v in (orig_uns or {}).items():
        if k not in out:
            out[k] = copy.deepcopy(v)
        else:
            # present in both: compare quickly (best-effort)
            try:
                equal = (out[k] == v)
            except Exception:
                equal = False
            if equal:
                continue
            # conflict
            warnings.warn(f".uns conflict for key '{k}'; keeping '{prefer}' value.")
            if prefer == "orig":
                out[k] = copy.deepcopy(v)
            else:
                # keep out[k] (the new one) and also stash orig under a suffix
                out[f"orig_uns__{k}"] = copy.deepcopy(v)
    return out

def flag_duplicate_reads(
    adata, 
    var_filters_sets, 
    distance_threshold=0.05, 
    obs_reference_col='Reference_strand', 
    sample_col='Barcode',
    output_directory=None,
    metric_keys=["Fraction_any_C_site_modified"],
    uns_flag="read_duplicate_detection_performed",
    uns_filtered_flag="read_duplicates_removed",
    bypass=False,
    force_redo=True,
    keep_best_metric: Optional[str] = None,
    keep_best_higher: bool = True,
    window_size: int = 50,
):
    """
    Duplicate-flagging pipeline.

    Notable behavior changes / guarantees:
      - sequence-derived columns are written with a 'sequence__' prefix (sequence__is_duplicate, sequence__merged_cluster_id, sequence__cluster_size)
      - window_size is passed into the sorted-neighborhood `cluster_pass`.
      - keep_best_metric selects the keeper in each union-find cluster (if present); otherwise first member kept.
      - plotting (hamming_vs_metric) is performed on the merged adata_full so columns exist for plotting.
    """
    # early exits
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo):
        if 'is_duplicate' in adata.obs.columns:
            adata_unique = adata[adata.obs['is_duplicate'] == False].copy()
            return adata_unique, adata
        else:
            return adata.copy(), adata.copy()
    if bypass:
        return None, adata

    class UnionFind:
        def __init__(self, size):
            self.parent = torch.arange(size)

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx = self.find(x); ry = self.find(y)
            if rx != ry:
                self.parent[ry] = rx

    adata_processed_list = []
    histograms = []

    samples = adata.obs[sample_col].astype('category').cat.categories
    references = adata.obs[obs_reference_col].astype('category').cat.categories

    for sample in samples:
        for ref in references:
            print(f'Processing {sample} on {ref}')
            sample_mask = adata.obs[sample_col] == sample
            ref_mask = adata.obs[obs_reference_col] == ref
            subset_mask = sample_mask & ref_mask
            adata_subset = adata[subset_mask].copy()

            if adata_subset.n_obs < 2:
                print(f'Skipping {sample}_{ref} (too few reads)')
                continue

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
            X_tensor = torch.from_numpy(X[:, col_indices].astype(np.float32))

            fwd_hamming_to_next = torch.full((N,), float('nan'))
            rev_hamming_to_prev = torch.full((N,), float('nan'))

            local_hamming_dists = []

            def cluster_pass(X_tensor_local, reverse=False, window_size=window_size, record_distances=False):
                N_local = X_tensor_local.shape[0]
                X_sortable = X_tensor_local.nan_to_num(-1)
                sort_keys = X_sortable.tolist()
                sorted_idx = sorted(range(N_local), key=lambda i: sort_keys[i], reverse=reverse)
                sorted_X = X_tensor_local[sorted_idx]
                cluster_pairs_local = []

                for i in tqdm(range(len(sorted_X)), desc=f"Pass {'rev' if reverse else 'fwd'} ({sample}_{ref})"):
                    row_i = sorted_X[i]
                    j_range = range(i + 1, min(i + 1 + window_size, len(sorted_X)))
                    if len(j_range) > 0:
                        row_i_exp = row_i.unsqueeze(0)
                        block_rows = sorted_X[j_range]
                        valid_mask = (~torch.isnan(row_i_exp)) & (~torch.isnan(block_rows))
                        valid_counts = valid_mask.sum(dim=1)
                        diffs = (row_i_exp != block_rows) & valid_mask
                        hamming_dists = diffs.sum(dim=1) / valid_counts.clamp(min=1)
                        local_hamming_dists.extend(hamming_dists.cpu().numpy().tolist())

                        matches = (hamming_dists < distance_threshold) & (valid_counts > 0)
                        for offset_idx, m in zip(j_range, matches):
                            if m:
                                cluster_pairs_local.append((sorted_idx[i], sorted_idx[offset_idx]))

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
                return cluster_pairs_local

            pairs_fwd = cluster_pass(X_tensor, reverse=False, record_distances=True)
            involved_in_fwd = set([p[0] for p in pairs_fwd] + [p[1] for p in pairs_fwd])
            mask_for_rev = torch.ones(N, dtype=torch.bool)
            if len(involved_in_fwd) > 0:
                mask_for_rev[list(involved_in_fwd)] = False
            pairs_rev = cluster_pass(X_tensor[mask_for_rev], reverse=True, record_distances=True)

            reverse_idx_map = mask_for_rev.nonzero(as_tuple=True)[0]
            remapped_rev_pairs = [(int(reverse_idx_map[i].item()), int(reverse_idx_map[j].item())) for (i, j) in pairs_rev]
            all_pairs = pairs_fwd + remapped_rev_pairs

            uf = UnionFind(N)
            for i, j in all_pairs:
                uf.union(i, j)

            merged_cluster = torch.zeros(N, dtype=torch.long)
            for i in range(N):
                merged_cluster[i] = uf.find(i)

            cluster_sizes = torch.zeros_like(merged_cluster)
            cluster_counts = []
            for cid in merged_cluster.unique():
                members = (merged_cluster == cid).nonzero(as_tuple=True)[0]
                csize = int(len(members))
                cluster_counts.append(csize)
                cluster_sizes[members] = csize

            # sequence-prefixed outputs so provenance is clear
            sequence_is_duplicate = torch.zeros(N, dtype=torch.bool)
            for cid in merged_cluster.unique():
                members_tensor = (merged_cluster == cid).nonzero(as_tuple=True)[0]
                members = members_tensor.tolist()
                if len(members) > 1:
                    # choose keeper
                    if keep_best_metric is None:
                        keeper_idx = members[0]
                    else:
                        # build member obs names
                        obs_index = list(adata_subset.obs.index)
                        member_names = [obs_index[m] for m in members]
                        try:
                            vals = pd.to_numeric(adata_subset.obs.loc[member_names, keep_best_metric], errors="coerce").to_numpy(dtype=float)
                        except Exception:
                            vals = np.array([np.nan] * len(members), dtype=float)
                        if np.all(np.isnan(vals)):
                            keeper_idx = members[0]
                        else:
                            if keep_best_higher:
                                nan_mask = np.isnan(vals)
                                vals[nan_mask] = -np.inf
                                rel_idx = int(np.nanargmax(vals))
                            else:
                                nan_mask = np.isnan(vals)
                                vals[nan_mask] = np.inf
                                rel_idx = int(np.nanargmin(vals))
                            keeper_idx = members[rel_idx]
                    others = [m for m in members if m != keeper_idx]
                    if others:
                        sequence_is_duplicate[others] = True

            # write sequence-prefixed columns
            adata_subset.obs['sequence__is_duplicate'] = sequence_is_duplicate.numpy()
            adata_subset.obs['sequence__merged_cluster_id'] = merged_cluster.numpy()
            adata_subset.obs['sequence__cluster_size'] = cluster_sizes.numpy()
            adata_subset.obs['fwd_hamming_to_next'] = fwd_hamming_to_next.numpy()
            adata_subset.obs['rev_hamming_to_prev'] = rev_hamming_to_prev.numpy()

            adata_processed_list.append(adata_subset)

            histograms.append({
                "sample": sample,
                "reference": ref,
                "distances": local_hamming_dists,
                "cluster_counts": cluster_counts,
            })

    # Merge annotated subsets back together BEFORE plotting so plotting sees fwd_hamming_to_next, etc.
    _original_uns = copy.deepcopy(adata.uns)
    adata_full = ad.concat(adata_processed_list, merge="same", join="outer", index_unique=None)
    adata_full.uns = merge_uns_preserve(_original_uns, adata_full.uns, prefer="orig")

    # Ensure expected numeric columns exist (create if missing)
    for col in ("fwd_hamming_to_next", "rev_hamming_to_prev"):
        if col not in adata_full.obs.columns:
            adata_full.obs[col] = np.nan

    # produce histogram pages (distance + cluster-size)
    saved_histograms_map = plot_histogram_pages(histograms, distance_threshold=distance_threshold, output_directory=output_directory)

    # produce hamming vs metric scatter + clusters and write cluster columns into adata_full.obs
    scatter_saved = plot_hamming_vs_metric_pages(
        adata_full,
        metric_keys=metric_keys,
        sample_col=sample_col,
        ref_col=obs_reference_col,
        hamming_col="fwd_hamming_to_next",
        rows_per_fig=6,
        output_dir=output_directory,
        kde=True,
        contour=True,
        clustering={"method":"gmm","dbscan_eps":0.05,"dbscan_min_samples":6,"min_points":8,"cmap":"tab10","hull":True},
        write_clusters_to_adata=False
    )

    # Mark duplicates: distance-threshold and hamming-vs-metric cluster columns
    fwd_vals = pd.to_numeric(adata_full.obs["fwd_hamming_to_next"], errors="coerce")
    rev_vals = pd.to_numeric(adata_full.obs["rev_hamming_to_prev"], errors="coerce")
    is_dup_dist = (fwd_vals < float(distance_threshold)) | (rev_vals < float(distance_threshold))
    is_dup_dist = is_dup_dist.fillna(False).astype(bool)
    adata_full.obs["is_duplicate_distance"] = is_dup_dist.values

    cluster_cols = [c for c in adata_full.obs.columns if c.startswith("hamming_cluster__")]
    if cluster_cols:
        cl_mask = pd.Series(False, index=adata_full.obs.index)
        for c in cluster_cols:
            vals = pd.to_numeric(adata_full.obs[c], errors="coerce")
            mask_pos = (vals > 0) & (vals != -1)
            mask_pos = mask_pos.fillna(False)
            cl_mask |= mask_pos
        adata_full.obs["is_duplicate_clustering"] = cl_mask.values
    else:
        adata_full.obs["is_duplicate_clustering"] = False

    # combine sequence-derived flag (if present) with the other flags into final is_duplicate
    if "sequence__is_duplicate" in adata_full.obs.columns:
        seq_dup = adata_full.obs["sequence__is_duplicate"].astype(bool)
    else:
        seq_dup = pd.Series(False, index=adata_full.obs.index)

    final_dup = seq_dup | adata_full.obs["is_duplicate_distance"].astype(bool) | adata_full.obs["is_duplicate_clustering"].astype(bool)
    adata_full.obs["is_duplicate"] = final_dup.values

    # reason column
    def _dup_reason_row(row):
        reasons = []
        if row.get("is_duplicate_distance", False):
            reasons.append("distance_thresh")
        if row.get("is_duplicate_clustering", False):
            reasons.append("hamming_metric_cluster")
        if bool(row.get("sequence__is_duplicate", False)):
            reasons.append("sequence_cluster")
        return ";".join(reasons) if reasons else ""

    try:
        reasons = adata_full.obs.apply(_dup_reason_row, axis=1)
        adata_full.obs["is_duplicate_reason"] = reasons.values
    except Exception:
        adata_full.obs["is_duplicate_reason"] = ""

    adata_unique = adata_full[~adata_full.obs["is_duplicate"].astype(bool)].copy()

    # mark flags in .uns
    adata_unique.uns[uns_flag] = True
    adata_unique.uns[uns_filtered_flag] = True
    adata_full.uns[uns_flag] = True

    return adata_unique, adata_full


def plot_histogram_pages(
    histograms,
    distance_threshold,
    output_directory=None,
    rows_per_page=6,
    bins=50,
    dpi=160,
    figsize_per_cell=(5, 3),
):
    """
    Plot Hamming-distance histograms as a grid: rows = samples, columns = references.
    Additionally plots cluster-size histograms (one per cell) using `cluster_counts` stored
    in each histogram entry.

    Returns:
      { "distance_pages": [...filenames...], "cluster_size_pages": [...filenames...] }
    """

    # Build set of unique samples and references (sorted for stable ordering)
    samples = sorted({h["sample"] for h in histograms})
    references = sorted({h["reference"] for h in histograms})

    if len(samples) == 0 or len(references) == 0:
        print("No histogram data to plot.")
        return {"distance_pages": [], "cluster_size_pages": []}

    # Map (sample, reference) -> distances list and cluster_counts
    grid_dists = defaultdict(list)
    grid_cluster_counts = defaultdict(list)
    counts = {}  # store number of distances for label
    for h in histograms:
        key = (h["sample"], h["reference"])
        d = h.get("distances") or []
        cc = h.get("cluster_counts") or []
        grid_dists[key].extend(d)
        # cluster_counts is a list where each entry is a cluster size (one per cluster)
        grid_cluster_counts[key].extend(cc)
    for s in samples:
        for r in references:
            counts[(s, r)] = len(grid_dists.get((s, r), []))

    distance_pages = []
    cluster_size_pages = []
    n_pages = math.ceil(len(samples) / rows_per_page)

    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, len(samples))
        chunk = samples[start:end]
        nrows = len(chunk)
        ncols = len(references)

        # -- Distance histogram page --
        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(references):
                ax = axes[r_idx][c_idx]
                dists = grid_dists.get((sample_name, ref_name), [])
                if dists:
                    ax.hist(dists, bins=bins, alpha=0.75)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                ax.axvline(distance_threshold, color="red", linestyle="--", linewidth=1)

                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=10)
                if c_idx == 0:
                    total_reads = sum(counts.get((sample_name, ref), 0) for ref in references)
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=9)
                if r_idx == nrows - 1:
                    ax.set_xlabel("Hamming Distance", fontsize=9)
                else:
                    ax.set_xticklabels([])

                ax.set_xlim(left=0.0, right=1.0)
                ax.grid(True, alpha=0.25)

        fig.suptitle(f"Hamming distance histograms (rows=samples, cols=references) — page {page+1}/{n_pages}", fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            fname = os.path.join(output_directory, f"hamming_histograms_page_{page+1}.png")
            plt.savefig(fname, bbox_inches="tight")
            distance_pages.append(fname)
        else:
            plt.show()
        plt.close(fig)

        # -- Cluster-size histogram page --
        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(references):
                ax = axes2[r_idx][c_idx]
                sizes = grid_cluster_counts.get((sample_name, ref_name), [])
                if sizes:
                    # plot histogram of cluster sizes (clusters counted once each)
                    ax.hist(sizes, bins=range(1, max(2, max(sizes)+1)), alpha=0.8, align='left')
                    ax.set_xlabel("Cluster size")
                    ax.set_ylabel("Count")
                else:
                    ax.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")

                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=10)
                if c_idx == 0:
                    total_reads = sum(counts.get((sample_name, ref), 0) for ref in references)
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=9)
                if r_idx != nrows - 1:
                    ax.set_xticklabels([])

                ax.grid(True, alpha=0.25)

        fig2.suptitle(f"Union-find cluster size histograms — page {page+1}/{n_pages}", fontsize=12, y=0.995)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])

        if output_directory:
            fname2 = os.path.join(output_directory, f"cluster_size_histograms_page_{page+1}.png")
            plt.savefig(fname2, bbox_inches="tight")
            cluster_size_pages.append(fname2)
        else:
            plt.show()
        plt.close(fig2)

    return {"distance_pages": distance_pages, "cluster_size_pages": cluster_size_pages}

def plot_hamming_vs_metric_pages(
    adata,
    metric_keys: Union[str, List[str]],
    hamming_col: str = "fwd_hamming_to_next",
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    references: Optional[List[str]] = None,
    samples: Optional[List[str]] = None,
    rows_per_fig: int = 6,
    dpi: int = 160,
    filename_prefix: str = "hamming_vs_metric",
    output_dir: Optional[str] = None,
    kde: bool = False,
    contour: bool = False,
    regression: bool = True,
    show_ticks: bool = True,
    clustering: Optional[Dict[str, Any]] = None,
    write_clusters_to_adata: bool = False,
    figsize_per_cell: Tuple[float, float] = (4.0, 3.0),
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Robust plotting of hamming (y) vs metric (x). Aligns by index; tolerates missing columns.
    """
    if isinstance(metric_keys, str):
        metric_keys = [metric_keys]
    metric_keys = list(metric_keys)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    obs = adata.obs
    if sample_col not in obs.columns or ref_col not in obs.columns:
        raise ValueError(f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs")

    # canonicalize samples and refs
    if samples is None:
        sseries = obs[sample_col]
        if not pd.api.types.is_categorical_dtype(sseries):
            sseries = sseries.astype("category")
        samples_all = list(sseries.cat.categories)
    else:
        samples_all = list(samples)

    if references is None:
        rseries = obs[ref_col]
        if not pd.api.types.is_categorical_dtype(rseries):
            rseries = rseries.astype("category")
        refs_all = list(rseries.cat.categories)
    else:
        refs_all = list(references)

    extra_col = "ALLREFS"
    cols = list(refs_all) + [extra_col]

    saved_map: Dict[str, Any] = {}

    for metric in metric_keys:
        clusters_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
        files: List[str] = []
        written_cols: List[str] = []

        # compute global x/y limits robustly by aligning Series
        global_xlim = None
        global_ylim = None

        metric_present = metric in obs.columns
        hamming_present = hamming_col in obs.columns

        if metric_present or hamming_present:
            sX = obs[metric].astype(float) if metric_present else None
            sY = obs[hamming_col].astype(float) if hamming_present else None

            if (sX is not None) and (sY is not None):
                valid_both = sX.notna() & sY.notna() & np.isfinite(sX.values) & np.isfinite(sY.values)
                if valid_both.any():
                    xvals = sX[valid_both].to_numpy(dtype=float)
                    yvals = sY[valid_both].to_numpy(dtype=float)
                    xmin, xmax = float(np.nanmin(xvals)), float(np.nanmax(xvals))
                    ymin, ymax = float(np.nanmin(yvals)), float(np.nanmax(yvals))
                    xpad = max(1e-6, (xmax - xmin) * 0.05) if xmax > xmin else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                    ypad = max(1e-6, (ymax - ymin) * 0.05) if ymax > ymin else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                    global_xlim = (xmin - xpad, xmax + xpad)
                    global_ylim = (ymin - ypad, ymax + ypad)
                else:
                    sX_finite = sX[np.isfinite(sX)]
                    sY_finite = sY[np.isfinite(sY)]
                    if sX_finite.size > 0:
                        xmin, xmax = float(np.nanmin(sX_finite)), float(np.nanmax(sX_finite))
                        xpad = max(1e-6, (xmax - xmin) * 0.05) if xmax > xmin else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                        global_xlim = (xmin - xpad, xmax + xpad)
                    if sY_finite.size > 0:
                        ymin, ymax = float(np.nanmin(sY_finite)), float(np.nanmax(sY_finite))
                        ypad = max(1e-6, (ymax - ymin) * 0.05) if ymax > ymin else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                        global_ylim = (ymin - ypad, ymax + ypad)
            elif sX is not None:
                sX_finite = sX[np.isfinite(sX)]
                if sX_finite.size > 0:
                    xmin, xmax = float(np.nanmin(sX_finite)), float(np.nanmax(sX_finite))
                    xpad = max(1e-6, (xmax - xmin) * 0.05) if xmax > xmin else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                    global_xlim = (xmin - xpad, xmax + xpad)
            elif sY is not None:
                sY_finite = sY[np.isfinite(sY)]
                if sY_finite.size > 0:
                    ymin, ymax = float(np.nanmin(sY_finite)), float(np.nanmax(sY_finite))
                    ypad = max(1e-6, (ymax - ymin) * 0.05) if ymax > ymin else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                    global_ylim = (ymin - ypad, ymax + ypad)

        # pagination
        for start in range(0, len(samples_all), rows_per_fig):
            chunk = samples_all[start : start + rows_per_fig]
            nrows = len(chunk)
            ncols = len(cols)
            fig_w = ncols * figsize_per_cell[0]
            fig_h = nrows * figsize_per_cell[1]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

            for r_idx, sample_name in enumerate(chunk):
                for c_idx, ref_name in enumerate(cols):
                    ax = axes[r_idx][c_idx]
                    if ref_name == extra_col:
                        mask = (obs[sample_col].values == sample_name)
                    else:
                        mask = (obs[sample_col].values == sample_name) & (obs[ref_col].values == ref_name)

                    sub = obs[mask]

                    if metric in sub.columns:
                        x_all = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
                    else:
                        x_all = np.array([], dtype=float)
                    if hamming_col in sub.columns:
                        y_all = pd.to_numeric(sub[hamming_col], errors="coerce").to_numpy(dtype=float)
                    else:
                        y_all = np.array([], dtype=float)

                    idxs = sub.index.to_numpy()

                    # drop nan pairs
                    if x_all.size and y_all.size and len(x_all) == len(y_all):
                        valid_pair_mask = np.isfinite(x_all) & np.isfinite(y_all)
                        x = x_all[valid_pair_mask]
                        y = y_all[valid_pair_mask]
                        idxs_valid = idxs[valid_pair_mask]
                    else:
                        # insufficient paired data
                        x = np.array([], dtype=float)
                        y = np.array([], dtype=float)
                        idxs_valid = np.array([], dtype=int)

                    if x.size == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        clusters_info[(sample_name, ref_name)] = {"diag": None, "n_points": 0}
                    else:
                        ax.scatter(x, y, s=12, alpha=0.6, rasterized=True)

                        if kde and gaussian_kde is not None and x.size >= 4:
                            try:
                                xy = np.vstack([x, y])
                                kde2 = gaussian_kde(xy)(xy)
                                if contour:
                                    xi = np.linspace(np.nanmin(x), np.nanmax(x), 80)
                                    yi = np.linspace(np.nanmin(y), np.nanmax(y), 80)
                                    xi_g, yi_g = np.meshgrid(xi, yi)
                                    coords = np.vstack([xi_g.ravel(), yi_g.ravel()])
                                    zi = gaussian_kde(np.vstack([x, y]))(coords).reshape(xi_g.shape)
                                    ax.contourf(xi_g, yi_g, zi, levels=8, alpha=0.35, cmap="Blues")
                                else:
                                    ax.scatter(x, y, c=kde2, s=16, cmap="viridis", alpha=0.7, linewidths=0)
                            except Exception:
                                pass

                        if regression and x.size >= 2:
                            try:
                                a, b = np.polyfit(x, y, 1)
                                xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                                ys = a * xs + b
                                ax.plot(xs, ys, linestyle="--", linewidth=1.2, alpha=0.9, color="red")
                                r = np.corrcoef(x, y)[0, 1]
                                ax.text(0.98, 0.02, f"r={float(r):.3f}", ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
                                        bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"))
                            except Exception:
                                pass

                        if clustering:
                            cl_labels, diag = _run_clustering(
                                x, y,
                                method=clustering.get("method", "dbscan"),
                                n_clusters=clustering.get("n_clusters", 2),
                                dbscan_eps=clustering.get("dbscan_eps", 0.05),
                                dbscan_min_samples=clustering.get("dbscan_min_samples", 5),
                                random_state=random_state,
                                min_points=clustering.get("min_points", 8),
                            )

                            remapped_labels = cl_labels.copy()
                            unique_nonnoise = sorted([u for u in np.unique(cl_labels) if u != -1])
                            if len(unique_nonnoise) > 0:
                                medians = {}
                                for lab in unique_nonnoise:
                                    mask_lab = (cl_labels == lab)
                                    medians[lab] = float(np.median(y[mask_lab])) if mask_lab.any() else float("nan")
                                sorted_by_median = sorted(unique_nonnoise, key=lambda l: (np.nan if np.isnan(medians[l]) else medians[l]), reverse=True)
                                mapping = {old: new for new, old in enumerate(sorted_by_median)}
                                for i_lab in range(len(remapped_labels)):
                                    if remapped_labels[i_lab] != -1:
                                        remapped_labels[i_lab] = mapping.get(remapped_labels[i_lab], -1)
                                diag = diag or {}
                                diag["cluster_median_hamming"] = {int(old): medians[old] for old in medians}
                                diag["cluster_old_to_new_map"] = {int(old): int(new) for old, new in mapping.items()}
                            else:
                                remapped_labels = cl_labels.copy()
                                diag = diag or {}
                                diag["cluster_median_hamming"] = {}
                                diag["cluster_old_to_new_map"] = {}

                            _overlay_clusters_on_ax(ax, x, y, remapped_labels, diag,
                                                    cmap=clustering.get("cmap", "tab10"),
                                                    hull=clustering.get("hull", True),
                                                    show_cluster_labels=True)

                            clusters_info[(sample_name, ref_name)] = {"diag": diag, "n_points": len(x)}

                            if write_clusters_to_adata and idxs_valid.size > 0:
                                colname_safe_ref = (ref_name if ref_name != extra_col else "ALLREFS")
                                colname = f"hamming_cluster__{metric}__{sample_name}__{colname_safe_ref}"
                                if colname not in adata.obs.columns:
                                    adata.obs[colname] = np.nan
                                lab_arr = remapped_labels.astype(float)
                                adata.obs.loc[idxs_valid, colname] = lab_arr
                                if colname not in written_cols:
                                    written_cols.append(colname)

                    if r_idx == 0:
                        ax.set_title(str(ref_name), fontsize=9)
                    if c_idx == 0:
                        total_reads = int((obs[sample_col] == sample_name).sum())
                        ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                    if r_idx == nrows - 1:
                        ax.set_xlabel(metric, fontsize=8)

                    if global_xlim is not None:
                        ax.set_xlim(global_xlim)
                    if global_ylim is not None:
                        ax.set_ylim(global_ylim)

                    if not show_ticks:
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])

                    ax.grid(True, alpha=0.25)

            fig.suptitle(f"Hamming ({hamming_col}) vs {metric}", y=0.995, fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            page_idx = start // rows_per_fig + 1
            fname = f"{filename_prefix}_{metric}_page{page_idx}.png"
            if output_dir:
                outpath = os.path.join(output_dir, fname)
                plt.savefig(outpath, bbox_inches="tight", dpi=dpi)
                files.append(outpath)
            else:
                plt.show()
            plt.close(fig)

        saved_map[metric] = {"files": files, "clusters_info": clusters_info, "written_cols": written_cols}

    return saved_map



def _run_clustering(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "kmeans",   # "kmeans", "dbscan", "gmm", "hdbscan"
    n_clusters: int = 2,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 5,
    random_state: int = 0,
    min_points: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run clustering on 2D points (x,y). Returns labels (len = npoints) and diagnostics dict.
    Labels follow sklearn conventions (noise -> -1 for DBSCAN/HDBSCAN).

    Ensures contiguous cluster numbers starting at 0 for non-noise clusters.
    """
    pts = np.column_stack([x, y])
    diagnostics: Dict[str, Any] = {"method": method, "n_input": len(x)}
    if len(x) < min_points:
        diagnostics["skipped"] = True
        return np.full(len(x), -1, dtype=int), diagnostics

    method = (method or "kmeans").lower()
    labels = np.full(len(x), -1, dtype=int)

    try:
        if method == "kmeans" and KMeans is not None:
            km = KMeans(n_clusters=max(1, int(n_clusters)), random_state=random_state)
            labels = km.fit_predict(pts)
            diagnostics["centers"] = km.cluster_centers_
            diagnostics["n_clusters_found"] = int(len(np.unique(labels)))
        elif method == "dbscan" and DBSCAN is not None:
            db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
            labels = db.fit_predict(pts)
            uniq = [u for u in np.unique(labels) if u != -1]
            diagnostics["n_clusters_found"] = int(len(uniq))
        elif method == "gmm" and GaussianMixture is not None:
            gm = GaussianMixture(n_components=max(1, int(n_clusters)), random_state=random_state)
            labels = gm.fit_predict(pts)
            diagnostics["means"] = gm.means_
            diagnostics["covariances"] = getattr(gm, "covariances_", None)
            diagnostics["n_clusters_found"] = int(len(np.unique(labels)))
        elif method == "hdbscan" and hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(dbscan_min_samples)))
            labels = clusterer.fit_predict(pts)
            diagnostics["n_clusters_found"] = int(len([u for u in np.unique(labels) if u != -1]))
        else:
            # fallback: try DBSCAN then KMeans
            if DBSCAN is not None:
                db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
                labels = db.fit_predict(pts)
                if (labels == -1).all() and KMeans is not None:
                    km = KMeans(n_clusters=max(1, int(n_clusters)), random_state=random_state)
                    labels = km.fit_predict(pts)
                    diagnostics["fallback_to"] = "kmeans"
                    diagnostics["centers"] = km.cluster_centers_
                    diagnostics["n_clusters_found"] = int(len(np.unique(labels)))
            elif KMeans is not None:
                km = KMeans(n_clusters=max(1, int(n_clusters)), random_state=random_state)
                labels = km.fit_predict(pts)
                diagnostics["n_clusters_found"] = int(len(np.unique(labels)))
            else:
                diagnostics["skipped"] = True
                return np.full(len(x), -1, dtype=int), diagnostics

    except Exception as e:
        diagnostics["error"] = str(e)
        diagnostics["skipped"] = True
        return np.full(len(x), -1, dtype=int), diagnostics

    # remap non-noise labels to contiguous ints starting at 0 (keep -1 for noise)
    unique_nonnoise = sorted([u for u in np.unique(labels) if u != -1])
    if unique_nonnoise:
        mapping = {old: new for new, old in enumerate(unique_nonnoise)}
        remapped = np.full_like(labels, -1)
        for i, lab in enumerate(labels):
            if lab != -1:
                remapped[i] = mapping.get(lab, -1)
        labels = remapped
        diagnostics["n_clusters_found"] = int(len(unique_nonnoise))
    else:
        diagnostics["n_clusters_found"] = 0

    # compute silhouette if suitable
    try:
        if diagnostics.get("n_clusters_found", 0) >= 2 and len(x) >= 3 and silhouette_score is not None:
            diagnostics["silhouette"] = float(silhouette_score(pts, labels))
        else:
            diagnostics["silhouette"] = None
    except Exception:
        diagnostics["silhouette"] = None

    diagnostics["skipped"] = False
    return labels.astype(int), diagnostics


def _overlay_clusters_on_ax(
    ax,
    x,
    y,
    labels,
    diagnostics,
    *,
    cmap="tab20",
    alpha_pts=0.6,
    marker="o",
    plot_centroids=True,
    centroid_marker="X",
    centroid_size=60,
    hull=True,
    hull_alpha=0.12,
    hull_edgecolor="k",
    show_cluster_labels=True,
    cluster_label_fontsize=8,
):
    """
    Color points by label, plot centroids and optional convex hulls.
    Labels == -1 are noise and drawn in grey.
    Also annotates cluster numbers near centroids (contiguous numbers starting at 0).
    """
    import matplotlib.colors as mcolors
    from scipy.spatial import ConvexHull

    labels = np.asarray(labels)
    pts = np.column_stack([x, y])

    unique = np.unique(labels)
    # sort so noise (-1) comes last for drawing
    unique = sorted(unique.tolist(), key=lambda v: (v == -1, v))
    cmap_obj = plt.get_cmap(cmap)
    ncolors = max(8, len(unique))
    colors = [cmap_obj(i / float(ncolors)) for i in range(ncolors)]

    for idx, lab in enumerate(unique):
        mask = labels == lab
        if not mask.any():
            continue
        col = (0.6, 0.6, 0.6, 0.6) if lab == -1 else colors[idx % ncolors]
        ax.scatter(x[mask], y[mask], s=20, c=[col], alpha=alpha_pts, marker=marker, linewidths=0.2, edgecolors="none", rasterized=True)

        if lab != -1:
            # centroid
            if plot_centroids:
                cx = float(np.mean(x[mask]))
                cy = float(np.mean(y[mask]))
                ax.scatter([cx], [cy], s=centroid_size, marker=centroid_marker, c=[col], edgecolor="k", linewidth=0.6, zorder=10)

                if show_cluster_labels:
                    ax.text(cx, cy, str(int(lab)), color="white", fontsize=cluster_label_fontsize,
                            ha="center", va="center", weight="bold", zorder=12,
                            bbox=dict(facecolor=(0,0,0,0.5), pad=0.3, boxstyle="round"))

            # hull
            if hull and np.sum(mask) >= 3:
                try:
                    ch_pts = pts[mask]
                    hull_idx = ConvexHull(ch_pts).vertices
                    hull_poly = ch_pts[hull_idx]
                    ax.fill(hull_poly[:, 0], hull_poly[:, 1], alpha=hull_alpha, facecolor=col, edgecolor=hull_edgecolor, linewidth=0.6, zorder=5)
                except Exception:
                    pass

    # return nothing; ax is modified in place
    return None