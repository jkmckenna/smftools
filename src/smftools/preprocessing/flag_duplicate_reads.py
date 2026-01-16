from __future__ import annotations

# duplicate_detection_with_hier_and_plots.py
import copy
import math
import os
import warnings
from collections import defaultdict
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

from ..readwrite import make_dirs

logger = get_logger(__name__)

plt = require("matplotlib.pyplot", extra="plotting", purpose="duplicate read plots")
torch = require("torch", extra="torch", purpose="duplicate read detection")

if TYPE_CHECKING:
    import anndata as ad

SCIPY_AVAILABLE = True
SKLEARN_AVAILABLE = find_spec("sklearn") is not None

PCA = None
KMeans = DBSCAN = GaussianMixture = silhouette_score = None
if SKLEARN_AVAILABLE:
    sklearn_cluster = require(
        "sklearn.cluster",
        extra="ml-base",
        purpose="duplicate read clustering",
    )
    sklearn_decomp = require(
        "sklearn.decomposition",
        extra="ml-base",
        purpose="duplicate read PCA",
    )
    sklearn_metrics = require(
        "sklearn.metrics",
        extra="ml-base",
        purpose="duplicate read clustering diagnostics",
    )
    sklearn_mixture = require(
        "sklearn.mixture",
        extra="ml-base",
        purpose="duplicate read clustering",
    )
    DBSCAN = sklearn_cluster.DBSCAN
    KMeans = sklearn_cluster.KMeans
    PCA = sklearn_decomp.PCA
    silhouette_score = sklearn_metrics.silhouette_score
    GaussianMixture = sklearn_mixture.GaussianMixture


def merge_uns_preserve(orig_uns: dict, new_uns: dict, prefer: str = "orig") -> dict:
    """Merge two ``.uns`` dictionaries while preserving preferred values.

    Args:
        orig_uns: Original ``.uns`` dictionary.
        new_uns: New ``.uns`` dictionary to merge.
        prefer: Which dictionary to prefer on conflict (``"orig"`` or ``"new"``).

    Returns:
        dict: Merged dictionary.
    """
    out = copy.deepcopy(new_uns) if new_uns is not None else {}
    for k, v in (orig_uns or {}).items():
        if k not in out:
            out[k] = copy.deepcopy(v)
        else:
            # present in both: compare quickly (best-effort)
            try:
                equal = out[k] == v
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
    adata: ad.AnnData,
    var_filters_sets: Sequence[dict[str, Any]],
    distance_threshold: float = 0.07,
    obs_reference_col: str = "Reference_strand",
    sample_col: str = "Barcode",
    output_directory: Optional[str] = None,
    metric_keys: Union[str, List[str]] = ("Fraction_any_C_site_modified",),
    uns_flag: str = "flag_duplicate_reads_performed",
    uns_filtered_flag: str = "read_duplicates_removed",
    bypass: bool = False,
    force_redo: bool = False,
    keep_best_metric: Optional[str] = "read_quality",
    keep_best_higher: bool = True,
    window_size: int = 50,
    min_overlap_positions: int = 20,
    do_pca: bool = False,
    pca_n_components: int = 50,
    pca_center: bool = True,
    do_hierarchical: bool = True,
    hierarchical_linkage: str = "average",
    hierarchical_metric: str = "euclidean",
    hierarchical_window: int = 50,
    random_state: int = 0,
    demux_types: Optional[Sequence[str]] = None,
    demux_col: str = "demux_type",
) -> ad.AnnData:
    """Flag duplicate reads with demux-aware keeper preference.

    Behavior:
        - All reads are processed (no masking by demux).
        - At each keeper decision, prefer reads whose ``demux_col`` value is in
          ``demux_types`` when present. Among candidates, choose by
          ``keep_best_metric``.

    Args:
        adata: AnnData object to process.
        var_filters_sets: Sequence of variable filter definitions.
        distance_threshold: Distance threshold for duplicate detection.
        obs_reference_col: Obs column containing reference identifiers.
        sample_col: Obs column containing sample identifiers.
        output_directory: Directory for output plots and artifacts.
        metric_keys: Metric key(s) used in processing.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        uns_filtered_flag: Flag to mark read duplicates removal.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        keep_best_metric: Obs column used to select best read within duplicates.
        keep_best_higher: Whether higher values in ``keep_best_metric`` are preferred.
        window_size: Window size for local comparisons.
        min_overlap_positions: Minimum overlapping positions required.
        do_pca: Whether to run PCA before clustering.
        pca_n_components: Number of PCA components.
        pca_center: Whether to center data before PCA.
        do_hierarchical: Whether to run hierarchical clustering.
        hierarchical_linkage: Linkage method for hierarchical clustering.
        hierarchical_metric: Distance metric for hierarchical clustering.
        hierarchical_window: Window size for hierarchical clustering.
        random_state: Random seed.
        demux_types: Preferred demux types for keeper selection.
        demux_col: Obs column containing demux type labels.

    Returns:
        anndata.AnnData: AnnData object with duplicate flags stored in ``adata.obs``.
    """
    import copy
    import warnings

    import numpy as np
    import pandas as pd

    # -------- helper: demux-aware keeper selection --------
    def _choose_keeper_with_demux_preference(
        members_idx: List[int],
        adata_subset: ad.AnnData,
        obs_index_list: List[Any],
        *,
        demux_col: str = "demux_type",
        preferred_types: Optional[Sequence[str]] = None,
        keep_best_metric: Optional[str] = None,
        keep_best_higher: bool = True,
        lex_keeper_mask: Optional[np.ndarray] = None,  # aligned to members order
    ) -> int:
        """
        Prefer members whose demux_col ∈ preferred_types.
        Among candidates, pick by keep_best_metric (higher/lower).
        If metric missing/NaN, prefer lex keeper (via mask) among candidates.
        Fallback: first candidate.
        Returns the chosen *member index* (int from members_idx).
        """
        # 1) demux-preferred candidates
        if preferred_types and (demux_col in adata_subset.obs.columns):
            preferred = set(map(str, preferred_types))
            demux_series = adata_subset.obs[demux_col].astype("string")
            names = [obs_index_list[m] for m in members_idx]
            is_pref = demux_series.loc[names].isin(preferred).to_numpy()
            candidates = [members_idx[i] for i, ok in enumerate(is_pref) if ok]
        else:
            candidates = []

        if not candidates:
            candidates = list(members_idx)

        # 2) metric-based within candidates
        if keep_best_metric and (keep_best_metric in adata_subset.obs.columns):
            cand_names = [obs_index_list[m] for m in candidates]
            try:
                vals = pd.to_numeric(
                    adata_subset.obs.loc[cand_names, keep_best_metric],
                    errors="coerce",
                ).to_numpy(dtype=float)
            except Exception:
                vals = np.array([np.nan] * len(candidates), dtype=float)

            if not np.all(np.isnan(vals)):
                if keep_best_higher:
                    vals = np.where(np.isnan(vals), -np.inf, vals)
                    return candidates[int(np.nanargmax(vals))]
                else:
                    vals = np.where(np.isnan(vals), np.inf, vals)
                    return candidates[int(np.nanargmin(vals))]

        # 3) metric unhelpful — prefer lex keeper if provided
        if lex_keeper_mask is not None:
            for i, midx in enumerate(members_idx):
                if (midx in candidates) and bool(lex_keeper_mask[i]):
                    return midx

        # 4) fallback
        return candidates[0]

    # -------- early exits --------
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        if "is_duplicate" in adata.obs.columns:
            adata_unique = adata[~adata.obs["is_duplicate"]].copy()
            return adata_unique, adata
        else:
            return adata.copy(), adata.copy()
    if bypass:
        return None, adata

    if isinstance(metric_keys, str):
        metric_keys = [metric_keys]

    # local UnionFind
    class UnionFind:
        """Disjoint-set union-find helper for clustering indices."""

        def __init__(self, size):
            """Initialize parent pointers for the union-find."""
            self.parent = list(range(size))

        def find(self, x):
            """Find the root for a member with path compression."""
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            """Union the sets that contain x and y."""
            rx = self.find(x)
            ry = self.find(y)
            if rx != ry:
                self.parent[ry] = rx

    adata_processed_list = []
    histograms = []

    samples = adata.obs[sample_col].astype("category").cat.categories
    references = adata.obs[obs_reference_col].astype("category").cat.categories

    for sample in samples:
        for ref in references:
            logger.info("Processing sample=%s ref=%s", sample, ref)
            sample_mask = adata.obs[sample_col] == sample
            ref_mask = adata.obs[obs_reference_col] == ref
            subset_mask = sample_mask & ref_mask
            adata_subset = adata[subset_mask].copy()

            if adata_subset.n_obs < 2:
                logger.info("  Skipping %s_%s (too few reads)", sample, ref)
                continue

            N = adata_subset.shape[0]

            # Build mask of columns (vars) to use
            combined_mask = np.zeros(len(adata.var), dtype=bool)
            for var_set in var_filters_sets:
                if any(str(ref) in str(v) for v in var_set):
                    per_col_mask = np.ones(len(adata.var), dtype=bool)
                    for key in var_set:
                        per_col_mask &= np.asarray(adata.var[key].values, dtype=bool)
                    combined_mask |= per_col_mask

            selected_cols = adata.var.index[combined_mask.tolist()].to_list()
            col_indices = [adata.var.index.get_loc(c) for c in selected_cols]
            logger.info(
                "  Selected %s columns out of %s for %s",
                len(col_indices),
                adata.var.shape[0],
                ref,
            )

            # Extract data matrix (dense numpy) for the subset
            X = adata_subset.X
            if not isinstance(X, np.ndarray):
                try:
                    X = X.toarray()
                except Exception:
                    X = np.asarray(X)
            X_sub = X[:, col_indices].astype(float)  # keep NaNs

            # convert to torch for some vector ops
            X_tensor = torch.from_numpy(X_sub.copy())

            # per-read nearest distances recorded
            fwd_hamming_to_next = np.full((N,), np.nan, dtype=float)
            rev_hamming_to_prev = np.full((N,), np.nan, dtype=float)
            hierarchical_min_pair = np.full((N,), np.nan, dtype=float)

            # legacy local lexographic pairwise hamming distances (for histogram)
            local_hamming_dists = []
            # hierarchical discovered dists (for histogram)
            hierarchical_found_dists = []

            # Lexicographic windowed pass function
            def cluster_pass(
                X_tensor_local, reverse=False, window=int(window_size), record_distances=False
            ):
                """Perform a lexicographic windowed clustering pass."""
                N_local = X_tensor_local.shape[0]
                X_sortable = X_tensor_local.clone().nan_to_num(-1.0)
                sort_keys = [tuple(row.numpy().tolist()) for row in X_sortable]
                sorted_idx = sorted(range(N_local), key=lambda i: sort_keys[i], reverse=reverse)
                sorted_X = X_tensor_local[sorted_idx]
                cluster_pairs_local = []

                for i in range(len(sorted_X)):
                    row_i = sorted_X[i]
                    j_range_local = range(i + 1, min(i + 1 + window, len(sorted_X)))
                    if len(j_range_local) == 0:
                        continue
                    block_rows = sorted_X[list(j_range_local)]
                    row_i_exp = row_i.unsqueeze(0)  # (1, D)
                    valid_mask = (~torch.isnan(row_i_exp)) & (~torch.isnan(block_rows))  # (M, D)
                    valid_counts = valid_mask.sum(dim=1).float()
                    enough_overlap = valid_counts >= float(min_overlap_positions)
                    if enough_overlap.any():
                        diffs = (row_i_exp != block_rows) & valid_mask
                        hamming_counts = diffs.sum(dim=1).float()
                        hamming_dists = torch.where(
                            valid_counts > 0,
                            hamming_counts / valid_counts,
                            torch.tensor(float("nan")),
                        )
                        # record distances (legacy list of all local comparisons)
                        hamming_np = hamming_dists.cpu().numpy().tolist()
                        local_hamming_dists.extend(
                            [float(x) for x in hamming_np if (not np.isnan(x))]
                        )
                        matches = (hamming_dists < distance_threshold) & (enough_overlap)
                        for offset_local, m in enumerate(matches):
                            if m:
                                i_global = sorted_idx[i]
                                j_global = sorted_idx[i + 1 + offset_local]
                                cluster_pairs_local.append((i_global, j_global))
                        if record_distances:
                            # record next neighbor distance for the item (global index)
                            next_local_idx = i + 1
                            if next_local_idx < len(sorted_X):
                                next_global = sorted_idx[next_local_idx]
                                vm_pair = (~torch.isnan(row_i)) & (
                                    ~torch.isnan(sorted_X[next_local_idx])
                                )
                                vc = vm_pair.sum().item()
                                if vc >= min_overlap_positions:
                                    d = float(
                                        (
                                            (row_i[vm_pair] != sorted_X[next_local_idx][vm_pair])
                                            .sum()
                                            .item()
                                        )
                                        / vc
                                    )
                                    if reverse:
                                        rev_hamming_to_prev[next_global] = d
                                    else:
                                        fwd_hamming_to_next[sorted_idx[i]] = d
                return cluster_pairs_local

            # run forward & reverse windows
            pairs_fwd = cluster_pass(X_tensor, reverse=False, record_distances=True)
            involved_in_fwd = set([p for pair in pairs_fwd for p in pair])
            mask_for_rev = np.ones(N, dtype=bool)
            if len(involved_in_fwd) > 0:
                for idx in involved_in_fwd:
                    mask_for_rev[idx] = False
            rev_idx_map = np.nonzero(mask_for_rev)[0].tolist()
            if len(rev_idx_map) > 0:
                reduced_tensor = X_tensor[rev_idx_map]
                pairs_rev_local = cluster_pass(reduced_tensor, reverse=True, record_distances=True)
                remapped_rev_pairs = [
                    (int(rev_idx_map[i]), int(rev_idx_map[j])) for (i, j) in pairs_rev_local
                ]
            else:
                remapped_rev_pairs = []

            all_pairs = pairs_fwd + remapped_rev_pairs

            # initial union-find based on lex pairs
            uf = UnionFind(N)
            for i, j in all_pairs:
                uf.union(i, j)

            # initial merged clusters (lex-level)
            merged_cluster = np.zeros((N,), dtype=int)
            for i in range(N):
                merged_cluster[i] = uf.find(i)
            unique_initial = np.unique(merged_cluster)
            id_map = {old: new for new, old in enumerate(sorted(unique_initial.tolist()))}
            merged_cluster_mapped = np.array([id_map[int(x)] for x in merged_cluster], dtype=int)

            # cluster sizes and choose lex-keeper per lex-cluster (demux-aware)
            cluster_sizes = np.zeros_like(merged_cluster_mapped)
            cluster_counts = []
            unique_clusters = np.unique(merged_cluster_mapped)
            keeper_for_cluster = {}

            obs_index = list(adata_subset.obs.index)
            for cid in unique_clusters:
                members = np.where(merged_cluster_mapped == cid)[0].tolist()
                csize = int(len(members))
                cluster_counts.append(csize)
                cluster_sizes[members] = csize

                keeper_for_cluster[cid] = _choose_keeper_with_demux_preference(
                    members,
                    adata_subset,
                    obs_index,
                    demux_col=demux_col,
                    preferred_types=demux_types,
                    keep_best_metric=keep_best_metric,
                    keep_best_higher=keep_best_higher,
                    lex_keeper_mask=None,  # no lex preference yet
                )

            # expose lex keeper info (record only)
            lex_is_keeper = np.zeros((N,), dtype=bool)
            lex_is_duplicate = np.zeros((N,), dtype=bool)
            for cid in unique_clusters:
                members = np.where(merged_cluster_mapped == cid)[0].tolist()
                keeper_idx = keeper_for_cluster[cid]
                lex_is_keeper[keeper_idx] = True
                for m in members:
                    if m != keeper_idx:
                        lex_is_duplicate[m] = True

            # record lex min pair (min of fwd/rev neighbor) for each read
            min_pair = np.full((N,), np.nan, dtype=float)
            for i in range(N):
                a = fwd_hamming_to_next[i]
                b = rev_hamming_to_prev[i]
                vals = []
                if not np.isnan(a):
                    vals.append(a)
                if not np.isnan(b):
                    vals.append(b)
                if vals:
                    min_pair[i] = float(np.nanmin(vals))

            # --- hierarchical on representatives only ---
            hierarchical_pairs = []  # (rep_global_i, rep_global_j, d)
            rep_global_indices = sorted(set(keeper_for_cluster.values()))
            if do_hierarchical and len(rep_global_indices) > 1:
                if not SKLEARN_AVAILABLE:
                    warnings.warn("sklearn not available; skipping PCA/hierarchical pass.")
                elif not SCIPY_AVAILABLE:
                    warnings.warn("scipy not available; skipping hierarchical pass.")
                else:
                    # build reps array and impute for PCA
                    reps_X = X_sub[rep_global_indices, :]
                    reps_arr = np.array(reps_X, dtype=float, copy=True)
                    col_means = np.nanmean(reps_arr, axis=0)
                    col_means = np.where(np.isnan(col_means), 0.0, col_means)
                    inds = np.where(np.isnan(reps_arr))
                    if inds[0].size > 0:
                        reps_arr[inds] = np.take(col_means, inds[1])

                    # PCA if requested
                    if do_pca and PCA is not None:
                        n_comp = min(int(pca_n_components), reps_arr.shape[1], reps_arr.shape[0])
                        if n_comp <= 0:
                            reps_for_clustering = reps_arr
                        else:
                            pca = PCA(
                                n_components=n_comp,
                                random_state=int(random_state),
                                svd_solver="auto",
                                copy=True,
                            )
                            reps_for_clustering = pca.fit_transform(reps_arr)
                    else:
                        reps_for_clustering = reps_arr

                    # linkage & leaves (ordering)
                    try:
                        pdist_vec = pdist(reps_for_clustering, metric=hierarchical_metric)
                        Z = sch.linkage(pdist_vec, method=hierarchical_linkage)
                        leaves = sch.leaves_list(Z)
                    except Exception as e:
                        warnings.warn(
                            f"hierarchical pass failed: {e}; skipping hierarchical stage."
                        )
                        leaves = np.arange(len(rep_global_indices), dtype=int)

                    # windowed hamming comparisons across ordered reps and union
                    order_global_reps = [rep_global_indices[i] for i in leaves]
                    n_reps = len(order_global_reps)
                    for pos in range(n_reps):
                        i_global = order_global_reps[pos]
                        for jpos in range(pos + 1, min(pos + 1 + hierarchical_window, n_reps)):
                            j_global = order_global_reps[jpos]
                            vi = X_sub[int(i_global), :]
                            vj = X_sub[int(j_global), :]
                            valid_mask = (~np.isnan(vi)) & (~np.isnan(vj))
                            overlap = int(valid_mask.sum())
                            if overlap < min_overlap_positions:
                                continue
                            diffs = (vi[valid_mask] != vj[valid_mask]).sum()
                            d = float(diffs) / float(overlap)
                            if d < distance_threshold:
                                uf.union(int(i_global), int(j_global))
                                hierarchical_pairs.append((int(i_global), int(j_global), float(d)))
                                hierarchical_found_dists.append(float(d))

            # after hierarchical unions, reconstruct merged clusters for all reads
            merged_cluster_after = np.zeros((N,), dtype=int)
            for i in range(N):
                merged_cluster_after[i] = uf.find(i)
            unique_final = np.unique(merged_cluster_after)
            id_map_final = {old: new for new, old in enumerate(sorted(unique_final.tolist()))}
            merged_cluster_mapped_final = np.array(
                [id_map_final[int(x)] for x in merged_cluster_after], dtype=int
            )

            # compute final cluster members and choose final keeper per final cluster (demux-aware)
            cluster_sizes_final = np.zeros_like(merged_cluster_mapped_final)
            final_keeper_for_cluster = {}
            cluster_members_map = {}

            obs_index = list(adata_subset.obs.index)
            lex_mask_full = lex_is_keeper  # use lex keeper as optional tiebreaker

            for cid in np.unique(merged_cluster_mapped_final):
                members = np.where(merged_cluster_mapped_final == cid)[0].tolist()
                cluster_members_map[cid] = members
                cluster_sizes_final[members] = len(members)

                lex_mask_members = np.array([bool(lex_mask_full[m]) for m in members], dtype=bool)

                keeper = _choose_keeper_with_demux_preference(
                    members,
                    adata_subset,
                    obs_index,
                    demux_col=demux_col,
                    preferred_types=demux_types,
                    keep_best_metric=keep_best_metric,
                    keep_best_higher=keep_best_higher,
                    lex_keeper_mask=lex_mask_members,
                )
                final_keeper_for_cluster[cid] = keeper

            # update sequence__is_duplicate based on final clusters
            sequence_is_duplicate = np.zeros((N,), dtype=bool)
            for cid in np.unique(merged_cluster_mapped_final):
                keeper = final_keeper_for_cluster[cid]
                members = cluster_members_map[cid]
                if len(members) > 1:
                    for m in members:
                        if m != keeper:
                            sequence_is_duplicate[m] = True

            # propagate hierarchical distances into hierarchical_min_pair for all cluster members
            for i_g, j_g, d in hierarchical_pairs:
                c_i = merged_cluster_mapped_final[int(i_g)]
                c_j = merged_cluster_mapped_final[int(j_g)]
                members_i = cluster_members_map.get(c_i, [int(i_g)])
                members_j = cluster_members_map.get(c_j, [int(j_g)])
                for mi in members_i:
                    if np.isnan(hierarchical_min_pair[mi]) or (d < hierarchical_min_pair[mi]):
                        hierarchical_min_pair[mi] = d
                for mj in members_j:
                    if np.isnan(hierarchical_min_pair[mj]) or (d < hierarchical_min_pair[mj]):
                        hierarchical_min_pair[mj] = d

            # combine min pairs
            combined_min = min_pair.copy()
            for i in range(N):
                hval = hierarchical_min_pair[i]
                if not np.isnan(hval):
                    if np.isnan(combined_min[i]) or (hval < combined_min[i]):
                        combined_min[i] = hval

            # write columns back into adata_subset.obs
            adata_subset.obs["sequence__is_duplicate"] = sequence_is_duplicate
            adata_subset.obs["sequence__merged_cluster_id"] = merged_cluster_mapped_final
            adata_subset.obs["sequence__cluster_size"] = cluster_sizes_final
            adata_subset.obs["fwd_hamming_to_next"] = fwd_hamming_to_next
            adata_subset.obs["rev_hamming_to_prev"] = rev_hamming_to_prev
            adata_subset.obs["sequence__hier_hamming_to_pair"] = hierarchical_min_pair
            adata_subset.obs["sequence__min_hamming_to_pair"] = combined_min
            # persist lex bookkeeping
            adata_subset.obs["sequence__lex_is_keeper"] = lex_is_keeper
            adata_subset.obs["sequence__lex_is_duplicate"] = lex_is_duplicate

            adata_processed_list.append(adata_subset)

            histograms.append(
                {
                    "sample": sample,
                    "reference": ref,
                    "distances": local_hamming_dists,
                    "cluster_counts": [
                        int(x) for x in np.unique(cluster_sizes_final[cluster_sizes_final > 0])
                    ],
                    "hierarchical_pairs": hierarchical_found_dists,
                }
            )

    # Merge annotated subsets back together BEFORE plotting
    _original_uns = copy.deepcopy(adata.uns)
    if len(adata_processed_list) == 0:
        return adata.copy(), adata.copy()

    adata_full = ad.concat(adata_processed_list, merge="same", join="outer", index_unique=None)

    # preserve uns (prefer original on conflicts)
    def merge_uns_preserve(orig_uns: dict, new_uns: dict, prefer="orig") -> dict:
        """Merge .uns dictionaries while preserving original on conflicts."""
        out = copy.deepcopy(new_uns) if new_uns is not None else {}
        for k, v in (orig_uns or {}).items():
            if k not in out:
                out[k] = copy.deepcopy(v)
            else:
                try:
                    equal = out[k] == v
                except Exception:
                    equal = False
                if equal:
                    continue
                if prefer == "orig":
                    out[k] = copy.deepcopy(v)
                else:
                    out[f"orig_uns__{k}"] = copy.deepcopy(v)
        return out

    adata_full.uns = merge_uns_preserve(_original_uns, adata_full.uns, prefer="orig")

    # Ensure expected numeric columns exist (create if missing)
    for col in (
        "fwd_hamming_to_next",
        "rev_hamming_to_prev",
        "sequence__min_hamming_to_pair",
        "sequence__hier_hamming_to_pair",
    ):
        if col not in adata_full.obs.columns:
            adata_full.obs[col] = np.nan

    # histograms
    hist_outs = (
        os.path.join(output_directory, "read_pair_hamming_distance_histograms")
        if output_directory
        else None
    )
    if hist_outs:
        make_dirs([hist_outs])
    plot_histogram_pages(
        histograms,
        distance_threshold=distance_threshold,
        adata=adata_full,
        output_directory=hist_outs,
        distance_types=["min", "fwd", "rev", "hier", "lex_local"],
        sample_key=sample_col,
    )

    # hamming vs metric scatter
    scatter_outs = (
        os.path.join(output_directory, "read_pair_hamming_distance_scatter_plots")
        if output_directory
        else None
    )
    if scatter_outs:
        make_dirs([scatter_outs])
    plot_hamming_vs_metric_pages(
        adata_full,
        metric_keys=metric_keys,
        output_dir=scatter_outs,
        hamming_col="sequence__min_hamming_to_pair",
        highlight_threshold=distance_threshold,
        highlight_color="red",
        sample_col=sample_col,
    )

    # boolean columns from neighbor distances
    fwd_vals = pd.to_numeric(
        adata_full.obs.get("fwd_hamming_to_next", pd.Series(np.nan, index=adata_full.obs.index)),
        errors="coerce",
    )
    rev_vals = pd.to_numeric(
        adata_full.obs.get("rev_hamming_to_prev", pd.Series(np.nan, index=adata_full.obs.index)),
        errors="coerce",
    )
    is_dup_dist = (fwd_vals < float(distance_threshold)) | (rev_vals < float(distance_threshold))
    is_dup_dist = is_dup_dist.fillna(False).astype(bool)
    adata_full.obs["is_duplicate_distance"] = is_dup_dist.values

    # combine with sequence flag and any clustering flags
    seq_dup = (
        adata_full.obs["sequence__is_duplicate"].astype(bool)
        if "sequence__is_duplicate" in adata_full.obs.columns
        else pd.Series(False, index=adata_full.obs.index)
    )
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

    final_dup = (
        seq_dup
        | adata_full.obs["is_duplicate_distance"].astype(bool)
        | adata_full.obs["is_duplicate_clustering"].astype(bool)
    )
    adata_full.obs["is_duplicate"] = final_dup.values

    # -------- Final keeper enforcement on adata_full (demux-aware) --------
    keeper_idx_by_cluster = {}
    metric_col = keep_best_metric

    # Build an index→row-number mapping
    name_to_pos = {name: i for i, name in enumerate(adata_full.obs.index)}
    obs_index_full = list(adata_full.obs.index)

    lex_col = (
        "sequence__lex_is_keeper" if "sequence__lex_is_keeper" in adata_full.obs.columns else None
    )

    for cid, sub in adata_full.obs.groupby("sequence__merged_cluster_id", dropna=False):
        members_names = sub.index.to_list()
        members_pos = [name_to_pos[n] for n in members_names]

        if lex_col:
            lex_mask_members = adata_full.obs.loc[members_names, lex_col].astype(bool).to_numpy()
        else:
            lex_mask_members = np.zeros(len(members_pos), dtype=bool)

        keeper_pos = _choose_keeper_with_demux_preference(
            members_pos,
            adata_full,
            obs_index_full,
            demux_col=demux_col,
            preferred_types=demux_types,
            keep_best_metric=metric_col,
            keep_best_higher=keep_best_higher,
            lex_keeper_mask=lex_mask_members,
        )
        keeper_name = obs_index_full[keeper_pos]
        keeper_idx_by_cluster[cid] = keeper_name

    # enforce: keepers are not duplicates
    is_dup_series = adata_full.obs["is_duplicate"].astype(bool)
    for cid, keeper_name in keeper_idx_by_cluster.items():
        if keeper_name in adata_full.obs.index:
            is_dup_series.at[keeper_name] = False
            if "sequence__is_duplicate" in adata_full.obs.columns:
                adata_full.obs.at[keeper_name, "sequence__is_duplicate"] = False
            if "sequence__lex_is_duplicate" in adata_full.obs.columns:
                adata_full.obs.at[keeper_name, "sequence__lex_is_duplicate"] = False
    adata_full.obs["is_duplicate"] = is_dup_series.values

    # reason column
    def _dup_reason_row(row):
        """Build a semi-colon delimited duplicate reason string."""
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


# ---------------------------
# Plot helpers (use adata_full as input)
# ---------------------------


def plot_histogram_pages(
    histograms,
    distance_threshold,
    output_directory=None,
    rows_per_page=6,
    bins=50,
    dpi=160,
    figsize_per_cell=(5, 3),
    adata: Optional[ad.AnnData] = None,
    sample_key: str = "Barcode",
    ref_key: str = "Reference_strand",
    distance_key: str = "sequence__min_hamming_to_pair",
    distance_types: Optional[List[str]] = None,
):
    """
    Plot Hamming-distance histograms as a grid (rows=samples, cols=references).

    Changes:
      - Ensures that every subplot in a column (same ref) uses the same X-axis range and the same bins,
        computed from the union of values for that reference across samples/dtypes (clamped to [0,1]).
    """
    if distance_types is None:
        distance_types = ["min", "fwd", "rev", "hier", "lex_local"]

    # canonicalize samples / refs
    if adata is not None and sample_key in adata.obs.columns and ref_key in adata.obs.columns:
        obs = adata.obs
        sseries = obs[sample_key]
        if not pd.api.types.is_categorical_dtype(sseries):
            sseries = sseries.astype("category")
        samples = list(sseries.cat.categories)
        rseries = obs[ref_key]
        if not pd.api.types.is_categorical_dtype(rseries):
            rseries = rseries.astype("category")
        references = list(rseries.cat.categories)
        use_adata = True
    else:
        samples = sorted({h["sample"] for h in histograms})
        references = sorted({h["reference"] for h in histograms})
        use_adata = False

    if len(samples) == 0 or len(references) == 0:
        logger.info("No histogram data to plot.")
        return {"distance_pages": [], "cluster_size_pages": []}

    def clean_array(arr):
        """Filter array values to finite [0, 1] range for plotting."""
        if arr is None or len(arr) == 0:
            return np.array([], dtype=float)
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        a = a[(a >= 0.0) & (a <= 1.0)]
        return a

    grid = defaultdict(lambda: defaultdict(list))
    # populate from adata if available
    if use_adata:
        obs = adata.obs
        try:
            grouped = obs.groupby([sample_key, ref_key])
        except Exception:
            grouped = []
            for s in samples:
                for r in references:
                    sub = obs[(obs[sample_key] == s) & (obs[ref_key] == r)]
                    if not sub.empty:
                        grouped.append(((s, r), sub))
        if isinstance(grouped, dict) or hasattr(grouped, "groups"):
            for (s, r), group in grouped:
                if "min" in distance_types and distance_key in group.columns:
                    grid[(s, r)]["min"].extend(clean_array(group[distance_key].to_numpy()))
                if "fwd" in distance_types and "fwd_hamming_to_next" in group.columns:
                    grid[(s, r)]["fwd"].extend(clean_array(group["fwd_hamming_to_next"].to_numpy()))
                if "rev" in distance_types and "rev_hamming_to_prev" in group.columns:
                    grid[(s, r)]["rev"].extend(clean_array(group["rev_hamming_to_prev"].to_numpy()))
                if "hier" in distance_types and "sequence__hier_hamming_to_pair" in group.columns:
                    grid[(s, r)]["hier"].extend(
                        clean_array(group["sequence__hier_hamming_to_pair"].to_numpy())
                    )
        else:
            for (s, r), group in grouped:
                if "min" in distance_types and distance_key in group.columns:
                    grid[(s, r)]["min"].extend(clean_array(group[distance_key].to_numpy()))
                if "fwd" in distance_types and "fwd_hamming_to_next" in group.columns:
                    grid[(s, r)]["fwd"].extend(clean_array(group["fwd_hamming_to_next"].to_numpy()))
                if "rev" in distance_types and "rev_hamming_to_prev" in group.columns:
                    grid[(s, r)]["rev"].extend(clean_array(group["rev_hamming_to_prev"].to_numpy()))
                if "hier" in distance_types and "sequence__hier_hamming_to_pair" in group.columns:
                    grid[(s, r)]["hier"].extend(
                        clean_array(group["sequence__hier_hamming_to_pair"].to_numpy())
                    )

    # legacy histograms fallback
    if histograms:
        for h in histograms:
            key = (h["sample"], h["reference"])
            if "lex_local" in distance_types:
                grid[key]["lex_local"].extend(clean_array(h.get("distances", [])))
            if "hier" in distance_types and "hierarchical_pairs" in h:
                grid[key]["hier"].extend(clean_array(h.get("hierarchical_pairs", [])))
            if "cluster_counts" in h:
                grid[key]["_legacy_cluster_counts"].extend(h.get("cluster_counts", []))

    # Compute per-reference global x-range and bin edges (so every subplot in a column uses same bins)
    ref_xmax = {}
    for ref in references:
        vals_for_ref = []
        for s in samples:
            for dt in distance_types:
                a = np.asarray(grid[(s, ref)].get(dt, []), dtype=float)
                if a.size:
                    a = a[np.isfinite(a)]
                    if a.size:
                        vals_for_ref.append(a)
        if vals_for_ref:
            allvals = np.concatenate(vals_for_ref)
            vmax = float(np.nanmax(allvals)) if np.isfinite(allvals).any() else 1.0
            # pad slightly to include uppermost bin and always keep at least distance_threshold
            vmax = max(vmax, float(distance_threshold))
            vmax = min(1.0, max(0.01, vmax))  # clamp to [0.01, 1.0] to avoid degenerate bins
        else:
            vmax = 1.0
        ref_xmax[ref] = vmax

    # counts (for labels)
    if use_adata:
        counts = {
            (s, r): int(((adata.obs[sample_key] == s) & (adata.obs[ref_key] == r)).sum())
            for s in samples
            for r in references
        }
    else:
        counts = {
            (s, r): sum(len(grid[(s, r)][dt]) for dt in distance_types)
            for s in samples
            for r in references
        }

    distance_pages = []
    cluster_size_pages = []
    n_pages = math.ceil(len(samples) / rows_per_page)
    palette = plt.get_cmap("tab10")
    dtype_colors = {dt: palette(i % 10) for i, dt in enumerate(distance_types)}

    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, len(samples))
        chunk = samples[start:end]
        nrows = len(chunk)
        ncols = len(references)

        # Distance histogram page
        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
        )

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(references):
                ax = axes[r_idx][c_idx]
                any_data = False
                # pick per-column bins based on ref_xmax
                ref_vmax = ref_xmax.get(ref_name, 1.0)
                bins_edges = np.linspace(0.0, ref_vmax, bins + 1)
                for dtype in distance_types:
                    vals = np.asarray(grid[(sample_name, ref_name)].get(dtype, []), dtype=float)
                    if vals.size > 0:
                        vals = vals[np.isfinite(vals)]
                        vals = vals[(vals >= 0.0) & (vals <= ref_vmax)]
                    if vals.size > 0:
                        any_data = True
                        ax.hist(
                            vals,
                            bins=bins_edges,
                            alpha=0.5,
                            label=dtype,
                            density=False,
                            stacked=False,
                            color=dtype_colors.get(dtype, None),
                        )
                if not any_data:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )
                # threshold line (make sure it is within axis)
                ax.axvline(distance_threshold, color="red", linestyle="--", linewidth=1)

                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=10)
                if c_idx == 0:
                    total_reads = (
                        sum(counts.get((sample_name, ref), 0) for ref in references)
                        if not use_adata
                        else int((adata.obs[sample_key] == sample_name).sum())
                    )
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=9)
                if r_idx == nrows - 1:
                    ax.set_xlabel("Hamming Distance", fontsize=9)
                else:
                    ax.set_xticklabels([])

                ax.set_xlim(left=0.0, right=ref_vmax)
                ax.grid(True, alpha=0.25)
                if r_idx == 0 and c_idx == 0:
                    ax.legend(fontsize=7, loc="upper right")

        fig.suptitle(
            f"Hamming distance histograms (rows=samples, cols=references) — page {page + 1}/{n_pages}",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            fname = os.path.join(output_directory, f"hamming_histograms_page_{page + 1}.png")
            plt.savefig(fname, bbox_inches="tight")
            distance_pages.append(fname)
        else:
            plt.show()
        plt.close(fig)

        # Cluster-size histogram page (unchanged except it uses adata-derived sizes per cluster if available)
        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig2, axes2 = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
        )

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(references):
                ax = axes2[r_idx][c_idx]
                sizes = []
                if use_adata and (
                    "sequence__merged_cluster_id" in adata.obs.columns
                    and "sequence__cluster_size" in adata.obs.columns
                ):
                    sub = adata.obs[
                        (adata.obs[sample_key] == sample_name) & (adata.obs[ref_key] == ref_name)
                    ]
                    if not sub.empty:
                        try:
                            grp = sub.groupby("sequence__merged_cluster_id")[
                                "sequence__cluster_size"
                            ].first()
                            sizes = [
                                int(x)
                                for x in grp.to_numpy().tolist()
                                if (pd.notna(x) and np.isfinite(x))
                            ]
                        except Exception:
                            try:
                                unique_pairs = sub[
                                    ["sequence__merged_cluster_id", "sequence__cluster_size"]
                                ].drop_duplicates()
                                sizes = [
                                    int(x)
                                    for x in unique_pairs["sequence__cluster_size"]
                                    .dropna()
                                    .astype(int)
                                    .tolist()
                                ]
                            except Exception:
                                sizes = []
                if (not sizes) and histograms:
                    for h in histograms:
                        if h.get("sample") == sample_name and h.get("reference") == ref_name:
                            sizes = h.get("cluster_counts", []) or []
                            break

                if sizes:
                    ax.hist(sizes, bins=range(1, max(2, max(sizes) + 1)), alpha=0.8, align="left")
                    ax.set_xlabel("Cluster size")
                    ax.set_ylabel("Count")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No clusters",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )

                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=10)
                if c_idx == 0:
                    total_reads = (
                        sum(counts.get((sample_name, ref), 0) for ref in references)
                        if not use_adata
                        else int((adata.obs[sample_key] == sample_name).sum())
                    )
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=9)
                if r_idx != nrows - 1:
                    ax.set_xticklabels([])

                ax.grid(True, alpha=0.25)

        fig2.suptitle(
            f"Union-find cluster size histograms — page {page + 1}/{n_pages}", fontsize=12, y=0.995
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.96])

        if output_directory:
            fname2 = os.path.join(output_directory, f"cluster_size_histograms_page_{page + 1}.png")
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
    highlight_threshold: Optional[float] = None,
    highlight_color: str = "red",
    color_by_duplicate: bool = False,
    duplicate_col: str = "is_duplicate",
) -> Dict[str, Any]:
    """
    Plot hamming (y) vs metric (x).

    New behavior:
      - If color_by_duplicate is True and adata.obs[duplicate_col] exists, points are colored by that boolean:
           duplicates -> highlight_color (with edge), non-duplicates -> gray
      - If color_by_duplicate is False, previous highlight_threshold behavior is preserved.
    """
    if isinstance(metric_keys, str):
        metric_keys = [metric_keys]
    metric_keys = list(metric_keys)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    obs = adata.obs
    if sample_col not in obs.columns or ref_col not in obs.columns:
        raise ValueError(
            f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs"
        )

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
            sY = pd.to_numeric(obs[hamming_col], errors="coerce") if hamming_present else None

            if (sX is not None) and (sY is not None):
                valid_both = (
                    sX.notna() & sY.notna() & np.isfinite(sX.values) & np.isfinite(sY.values)
                )
                if valid_both.any():
                    xvals = sX[valid_both].to_numpy(dtype=float)
                    yvals = sY[valid_both].to_numpy(dtype=float)
                    xmin, xmax = float(np.nanmin(xvals)), float(np.nanmax(xvals))
                    ymin, ymax = float(np.nanmin(yvals)), float(np.nanmax(yvals))
                    xpad = (
                        max(1e-6, (xmax - xmin) * 0.05)
                        if xmax > xmin
                        else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                    )
                    ypad = (
                        max(1e-6, (ymax - ymin) * 0.05)
                        if ymax > ymin
                        else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                    )
                    global_xlim = (xmin - xpad, xmax + xpad)
                    global_ylim = (ymin - ypad, ymax + ypad)
                else:
                    sX_finite = sX[np.isfinite(sX)]
                    sY_finite = sY[np.isfinite(sY)]
                    if sX_finite.size > 0:
                        xmin, xmax = float(np.nanmin(sX_finite)), float(np.nanmax(sX_finite))
                        xpad = (
                            max(1e-6, (xmax - xmin) * 0.05)
                            if xmax > xmin
                            else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                        )
                        global_xlim = (xmin - xpad, xmax + xpad)
                    if sY_finite.size > 0:
                        ymin, ymax = float(np.nanmin(sY_finite)), float(np.nanmax(sY_finite))
                        ypad = (
                            max(1e-6, (ymax - ymin) * 0.05)
                            if ymax > ymin
                            else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                        )
                        global_ylim = (ymin - ypad, ymax + ypad)
            elif sX is not None:
                sX_finite = sX[np.isfinite(sX)]
                if sX_finite.size > 0:
                    xmin, xmax = float(np.nanmin(sX_finite)), float(np.nanmax(sX_finite))
                    xpad = (
                        max(1e-6, (xmax - xmin) * 0.05)
                        if xmax > xmin
                        else max(1e-3, abs(xmin) * 0.05 + 1e-3)
                    )
                    global_xlim = (xmin - xpad, xmax + xpad)
            elif sY is not None:
                sY_finite = sY[np.isfinite(sY)]
                if sY_finite.size > 0:
                    ymin, ymax = float(np.nanmin(sY_finite)), float(np.nanmax(sY_finite))
                    ypad = (
                        max(1e-6, (ymax - ymin) * 0.05)
                        if ymax > ymin
                        else max(1e-3, abs(ymin) * 0.05 + 1e-3)
                    )
                    global_ylim = (ymin - ypad, ymax + ypad)

        # pagination
        for start in range(0, len(samples_all), rows_per_fig):
            chunk = samples_all[start : start + rows_per_fig]
            nrows = len(chunk)
            ncols = len(cols)
            fig_w = ncols * figsize_per_cell[0]
            fig_h = nrows * figsize_per_cell[1]
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False
            )

            for r_idx, sample_name in enumerate(chunk):
                for c_idx, ref_name in enumerate(cols):
                    ax = axes[r_idx][c_idx]
                    if ref_name == extra_col:
                        mask = obs[sample_col].values == sample_name
                    else:
                        mask = (obs[sample_col].values == sample_name) & (
                            obs[ref_col].values == ref_name
                        )

                    sub = obs[mask]

                    if metric in sub.columns:
                        x_all = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
                    else:
                        x_all = np.array([], dtype=float)
                    if hamming_col in sub.columns:
                        y_all = pd.to_numeric(sub[hamming_col], errors="coerce").to_numpy(
                            dtype=float
                        )
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
                        x = np.array([], dtype=float)
                        y = np.array([], dtype=float)
                        idxs_valid = np.array([], dtype=int)

                    if x.size == 0:
                        ax.text(
                            0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                        )
                        clusters_info[(sample_name, ref_name)] = {"diag": None, "n_points": 0}
                    else:
                        # Decide color mapping
                        if (
                            color_by_duplicate
                            and duplicate_col in adata.obs.columns
                            and idxs_valid.size > 0
                        ):
                            # get boolean series aligned to idxs_valid
                            try:
                                dup_flags = (
                                    adata.obs.loc[idxs_valid, duplicate_col].astype(bool).to_numpy()
                                )
                            except Exception:
                                dup_flags = np.zeros(len(idxs_valid), dtype=bool)
                            mask_dup = dup_flags
                            mask_nondup = ~mask_dup
                            # plot non-duplicates first in gray, duplicates in highlight color
                            if mask_nondup.any():
                                ax.scatter(
                                    x[mask_nondup],
                                    y[mask_nondup],
                                    s=12,
                                    alpha=0.6,
                                    rasterized=True,
                                    c="lightgray",
                                )
                            if mask_dup.any():
                                ax.scatter(
                                    x[mask_dup],
                                    y[mask_dup],
                                    s=20,
                                    alpha=0.9,
                                    rasterized=True,
                                    c=highlight_color,
                                    edgecolors="k",
                                    linewidths=0.3,
                                )
                        else:
                            # old behavior: highlight by threshold if requested
                            if highlight_threshold is not None and y.size:
                                mask_low = (y < float(highlight_threshold)) & np.isfinite(y)
                                mask_high = ~mask_low
                                if mask_high.any():
                                    ax.scatter(
                                        x[mask_high], y[mask_high], s=12, alpha=0.6, rasterized=True
                                    )
                                if mask_low.any():
                                    ax.scatter(
                                        x[mask_low],
                                        y[mask_low],
                                        s=18,
                                        alpha=0.9,
                                        rasterized=True,
                                        c=highlight_color,
                                        edgecolors="k",
                                        linewidths=0.3,
                                    )
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
                                    ax.scatter(
                                        x, y, c=kde2, s=16, cmap="viridis", alpha=0.7, linewidths=0
                                    )
                            except Exception:
                                pass

                        if regression and x.size >= 2:
                            try:
                                a, b = np.polyfit(x, y, 1)
                                xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                                ys = a * xs + b
                                ax.plot(
                                    xs, ys, linestyle="--", linewidth=1.2, alpha=0.9, color="red"
                                )
                                r = np.corrcoef(x, y)[0, 1]
                                ax.text(
                                    0.98,
                                    0.02,
                                    f"r={float(r):.3f}",
                                    ha="right",
                                    va="bottom",
                                    transform=ax.transAxes,
                                    fontsize=8,
                                    bbox=dict(
                                        facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"
                                    ),
                                )
                            except Exception:
                                pass

                        if clustering:
                            cl_labels, diag = _run_clustering(
                                x,
                                y,
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
                                    mask_lab = cl_labels == lab
                                    medians[lab] = (
                                        float(np.median(y[mask_lab]))
                                        if mask_lab.any()
                                        else float("nan")
                                    )
                                sorted_by_median = sorted(
                                    unique_nonnoise,
                                    key=lambda idx: (
                                        np.nan if np.isnan(medians[idx]) else medians[idx]
                                    ),
                                    reverse=True,
                                )
                                mapping = {old: new for new, old in enumerate(sorted_by_median)}
                                for i_lab in range(len(remapped_labels)):
                                    if remapped_labels[i_lab] != -1:
                                        remapped_labels[i_lab] = mapping.get(
                                            remapped_labels[i_lab], -1
                                        )
                                diag = diag or {}
                                diag["cluster_median_hamming"] = {
                                    int(old): medians[old] for old in medians
                                }
                                diag["cluster_old_to_new_map"] = {
                                    int(old): int(new) for old, new in mapping.items()
                                }
                            else:
                                remapped_labels = cl_labels.copy()
                                diag = diag or {}
                                diag["cluster_median_hamming"] = {}
                                diag["cluster_old_to_new_map"] = {}

                            _overlay_clusters_on_ax(
                                ax,
                                x,
                                y,
                                remapped_labels,
                                diag,
                                cmap=clustering.get("cmap", "tab10"),
                                hull=clustering.get("hull", True),
                                show_cluster_labels=True,
                            )

                            clusters_info[(sample_name, ref_name)] = {
                                "diag": diag,
                                "n_points": len(x),
                            }

                            if write_clusters_to_adata and idxs_valid.size > 0:
                                colname_safe_ref = ref_name if ref_name != extra_col else "ALLREFS"
                                colname = (
                                    f"hamming_cluster__{metric}__{sample_name}__{colname_safe_ref}"
                                )
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

        saved_map[metric] = {
            "files": files,
            "clusters_info": clusters_info,
            "written_cols": written_cols,
        }

    return saved_map


def _run_clustering(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "kmeans",  # "kmeans", "dbscan", "gmm", "hdbscan"
    n_clusters: int = 2,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 5,
    random_state: int = 0,
    min_points: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run clustering on 2D points (x,y). Returns labels (len = npoints) and diagnostics dict.
    Labels follow sklearn conventions (noise -> -1 for DBSCAN/HDBSCAN).
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
        if (
            diagnostics.get("n_clusters_found", 0) >= 2
            and len(x) >= 3
            and silhouette_score is not None
        ):
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
        ax.scatter(
            x[mask],
            y[mask],
            s=20,
            c=[col],
            alpha=alpha_pts,
            marker=marker,
            linewidths=0.2,
            edgecolors="none",
            rasterized=True,
        )

        if lab != -1:
            # centroid
            if plot_centroids:
                cx = float(np.mean(x[mask]))
                cy = float(np.mean(y[mask]))
                ax.scatter(
                    [cx],
                    [cy],
                    s=centroid_size,
                    marker=centroid_marker,
                    c=[col],
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=10,
                )

                if show_cluster_labels:
                    ax.text(
                        cx,
                        cy,
                        str(int(lab)),
                        color="white",
                        fontsize=cluster_label_fontsize,
                        ha="center",
                        va="center",
                        weight="bold",
                        zorder=12,
                        bbox=dict(facecolor=(0, 0, 0, 0.5), pad=0.3, boxstyle="round"),
                    )

            # hull
            if hull and np.sum(mask) >= 3:
                try:
                    ch_pts = pts[mask]
                    hull_idx = ConvexHull(ch_pts).vertices
                    hull_poly = ch_pts[hull_idx]
                    ax.fill(
                        hull_poly[:, 0],
                        hull_poly[:, 1],
                        alpha=hull_alpha,
                        facecolor=col,
                        edgecolor=hull_edgecolor,
                        linewidth=0.6,
                        zorder=5,
                    )
                except Exception:
                    pass

    return None
