"""Preprocessing functions for UMI annotations in AnnData objects."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)

if TYPE_CHECKING:
    import anndata as ad


def _has_homopolymer_run(seq: str, max_run: int = 4) -> bool:
    """Check if sequence contains a homopolymer run longer than max_run."""
    if not seq:
        return False
    seq = seq.upper()
    for base in "ACGT":
        pattern = base * (max_run + 1)
        if pattern in seq:
            return True
    return False


def _homopolymer_fraction(seq: str) -> float:
    """Calculate the fraction of the sequence that is the most common base."""
    if not seq:
        return 1.0
    seq = seq.upper()
    counts = {base: seq.count(base) for base in "ACGT"}
    max_count = max(counts.values())
    return max_count / len(seq)


def _unique_base_count(seq: str) -> int:
    """Count number of unique bases (A, C, G, T) in sequence."""
    if not seq:
        return 0
    seq = seq.upper()
    return len(set(c for c in seq if c in "ACGT"))


def _contains_n(seq: str) -> bool:
    """Check if sequence contains N or other ambiguous bases."""
    if not seq:
        return False
    seq = seq.upper()
    valid_bases = set("ACGT")
    return any(c not in valid_bases for c in seq)


def _edit_distance(s1: str, s2: str, max_dist: Optional[int] = None) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Uses edlib if available for speed, otherwise falls back to dynamic programming.
    If max_dist is provided, returns max_dist + 1 if distance exceeds threshold.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Try to use edlib for faster computation
    try:
        edlib = require("edlib", extra="umi", purpose="UMI edit distance calculation")
        k = max_dist if max_dist is not None else max(len(s1), len(s2))
        result = edlib.align(s1, s2, mode="NW", task="distance", k=k)
        dist = result.get("editDistance", -1)
        if dist == -1:
            # Distance exceeds k
            return (max_dist + 1) if max_dist is not None else max(len(s1), len(s2))
        return dist
    except Exception:
        pass

    # Fallback to standard DP algorithm with early termination
    len1, len2 = len(s1), len(s2)
    if max_dist is not None and abs(len1 - len2) > max_dist:
        return max_dist + 1

    # Use two-row DP for memory efficiency
    prev_row = list(range(len2 + 1))
    curr_row = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr_row[0] = i
        row_min = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,  # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution
            )
            row_min = min(row_min, curr_row[j])

        if max_dist is not None and row_min > max_dist:
            return max_dist + 1

        prev_row, curr_row = curr_row, prev_row

    return prev_row[len2]


def _cluster_umis_by_edit_distance(
    umis: List[str],
    max_edit_distance: int = 1,
    directional: bool = True,
) -> Dict[str, str]:
    """
    Cluster UMIs by edit distance and return mapping from UMI to cluster representative.

    Parameters
    ----------
    umis : List[str]
        List of UMI sequences to cluster.
    max_edit_distance : int
        Maximum edit distance to consider UMIs as belonging to same cluster.
    directional : bool
        If True, use directional clustering (high-count UMIs absorb low-count).
        If False, use simple hierarchical clustering.

    Returns
    -------
    Dict[str, str]
        Mapping from each UMI to its cluster representative (consensus).
    """
    if not umis:
        return {}

    # Count occurrences
    umi_counts = defaultdict(int)
    for umi in umis:
        umi_counts[umi] += 1

    unique_umis = list(umi_counts.keys())
    n_umis = len(unique_umis)

    if n_umis == 0:
        return {}
    if n_umis == 1:
        return {unique_umis[0]: unique_umis[0]}

    # Sort by count (descending) for directional clustering
    sorted_umis = sorted(unique_umis, key=lambda x: umi_counts[x], reverse=True)

    # Initialize: each UMI is its own cluster representative
    umi_to_cluster = {umi: umi for umi in unique_umis}

    if directional:
        # Directional clustering: high-count UMIs absorb low-count neighbors
        # Process from highest to lowest count
        for i, umi_i in enumerate(sorted_umis):
            if umi_to_cluster[umi_i] != umi_i:
                # Already assigned to another cluster
                continue

            # Find all UMIs within edit distance that have lower count
            for j in range(i + 1, n_umis):
                umi_j = sorted_umis[j]
                if umi_to_cluster[umi_j] != umi_j:
                    # Already assigned
                    continue

                dist = _edit_distance(umi_i, umi_j, max_dist=max_edit_distance)
                if dist <= max_edit_distance:
                    # Assign umi_j to umi_i's cluster
                    umi_to_cluster[umi_j] = umi_i
    else:
        # Simple union-find clustering
        parent = {umi: umi for umi in unique_umis}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Union by count (higher count becomes parent)
                if umi_counts[px] >= umi_counts[py]:
                    parent[py] = px
                else:
                    parent[px] = py

        # Compare all pairs (expensive for large sets)
        for i in range(n_umis):
            for j in range(i + 1, n_umis):
                umi_i, umi_j = unique_umis[i], unique_umis[j]
                dist = _edit_distance(umi_i, umi_j, max_dist=max_edit_distance)
                if dist <= max_edit_distance:
                    union(umi_i, umi_j)

        # Build final mapping
        for umi in unique_umis:
            umi_to_cluster[umi] = find(umi)

    return umi_to_cluster


def validate_umi(
    umi: Optional[str],
    expected_length: Optional[int] = None,
    max_homopolymer_fraction: float = 0.7,
    min_unique_bases: int = 2,
    max_homopolymer_run: int = 4,
) -> Tuple[bool, str]:
    """
    Validate a single UMI sequence.

    Parameters
    ----------
    umi : str or None
        UMI sequence to validate.
    expected_length : int, optional
        Expected UMI length. If provided, UMIs of different length are invalid.
    max_homopolymer_fraction : float
        Maximum fraction of UMI that can be a single base (default 0.7).
    min_unique_bases : int
        Minimum number of unique bases required (default 2).
    max_homopolymer_run : int
        Maximum allowed homopolymer run length (default 4).

    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason) - whether UMI is valid and reason if invalid.
    """
    if umi is None or pd.isna(umi):
        return False, "missing"

    umi = str(umi).strip().upper()

    if not umi:
        return False, "empty"

    if expected_length is not None and len(umi) != expected_length:
        return False, f"wrong_length_{len(umi)}"

    if _contains_n(umi):
        return False, "contains_N"

    if _homopolymer_fraction(umi) > max_homopolymer_fraction:
        return False, "high_homopolymer_fraction"

    if _unique_base_count(umi) < min_unique_bases:
        return False, "low_complexity"

    if _has_homopolymer_run(umi, max_homopolymer_run):
        return False, f"homopolymer_run_gt{max_homopolymer_run}"

    return True, "valid"


def preprocess_umi_annotations(
    adata: "ad.AnnData",
    umi_cols: Sequence[str] = ("U1", "U2"),
    combined_col: str = "RX",
    expected_length: Optional[int] = None,
    max_homopolymer_fraction: float = 0.7,
    min_unique_bases: int = 2,
    max_homopolymer_run: int = 4,
    cluster_max_edit_distance: int = 1,
    cluster_directional: bool = True,
    sample_col: str = "Barcode",
    reference_col: str = "Reference_strand",
    cluster_within_groups: bool = True,
    min_cluster_size: int = 1,
    uns_flag: str = "preprocess_umi_annotations_performed",
    bypass: bool = False,
    force_redo: bool = False,
) -> "ad.AnnData":
    """
    Preprocess UMI annotations: validate, filter, and cluster UMIs.

    This function:
    1. Validates each UMI (filters homopolymers, low complexity, Ns, wrong length)
    2. Clusters valid UMIs by edit distance to correct sequencing errors
    3. Assigns cluster IDs/consensus sequences
    4. Adds cleaned columns to adata.obs

    Parameters
    ----------
    adata : AnnData
        AnnData object with UMI columns in obs.
    umi_cols : Sequence[str]
        Names of UMI columns to process (default: ["U1", "U2"]).
    combined_col : str
        Name of combined UMI column (default: "RX").
    expected_length : int, optional
        Expected UMI length. If None, inferred from most common length.
    max_homopolymer_fraction : float
        Maximum fraction of UMI that can be a single base (default 0.7).
    min_unique_bases : int
        Minimum number of unique bases required (default 2).
    max_homopolymer_run : int
        Maximum allowed homopolymer run length (default 4).
    cluster_max_edit_distance : int
        Maximum edit distance for clustering UMIs (default 1).
    cluster_directional : bool
        Use directional clustering where high-count UMIs absorb low-count neighbors.
    sample_col : str
        Column name for sample/barcode grouping.
    reference_col : str
        Column name for reference/strand grouping.
    cluster_within_groups : bool
        If True, cluster UMIs separately within each sample/reference group.
    min_cluster_size : int
        Minimum cluster size to keep (default 1, keep all).
    uns_flag : str
        Key in adata.uns to mark completion.
    bypass : bool
        If True, skip processing entirely.
    force_redo : bool
        If True, rerun even if uns_flag is set.

    Returns
    -------
    AnnData
        Modified AnnData with new columns:
        - {umi_col}_valid: bool - whether UMI passed validation
        - {umi_col}_invalid_reason: str - reason for invalidity
        - {umi_col}_cluster: str - clustered/corrected UMI
        - {umi_col}_cluster_size: int - size of the UMI cluster
        - {combined_col}_cluster: str - combined clustered UMI
        - umi_cluster_key: str - combined key for deduplication
    """
    # Early exits
    if bypass:
        logger.info("Bypassing UMI preprocessing (bypass=True)")
        return adata

    if adata.uns.get(uns_flag, False) and not force_redo:
        logger.info("UMI preprocessing already performed (set force_redo=True to rerun)")
        return adata

    # Check which columns exist
    existing_cols = [col for col in umi_cols if col in adata.obs.columns]
    if not existing_cols:
        logger.warning(f"No UMI columns found in adata.obs. Expected: {umi_cols}")
        return adata

    logger.info(f"Preprocessing UMI annotations for columns: {existing_cols}")

    n_obs = adata.n_obs

    # Initialize output columns
    for col in existing_cols:
        adata.obs[f"{col}_valid"] = False
        adata.obs[f"{col}_invalid_reason"] = ""
        adata.obs[f"{col}_cluster"] = None
        adata.obs[f"{col}_cluster_size"] = 0

    # Infer expected length if not provided
    inferred_lengths = {}
    for col in existing_cols:
        lengths = adata.obs[col].dropna().astype(str).str.len()
        if len(lengths) > 0:
            # Use most common length
            length_counts = lengths.value_counts()
            if len(length_counts) > 0:
                inferred_lengths[col] = length_counts.index[0]
                logger.info(
                    f"Inferred UMI length for {col}: {inferred_lengths[col]} "
                    f"(from {length_counts.iloc[0]}/{len(lengths)} UMIs)"
                )

    # Step 1: Validate each UMI
    validation_stats = defaultdict(lambda: defaultdict(int))

    for col in existing_cols:
        col_length = expected_length if expected_length else inferred_lengths.get(col)

        valid_flags = []
        invalid_reasons = []

        for idx, umi in enumerate(adata.obs[col]):
            is_valid, reason = validate_umi(
                umi,
                expected_length=col_length,
                max_homopolymer_fraction=max_homopolymer_fraction,
                min_unique_bases=min_unique_bases,
                max_homopolymer_run=max_homopolymer_run,
            )
            valid_flags.append(is_valid)
            invalid_reasons.append(reason if not is_valid else "")
            validation_stats[col][reason] += 1

        adata.obs[f"{col}_valid"] = valid_flags
        adata.obs[f"{col}_invalid_reason"] = invalid_reasons

        n_valid = sum(valid_flags)
        logger.info(
            f"{col}: {n_valid}/{n_obs} ({100 * n_valid / n_obs:.1f}%) UMIs passed validation"
        )
        for reason, count in sorted(validation_stats[col].items(), key=lambda x: -x[1]):
            if reason != "valid":
                logger.debug(f"  {col} - {reason}: {count}")

    # Step 2: Cluster UMIs
    if cluster_within_groups:
        # Cluster within each sample/reference group
        group_cols = []
        if sample_col in adata.obs.columns:
            group_cols.append(sample_col)
        if reference_col in adata.obs.columns:
            group_cols.append(reference_col)

        if group_cols:
            groups = adata.obs.groupby(group_cols, observed=True)
        else:
            # No grouping columns, cluster all together
            groups = [(None, adata.obs)]
    else:
        groups = [(None, adata.obs)]

    # Initialize cluster columns with None
    for col in existing_cols:
        adata.obs[f"{col}_cluster"] = pd.Series([None] * n_obs, index=adata.obs.index, dtype=object)
        adata.obs[f"{col}_cluster_size"] = 0

    total_clusters = defaultdict(int)
    total_singletons = defaultdict(int)

    for group_key, group_df in groups:
        group_indices = group_df.index

        for col in existing_cols:
            # Get valid UMIs for this group
            valid_mask = adata.obs.loc[group_indices, f"{col}_valid"]
            valid_indices = group_indices[valid_mask]

            if len(valid_indices) == 0:
                continue

            valid_umis = adata.obs.loc[valid_indices, col].astype(str).str.upper().tolist()

            # Cluster UMIs
            umi_to_cluster = _cluster_umis_by_edit_distance(
                valid_umis,
                max_edit_distance=cluster_max_edit_distance,
                directional=cluster_directional,
            )

            # Count cluster sizes
            cluster_counts = defaultdict(int)
            for umi in valid_umis:
                cluster_rep = umi_to_cluster.get(umi, umi)
                cluster_counts[cluster_rep] += 1

            # Assign cluster representatives and sizes
            for idx, umi in zip(valid_indices, valid_umis):
                cluster_rep = umi_to_cluster.get(umi, umi)
                cluster_size = cluster_counts[cluster_rep]

                if cluster_size >= min_cluster_size:
                    adata.obs.at[idx, f"{col}_cluster"] = cluster_rep
                    adata.obs.at[idx, f"{col}_cluster_size"] = cluster_size

            # Statistics
            unique_clusters = set(umi_to_cluster.values())
            total_clusters[col] += len(unique_clusters)
            total_singletons[col] += sum(1 for c in cluster_counts.values() if c == 1)

    for col in existing_cols:
        n_clustered = adata.obs[f"{col}_cluster"].notna().sum()
        logger.info(
            f"{col}: {total_clusters[col]} unique clusters, "
            f"{total_singletons[col]} singletons, "
            f"{n_clustered} reads with cluster assignment"
        )

    # Step 3: Create combined cluster column
    if len(existing_cols) >= 2:
        # Combine U1 and U2 clusters
        u1_col = existing_cols[0]
        u2_col = existing_cols[1] if len(existing_cols) > 1 else None

        combined_clusters = []
        for idx in adata.obs.index:
            u1_cluster = adata.obs.at[idx, f"{u1_col}_cluster"]
            u2_cluster = adata.obs.at[idx, f"{u2_col}_cluster"] if u2_col else None

            if u1_cluster and u2_cluster:
                combined = f"{u1_cluster}-{u2_cluster}"
            elif u1_cluster:
                combined = u1_cluster
            elif u2_cluster:
                combined = u2_cluster
            else:
                combined = None
            combined_clusters.append(combined)

        adata.obs[f"{combined_col}_cluster"] = combined_clusters
    elif len(existing_cols) == 1:
        adata.obs[f"{combined_col}_cluster"] = adata.obs[f"{existing_cols[0]}_cluster"]

    # Step 4: Create UMI cluster key for deduplication
    # Combine sample, reference, and UMI cluster for unique molecule identification
    key_parts = []
    if sample_col in adata.obs.columns:
        key_parts.append(adata.obs[sample_col].astype(str))
    if reference_col in adata.obs.columns:
        key_parts.append(adata.obs[reference_col].astype(str))
    if f"{combined_col}_cluster" in adata.obs.columns:
        key_parts.append(adata.obs[f"{combined_col}_cluster"].fillna("NO_UMI").astype(str))

    if key_parts:
        adata.obs["umi_cluster_key"] = key_parts[0]
        for part in key_parts[1:]:
            adata.obs["umi_cluster_key"] = adata.obs["umi_cluster_key"] + "_" + part

    # Summary statistics
    n_with_valid_umi = (adata.obs[[f"{col}_valid" for col in existing_cols]].any(axis=1)).sum()
    n_with_cluster = adata.obs[f"{combined_col}_cluster"].notna().sum()

    logger.info(f"UMI preprocessing complete:")
    logger.info(
        f"  Reads with any valid UMI: {n_with_valid_umi}/{n_obs} ({100 * n_with_valid_umi / n_obs:.1f}%)"
    )
    logger.info(
        f"  Reads with cluster assignment: {n_with_cluster}/{n_obs} ({100 * n_with_cluster / n_obs:.1f}%)"
    )

    # Store validation stats in uns
    adata.uns["umi_preprocessing_stats"] = {
        "validation_stats": dict(validation_stats),
        "total_clusters": dict(total_clusters),
        "total_singletons": dict(total_singletons),
        "n_with_valid_umi": int(n_with_valid_umi),
        "n_with_cluster": int(n_with_cluster),
        "params": {
            "umi_cols": list(existing_cols),
            "expected_length": expected_length,
            "max_homopolymer_fraction": max_homopolymer_fraction,
            "min_unique_bases": min_unique_bases,
            "max_homopolymer_run": max_homopolymer_run,
            "cluster_max_edit_distance": cluster_max_edit_distance,
            "cluster_directional": cluster_directional,
            "cluster_within_groups": cluster_within_groups,
            "min_cluster_size": min_cluster_size,
        },
    }

    adata.uns[uns_flag] = True

    return adata
