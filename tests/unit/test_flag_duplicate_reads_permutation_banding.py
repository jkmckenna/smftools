"""Regression tests for the K-random-permutation lex-sort banding generalization.

Replaces the uncapped O(n^2) hierarchical clustering pass as the primary
mechanism for catching duplicate reads that a single fixed sort order misses.
See dev/duplicate_detection_scaling.md and the plan this was built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smftools.preprocessing.flag_duplicate_reads import _process_group


def _args_with_early_divergent_pair(n_permutation_passes: int, *, n_sites: int = 200) -> dict:
    """Build a group where read0/read1 are true near-duplicates (differ only at
    column 0) but are separated by more reads than window_size under natural
    sort order -- so only the permutation-banding generalization can catch them
    (hierarchical clustering is disabled so it can't mask the result).
    """
    rng = np.random.default_rng(7)

    read0 = np.zeros(n_sites, dtype=float)
    read1 = np.zeros(n_sites, dtype=float)
    read1[0] = 1.0  # differs from read0 only at column 0 -> distance 1/n_sites

    # Fillers with column0=0: none all-zero, so read0 (the true all-zero key)
    # sorts strictly first within this block. 60 of them (> window_size=50)
    # guarantees read0 and read1 are separated by more than one window under
    # natural forward/reverse order.
    n_fillers_block0 = 60
    fillers_block0 = rng.integers(0, 2, size=(n_fillers_block0, n_sites)).astype(float)
    fillers_block0[:, 0] = 0.0
    fillers_block0[np.all(fillers_block0[:, 1:] == 0.0, axis=1), 1] = 1.0  # avoid accidental all-zero ties

    n_fillers_block1 = 20
    fillers_block1 = rng.integers(0, 2, size=(n_fillers_block1, n_sites)).astype(float)
    fillers_block1[:, 0] = 1.0

    x_sub = np.vstack([read0, read1, fillers_block0, fillers_block1])
    n_reads = x_sub.shape[0]
    obs_index = [f"read{i}" for i in range(n_reads)]
    obs_df = pd.DataFrame(index=obs_index)

    return {
        "X_sub": x_sub,
        "obs_df": obs_df,
        "obs_index": obs_index,
        "sample": "bc1",
        "ref": "ref",
        "distance_threshold": 0.07,
        "window_size": 50,
        "min_overlap_positions": 20,
        "keep_best_metric": None,
        "keep_best_higher": True,
        "do_hierarchical": False,  # isolate the permutation-passes' own contribution
        "hierarchical_linkage": "average",
        "hierarchical_metric": "euclidean",
        "hierarchical_window": 50,
        "hierarchical_max_representatives": 5000,
        "do_pca": False,
        "pca_n_components": 50,
        "random_state": 0,
        "demux_col": "demux_type",
        "demux_types": None,
        "n_permutation_passes": n_permutation_passes,
        "permutation_seed": 0,
    }


def test_natural_order_only_misses_early_divergent_pair():
    result = _process_group(_args_with_early_divergent_pair(n_permutation_passes=0))
    assert result is not None
    cluster_ids = result["sequence__merged_cluster_id"]
    assert cluster_ids[0] != cluster_ids[1]


def test_permutation_passes_catch_early_divergent_pair():
    result = _process_group(_args_with_early_divergent_pair(n_permutation_passes=6))
    assert result is not None
    cluster_ids = result["sequence__merged_cluster_id"]
    assert cluster_ids[0] == cluster_ids[1]
