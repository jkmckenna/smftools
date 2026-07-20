"""Regression tests for the hierarchical-clustering representative-count cap.

Exact pdist+linkage is O(n^2) in the representative count -- uncapped, this
crashed a real production run (~40,000 representatives). See
dev/duplicate_detection_scaling.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.flag_duplicate_reads import _process_group


def _distinct_reads_args(n_reads: int, *, hierarchical_max_representatives: int) -> dict:
    """n_reads fully distinct reads (no lex duplicates), all becoming their own
    lex-cluster representative -- so the representative count fed to the
    hierarchical block equals n_reads.
    """
    rng = np.random.default_rng(3)
    n_sites = 100
    x_sub = rng.integers(0, 2, size=(n_reads, n_sites)).astype(float)
    obs_index = [f"read{i}" for i in range(n_reads)]
    obs_df = pd.DataFrame(index=obs_index)

    return {
        "X_sub": x_sub,
        "obs_df": obs_df,
        "obs_index": obs_index,
        "sample": "bc1",
        "ref": "ref",
        "distance_threshold": 0.01,  # tight -- these random reads shouldn't lex-cluster
        "window_size": 5,
        "min_overlap_positions": 20,
        "keep_best_metric": None,
        "keep_best_higher": True,
        "do_hierarchical": True,
        "hierarchical_linkage": "average",
        "hierarchical_metric": "euclidean",
        "hierarchical_window": 5,
        "hierarchical_max_representatives": hierarchical_max_representatives,
        "do_pca": False,
        "pca_n_components": 50,
        "random_state": 0,
        "demux_col": "demux_type",
        "demux_types": None,
        "n_permutation_passes": 0,
        "permutation_seed": 0,
    }


def test_hierarchical_topup_skipped_above_representative_cap(monkeypatch):
    import smftools.preprocessing.flag_duplicate_reads as module

    linkage_calls = []
    monkeypatch.setattr(
        module.sch, "linkage", lambda *a, **k: (linkage_calls.append(1), np.zeros((0, 4)))[1]
    )

    with pytest.warns(UserWarning, match="skipping hierarchical top-up"):
        result = _process_group(_distinct_reads_args(50, hierarchical_max_representatives=10))

    assert result is not None
    assert linkage_calls == []


def test_hierarchical_topup_runs_below_representative_cap(monkeypatch):
    import smftools.preprocessing.flag_duplicate_reads as module

    linkage_calls = []
    real_linkage = module.sch.linkage

    def spy_linkage(*a, **k):
        linkage_calls.append(1)
        return real_linkage(*a, **k)

    monkeypatch.setattr(module.sch, "linkage", spy_linkage)

    result = _process_group(_distinct_reads_args(20, hierarchical_max_representatives=50))

    assert result is not None
    assert linkage_calls == [1]
