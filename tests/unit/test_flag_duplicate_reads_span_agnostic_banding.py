"""Regression tests for span-agnostic anchor-window banding.

`F51`: duplicate detection's sort key fills unmeasured positions with -1, so two
reads that agree perfectly over their overlap but differ in span diverge in the
leading key columns and sort far apart -- the comparison window never brings
them together, even though the distance function scores overlap only. The
hierarchical top-up was the sole span-blind step, and it is skipped above
`hierarchical_max_representatives`, so recall on fragmented libraries collapsed
once groups grew past that cap.

Anchor windows key on their own columns only, over only the reads whose measured
extent covers the whole window, so span drops out of the ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.flag_duplicate_reads import (
    _plan_anchor_windows,
    _process_group,
)

pytestmark = pytest.mark.unit

N_SITES = 400


def _group_with_span_offset(n_molecules: int, span_offset: int, seed: int = 0) -> np.ndarray:
    """Two reads per molecule: one full-span, one truncated at both ends.

    The truncated read shares its measured calls exactly with its full-span
    partner, so every pair is a true duplicate over its overlap.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_molecules):
        full = rng.integers(0, 2, N_SITES).astype(float)
        truncated = full.copy()
        truncated[:span_offset] = np.nan
        truncated[N_SITES - span_offset // 2 :] = np.nan
        rows += [full, truncated]
    return np.asarray(rows, dtype=np.float32)


def _run(x_sub: np.ndarray, *, span_agnostic_banding: bool, anchor_window_sites: int = 0):
    """Run one group with the hierarchical top-up capped out, as at scale."""
    n_reads = x_sub.shape[0]
    obs_index = [f"read{i}" for i in range(n_reads)]
    obs_df = pd.DataFrame(
        {"read_quality": np.ones(n_reads), "demux_type": ["double"] * n_reads},
        index=obs_index,
    )
    result = _process_group(
        {
            "X_sub": x_sub,
            "obs_df": obs_df,
            "obs_index": obs_index,
            "sample": "bc1",
            "ref": "ref",
            "distance_threshold": 0.07,
            "window_size": 50,
            "min_overlap_positions": 20,
            "keep_best_metric": "read_quality",
            "keep_best_higher": True,
            # Capped out on purpose: this is the regime the regression lives in,
            # and it keeps the hierarchical pass from masking the result.
            "do_hierarchical": True,
            "hierarchical_linkage": "average",
            "hierarchical_metric": "euclidean",
            "hierarchical_window": 50,
            "hierarchical_max_representatives": 1,
            "do_pca": False,
            "pca_n_components": 50,
            "random_state": 0,
            "demux_col": "demux_type",
            "demux_types": ["double"],
            "n_permutation_passes": 4,
            "permutation_seed": 0,
            "span_agnostic_banding": span_agnostic_banding,
            "anchor_window_sites": anchor_window_sites,
            "anchor_window_stride_sites": 0,
            "max_anchor_windows": 512,
        }
    )
    return np.asarray(result["sequence__merged_cluster_id"])


def _pair_recall(cluster_ids: np.ndarray, n_molecules: int) -> float:
    return (
        sum(cluster_ids[2 * m] == cluster_ids[2 * m + 1] for m in range(n_molecules)) / n_molecules
    )


@pytest.mark.parametrize("span_offset", [120, 200])
def test_differing_span_duplicates_are_clustered(span_offset):
    """The regression itself: same overlap, different span, one cluster each."""
    n_molecules = 500
    cluster_ids = _run(
        _group_with_span_offset(n_molecules, span_offset), span_agnostic_banding=True
    )
    assert _pair_recall(cluster_ids, n_molecules) == 1.0
    # Exactly one cluster per molecule -- full recall via over-merging would
    # show up here as a cluster count below the molecule count.
    assert len(np.unique(cluster_ids)) == n_molecules


@pytest.mark.parametrize("span_offset", [120, 200])
def test_banding_disabled_reproduces_the_regression(span_offset):
    """Guard the guard: without anchors these pairs are missed, so the test above
    is measuring the new pass rather than something else that already worked."""
    n_molecules = 500
    cluster_ids = _run(
        _group_with_span_offset(n_molecules, span_offset), span_agnostic_banding=False
    )
    assert _pair_recall(cluster_ids, n_molecules) < 0.5


def test_equal_span_duplicates_still_cluster():
    """Anchors must not disturb the case the lex passes already handled."""
    n_molecules = 500
    cluster_ids = _run(_group_with_span_offset(n_molecules, 0), span_agnostic_banding=True)
    assert _pair_recall(cluster_ids, n_molecules) == 1.0
    assert len(np.unique(cluster_ids)) == n_molecules


def test_distinct_reads_are_not_merged():
    """No duplicates present: every read must remain its own cluster."""
    rng = np.random.default_rng(1)
    x_sub = np.asarray(rng.integers(0, 2, (2000, N_SITES)), dtype=np.float32)
    cluster_ids = _run(x_sub, span_agnostic_banding=True)
    assert len(np.unique(cluster_ids)) == 2000


def test_anchor_width_narrows_to_short_fragments():
    """A width above the median span would leave most reads in no window at all.

    Without auto-narrowing this silently plans nothing and restores the bug, so
    the default width must still work on a fragmented library.
    """
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(500):
        molecule = rng.integers(0, 2, N_SITES).astype(float)
        first = np.full(N_SITES, np.nan)
        first[100:160] = molecule[100:160]
        second = np.full(N_SITES, np.nan)
        second[120:180] = molecule[120:180]
        rows += [first, second]
    cluster_ids = _run(
        np.asarray(rows, dtype=np.float32),
        span_agnostic_banding=True,
        anchor_window_sites=100,  # wider than the 60-site fragments
    )
    assert _pair_recall(cluster_ids, 500) == 1.0
    assert len(np.unique(cluster_ids)) == 500


def test_plan_anchor_windows_selects_only_covering_reads():
    coverage_start = np.array([0, 120, 0, 300])
    coverage_end = np.array([400, 340, 400, 400])
    windows = _plan_anchor_windows(
        coverage_start,
        coverage_end,
        n_sites=400,
        anchor_window_sites=100,
        anchor_window_stride_sites=0,
        max_anchor_windows=512,
    )
    covered = {(start, end): rows.tolist() for start, end, rows in windows}
    # Read 1 spans [120, 340): it covers [200,300) but not [100,200).
    assert covered[(100, 200)] == [0, 2]
    assert covered[(200, 300)] == [0, 1, 2]
    # Read 3 spans [300, 400): only the last window.
    assert covered[(300, 400)] == [0, 2, 3]


def test_plan_anchor_windows_disabled_returns_nothing():
    windows = _plan_anchor_windows(
        np.array([0, 0]),
        np.array([400, 400]),
        n_sites=400,
        anchor_window_sites=100,
        anchor_window_stride_sites=0,
        max_anchor_windows=512,
        enabled=False,
    )
    assert windows == []


def test_max_anchor_windows_widens_stride_instead_of_truncating():
    """The cap must not leave one end of the reference with no anchored pass."""
    windows = _plan_anchor_windows(
        np.zeros(4, dtype=np.int64),
        np.full(4, 4000, dtype=np.int64),
        n_sites=4000,
        anchor_window_sites=100,
        anchor_window_stride_sites=0,
        max_anchor_windows=8,
    )
    assert len(windows) <= 8
    assert windows[-1][1] > 3000


def _random_fragmentation(n_molecules: int, min_len: int, max_len: int, seed: int = 0):
    """Each molecule observed twice at independent random spans.

    The two-size-class fixture above is the easy case. Real fragmentation puts
    every pair at a different overlap length, and a pair is only *eligible* to
    be called duplicate when its overlap reaches `min_overlap_positions` -- so
    recall is scored over eligible pairs only.
    """
    rng = np.random.default_rng(seed)
    n_sites = 2000
    rows, overlaps = [], []
    for _ in range(n_molecules):
        molecule = rng.integers(0, 2, n_sites).astype(float)
        spans = []
        for _ in range(2):
            length = int(rng.integers(min_len, max_len + 1))
            start = int(rng.integers(0, n_sites - length + 1))
            read = np.full(n_sites, np.nan)
            read[start : start + length] = molecule[start : start + length]
            rows.append(read)
            spans.append((start, start + length))
        (a_start, a_end), (b_start, b_end) = spans
        overlaps.append(max(0, min(a_end, b_end) - max(a_start, b_start)))
    return np.asarray(rows, dtype=np.float32), overlaps


def _eligible_recall(cluster_ids, overlaps, min_overlap: int = 20):
    eligible = [m for m, overlap in enumerate(overlaps) if overlap >= min_overlap]
    assert eligible, "fixture produced no comparable pairs"
    hits = sum(cluster_ids[2 * m] == cluster_ids[2 * m + 1] for m in eligible)
    return hits / len(eligible)


@pytest.mark.parametrize(
    ("min_len", "max_len"),
    [(50, 1500), (50, 300)],
    ids=["wide-size-range", "short-fragments"],
)
def test_random_fragmentation_recovers_every_comparable_pair(min_len, max_len):
    """Fragment sizes spanning many size classes, not two clean classes.

    The derived window geometry exists for this case: with the first draft's
    fixed 100-site window at stride 100, a pair needed 199 positions of overlap
    to share a window, so pairs overlapping by 50-99 were recovered at 0.08.
    """
    x_sub, overlaps = _random_fragmentation(1000, min_len, max_len)
    cluster_ids = _run(x_sub, span_agnostic_banding=True)
    assert _eligible_recall(cluster_ids, overlaps) == 1.0


def test_random_fragmentation_does_not_merge_distinct_molecules():
    """Short overlaps mean fewer positions to disagree on; check precision too."""
    x_sub, _ = _random_fragmentation(400, 50, 1500, seed=11)
    # Re-key every read to its own molecule by rebuilding without duplication.
    rng = np.random.default_rng(5)
    n_sites = 2000
    rows = []
    for _ in range(800):
        molecule = rng.integers(0, 2, n_sites).astype(float)
        length = int(rng.integers(50, 1501))
        start = int(rng.integers(0, n_sites - length + 1))
        read = np.full(n_sites, np.nan)
        read[start : start + length] = molecule[start : start + length]
        rows.append(read)
    cluster_ids = _run(np.asarray(rows, dtype=np.float32), span_agnostic_banding=True)
    assert len(np.unique(cluster_ids)) == 800


def test_derived_geometry_reaches_the_configured_minimum_overlap():
    """width + stride - 1 must not exceed min_overlap_positions.

    Windows start at multiples of the stride, so a pair overlapping by fewer
    positions than that can never contain an aligned window no matter how well
    its calls agree.
    """
    for min_overlap in (10, 20, 50):
        windows = _plan_anchor_windows(
            np.zeros(4, dtype=np.int64),
            np.full(4, 2000, dtype=np.int64),
            n_sites=2000,
            anchor_window_sites=0,
            anchor_window_stride_sites=0,
            max_anchor_windows=10_000,
            min_overlap_positions=min_overlap,
        )
        width = windows[0][1] - windows[0][0]
        stride = windows[1][0] - windows[0][0]
        assert width + stride - 1 <= min_overlap


def test_unreachable_explicit_geometry_is_logged(caplog):
    """A hand-set width too wide for the minimum overlap must not fail silently."""
    import logging

    import smftools.preprocessing.flag_duplicate_reads as module

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        _plan_anchor_windows(
            np.zeros(4, dtype=np.int64),
            np.full(4, 2000, dtype=np.int64),
            n_sites=2000,
            anchor_window_sites=100,
            anchor_window_stride_sites=0,
            max_anchor_windows=10_000,
            min_overlap_positions=20,
        )
    assert any("reaches only pairs overlapping" in record.message for record in caplog.records)


def test_window_ceiling_that_breaks_reach_is_logged(caplog):
    """Widening the stride to honour the ceiling costs short-overlap recall."""
    import logging

    import smftools.preprocessing.flag_duplicate_reads as module

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        _plan_anchor_windows(
            np.zeros(4, dtype=np.int64),
            np.full(4, 20_000, dtype=np.int64),
            n_sites=20_000,
            anchor_window_sites=0,
            anchor_window_stride_sites=0,
            max_anchor_windows=8,
            min_overlap_positions=20,
        )
    assert any("widened the anchor stride" in record.message for record in caplog.records)
