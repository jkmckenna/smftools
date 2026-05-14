from __future__ import annotations

import numpy as np
import pytest

from smftools.analysis.compute.hmm_features import extract_intervals_from_row

# ---------------------------------------------------------------------------
# extract_intervals_from_row
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_row_returns_empty() -> None:
    sizes, dists = extract_intervals_from_row(np.array([]), np.array([]))
    assert sizes == []
    assert dists == []


@pytest.mark.unit
def test_all_zero_row_returns_empty() -> None:
    row = np.zeros(10)
    coords = np.arange(10, dtype=float)
    sizes, dists = extract_intervals_from_row(row, coords)
    assert sizes == []
    assert dists == []


@pytest.mark.unit
def test_single_interval() -> None:
    #  0  0  1  1  1  0  0
    row = np.array([0, 0, 1, 1, 1, 0, 0], dtype=float)
    coords = np.arange(7, dtype=float) * 10  # 0, 10, 20, 30, 40, 50, 60
    sizes, dists = extract_intervals_from_row(row, coords)
    # coords[2]=20, coords[4]=40 → size = 40 - 20 + 1 = 21 bp
    assert sizes == [21]
    assert dists == []  # no neighbors


@pytest.mark.unit
def test_two_intervals_neighbor_distance() -> None:
    # interval at positions 0-10, gap, interval at 30-40
    row = np.array([1, 1, 0, 1, 1], dtype=float)
    coords = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    sizes, dists = extract_intervals_from_row(row, coords)
    assert len(sizes) == 2
    assert len(dists) == 1
    # Centers: (0+10)/2=5, (30+40)/2=35, distance=30
    assert dists[0] == pytest.approx(30.0)


@pytest.mark.unit
def test_nan_treated_as_absent() -> None:
    # NaN positions are not > 0, so treated as 0 (not an interval)
    row = np.array([1.0, np.nan, 1.0], dtype=float)
    coords = np.array([0.0, 10.0, 20.0])
    sizes, dists = extract_intervals_from_row(row, coords)
    # NaN breaks the run → two separate single-position intervals
    assert len(sizes) == 2


@pytest.mark.unit
def test_interval_sizes_are_integers() -> None:
    row = np.array([1, 1, 1], dtype=float)
    coords = np.array([0.0, 5.0, 10.0])
    sizes, _ = extract_intervals_from_row(row, coords)
    assert all(isinstance(s, int) for s in sizes)


@pytest.mark.unit
def test_three_intervals_two_distances() -> None:
    row = np.array([1, 0, 1, 0, 1], dtype=float)
    coords = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    sizes, dists = extract_intervals_from_row(row, coords)
    assert len(sizes) == 3
    assert len(dists) == 2


# ---------------------------------------------------------------------------
# extract_intervals_from_row — full_row boundary clipping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_row_clipping_drops_mostly_external_interval() -> None:
    # Full locus: one long interval spanning 0..40
    full_row = np.array([1, 1, 1, 1, 1], dtype=float)
    full_coords = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    # Analysis window only covers 20..40 (last 3 positions)
    # The interval appears to start at 20 but extends left to 0 in the full row
    # so 20 bp out of 40 bp is outside → outside_frac = 0.5 exactly
    sub_row = np.array([1, 1, 1], dtype=float)
    sub_coords = np.array([20.0, 30.0, 40.0])

    sizes_no_clip, _ = extract_intervals_from_row(sub_row, sub_coords)
    sizes_clip, _ = extract_intervals_from_row(
        sub_row,
        sub_coords,
        full_row=full_row,
        full_coords=full_coords,
        max_mask_overlap_frac=0.4,  # drop if >40% outside → 50% > 40% → dropped
    )
    assert len(sizes_no_clip) == 1  # without clipping: interval retained
    assert len(sizes_clip) == 0  # with clipping: interval dropped (50% outside > 40% threshold)
