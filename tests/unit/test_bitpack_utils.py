from __future__ import annotations

import numpy as np

from smftools.preprocessing._bitpack_utils import (
    pack_calls_and_valid_mask,
    popcount_hamming_windowed,
    unpack_to_float,
)


def _float_nan_aware_hamming(x_sub: np.ndarray, i: int, j_indices: np.ndarray, min_overlap: int):
    """Reference implementation matching the pre-bitpack torch/NaN comparison semantics."""
    row_i = x_sub[i]
    distances = np.full(len(j_indices), np.nan, dtype=float)
    overlaps = np.zeros(len(j_indices), dtype=int)
    for pos, j in enumerate(j_indices):
        row_j = x_sub[j]
        valid = ~np.isnan(row_i) & ~np.isnan(row_j)
        overlap = int(valid.sum())
        overlaps[pos] = overlap
        if overlap >= min_overlap:
            mismatches = (row_i[valid] != row_j[valid]).sum()
            distances[pos] = mismatches / overlap
    return distances, overlaps


def test_popcount_hamming_matches_float_reference_with_nans():
    rng = np.random.default_rng(0)
    n_reads, n_sites = 40, 137  # non-multiple-of-64 width exercises the packing pad path
    x_sub = rng.choice([0.0, 1.0], size=(n_reads, n_sites))
    nan_mask = rng.random((n_reads, n_sites)) < 0.2
    x_sub[nan_mask] = np.nan

    calls_u64, valid_u64 = pack_calls_and_valid_mask(x_sub)

    j_indices = np.arange(1, n_reads)
    packed_distances, packed_overlaps = popcount_hamming_windowed(
        calls_u64, valid_u64, 0, j_indices, min_overlap_positions=5
    )
    float_distances, float_overlaps = _float_nan_aware_hamming(x_sub, 0, j_indices, min_overlap=5)

    np.testing.assert_array_equal(packed_overlaps, float_overlaps)
    np.testing.assert_allclose(packed_distances, float_distances, equal_nan=True)


def test_unpack_to_float_round_trips_including_non_multiple_of_64_width():
    rng = np.random.default_rng(1)
    n_reads, n_sites = 17, 101  # deliberately not a multiple of 8 or 64
    x_sub = rng.choice([0.0, 1.0], size=(n_reads, n_sites))
    nan_mask = rng.random((n_reads, n_sites)) < 0.3
    x_sub[nan_mask] = np.nan

    calls_u64, valid_u64 = pack_calls_and_valid_mask(x_sub)
    reconstructed = unpack_to_float(calls_u64, valid_u64, n_sites)

    np.testing.assert_array_equal(np.isnan(reconstructed), np.isnan(x_sub))
    finite = ~np.isnan(x_sub)
    np.testing.assert_array_equal(reconstructed[finite], x_sub[finite])


def test_unpack_to_float_row_subset():
    x_sub = np.array(
        [[0.0, 1.0, np.nan], [1.0, np.nan, 0.0], [np.nan, np.nan, 1.0]], dtype=np.float32
    )
    calls_u64, valid_u64 = pack_calls_and_valid_mask(x_sub)
    subset = unpack_to_float(calls_u64, valid_u64, 3, row_indices=[2, 0])
    np.testing.assert_array_equal(subset[0], x_sub[2])
    np.testing.assert_array_equal(subset[1], x_sub[0])


def test_pack_calls_and_valid_mask_drops_precision_safely():
    x_sub = np.array([[0.0, 1.0, np.nan, 1.0, 0.0]], dtype=np.float32)
    calls_u64, valid_u64 = pack_calls_and_valid_mask(x_sub)
    # Round-trip via popcount against itself: distance to itself must be exactly 0.
    distances, overlaps = popcount_hamming_windowed(
        calls_u64, valid_u64, 0, np.array([0]), min_overlap_positions=1
    )
    assert overlaps[0] == 4  # 5 sites, 1 NaN
    assert distances[0] == 0.0
