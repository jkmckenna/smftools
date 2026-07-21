"""Bit-packing and popcount helpers for compact binary-with-missing-data comparisons.

Shared by UMI annotation (fully-populated one-hot DNA vectors) and duplicate-read
detection (binary methylation calls with NaN-valued missing positions). Packing
into ``uint64`` words keeps the resident representation of a read's per-site data
close to 1 bit/site instead of 4-8 bytes/site, and lets pairwise/windowed Hamming
distance be computed via vectorized XOR + popcount instead of per-element float
comparison.
"""

from __future__ import annotations

import numpy as np


def _pack_bool_to_u64(b: np.ndarray) -> np.ndarray:
    """Pack boolean matrix (n, w) into uint64 blocks (n, ceil(w/64))."""
    b = np.asarray(b, dtype=np.uint8)
    packed_u8 = np.packbits(b, axis=1)
    n, nb = packed_u8.shape
    pad = (-nb) % 8
    if pad:
        packed_u8 = np.pad(packed_u8, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    packed_u8 = np.ascontiguousarray(packed_u8)
    return packed_u8.reshape(n, -1, 8).view(np.uint64).reshape(n, -1)


def _popcount_u64_matrix(a_u64: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint64 arrays."""
    b = a_u64.view(np.uint8).reshape(a_u64.shape + (8,))
    return np.unpackbits(b, axis=-1).sum(axis=-1)


def pack_calls_and_valid_mask(x_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ``{0.0, 1.0, NaN}`` read x site matrix into bit-packed calls + valid mask.

    ``calls_u64``: bit set where the call is ``1.0`` (NaN positions get an arbitrary
    but ignored bit, masked out via ``valid_u64`` at comparison time).
    ``valid_u64``: bit set where the position is not ``NaN``.

    Callers should pack immediately after materializing a read's data and hold only
    the packed arrays for the rest of a comparison pass -- never keep the source
    float array resident once packing is done.
    """
    x_sub = np.asarray(x_sub, dtype=np.float32)
    valid = ~np.isnan(x_sub)
    calls = np.zeros_like(x_sub, dtype=bool)
    calls[valid] = x_sub[valid] != 0.0
    return _pack_bool_to_u64(calls), _pack_bool_to_u64(valid)


def unpack_to_float(
    calls_u64: np.ndarray,
    valid_u64: np.ndarray,
    n_sites: int,
    row_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Reconstruct a ``{0.0, 1.0, NaN}`` float matrix from packed calls/valid.

    Used only where a downstream algorithm (PCA, ``pdist``) genuinely needs float
    input -- callers should keep this restricted to small, bounded row subsets
    (e.g. capped hierarchical-clustering representatives), never the full chunk.
    """
    if row_indices is not None:
        calls_u64 = calls_u64[np.asarray(row_indices)]
        valid_u64 = valid_u64[np.asarray(row_indices)]
    calls_bits = np.unpackbits(calls_u64.view(np.uint8), axis=1)[:, :n_sites]
    valid_bits = np.unpackbits(valid_u64.view(np.uint8), axis=1)[:, :n_sites].astype(bool)
    out = np.full(calls_bits.shape, np.nan, dtype=float)
    out[valid_bits] = calls_bits[valid_bits].astype(float)
    return out


def popcount_hamming_windowed(
    calls_u64: np.ndarray,
    valid_u64: np.ndarray,
    i: int,
    j_indices: np.ndarray,
    *,
    min_overlap_positions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware fractional Hamming distance from read ``i`` to reads at ``j_indices``.

    Vectorized ``popcount((calls_i XOR calls_j) & valid_i & valid_j) / popcount(valid_i & valid_j)``.
    Returns ``(distances, overlap_counts)``, both length ``len(j_indices)``; ``distances``
    is ``NaN`` wherever ``overlap_counts < min_overlap_positions``.
    """
    calls_i = calls_u64[i : i + 1, :]
    valid_i = valid_u64[i : i + 1, :]
    calls_j = calls_u64[j_indices, :]
    valid_j = valid_u64[j_indices, :]

    joint_valid = np.bitwise_and(valid_i, valid_j)
    overlap_counts = _popcount_u64_matrix(joint_valid).sum(axis=1)

    mismatch_bits = np.bitwise_and(np.bitwise_xor(calls_i, calls_j), joint_valid)
    mismatch_counts = _popcount_u64_matrix(mismatch_bits).sum(axis=1)

    distances = np.full(len(j_indices), np.nan, dtype=float)
    enough_overlap = overlap_counts >= min_overlap_positions
    distances[enough_overlap] = mismatch_counts[enough_overlap] / overlap_counts[enough_overlap]
    return distances, overlap_counts
