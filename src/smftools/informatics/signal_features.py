"""Per-base nanopore current features from the dorado move table.

The dorado ``mv`` BAM tag (emitted with ``--emit-moves``) is ``[stride, m0, m1, ...]``
where each ``mi`` is a 0/1 flag per signal *block* of ``stride`` samples and a ``1``
marks the start of a new basecalled base. The ``ts`` tag is the number of samples
trimmed from the start of the raw signal before basecalling. Composing the move
table with the raw POD5 signal yields, per basecalled base, the signal sample range
that produced it -- and hence per-base current statistics (mean/std/dwell). These
are read-relative (query/basecall order); downstream CIGAR placement scatters them
onto reference positions.
"""

from __future__ import annotations

import numpy as np

# Read-relative signal-feature column names (query/basecall order).
CURRENT_MEAN = "current_mean"
CURRENT_STD = "current_std"
DWELL = "dwell"
SIGNAL_START = "signal_start"
SIGNAL_FEATURE_COLUMNS = (CURRENT_MEAN, CURRENT_STD, DWELL, SIGNAL_START)


def move_table_to_base_ranges(mv, ts: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return per-base ``(start, end)`` signal sample indices in basecall order.

    Args:
        mv: The dorado move table (``mv[0]`` is the stride; ``mv[1:]`` are 0/1
            per-block move flags).
        ts: Samples trimmed from the signal start (the ``ts`` tag).

    Returns:
        Two int arrays (one entry per basecalled base) giving inclusive-start,
        exclusive-end sample indices into the full (untrimmed) POD5 signal.
    """
    mv = np.asarray(mv, dtype=np.int64)
    if mv.size < 1:
        raise ValueError("empty move table")
    stride = int(mv[0])
    if stride <= 0:
        raise ValueError(f"invalid move-table stride: {stride}")
    flags = mv[1:]
    n_bases = int(flags.sum())
    if n_bases == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    base_index = np.cumsum(flags) - 1  # block -> 0-based base index (non-decreasing)
    bases = np.arange(n_bases)
    first_block = np.searchsorted(base_index, bases, side="left")
    end_block = np.searchsorted(base_index, bases, side="right")
    start = int(ts) + first_block.astype(np.int64) * stride
    end = int(ts) + end_block.astype(np.int64) * stride
    return start, end


def base_signal_features(signal, start: np.ndarray, end: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-base current mean/std/dwell/start via prefix sums (vectorized).

    Args:
        signal: 1-D current samples (raw or picoamps) for the whole read.
        start: Per-base inclusive start sample indices.
        end: Per-base exclusive end sample indices.

    Returns:
        Dict with ``current_mean``/``current_std`` (float32), ``dwell`` (samples)
        and ``signal_start`` (clipped start index), one value per base.
    """
    signal = np.asarray(signal, dtype=np.float64)
    size = signal.size
    starts = np.clip(start, 0, size).astype(np.int64)
    ends = np.clip(end, 0, size).astype(np.int64)
    counts = (ends - starts).astype(np.int64)

    prefix = np.concatenate(([0.0], np.cumsum(signal)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(signal * signal)))
    sums = prefix[ends] - prefix[starts]
    sums_sq = prefix_sq[ends] - prefix_sq[starts]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
        var = np.where(counts > 0, sums_sq / counts - mean * mean, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return {
        CURRENT_MEAN: mean.astype(np.float32),
        CURRENT_STD: std.astype(np.float32),
        DWELL: counts.astype(np.float32),
        SIGNAL_START: starts.astype(np.float32),
    }


def base_signal_trace(signal, start: np.ndarray, end: np.ndarray, k: int) -> np.ndarray:
    """Per-base linearly-resampled current trace, ``k`` points per base.

    Unlike ``base_signal_features`` (which reduces each base's sample range
    to a single mean), this preserves within-base current dynamics by
    resampling to a fixed width so bases with different dwell times still
    produce comparable, stackable rows.

    Args:
        signal: 1-D current samples (raw or picoamps) for the whole read.
        start: Per-base inclusive start sample indices.
        end: Per-base exclusive end sample indices.
        k: Number of resampled points per base.

    Returns:
        ``(n_bases, k)`` float32 array; a base with zero samples is all-NaN.
    """
    signal = np.asarray(signal, dtype=np.float64)
    size = signal.size
    starts = np.clip(start, 0, size).astype(np.int64)
    ends = np.clip(end, 0, size).astype(np.int64)
    n_bases = starts.size
    trace = np.full((n_bases, k), np.nan, dtype=np.float32)
    query = np.linspace(0.0, 1.0, k)
    for i in range(n_bases):
        count = int(ends[i] - starts[i])
        if count <= 0:
            continue
        if count == 1:
            trace[i, :] = signal[starts[i]]
            continue
        base_samples = signal[starts[i] : ends[i]]
        sample_positions = np.linspace(0.0, 1.0, count)
        trace[i, :] = np.interp(query, sample_positions, base_samples).astype(np.float32)
    return trace


def read_signal_trace(
    mv,
    ts: int,
    reverse: bool,
    signal,
    k: int,
    *,
    expected_bases: int | None = None,
) -> np.ndarray | None:
    """Per-base resampled current trace for one read, oriented to BAM query order.

    Mirrors ``read_signal_features``: move tables index bases in basecall
    order; for reverse-mapped reads the BAM query is reverse-complemented,
    so the per-base trace rows are flipped (base order only -- each row's
    ``k`` samples are already in forward-time order and are not reversed)
    to match the query coordinate arrays produced by
    ``alignment_to_ragged_record``.

    Args:
        mv: Move table tag.
        ts: Trim offset (``ts`` tag).
        reverse: Whether the read is reverse-mapped.
        signal: Full current sample array for the read.
        k: Number of resampled points per base.
        expected_bases: If given, return ``None`` when the basecalled base
            count does not match (e.g. hard-clipped/supplementary reads).

    Returns:
        ``(n_bases, k)`` float32 array in BAM query order, or ``None`` if
        the move table is unusable or the base count mismatches
        ``expected_bases``.
    """
    start, end = move_table_to_base_ranges(mv, ts)
    n_bases = int(start.size)
    if n_bases == 0:
        return None
    if expected_bases is not None and n_bases != int(expected_bases):
        return None
    trace = base_signal_trace(signal, start, end, k)
    if reverse:
        trace = trace[::-1, :].copy()
    return trace


def read_signal_features(
    mv,
    ts: int,
    reverse: bool,
    signal,
    *,
    expected_bases: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Per-base current features for one read, oriented to BAM query order.

    Move tables index bases in basecall order; for reverse-mapped reads the BAM
    query is reverse-complemented, so features are flipped to match the query
    coordinate arrays produced by ``alignment_to_ragged_record``.

    Args:
        mv: Move table tag.
        ts: Trim offset (``ts`` tag).
        reverse: Whether the read is reverse-mapped.
        signal: Full current sample array for the read.
        expected_bases: If given, return ``None`` when the basecalled base count
            does not match (e.g. hard-clipped/supplementary reads).

    Returns:
        Dict of per-base feature arrays in BAM query order, or ``None`` if the
        move table is unusable or the base count mismatches ``expected_bases``.
    """
    start, end = move_table_to_base_ranges(mv, ts)
    n_bases = int(start.size)
    if n_bases == 0:
        return None
    if expected_bases is not None and n_bases != int(expected_bases):
        return None
    features = base_signal_features(signal, start, end)
    if reverse:
        features = {key: value[::-1].copy() for key, value in features.items()}
    return features
