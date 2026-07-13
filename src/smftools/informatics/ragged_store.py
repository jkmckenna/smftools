"""Read-relative molecule storage and reference-grid materialization.

Ragged records keep per-base arrays in query coordinates.  CIGAR placement is
performed only when a dense reference-grid slice is requested, avoiding the
mandatory padded matrices produced by the legacy load path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from smftools.constants import (
    BASE_QUALITY_SCORES,
    MISMATCH_INTEGER_ENCODING,
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    READ_SPAN_MASK,
    REFERENCE_STRAND,
    SEQUENCE_INTEGER_ENCODING,
)

from .signal_features import SIGNAL_FEATURE_COLUMNS

if TYPE_CHECKING:
    import anndata as ad

READ_ID = "read_id"
REFERENCE = "reference"
REFERENCE_START = "reference_start"
CIGAR = "cigar"
ALIGNED_LENGTH = "aligned_length"
SEQUENCE = "sequence"
QUALITY = "quality"
MISMATCH = "mismatch"
MODIFICATION_SIGNAL = "modification_signal"

RAGGED_REQUIRED_COLUMNS = (
    READ_ID,
    REFERENCE,
    REFERENCE_STRAND,
    REFERENCE_START,
    CIGAR,
    ALIGNED_LENGTH,
)
RAGGED_ARRAY_COLUMNS = (SEQUENCE, QUALITY, MISMATCH, MODIFICATION_SIGNAL) + SIGNAL_FEATURE_COLUMNS

# Read-relative signal-feature column -> dense reference-grid layer name. These are
# only present when raw captured the move table (dorado --emit-moves) + POD5 signal.
# All are float32 with a NaN default (unobserved positions), like X.
SIGNAL_FEATURE_LAYERS = {
    "current_mean": "current_mean",
    "current_std": "current_std",
    "dwell": "current_dwell",
    "signal_start": "current_signal_start",
}

_CIGAR_TOKEN = re.compile(r"(\d+)([MIDNSHP=X])")
_QUERY_CONSUMING = frozenset({"M", "I", "S", "=", "X"})
_REFERENCE_CONSUMING = frozenset({"M", "D", "N", "=", "X"})
_ALIGNED = frozenset({"M", "=", "X"})


def parse_cigar(cigar: str) -> tuple[tuple[int, str], ...]:
    """Parse a SAM CIGAR string and reject malformed or unsupported input."""
    if not isinstance(cigar, str) or not cigar or cigar == "*":
        raise ValueError(f"invalid CIGAR: {cigar!r}")
    tokens = tuple((int(length), op) for length, op in _CIGAR_TOKEN.findall(cigar))
    if not tokens or "".join(f"{length}{op}" for length, op in tokens) != cigar:
        raise ValueError(f"invalid CIGAR: {cigar!r}")
    if any(length <= 0 for length, _op in tokens):
        raise ValueError(f"invalid CIGAR operation length in {cigar!r}")
    return tokens


def cigar_query_length(cigar: str) -> int:
    """Return the number of query bases consumed by *cigar*."""
    return sum(length for length, op in parse_cigar(cigar) if op in _QUERY_CONSUMING)


def cigar_reference_length(cigar: str) -> int:
    """Return the number of reference positions consumed by *cigar*."""
    return sum(length for length, op in parse_cigar(cigar) if op in _REFERENCE_CONSUMING)


def iter_cigar_aligned_pairs(cigar: str, reference_start: int) -> Iterator[tuple[int, int]]:
    """Yield ``(query_position, reference_position)`` for aligned CIGAR bases."""
    query_position = 0
    reference_position = int(reference_start)
    for length, op in parse_cigar(cigar):
        if op in _ALIGNED:
            for offset in range(length):
                yield query_position + offset, reference_position + offset
        if op in _QUERY_CONSUMING:
            query_position += length
        if op in _REFERENCE_CONSUMING:
            reference_position += length


def _as_list(value: object) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"ragged array values must be list-like, got {type(value).__name__}")


def validate_ragged_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a ragged molecule DataFrame.

    Returns a copy so callers never have their input DataFrame mutated.
    """
    missing = [column for column in RAGGED_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"ragged frame missing required columns: {missing}")
    if frame[READ_ID].isna().any() or frame[READ_ID].astype(str).duplicated().any():
        raise ValueError("ragged frame read_id values must be non-null and unique")

    result = frame.copy()
    result[READ_ID] = result[READ_ID].astype(str)
    result[REFERENCE] = result[REFERENCE].astype(str)
    result[REFERENCE_STRAND] = result[REFERENCE_STRAND].astype(str)
    result[REFERENCE_START] = result[REFERENCE_START].astype(np.int64)
    result[ALIGNED_LENGTH] = result[ALIGNED_LENGTH].astype(np.int64)
    if (result[REFERENCE_START] < 0).any() or (result[ALIGNED_LENGTH] < 0).any():
        raise ValueError("reference_start and aligned_length must be non-negative")

    normalized_arrays: dict[str, list[list]] = {
        column: [] for column in RAGGED_ARRAY_COLUMNS if column in result.columns
    }
    for row_index, row in result.iterrows():
        cigar = str(row[CIGAR])
        expected_query_length = cigar_query_length(cigar)
        expected_aligned_length = cigar_reference_length(cigar)
        if int(row[ALIGNED_LENGTH]) != expected_aligned_length:
            raise ValueError(
                f"row {row_index!r} aligned_length does not match CIGAR reference span"
            )
        for column in RAGGED_ARRAY_COLUMNS:
            if column not in result.columns:
                continue
            values = _as_list(row[column])
            if values and len(values) != expected_query_length:
                raise ValueError(
                    f"row {row_index!r} column {column!r} has {len(values)} values; "
                    f"expected {expected_query_length} from CIGAR"
                )
            normalized_arrays[column].append(values)
    for column, values in normalized_arrays.items():
        result[column] = pd.Series(values, index=result.index, dtype=object)
    return result


def write_ragged_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write one shard of read-relative molecule records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_ragged_frame(frame).to_parquet(path, index=False)
    return path


def read_ragged_parquet(
    paths: str | Path | Sequence[str | Path],
    *,
    read_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read ragged parquet shard(s), optionally retaining selected read IDs."""
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not path_list:
        raise ValueError("at least one ragged parquet path is required")
    selected = None if read_ids is None else {str(read_id) for read_id in read_ids}
    frames: list[pd.DataFrame] = []
    for path in path_list:
        frame = pd.read_parquet(path)
        if selected is not None:
            frame = frame.loc[frame[READ_ID].astype(str).isin(selected)]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return validate_ragged_frame(result)


def alignment_to_ragged_record(
    read,
    reference_sequence: str,
    *,
    reference: str | None = None,
    reference_strand: str | None = None,
) -> dict[str, object]:
    """Convert a pysam-like aligned segment to one read-relative record."""
    if read.is_unmapped:
        raise ValueError("cannot extract a ragged record from an unmapped read")
    cigar = read.cigarstring
    query_sequence = (read.query_sequence or "").upper()
    if len(query_sequence) != cigar_query_length(cigar):
        raise ValueError("query sequence length does not match CIGAR query span")

    unknown = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]
    sequence = [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT.get(base, unknown) for base in query_sequence]
    qualities = list(read.query_qualities or [])
    if not qualities:
        qualities = [-1] * len(query_sequence)
    if len(qualities) != len(query_sequence):
        raise ValueError("query quality length does not match query sequence length")

    mismatch = [unknown] * len(query_sequence)
    c_to_t = 0
    g_to_a = 0
    reference_sequence = reference_sequence.upper()
    for query_position, reference_position in iter_cigar_aligned_pairs(cigar, read.reference_start):
        if reference_position >= len(reference_sequence):
            continue
        query_base = query_sequence[query_position]
        reference_base = reference_sequence[reference_position]
        if query_base != reference_base and query_base != "N" and reference_base != "N":
            mismatch[query_position] = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT.get(query_base, unknown)
            c_to_t += int(reference_base == "C" and query_base == "T")
            g_to_a += int(reference_base == "G" and query_base == "A")

    if c_to_t == g_to_a and c_to_t > 0:
        mismatch_trend = "equal"
    elif c_to_t > g_to_a:
        mismatch_trend = "C->T"
    elif g_to_a > c_to_t:
        mismatch_trend = "G->A"
    else:
        mismatch_trend = "none"

    reference_name = reference or str(read.reference_name)
    strand = "bottom" if read.is_reverse else "top"
    return {
        READ_ID: str(read.query_name),
        REFERENCE: reference_name,
        REFERENCE_STRAND: reference_strand or f"{reference_name}_{strand}",
        "strand": strand,
        "mapping_direction": "rev" if read.is_reverse else "fwd",
        "Read_mismatch_trend": mismatch_trend,
        REFERENCE_START: int(read.reference_start),
        CIGAR: cigar,
        ALIGNED_LENGTH: cigar_reference_length(cigar),
        SEQUENCE: sequence,
        QUALITY: qualities,
        MISMATCH: mismatch,
    }


def _reference_lengths_for_rows(
    frame: pd.DataFrame, reference_lengths: Mapping[str, int]
) -> tuple[dict[str, int], int]:
    lengths: dict[str, int] = {}
    for reference_strand in frame[REFERENCE_STRAND].unique():
        key = str(reference_strand)
        if key not in reference_lengths:
            raise KeyError(f"missing reference length for {key!r}")
        lengths[key] = int(reference_lengths[key])
    return lengths, max(lengths.values())


def materialize_ragged(
    frame: pd.DataFrame,
    *,
    obs: pd.DataFrame,
    reference_lengths: Mapping[str, int],
    layers: Iterable[str] | None = None,
    uns: Mapping[str, object] | None = None,
    start: int | None = None,
    end: int | None = None,
) -> "ad.AnnData":
    """Scatter selected read-relative records onto a dense reference grid."""
    import anndata as ad

    frame = validate_ragged_frame(frame).set_index(READ_ID, drop=False)
    read_ids = list(map(str, obs.index))
    missing = [read_id for read_id in read_ids if read_id not in frame.index]
    if missing:
        raise KeyError(f"ragged store lacks {len(missing)} selected read(s), e.g. {missing[0]!r}")
    frame = frame.loc[read_ids]
    lengths, full_n_positions = _reference_lengths_for_rows(frame, reference_lengths)
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    window_start = 0 if start is None else int(start)
    window_end = full_n_positions if end is None else int(end)
    if window_start < 0 or window_end <= window_start:
        raise ValueError("materialization interval must satisfy 0 <= start < end")
    if end is not None:
        if len(lengths) != 1:
            raise ValueError("interval materialization requires exactly one reference")
        reference_length = next(iter(lengths.values()))
        if window_end > reference_length:
            raise ValueError(
                f"materialization interval ends at {window_end}, beyond reference length "
                f"{reference_length}"
            )
    n_positions = window_end - window_start
    n_reads = len(read_ids)
    unknown = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]
    padding = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"]

    signal = np.full((n_reads, n_positions), np.nan, dtype=np.float32)
    sequence = np.full((n_reads, n_positions), unknown, dtype=np.int8)
    mismatch = np.full((n_reads, n_positions), unknown, dtype=np.int8)
    quality = np.full((n_reads, n_positions), -1, dtype=np.int8)
    span = np.zeros((n_reads, n_positions), dtype=np.int8)

    # Optional current signal-feature layers (only when raw captured move tables).
    requested = None if layers is None else set(layers)
    signal_feature_layers: dict[str, np.ndarray] = {}
    for column, layer_name in SIGNAL_FEATURE_LAYERS.items():
        if column not in frame.columns:
            continue
        if requested is not None and layer_name not in requested:
            continue
        if not frame[column].notna().any():
            continue
        signal_feature_layers[layer_name] = np.full(
            (n_reads, n_positions), np.nan, dtype=np.float32
        )

    for row_number, (_read_id, row) in enumerate(frame.iterrows()):
        ref_length = lengths[str(row[REFERENCE_STRAND])]
        if end is None and ref_length < full_n_positions:
            sequence[row_number, ref_length:] = padding
            mismatch[row_number, ref_length:] = padding
        read_start = int(row[REFERENCE_START])
        read_end = min(read_start + int(row[ALIGNED_LENGTH]), ref_length)
        span_start = max(read_start, window_start)
        span_end = min(read_end, window_end)
        if span_start < span_end:
            span[row_number, span_start - window_start : span_end - window_start] = 1

        arrays = {
            MODIFICATION_SIGNAL: _as_list(row.get(MODIFICATION_SIGNAL)),
            SEQUENCE: _as_list(row.get(SEQUENCE)),
            MISMATCH: _as_list(row.get(MISMATCH)),
            QUALITY: _as_list(row.get(QUALITY)),
        }
        feature_lists = {
            layer_name: _as_list(row.get(column))
            for column, layer_name in SIGNAL_FEATURE_LAYERS.items()
            if layer_name in signal_feature_layers
        }
        for query_position, reference_position in iter_cigar_aligned_pairs(
            str(row[CIGAR]), read_start
        ):
            if (
                reference_position < window_start
                or reference_position >= window_end
                or reference_position >= ref_length
            ):
                continue
            target_position = reference_position - window_start
            if arrays[MODIFICATION_SIGNAL]:
                signal[row_number, target_position] = arrays[MODIFICATION_SIGNAL][query_position]
            if arrays[SEQUENCE]:
                sequence[row_number, target_position] = arrays[SEQUENCE][query_position]
            if arrays[MISMATCH]:
                mismatch[row_number, target_position] = arrays[MISMATCH][query_position]
            if arrays[QUALITY]:
                quality[row_number, target_position] = arrays[QUALITY][query_position]
            for layer_name, values in feature_lists.items():
                if values:
                    signal_feature_layers[layer_name][row_number, target_position] = values[
                        query_position
                    ]

    available_layers = {
        SEQUENCE_INTEGER_ENCODING: sequence,
        MISMATCH_INTEGER_ENCODING: mismatch,
        BASE_QUALITY_SCORES: quality,
        READ_SPAN_MASK: span,
        **signal_feature_layers,
    }
    keep = set(available_layers) if layers is None else set(layers)
    unknown_layers = keep.difference(available_layers)
    if unknown_layers:
        raise KeyError(f"ragged store cannot materialize layers: {sorted(unknown_layers)}")
    result = ad.AnnData(
        X=signal,
        obs=obs.copy(),
        layers={key: value for key, value in available_layers.items() if key in keep},
    )
    result.var_names = [str(position) for position in range(window_start, window_end)]
    if uns:
        result.uns.update(dict(uns))
    return result
