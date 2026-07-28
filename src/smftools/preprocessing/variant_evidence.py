"""Pure per-read variant calling and segmentation kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..constants import VARIANT_CALL_SCHEMA_VERSION
from .variant_reference import VariantInformativeSiteCatalog

UNINFORMATIVE_CALL = -1
NO_CALL = 0
TRANSITION_SEGMENT = 3


def classify_variant_bases(
    read_bases: Sequence[object],
    covered: Sequence[bool],
    *,
    first_member_values: Sequence[object],
    second_member_values: Sequence[object],
    no_call_values: Sequence[object] = (),
) -> np.ndarray:
    """Classify encoded or textual bases for one informative substitution."""
    bases = np.asarray(read_bases)
    coverage = np.asarray(covered, dtype=bool)
    if bases.ndim != 1 or coverage.ndim != 1 or bases.shape != coverage.shape:
        raise ValueError("read_bases and covered must be equal-length one-dimensional arrays")
    first = np.asarray(tuple(first_member_values), dtype=bases.dtype)
    second = np.asarray(tuple(second_member_values), dtype=bases.dtype)
    calls = np.zeros(bases.shape, dtype=np.int8)
    calls[np.isin(bases, first) & coverage] = 1
    calls[np.isin(bases, second) & coverage] = 2
    calls[~coverage | np.isin(bases, tuple(no_call_values))] = NO_CALL
    return calls


@dataclass(frozen=True)
class ReadVariantCalls:
    """Dense calls for one read plus explicit evidence counts."""

    calls: np.ndarray
    informative_site_count: int
    callable_site_count: int
    no_call_count: int
    member_call_counts: tuple[int, int]
    schema_version: int = VARIANT_CALL_SCHEMA_VERSION


@dataclass(frozen=True)
class VariantSegmentationResult:
    """Dense segments and distinct transition/other-reference summaries."""

    segments: np.ndarray
    breakpoints: tuple[float | int, ...]
    breakpoint_count: int
    has_breakpoint: bool
    has_other_reference_segment: bool
    other_reference_segment_type: str
    self_base_count: int
    other_base_count: int
    segment_cigar: str
    schema_version: int = VARIANT_CALL_SCHEMA_VERSION


def call_read_variant_sites(
    read_bases: Sequence[str],
    covered: Sequence[bool],
    *,
    aligned_member_index: int,
    catalog: VariantInformativeSiteCatalog,
) -> ReadVariantCalls:
    """Call substitution evidence for one read in its aligned coordinate system."""
    bases = np.asarray(read_bases, dtype=object)
    coverage = np.asarray(covered, dtype=bool)
    if bases.ndim != 1 or coverage.ndim != 1 or bases.shape != coverage.shape:
        raise ValueError("read_bases and covered must be equal-length one-dimensional arrays")
    if aligned_member_index not in (0, 1):
        raise ValueError("aligned_member_index must be 0 or 1")

    calls = np.full(bases.shape, UNINFORMATIVE_CALL, dtype=np.int8)
    no_call_count = 0
    member_counts = [0, 0]
    for site in catalog.informative_sites:
        position = site.member_positions[aligned_member_index]
        if position >= len(calls):
            raise ValueError("informative-site position exceeds read coordinate length")
        call = NO_CALL
        if coverage[position]:
            base = str(bases[position]).upper()
            matches = [
                member_index
                for member_index, accepted in enumerate(site.accepted_bases)
                if base in accepted
            ]
            if len(matches) == 1:
                call = matches[0] + 1
                member_counts[matches[0]] += 1
        if call == NO_CALL:
            no_call_count += 1
        calls[position] = call

    callable_count = sum(member_counts)
    return ReadVariantCalls(
        calls=calls,
        informative_site_count=len(catalog.informative_sites),
        callable_site_count=callable_count,
        no_call_count=no_call_count,
        member_call_counts=(member_counts[0], member_counts[1]),
    )


def _segment_cigar(values: np.ndarray, self_value: int, other_value: int) -> str:
    runs: list[str] = []
    run_symbol: str | None = None
    run_length = 0
    for value in values:
        symbol = "S" if value == self_value else "X" if value == other_value else None
        if symbol is None:
            if run_symbol is not None:
                runs.append(f"{run_length}{run_symbol}")
            run_symbol = None
            run_length = 0
        elif symbol == run_symbol:
            run_length += 1
        else:
            if run_symbol is not None:
                runs.append(f"{run_length}{run_symbol}")
            run_symbol = symbol
            run_length = 1
    if run_symbol is not None:
        runs.append(f"{run_length}{run_symbol}")
    return "".join(runs)


def segment_variant_calls(
    calls: Sequence[int],
    covered: Sequence[bool],
    *,
    aligned_member_index: int,
) -> VariantSegmentationResult:
    """Interpolate calls across a read span using the legacy breakpoint semantics."""
    call_row = np.asarray(calls)
    span_row = np.asarray(covered, dtype=bool)
    if call_row.ndim != 1 or span_row.ndim != 1 or call_row.shape != span_row.shape:
        raise ValueError("calls and covered must be equal-length one-dimensional arrays")
    if aligned_member_index not in (0, 1):
        raise ValueError("aligned_member_index must be 0 or 1")

    segments = np.zeros(call_row.shape, dtype=np.int8)
    covered_positions = np.flatnonzero(span_row)
    breakpoints: list[float | int] = []
    if covered_positions.size:
        span_start = int(covered_positions[0])
        span_end = int(covered_positions[-1])
        positions = np.flatnonzero((call_row == 1) | (call_row == 2))
        positions = positions[(positions >= span_start) & (positions <= span_end)]
        if positions.size:
            previous_position = int(positions[0])
            previous_class = int(call_row[previous_position])
            segments[span_start:previous_position] = previous_class
            for raw_position in positions[1:]:
                position = int(raw_position)
                current_class = int(call_row[position])
                if current_class == previous_class:
                    segments[previous_position:position] = previous_class
                else:
                    segments[previous_position] = previous_class
                    segments[previous_position + 1 : position] = TRANSITION_SEGMENT
                    midpoint = (previous_position + position) / 2.0
                    breakpoints.append(int(midpoint) if midpoint.is_integer() else float(midpoint))
                previous_position = position
                previous_class = current_class
            segments[previous_position : span_end + 1] = previous_class

    self_value = aligned_member_index + 1
    other_value = 2 if self_value == 1 else 1
    in_span = (
        segments[int(covered_positions[0]) : int(covered_positions[-1]) + 1]
        if covered_positions.size
        else segments[:0]
    )
    mismatch_mask = in_span == other_value
    self_count = int(np.sum(in_span == self_value))
    other_count = int(np.sum(mismatch_mask))
    mismatch_type = "no_segment_mismatch"
    if np.any(mismatch_mask):
        starts = np.flatnonzero(mismatch_mask & ~np.r_[False, mismatch_mask[:-1]])
        ends = np.flatnonzero(mismatch_mask & ~np.r_[mismatch_mask[1:], False])
        if len(starts) >= 2:
            mismatch_type = "multi_segment_mismatch"
        elif int(starts[0]) == 0:
            mismatch_type = "left_segment_mismatch"
        elif int(ends[0]) == len(in_span) - 1:
            mismatch_type = "right_segment_mismatch"
        else:
            mismatch_type = "middle_segment_mismatch"

    return VariantSegmentationResult(
        segments=segments,
        breakpoints=tuple(breakpoints),
        breakpoint_count=len(breakpoints),
        has_breakpoint=bool(breakpoints),
        has_other_reference_segment=bool(np.any(mismatch_mask)),
        other_reference_segment_type=mismatch_type,
        self_base_count=self_count,
        other_base_count=other_count,
        segment_cigar=_segment_cigar(in_span, self_value, other_value),
    )
