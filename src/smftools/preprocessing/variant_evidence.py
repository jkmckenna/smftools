"""Pure per-read variant calling and segmentation kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class SparseVariantCall:
    """One informative-site observation in aligned-member coordinates."""

    site_id: str
    position: int
    call: int
    observed_base: str | None


@dataclass(frozen=True)
class SparseVariantSegment:
    """One half-open classified or transition interval."""

    start: int
    end: int
    state: int


@dataclass(frozen=True)
class SparseVariantSegmentationResult:
    """Sparse segmentation without allocating a reference-length dense row."""

    segments: tuple[SparseVariantSegment, ...]
    breakpoints: tuple[float | int, ...]
    has_breakpoint: bool
    has_other_reference_segment: bool
    other_reference_segment_type: str
    self_base_count: int
    other_base_count: int
    segment_cigar: str
    schema_version: int = VARIANT_CALL_SCHEMA_VERSION


def call_observed_variant_sites(
    observed_bases: Mapping[int, str],
    *,
    aligned_member_index: int,
    catalog: VariantInformativeSiteCatalog,
) -> tuple[tuple[SparseVariantCall, ...], ReadVariantCalls]:
    """Call sites from a sparse reference-position-to-base mapping."""
    if aligned_member_index not in (0, 1):
        raise ValueError("aligned_member_index must be 0 or 1")
    calls: list[SparseVariantCall] = []
    member_counts = [0, 0]
    no_call_count = 0
    for site in catalog.informative_sites:
        position = site.member_positions[aligned_member_index]
        observed = observed_bases.get(position)
        call = NO_CALL
        if observed is not None:
            base = str(observed).upper()
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
        calls.append(
            SparseVariantCall(
                site_id=site.site_id,
                position=position,
                call=call,
                observed_base=None if observed is None else str(observed).upper(),
            )
        )
    summary = ReadVariantCalls(
        calls=np.asarray([call.call for call in calls], dtype=np.int8),
        informative_site_count=len(calls),
        callable_site_count=sum(member_counts),
        no_call_count=no_call_count,
        member_call_counts=(member_counts[0], member_counts[1]),
    )
    return tuple(calls), summary


def segment_sparse_variant_calls(
    calls: Sequence[SparseVariantCall],
    *,
    span_start: int,
    span_end: int,
    aligned_member_index: int,
    min_adjacent_sites: int = 1,
) -> SparseVariantSegmentationResult:
    """Segment sparse calls across a half-open aligned read span.

    ``min_adjacent_sites`` is how many *consecutive informative sites* must call
    the other reference before the read is reported as reference-switching; see
    :func:`segment_variant_calls` for why this is counted in sites rather than
    bases and what happens without it (`F14`). This is the path the partitioned
    pipeline uses, so it is the one that produced the pilot's 100%-chimeric
    result.

    Counts and segments stay raw; only the interpretation is gated.
    """
    if span_start < 0 or span_end < span_start:
        raise ValueError("read span must be a valid half-open interval")
    if aligned_member_index not in (0, 1):
        raise ValueError("aligned_member_index must be 0 or 1")
    informative = sorted(
        (
            (int(call.position), int(call.call))
            for call in calls
            if call.call in (1, 2) and span_start <= call.position < span_end
        ),
        key=lambda value: value[0],
    )
    segments: list[SparseVariantSegment] = []
    breakpoints: list[float | int] = []
    if informative:
        current_start = span_start
        previous_position, previous_class = informative[0]
        for position, current_class in informative[1:]:
            if current_class != previous_class:
                segments.append(
                    SparseVariantSegment(current_start, previous_position + 1, previous_class)
                )
                if previous_position + 1 < position:
                    segments.append(
                        SparseVariantSegment(
                            previous_position + 1,
                            position,
                            TRANSITION_SEGMENT,
                        )
                    )
                midpoint = (previous_position + position) / 2.0
                breakpoints.append(int(midpoint) if midpoint.is_integer() else float(midpoint))
                current_start = position
            previous_position = position
            previous_class = current_class
        segments.append(SparseVariantSegment(current_start, span_end, previous_class))

    self_value = aligned_member_index + 1
    other_value = 2 if self_value == 1 else 1
    self_count = sum(
        segment.end - segment.start for segment in segments if segment.state == self_value
    )
    other_segments = [segment for segment in segments if segment.state == other_value]
    other_count = sum(segment.end - segment.start for segment in other_segments)

    # Site support per segment. Segments are emitted one per run of
    # same-class informative sites, in order, so the k-th non-transition
    # segment is backed by the k-th run -- which makes the run length the
    # segment's evidence count.
    minimum_sites = max(1, int(min_adjacent_sites))
    run_lengths: list[int] = []
    if informative:
        run_class = informative[0][1]
        run_length = 1
        for _position, current_class in informative[1:]:
            if current_class == run_class:
                run_length += 1
            else:
                run_lengths.append(run_length)
                run_class = current_class
                run_length = 1
        run_lengths.append(run_length)
    called_segments = [segment for segment in segments if segment.state in (1, 2)]
    supported_other = [
        segment
        for segment, support in zip(called_segments, run_lengths)
        if segment.state == other_value and support >= minimum_sites
    ]

    mismatch_type = "no_segment_mismatch"
    if supported_other:
        if len(supported_other) >= 2:
            mismatch_type = "multi_segment_mismatch"
        elif supported_other[0].start == span_start:
            mismatch_type = "left_segment_mismatch"
        elif supported_other[0].end == span_end:
            mismatch_type = "right_segment_mismatch"
        else:
            mismatch_type = "middle_segment_mismatch"
    cigar = "".join(
        f"{segment.end - segment.start}{'S' if segment.state == self_value else 'X'}"
        for segment in segments
        if segment.state in (self_value, other_value)
    )
    return SparseVariantSegmentationResult(
        segments=tuple(segments),
        breakpoints=tuple(breakpoints),
        has_breakpoint=bool(breakpoints),
        has_other_reference_segment=bool(supported_other),
        other_reference_segment_type=mismatch_type,
        self_base_count=self_count,
        other_base_count=other_count,
        segment_cigar=cigar,
    )


def build_segment_aware_site_index(
    catalogs_by_strand: Mapping[str, "VariantInformativeSiteCatalog"],
) -> tuple[tuple[tuple[int, int], dict[str, tuple[frozenset[str], frozenset[str]]], str], ...]:
    """Index candidate sites across per-strand catalogs, keyed by position.

    `EGL-18` builds one catalog per strand, and a site ambiguous under one
    chemistry is simply absent from that catalog. To call a read whose chemistry
    *varies along its length*, both catalogs must be consulted per position, so
    the sites have to be matched across them.

    They are matched on ``member_positions``, not ``site_id``: ids are assigned
    by enumeration over the sites that survive
    (``site-{len(informative_sites):06d}``), so the same id denotes different
    positions in catalogs that excluded different sites. Matching on ids would
    silently pair unrelated sites.

    Returns one entry per candidate position pair: its per-strand accepted-base
    sets (absent where that chemistry makes it unreadable) and a stable site id.
    """
    merged: dict[tuple[int, int], dict[str, tuple[frozenset[str], frozenset[str]]]] = {}
    for strand, catalog in catalogs_by_strand.items():
        for site in catalog.informative_sites:
            key = (int(site.member_positions[0]), int(site.member_positions[1]))
            merged.setdefault(key, {})[str(strand)] = site.accepted_bases
    return tuple(
        (key, by_strand, f"site-{index:06d}")
        for index, (key, by_strand) in enumerate(sorted(merged.items()))
    )


def call_observed_variant_sites_by_segment(
    observed_bases: Mapping[int, str],
    *,
    aligned_member_index: int,
    site_index: Sequence[tuple[tuple[int, int], Mapping[str, Any], str]],
    strand_at_position: Callable[[int], str | None],
    default_strand: str,
) -> tuple[tuple[SparseVariantCall, ...], ReadVariantCalls]:
    """Call sites using the chemistry local to each position (`EGL-20a`).

    In a conversion experiment the applicable chemistry is fixed for a whole
    read by its strand, which is what `EGL-18` exploits. In deaminase it is
    *positional*: a molecule can carry `C->T` over one stretch and `G->A` over
    another -- that is what makes it a chimera -- so a single per-read
    acceptance rule is wrong by construction.

    ``strand_at_position`` resolves the deamination segment covering a position;
    ``default_strand`` applies where no segment covers it. A site the local
    chemistry makes unreadable is a no-call rather than a guess, which is the
    conservative direction: it withholds evidence instead of inventing an
    allele.
    """
    if aligned_member_index not in (0, 1):
        raise ValueError("aligned_member_index must be 0 or 1")
    calls: list[SparseVariantCall] = []
    member_counts = [0, 0]
    no_call_count = 0
    for positions, by_strand, site_id in site_index:
        position = int(positions[aligned_member_index])
        observed = observed_bases.get(position)
        strand = strand_at_position(position) or default_strand
        accepted = by_strand.get(str(strand))
        call = NO_CALL
        if observed is not None and accepted is not None:
            base = str(observed).upper()
            matches = [
                member_index for member_index, allowed in enumerate(accepted) if base in allowed
            ]
            if len(matches) == 1:
                call = matches[0] + 1
                member_counts[matches[0]] += 1
        if call == NO_CALL:
            no_call_count += 1
        calls.append(
            SparseVariantCall(
                site_id=site_id,
                position=position,
                call=call,
                observed_base=None if observed is None else str(observed).upper(),
            )
        )
    summary = ReadVariantCalls(
        calls=np.asarray([call.call for call in calls], dtype=np.int8),
        informative_site_count=len(calls),
        callable_site_count=sum(member_counts),
        no_call_count=no_call_count,
        member_call_counts=(member_counts[0], member_counts[1]),
    )
    return tuple(calls), summary


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
    min_adjacent_sites: int = 1,
) -> VariantSegmentationResult:
    """Interpolate calls across a read span using the legacy breakpoint semantics.

    Args:
        calls: Per-position variant call classes (1 or 2 at informative sites).
        covered: Per-position read-span mask.
        aligned_member_index: 0 or 1 -- which reference this read aligned to.
        min_adjacent_sites: How many *consecutive informative sites* must call
            the other reference before the read is reported as
            reference-switching. Sites, not bases: variant sites are sparse, so
            the run is counted over the informative-site sequence rather than
            over interpolated genomic positions.

            The default of 1 preserves the historical behavior, in which a
            single discordant site was enough. That is what made the flag
            useless on real data: on the `241213` pilot it called 100% of
            QC-passing reads chimeric on the strength of 2-3 discordant bases
            out of ~2,300, because one isolated base opens a segment of the
            other reference (`F14`). Callers that care should pass the
            configured value; `ExperimentConfig.variant_chimera_min_adjacent
            _sites` defaults to 2.

            Counts (``self_base_count``, ``other_base_count``) and the segment
            layer are deliberately left raw. This gates the *interpretation* --
            whether the read is called chimeric and how the mismatch is typed --
            so a read with a couple of stray discordant bases still shows them
            in the plotted segments while being typed ``no_segment_mismatch``.
    """
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

    # Which stretches of other-reference segment are actually *supported*.
    # Support is counted in informative sites, so a long interpolated stretch
    # resting on one discordant site does not qualify while a short stretch
    # between two adjacent discordant sites does.
    minimum_sites = max(1, int(min_adjacent_sites))
    supported_mask = np.zeros_like(mismatch_mask)
    if covered_positions.size and np.any(mismatch_mask):
        span_offset = int(covered_positions[0])
        site_positions = np.flatnonzero((call_row == 1) | (call_row == 2))
        site_positions = site_positions[
            (site_positions >= span_offset) & (site_positions <= int(covered_positions[-1]))
        ]
        other_sites = site_positions[call_row[site_positions] == other_value] - span_offset
        # Keep or drop each contiguous stretch *whole*. Trimming a stretch to
        # its supporting sites would silently re-type it -- a segment running
        # to the end of the span would stop looking like a right-edge segment
        # -- so support decides membership only, never geometry.
        stretch_starts = np.flatnonzero(mismatch_mask & ~np.r_[False, mismatch_mask[:-1]])
        stretch_ends = np.flatnonzero(mismatch_mask & ~np.r_[mismatch_mask[1:], False])
        for low, high in zip(stretch_starts, stretch_ends):
            # Sites inside one stretch are consecutive informative sites all
            # calling the other reference, so this count is the run length.
            support = int(np.sum((other_sites >= low) & (other_sites <= high)))
            if support >= minimum_sites:
                supported_mask[low : high + 1] = True

    mismatch_type = "no_segment_mismatch"
    if np.any(supported_mask):
        starts = np.flatnonzero(supported_mask & ~np.r_[False, supported_mask[:-1]])
        ends = np.flatnonzero(supported_mask & ~np.r_[supported_mask[1:], False])
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
        has_other_reference_segment=bool(np.any(supported_mask)),
        other_reference_segment_type=mismatch_type,
        self_base_count=self_count,
        other_base_count=other_count,
        segment_cigar=_segment_cigar(in_span, self_value, other_value),
    )
