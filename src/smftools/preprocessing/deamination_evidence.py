"""Per-molecule deamination strand state via penalized change-point detection.

A deaminase library deaminates cytosines on whichever strand the molecule came
from. In top-reference coordinates that reads as `C->T`; a bottom-strand
molecule's cytosines sit opposite reference Gs and read as `G->A`. One template
carries one direction, so a molecule containing well-supported stretches of both
is a PCR chimera.

**Every C and G position is evidence, not only the converted ones.** A retained
G on a top-strand molecule is expected -- G is not substrate there -- and
directly contradicts bottom. Ignoring retained positions was the defect in the
first version of this module: on the `251105` pilot, molecules it called
chimeric carried a mean of 32 `G->A` events against **651 retained Gs**, and
counting only the 32 over-called chimeras by roughly 300x (5,965 against the
18 the existing scalar method finds). The G-position conversion rate in those
molecules is 0.036 versus 0.317 at C positions -- error, not chemistry.

The evidence is asymmetric, and one class deliberately carries less weight:

| reference | read | evidence |
|---|---|---|
| C | T | strong, top |
| G | A | strong, bottom |
| G | G | moderate, top -- on bottom it should have converted |
| C | C | weak, bottom -- on top this is also what a footprint looks like |

That last row is the only class where the assay's biological signal and the
strand signal are confounded, which is why conversion efficiency is estimated
per read rather than fixed: a globally-fixed efficiency reads footprints as
strand switches.

Segmentation is penalized binary segmentation, so the **number** of switches is
inferred rather than assumed. Long reads may carry several joins; a two-segment
model cannot express that. The penalty -- not a segment count -- says how much
evidence a further change point must supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..constants import CONVERSION_BASE_SUBSTITUTIONS

DEAMINATION_SCHEMA_VERSION = 2

# Conversion efficiency and error are estimated per read, then clamped. The
# bounds keep a read with almost no substrate, or one that is entirely
# footprinted, from producing a degenerate likelihood that makes every split
# look decisive.
MIN_EFFICIENCY, MAX_EFFICIENCY = 0.05, 0.99
MIN_ERROR, MAX_ERROR = 0.001, 0.10


@dataclass(frozen=True)
class DeaminationSubstitution:
    """One chemistry a read could carry, e.g. 5mC on the top strand."""

    modification: str
    strand: str
    source_base: str
    converted_base: str


@dataclass(frozen=True)
class DeaminationObservation:
    """One informative position: which strand it is substrate for, and its state."""

    position: int
    strand: str
    converted: bool


@dataclass(frozen=True)
class DeaminationSegment:
    """A stretch assigned to one strand chemistry."""

    start: int
    end: int
    strand: str
    n_observations: int
    n_converted: int


@dataclass(frozen=True)
class DeaminationSummary:
    """Per-molecule rollup, including the chimera call and its evidence."""

    n_observations: int
    efficiency: float
    error_rate: float
    segment_count: int
    strands_present: tuple[str, ...]
    dominant_strand: str | None
    switch_positions: tuple[int, ...]
    is_chimeric: bool
    schema_version: int = DEAMINATION_SCHEMA_VERSION


def deamination_substitutions(
    modality: str | None,
    conversion_types: Sequence[str] | None,
) -> tuple[DeaminationSubstitution, ...]:
    """Every chemistry a read in this experiment could carry, both strands.

    Both directions are always sought: finding both in one molecule is the
    signal. `direct` has no deamination chemistry and yields nothing, so callers
    gate the lane on an empty result rather than testing modality separately.
    """
    normalized = str(modality or "").strip().lower()
    if normalized not in {"conversion", "deaminase"}:
        return ()
    found: list[DeaminationSubstitution] = []
    for modification in conversion_types or ():
        for strand in ("top", "bottom"):
            pair = CONVERSION_BASE_SUBSTITUTIONS.get((str(modification).strip(), strand))
            if pair is None:
                continue
            candidate = DeaminationSubstitution(
                modification=str(modification).strip(),
                strand=strand,
                source_base=pair[0],
                converted_base=pair[1],
            )
            if candidate not in found:
                found.append(candidate)
    return tuple(found)


def observe_read_deamination(
    reference_sequence: str,
    observed_bases: Mapping[int, str],
    substitutions: Sequence[DeaminationSubstitution],
    *,
    excluded_positions: Iterable[int] = (),
) -> tuple[DeaminationObservation, ...]:
    """Every deamination-informative position on one read, converted or not.

    A position is informative when the *reference* base is substrate for one of
    the applicable chemistries. Retained positions are kept -- they are the bulk
    of the evidence. A third base is not evidence either way and is skipped
    rather than guessed at.

    ``excluded_positions`` drops known variant informative sites, where a
    genuine reference difference is indistinguishable from a deamination event
    (`EGL-20a`).
    """
    if not substitutions:
        return ()
    excluded = {int(position) for position in excluded_positions}
    by_source: dict[str, DeaminationSubstitution] = {}
    for substitution in substitutions:
        by_source.setdefault(substitution.source_base, substitution)

    reference = str(reference_sequence).upper()
    observations: list[DeaminationObservation] = []
    for position in sorted(observed_bases):
        if position in excluded or position < 0 or position >= len(reference):
            continue
        substitution = by_source.get(reference[position])
        if substitution is None:
            continue
        observed = str(observed_bases[position]).upper()
        if observed == substitution.converted_base:
            observations.append(DeaminationObservation(position, substitution.strand, True))
        elif observed == substitution.source_base:
            observations.append(DeaminationObservation(position, substitution.strand, False))
    return tuple(observations)


def estimate_conversion_rates(
    observations: Sequence[DeaminationObservation],
) -> tuple[float, float]:
    """Estimate (efficiency, error) for one read from its own observations.

    Efficiency is the conversion rate at the more-converted strand's substrate
    positions; error is the rate at the other's. Per read rather than global
    because efficiency is the footprint signal itself -- it varies by molecule
    and by region, and a fixed value would read protection as a strand switch.

    On a chimeric read the minority direction inflates the error estimate
    slightly. That is tolerable: it makes the model *more* conservative about
    declaring a switch, and the clamps bound how far it can drift.
    """
    rates: dict[str, tuple[int, int]] = {}
    for observation in observations:
        converted, total = rates.get(observation.strand, (0, 0))
        rates[observation.strand] = (converted + int(observation.converted), total + 1)
    fractions = [
        (converted / total, strand) for strand, (converted, total) in rates.items() if total
    ]
    if not fractions:
        return MIN_EFFICIENCY, MIN_ERROR
    fractions.sort(reverse=True)
    efficiency = fractions[0][0]
    error = fractions[-1][0] if len(fractions) > 1 else MIN_ERROR
    efficiency = min(MAX_EFFICIENCY, max(MIN_EFFICIENCY, efficiency))
    error = min(MAX_ERROR, max(MIN_ERROR, error))
    if error >= efficiency:
        error = max(MIN_ERROR, min(error, efficiency / 2))
    return efficiency, error


class _CostModel:
    """O(1) segment cost from prefix sums.

    The cost of a stretch under one strand is a sum of four log-likelihood
    terms -- substrate-converted, substrate-retained, other-converted,
    other-retained -- so it depends on the stretch only through those four
    counts. Prefix sums make every candidate split O(1) instead of O(n),
    which is the difference between O(n log n) and O(n^2) overall. With ~1,283
    informative positions per read across 28,302 reads, the naive form is
    ~46 billion operations; this is the same restructuring
    `ragged_store.strand_switch_metrics` already uses for its vote track.
    """

    __slots__ = ("strands", "prefix", "log_rates")

    def __init__(
        self,
        observations: Sequence[DeaminationObservation],
        efficiency: float,
        error_rate: float,
    ) -> None:
        self.strands = ("top", "bottom")
        # prefix[strand][i] = (converted, total) over observations[:i]
        self.prefix = {strand: [(0, 0)] * (len(observations) + 1) for strand in self.strands}
        for strand in self.strands:
            converted = total = 0
            column = self.prefix[strand]
            for index, observation in enumerate(observations):
                if observation.strand == strand:
                    total += 1
                    converted += int(observation.converted)
                column[index + 1] = (converted, total)
        self.log_rates = {
            True: (math.log(max(efficiency, 1e-12)), math.log(max(1.0 - efficiency, 1e-12))),
            False: (math.log(max(error_rate, 1e-12)), math.log(max(1.0 - error_rate, 1e-12))),
        }

    def cost(self, low: int, high: int) -> tuple[float, str | None]:
        """Negative log-likelihood of observations[low:high] under its best strand."""
        if high <= low:
            return 0.0, None
        best_cost = math.inf
        best_strand = None
        for candidate in self.strands:
            total = 0.0
            for strand in self.strands:
                converted_hi, total_hi = self.prefix[strand][high]
                converted_lo, total_lo = self.prefix[strand][low]
                converted = converted_hi - converted_lo
                retained = (total_hi - total_lo) - converted
                log_converted, log_retained = self.log_rates[strand == candidate]
                total -= converted * log_converted + retained * log_retained
            if total < best_cost:
                best_cost = total
                best_strand = candidate
        return best_cost, best_strand


def segment_deamination(
    observations: Sequence[DeaminationObservation],
    *,
    penalty_scale: float = 3.0,
    min_segment_size: int = 3,
    rates: tuple[float, float] | None = None,
) -> tuple[tuple[DeaminationSegment, ...], DeaminationSummary]:
    """Split a read into strand segments by penalized binary segmentation.

    The number of change points is *inferred*, not supplied: a split is accepted
    only when it reduces the negative log-likelihood by more than
    ``penalty_scale * log(n)``, a BIC-style cost per added change point. Long
    reads may carry several joins, which a fixed two-segment model cannot
    express, and the penalty rather than a segment count decides how many are
    justified.
    """
    if not observations:
        empty = DeaminationSummary(0, MIN_EFFICIENCY, MIN_ERROR, 0, (), None, (), False)
        return (), empty
    efficiency, error_rate = rates if rates is not None else estimate_conversion_rates(observations)
    ordered = sorted(observations, key=lambda item: item.position)
    penalty = float(penalty_scale) * math.log(max(len(ordered), 2))
    minimum = max(1, int(min_segment_size))
    model = _CostModel(ordered, efficiency, error_rate)

    # Optimal partitioning with PELT pruning, not greedy binary segmentation.
    # Binary segmentation is greedy and fails on alternating patterns: for a
    # top|bottom|top read no *single* split improves the fit, so it stops after
    # one change point and reports one switch where there are two. Measured on
    # a constructed four-block read, its best sub-window gain was 1.15e-14.
    # Optimal partitioning minimises the total cost over *all* partitions, so
    # alternating structure is found; PELT's pruning keeps it near-linear
    # instead of the O(n^2) that would be unusable at ~1,283 observations per
    # read across tens of thousands of reads.
    n = len(ordered)
    best_cost = [0.0] * (n + 1)
    best_cost[0] = -penalty
    previous = [0] * (n + 1)
    candidates = [0]
    for end_index in range(minimum, n + 1):
        best_value = math.inf
        best_start = 0
        # Segment cost is needed twice per candidate -- once to score it, once
        # to prune it -- so compute it once and keep it. Recomputing was costing
        # more than the pruning saved.
        scored: list[tuple[int, float]] = []
        # Start points still too close to `end_index` to form a segment are held
        # back unscored: they become valid as `end_index` advances, and dropping
        # them here silently removes the correct answer for later positions.
        too_short: list[int] = []
        for start_index in candidates:
            if end_index - start_index < minimum:
                too_short.append(start_index)
                continue
            segment_cost = model.cost(start_index, end_index)[0]
            scored.append((start_index, segment_cost))
            value = best_cost[start_index] + segment_cost + penalty
            if value < best_value:
                best_value = value
                best_start = start_index
        if not scored:
            best_cost[end_index] = math.inf
            candidates = [*too_short, end_index]
            continue
        best_cost[end_index] = best_value
        previous[end_index] = best_start
        # PELT pruning: a start point that cannot reach the current optimum even
        # with a zero-cost continuation can never be optimal later.
        candidates = [
            start_index
            for start_index, segment_cost in scored
            if best_cost[start_index] + segment_cost <= best_value
        ]
        candidates.extend(too_short)
        candidates.append(end_index)

    boundaries = []
    position = n
    while position > 0:
        start_index = previous[position]
        if start_index > 0:
            boundaries.append(start_index)
        position = start_index
    boundaries.reverse()

    cuts = [0, *sorted(boundaries), len(ordered)]

    segments: list[DeaminationSegment] = []
    for low, high in zip(cuts, cuts[1:]):
        window = ordered[low:high]
        if not window:
            continue
        _cost, strand = model.cost(low, high)
        segments.append(
            DeaminationSegment(
                start=window[0].position,
                end=window[-1].position,
                strand=str(strand),
                n_observations=len(window),
                n_converted=sum(1 for item in window if item.converted),
            )
        )

    # Merge neighbours the split assigned to the same strand: a change point can
    # be justified by a rate shift (a footprint) without being a strand switch,
    # and only strand changes are chimera evidence.
    merged: list[DeaminationSegment] = []
    for segment in segments:
        if merged and merged[-1].strand == segment.strand:
            previous = merged.pop()
            merged.append(
                DeaminationSegment(
                    start=previous.start,
                    end=segment.end,
                    strand=segment.strand,
                    n_observations=previous.n_observations + segment.n_observations,
                    n_converted=previous.n_converted + segment.n_converted,
                )
            )
        else:
            merged.append(segment)

    strands_present = tuple(dict.fromkeys(segment.strand for segment in merged))
    dominant = None
    if merged:
        totals: dict[str, int] = {}
        for segment in merged:
            totals[segment.strand] = totals.get(segment.strand, 0) + segment.n_observations
        dominant = max(totals, key=lambda strand: (totals[strand], strand))
    switches = tuple(
        (earlier.end + later.start) // 2
        for earlier, later in zip(merged, merged[1:])
        if earlier.strand != later.strand
    )
    summary = DeaminationSummary(
        n_observations=len(ordered),
        efficiency=efficiency,
        error_rate=error_rate,
        segment_count=len(merged),
        strands_present=strands_present,
        dominant_strand=dominant,
        switch_positions=switches,
        is_chimeric=len(strands_present) > 1,
    )
    return tuple(merged), summary
