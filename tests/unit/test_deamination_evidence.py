"""Per-molecule deamination strand state (`EGL-19`).

A deaminase library deaminates cytosines on whichever strand the molecule came
from: `C->T` in top-reference coordinates, `G->A` for a bottom-derived molecule.
One template carries one direction, so well-supported stretches of both mean a
PCR chimera.

The tests below pin the three decisions that make that call work on real data,
each of which was reached by getting it wrong first:

1. Every C and G position is evidence, not only the converted ones. Counting
   only events over-called chimeras ~300x on the `251105` pilot (5,965 against
   the 18 the scalar method finds), because 3 stray `G->A` among 651 retained
   Gs formed a "run".
2. Conversion efficiency is estimated per read. It *is* the footprint signal, so
   a fixed value reads protection as a strand switch.
3. The number of change points is inferred from a penalty, not assumed. Long
   reads may carry several joins.
"""

from __future__ import annotations

import pytest

from smftools.preprocessing.deamination_evidence import (
    deamination_substitutions,
    estimate_conversion_rates,
    observe_read_deamination,
    segment_deamination,
)

pytestmark = pytest.mark.unit

# Alternating C/G so every position is substrate for one strand or the other.
REFERENCE = "CG" * 120


def _observe(read: str, *, excluded=()):
    subs = deamination_substitutions("deaminase", ["5mC"])
    bases = {index: base for index, base in enumerate(read)}
    return observe_read_deamination(REFERENCE, bases, subs, excluded_positions=excluded)


def _top(n: int, offset: int = 0, *, efficiency: float = 1.0) -> str:
    """A top-derived stretch: Cs convert, Gs retained.

    ``offset`` is where the stretch sits in the read, and must match, because
    observations are classified against the reference base at that position.
    """
    stride = max(1, int(round(1 / efficiency))) if efficiency > 0 else 1
    out = []
    for index in range(offset, offset + n):
        if REFERENCE[index] == "C":
            out.append("T" if (index % stride) == 0 else "C")
        else:
            out.append("G")
    return "".join(out)


def _bottom(n: int, offset: int = 0) -> str:
    """A bottom-derived stretch: Gs convert, Cs retained."""
    return "".join("A" if REFERENCE[offset + i] == "G" else "C" for i in range(n))


# --- which chemistries apply -------------------------------------------------


def test_both_strand_directions_are_sought():
    """Restricting to the read's own strand would make chimeras undetectable."""
    subs = deamination_substitutions("deaminase", ["5mC"])
    assert {(s.strand, s.source_base, s.converted_base) for s in subs} == {
        ("top", "C", "T"),
        ("bottom", "G", "A"),
    }


def test_6ma_generalizes():
    """States come from the conversion map, not hard-coded C and G."""
    subs = deamination_substitutions("deaminase", ["6mA"])
    assert {(s.strand, s.source_base) for s in subs} == {("top", "A"), ("bottom", "T")}


def test_direct_modality_yields_nothing():
    assert deamination_substitutions("direct", ["5mC"]) == ()


# --- observations ------------------------------------------------------------


def test_retained_positions_are_recorded_as_evidence():
    """The defect that caused the 300x over-call: these were being discarded."""
    observations = _observe(_top(20))
    assert any(not item.converted for item in observations)
    assert {item.strand for item in observations} == {"top", "bottom"}


def test_third_base_is_not_evidence():
    """A mismatch unrelated to the chemistry must not be guessed at."""
    read = "".join("A" if REFERENCE[i] == "C" else "T" for i in range(20))
    assert _observe(read) == ()


def test_excluded_positions_are_dropped():
    """Variant sites must not masquerade as deamination events (`EGL-20a`)."""
    full = _observe(_top(20))
    trimmed = _observe(_top(20), excluded=[0, 1, 2, 3])
    assert len(trimmed) == len(full) - 4


# --- rate estimation ---------------------------------------------------------


def test_efficiency_is_estimated_per_read():
    observations = _observe(_top(40))
    efficiency, error = estimate_conversion_rates(observations)
    assert efficiency > error


def test_error_never_exceeds_efficiency():
    """A degenerate read must not invert the model."""
    efficiency, error = estimate_conversion_rates(_observe(_top(6)))
    assert error < efficiency


# --- segmentation ------------------------------------------------------------


def test_pure_read_is_one_segment_and_not_chimeric():
    segments, summary = segment_deamination(_observe(_top(60)))
    assert summary.segment_count == 1
    assert summary.is_chimeric is False
    assert summary.strands_present == ("top",)


def test_stray_opposite_events_do_not_make_a_chimera():
    """The exact pattern that produced 5,965 false positives.

    A handful of `G->A` among many retained Gs is error, and the retained
    positions say so.
    """
    read = list(_top(60))
    for index in range(1, 8, 2):  # a few G positions flipped to A
        read[index] = "A"
    segments, summary = segment_deamination(_observe("".join(read)))
    assert summary.is_chimeric is False


def test_genuine_switch_is_detected_and_located():
    """Half top, half bottom -- the defining case."""
    read = _top(60) + _bottom(60, offset=60)
    segments, summary = segment_deamination(_observe(read))
    assert summary.is_chimeric is True
    assert summary.strands_present in (("top", "bottom"), ("bottom", "top"))
    assert len(summary.switch_positions) == 1
    assert 40 < summary.switch_positions[0] < 80


def test_number_of_switches_is_inferred_not_assumed():
    """Three joins must yield three switches, which a two-segment model cannot."""
    read = _top(40) + _bottom(40, 40) + _top(40, 80) + _bottom(40, 120)
    _segments, summary = segment_deamination(_observe(read))
    assert len(summary.switch_positions) >= 2


def test_penalty_controls_sensitivity():
    """The knob is evidence-per-change-point, not a segment count."""
    observations = _observe(_top(40) + _bottom(40, offset=40))
    strict = segment_deamination(observations, penalty_scale=1e6)[1]
    assert strict.segment_count == 1


def test_same_strand_neighbours_are_merged():
    """A rate shift is a footprint, not a strand switch.

    Only strand changes are chimera evidence, so splits that keep the same
    strand must not inflate the segment count.
    """
    read = _top(30) + _top(30, 30, efficiency=0.34)
    _segments, summary = segment_deamination(_observe(read))
    assert summary.strands_present == ("top",)
    assert summary.is_chimeric is False


def test_empty_input_is_handled():
    segments, summary = segment_deamination(())
    assert segments == () and summary.is_chimeric is False and summary.n_observations == 0


def test_summary_carries_the_evidence_for_redecision():
    """Efficiency, error and switch positions are stored so a different
    threshold never needs a recompute."""
    _segments, summary = segment_deamination(_observe(_top(60)))
    assert summary.n_observations > 0
    assert 0.0 < summary.efficiency <= 1.0
    assert 0.0 < summary.error_rate < summary.efficiency
