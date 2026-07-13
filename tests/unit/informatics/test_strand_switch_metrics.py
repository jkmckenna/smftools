from __future__ import annotations

from smftools.informatics.ragged_store import (
    alignment_to_ragged_record,
    strand_switch_metrics,
)

# Reference with interleaved C and G blocks so C->T (top) and G->A (bottom)
# deamination events can be placed at known positions.
#   index:  0 1 2 3 4 5 6 7 8 9 10 11
#   base :  C C C G G G C C C G G  G
REFERENCE = "CCCGGGCCCGGG"


class _Read:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    is_reverse = False
    reference_name = "chr1"
    reference_start = 0
    cigarstring = "12M"
    query_qualities = list(range(20, 32))

    def __init__(self, query_sequence: str, name: str = "read"):
        self.query_sequence = query_sequence
        self.query_name = name


def _record(query_sequence: str) -> dict:
    return alignment_to_ragged_record(_Read(query_sequence), REFERENCE)


def test_strand_switch_metrics_pure_and_switch():
    # pure single strand -> high purity, no opposite-majority split
    assert strand_switch_metrics([1, 1, 1, 1]) == (1.0, -1)
    assert strand_switch_metrics([-1, -1, -1]) == (1.0, -1)
    # clean chimera -> full purity, split at the sign boundary
    assert strand_switch_metrics([1, 1, 1, -1, -1, -1]) == (1.0, 3)
    # randomly interspersed -> lower purity, below the labeling threshold
    purity, _ = strand_switch_metrics([1, -1, 1, -1])
    assert purity < 0.9
    # trivial inputs
    assert strand_switch_metrics([]) == (1.0, -1)
    assert strand_switch_metrics([1]) == (1.0, -1)


def test_pure_c_to_t_read_has_no_switch():
    # every reference C deaminated to T; reference G positions unchanged
    record = _record("TTTGGGTTTGGG")
    assert record["ct_event_count"] == 6
    assert record["ga_event_count"] == 0
    assert record["strand_segment_purity"] == 1.0
    assert record["strand_switch_position"] == -1
    assert record["Read_mismatch_trend"] == "C->T"


def test_pure_g_to_a_read_has_no_switch():
    record = _record("CCCAAACCCAAA")
    assert record["ct_event_count"] == 0
    assert record["ga_event_count"] == 6
    assert record["strand_segment_purity"] == 1.0
    assert record["strand_switch_position"] == -1
    assert record["Read_mismatch_trend"] == "G->A"


def test_clean_chimera_read_flags_switch():
    # first block C->T (positions 0,1,2), last block G->A (positions 9,10,11)
    record = _record("TTTGGGCCCAAA")
    assert record["ct_event_count"] == 3
    assert record["ga_event_count"] == 3
    assert record["strand_segment_purity"] == 1.0
    # boundary midpoint between last C->T (pos 2) and first G->A (pos 9)
    assert record["strand_switch_position"] == 5
    assert record["Read_mismatch_trend"] == "equal"


def test_noisy_read_has_low_segment_purity():
    # interspersed C->T and G->A events -> not a clean two-span switch
    record = _record("TCCAGGTCCAGG")
    assert record["ct_event_count"] == 2
    assert record["ga_event_count"] == 2
    assert record["strand_segment_purity"] < 0.9
