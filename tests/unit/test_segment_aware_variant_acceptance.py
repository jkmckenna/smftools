"""Variant acceptance follows the local chemistry, not the read (`EGL-20a`).

In a conversion experiment the applicable chemistry is fixed for a whole read by
its strand, which is what `EGL-18` exploits. In deaminase it is *positional*: a
molecule can carry `C->T` over one stretch and `G->A` over another -- that is
precisely what makes it a chimera -- so a single per-read acceptance rule is
wrong by construction.
"""

from __future__ import annotations

import pytest

from smftools.preprocessing.variant_evidence import (
    build_segment_aware_site_index,
    call_observed_variant_sites_by_segment,
)
from smftools.preprocessing.variant_reference import (
    calculate_variant_informative_sites,
    conversion_substitutions_for_strand,
    variant_reference_set_from_legacy,
)

pytestmark = pytest.mark.unit

# Position 3 is C/T (unreadable under C->T) and position 7 is G/A (unreadable
# under G->A), so each strand's chemistry obscures a different site.
SEQ1 = "AAACAAAGAAA"
SEQ2 = "AAATAAAAAAA"


def _catalogs():
    reference_set = variant_reference_set_from_legacy(
        ["r1_top_strand_FASTA_base", "r2_top_strand_FASTA_base"],
        {"r1_top_strand_FASTA_base": SEQ1, "r2_top_strand_FASTA_base": SEQ2},
    )
    return {
        strand: calculate_variant_informative_sites(
            reference_set,
            conversion_substitutions=conversion_substitutions_for_strand(
                "deaminase", ["5mC"], strand
            ),
        )
        for strand in ("top", "bottom")
    }


def _call(strand_at_position, default="top", observed=None):
    index = build_segment_aware_site_index(_catalogs())
    return call_observed_variant_sites_by_segment(
        observed if observed is not None else {3: "C", 7: "G"},
        aligned_member_index=0,
        site_index=index,
        strand_at_position=strand_at_position,
        default_strand=default,
    )


# --- the index ---------------------------------------------------------------


def test_index_carries_sites_from_both_catalogs():
    """Each strand's catalog omits the site its own chemistry obscures."""
    catalogs = _catalogs()
    assert [s.member_positions[0] for s in catalogs["top"].informative_sites] == [7]
    assert [s.member_positions[0] for s in catalogs["bottom"].informative_sites] == [3]
    index = build_segment_aware_site_index(catalogs)
    assert [key[0] for key, _by_strand, _sid in index] == [3, 7]


def test_index_records_which_strands_can_read_each_site():
    index = {
        key[0]: set(by_strand)
        for key, by_strand, _sid in build_segment_aware_site_index(_catalogs())
    }
    assert index[3] == {"bottom"}
    assert index[7] == {"top"}


def test_sites_are_matched_on_position_not_site_id():
    """Ids are assigned by enumeration over *surviving* sites.

    `site-000000` is position 3 in the bottom catalog and position 7 in the top
    one, so matching on ids would silently pair unrelated sites -- producing
    plausible wrong calls rather than an error.
    """
    catalogs = _catalogs()
    top_first = catalogs["top"].informative_sites[0]
    bottom_first = catalogs["bottom"].informative_sites[0]
    assert top_first.site_id == bottom_first.site_id
    assert top_first.member_positions != bottom_first.member_positions


# --- per-position acceptance -------------------------------------------------


def test_uniform_top_chemistry_reads_only_the_ga_site():
    _calls, summary = _call(lambda position: "top")
    assert summary.callable_site_count == 1


def test_uniform_bottom_chemistry_reads_only_the_ct_site():
    _calls, summary = _call(lambda position: "bottom")
    assert summary.callable_site_count == 1


def test_a_chimera_can_read_both_sites():
    """The payoff, and what a per-read rule structurally cannot do.

    When each site falls in the segment whose chemistry does *not* obscure it,
    both are callable. Any single-strand rule gets at most one.
    """
    _calls, summary = _call(lambda position: "bottom" if position < 5 else "top")
    assert summary.callable_site_count == 2


def test_the_opposite_chimera_reads_neither():
    """The mirror case must lose both, or the model is not really positional."""
    _calls, summary = _call(lambda position: "top" if position < 5 else "bottom")
    assert summary.callable_site_count == 0


def test_unreadable_site_is_a_no_call_not_a_guess():
    """Withholding evidence is the conservative direction for chimera input."""
    calls, _summary = _call(lambda position: "top")
    by_position = {call.position: call.call for call in calls}
    assert by_position[3] == 0


def test_default_strand_applies_where_no_segment_covers():
    """Reads are not segmented end to end; uncovered positions need a fallback."""
    _calls, summary = _call(lambda position: None, default="bottom")
    assert summary.callable_site_count == 1


def test_absent_observation_is_a_no_call():
    _calls, summary = _call(lambda position: "bottom", observed={})
    assert summary.callable_site_count == 0
    assert summary.no_call_count == 2


def test_rejects_an_invalid_member_index():
    with pytest.raises(ValueError, match="aligned_member_index"):
        call_observed_variant_sites_by_segment(
            {},
            aligned_member_index=2,
            site_index=build_segment_aware_site_index(_catalogs()),
            strand_at_position=lambda position: "top",
            default_strand="top",
        )
