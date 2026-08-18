"""Conversion acceptance follows the read's strand (`F16`, `EGL-18`).

A conversion chemistry makes some reference differences unreadable: under
`C->T`, a `C` on one reference and a `T` on the other are indistinguishable
once the C converts, so calling a variant there measures methylation rather
than genotype. On the `241213` pilot those sites miscalled at up to 70.7%.

Which chemistry applies is chosen by the *reference-strand assignment* -- top
converts `C->T`, bottom `G->A` -- so a site lost to one strand stays usable for
the other. That asymmetry is the whole point of doing this per strand instead of
excluding both classes everywhere.
"""

from __future__ import annotations

import pytest

from smftools.preprocessing.partitioned_variant import strand_of_reference
from smftools.preprocessing.variant_reference import (
    calculate_variant_informative_sites,
    conversion_substitutions_for_strand,
    variant_reference_set_from_legacy,
)

pytestmark = pytest.mark.unit

# Position 3 is C/T (ambiguous under C->T), position 7 is G/A (ambiguous under
# G->A), position 11 is A/T (never ambiguous under either).
SEQ1 = "AAACAAAGAAATAA"
SEQ2 = "AAATAAAAAAAAAA"


def _reference_set():
    return variant_reference_set_from_legacy(
        ["ref1_top_strand_FASTA_base", "ref2_top_strand_FASTA_base"],
        {
            "ref1_top_strand_FASTA_base": SEQ1,
            "ref2_top_strand_FASTA_base": SEQ2,
        },
    )


def _pairs(strand, modality="conversion", conversion_types=("5mC",)):
    substitutions = conversion_substitutions_for_strand(modality, conversion_types, strand)
    catalog = calculate_variant_informative_sites(
        _reference_set(), conversion_substitutions=substitutions
    )
    return {tuple(site.member_bases) for site in catalog.informative_sites}


@pytest.mark.parametrize(
    ("modality", "modification", "strand", "expected"),
    [
        ("conversion", "5mC", "top", (("C", "T"),)),
        ("conversion", "5mC", "bottom", (("G", "A"),)),
        ("deaminase", "5mC", "top", (("C", "T"),)),
        ("conversion", "6mA", "top", (("A", "G"),)),
        ("conversion", "6mA", "bottom", (("T", "C"),)),
    ],
)
def test_substitution_lookup(modality, modification, strand, expected):
    assert conversion_substitutions_for_strand(modality, [modification], strand) == expected


def test_direct_modality_has_no_conversion():
    """Direct SMF has no conversion chemistry; nothing may be excluded for it."""
    assert conversion_substitutions_for_strand("direct", ["5mC"], "top") == ()


def test_unknown_strand_yields_canonical_acceptance():
    """An unrecognised strand must not silently guess a chemistry."""
    assert conversion_substitutions_for_strand("conversion", ["5mC"], "") == ()


def test_unmapped_modification_is_ignored():
    """`conversion_types` really does carry entries like 'unconverted'."""
    assert conversion_substitutions_for_strand("conversion", ["unconverted"], "top") == ()
    assert conversion_substitutions_for_strand("conversion", ["unconverted", "5mC"], "top") == (
        ("C", "T"),
    )


def test_top_strand_drops_ct_and_keeps_ga():
    pairs = _pairs("top")
    assert ("C", "T") not in pairs, "C/T is unreadable under C->T and must be excluded"
    assert ("G", "A") in pairs, "G/A is unaffected by C->T and must survive"


def test_bottom_strand_drops_ga_and_keeps_ct():
    """The mirror case: the asymmetry is the reason for per-strand catalogs."""
    pairs = _pairs("bottom")
    assert ("G", "A") not in pairs
    assert ("C", "T") in pairs


def test_unambiguous_sites_survive_both_strands():
    """Guard against over-correcting into excluding everything."""
    for strand in ("top", "bottom"):
        assert ("T", "A") in _pairs(strand)


def test_no_conversion_keeps_every_site():
    """Pins the previous behavior, which is what `direct` still gets."""
    pairs = _pairs("top", modality="direct")
    assert {("C", "T"), ("G", "A"), ("T", "A")} <= pairs


def test_reference_set_id_is_stable_across_strands():
    """The design invariant that keeps task grouping intact.

    Widening acceptance must not look like a different reference set: task
    planning groups on `reference_set_id`, so two strands over one pair of
    references have to share it.
    """
    reference_set = _reference_set()
    top = calculate_variant_informative_sites(reference_set, conversion_substitutions=(("C", "T"),))
    bottom = calculate_variant_informative_sites(
        reference_set, conversion_substitutions=(("G", "A"),)
    )
    assert top.reference_set_id == bottom.reference_set_id


def test_catalog_id_differs_across_strands():
    """...but the catalogs themselves must not collide as one cached identity."""
    reference_set = _reference_set()
    top = calculate_variant_informative_sites(
        reference_set, conversion_substitutions=(("C", "T"),), conversion_semantics="5mC:top"
    )
    bottom = calculate_variant_informative_sites(
        reference_set, conversion_substitutions=(("G", "A"),), conversion_semantics="5mC:bottom"
    )
    assert top.catalog_id != bottom.catalog_id
    assert top.to_dict()["conversion_semantics"] == "5mC:top"


def test_excluded_sites_are_recorded_with_a_reason():
    """An excluded site must be explicable, not just absent."""
    catalog = calculate_variant_informative_sites(
        _reference_set(), conversion_substitutions=(("C", "T"),)
    )
    reasons = {event.exclusion_reason for event in catalog.events if not event.callable}
    assert "accepted_base_sets_overlap" in reasons


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("6B6_top", "top"),
        ("6BALB_cJ_bottom", "bottom"),
        ("ctcf_mNanog_top", "top"),
        ("weird_name", ""),
    ],
)
def test_strand_of_reference(reference, expected):
    assert strand_of_reference(reference) == expected
