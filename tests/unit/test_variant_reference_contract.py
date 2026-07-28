from dataclasses import replace

import numpy as np
import pytest

from smftools.config import ExperimentConfig
from smftools.preprocessing.variant_evidence import (
    call_read_variant_sites,
    segment_variant_calls,
)
from smftools.preprocessing.variant_reference import (
    SUBSTITUTIONS_ONLY_POLICY,
    VariantAlignmentScoring,
    VariantReferenceMember,
    VariantReferenceSet,
    calculate_variant_informative_sites,
    normalize_legacy_variant_pair,
    variant_reference_set_from_legacy,
)


def _reference_set(
    first: str = "ACGT",
    second: str = "ATGT",
    **kwargs,
) -> VariantReferenceSet:
    return VariantReferenceSet(
        members=(
            VariantReferenceMember(member_id="refA", sequence=first),
            VariantReferenceMember(member_id="refB", sequence=second),
        ),
        **kwargs,
    )


def test_legacy_pair_normalization_is_strict_and_deterministic() -> None:
    assert normalize_legacy_variant_pair([" refA ", "refB"]) == ("refA", "refB")
    assert normalize_legacy_variant_pair([None, None]) is None
    assert normalize_legacy_variant_pair([]) is None

    with pytest.raises(ValueError, match="exactly two"):
        normalize_legacy_variant_pair(["refA"])
    with pytest.raises(ValueError, match="both members or neither"):
        normalize_legacy_variant_pair(["refA", None])
    with pytest.raises(ValueError, match="distinct"):
        normalize_legacy_variant_pair(["refA", "refA"])


def test_config_normalizes_and_rejects_partial_legacy_pair() -> None:
    config, _ = ExperimentConfig.from_var_dict(
        {"references_to_align_for_variant_annotation": '["refA", "refB"]'},
        defaults_map={},
    )
    assert config.references_to_align_for_variant_annotation == ["refA", "refB"]

    with pytest.raises(ValueError, match="both members or neither"):
        ExperimentConfig.from_var_dict(
            {"references_to_align_for_variant_annotation": ["refA", None]},
            defaults_map={},
        )


def test_legacy_reference_resolution_rejects_missing_and_ambiguous_sources() -> None:
    sources = {
        "refA_top_strand_FASTA_base": ["A", "C", "N", "G"],
        "refB_top_strand_FASTA_base": ["A", "T", "N", "G"],
    }
    reference_set = variant_reference_set_from_legacy(["refA_top", "refB_top"], sources)
    assert reference_set is not None
    assert [member.sequence for member in reference_set.members] == ["ACG", "ATG"]

    with pytest.raises(ValueError, match="missing"):
        variant_reference_set_from_legacy(["refA_top", "missing"], sources)

    ambiguous = {
        **sources,
        "refA_bottom_strand_FASTA_base": ["A", "C", "N", "G"],
    }
    with pytest.raises(ValueError, match="ambiguous"):
        variant_reference_set_from_legacy(["refA", "refB_top"], ambiguous)


def test_reference_set_id_tracks_scientific_semantics_but_not_location() -> None:
    baseline = _reference_set()
    relocated = VariantReferenceSet(
        members=tuple(
            replace(
                member,
                member_id=f"renamed-{index}",
                source_id=f"/relocated/{member.member_id}.fa",
                aliases=("new",),
            )
            for index, member in enumerate(baseline.members)
        )
    )
    assert relocated.reference_set_id == baseline.reference_set_id
    assert _reference_set(first="AGGT").reference_set_id != baseline.reference_set_id
    assert (
        VariantReferenceSet(
            members=(
                replace(baseline.members[0], orientation="reverse_complement"),
                baseline.members[1],
            )
        ).reference_set_id
        != baseline.reference_set_id
    )
    assert (
        replace(baseline, scoring=VariantAlignmentScoring(gap=-3)).reference_set_id
        != baseline.reference_set_id
    )
    assert (
        replace(baseline, conversion_semantics="cytosine_to_thymine").reference_set_id
        != baseline.reference_set_id
    )
    assert (
        replace(baseline, informative_site_policy="future-policy").reference_set_id
        != baseline.reference_set_id
    )


def test_substitutions_are_callable_and_indels_are_explicitly_excluded() -> None:
    substitution_catalog = calculate_variant_informative_sites(_reference_set())
    assert len(substitution_catalog.informative_sites) == 1
    assert substitution_catalog.events[0].event == "substitution"
    assert substitution_catalog.events[0].callable is True

    indel_catalog = calculate_variant_informative_sites(_reference_set(second="ACGGT"))
    indels = [event for event in indel_catalog.events if event.event != "substitution"]
    assert len(indels) == 1
    assert indels[0].callable is False
    assert indels[0].exclusion_reason == "per_read_indel_calling_excluded"


def test_conversion_overlap_collapses_an_otherwise_informative_site() -> None:
    reference_set = VariantReferenceSet(
        members=(
            VariantReferenceMember(
                member_id="refA",
                sequence="ACG",
                accepted_sequences=("ACG", "ATG"),
            ),
            VariantReferenceMember(member_id="refB", sequence="ATG"),
        ),
        conversion_semantics="cytosine_to_thymine",
    )
    catalog = calculate_variant_informative_sites(reference_set)
    assert catalog.informative_sites == ()
    assert catalog.events[0].exclusion_reason == "accepted_base_sets_overlap"


def test_unsupported_policy_has_identity_but_cannot_be_calculated() -> None:
    reference_set = replace(
        _reference_set(),
        informative_site_policy=f"{SUBSTITUTIONS_ONLY_POLICY}-future",
    )
    with pytest.raises(ValueError, match="does not support policy"):
        calculate_variant_informative_sites(reference_set)


def test_pure_read_calls_distinguish_no_call_and_uninformative_positions() -> None:
    catalog = calculate_variant_informative_sites(_reference_set())
    result = call_read_variant_sites(
        ["A", "T", "G", "T"],
        [True, True, True, True],
        aligned_member_index=0,
        catalog=catalog,
    )
    assert result.calls.tolist() == [-1, 2, -1, -1]
    assert result.member_call_counts == (0, 1)
    assert result.callable_site_count == 1
    assert result.no_call_count == 0

    no_call = call_read_variant_sites(
        ["A", "N", "G", "T"],
        [True, False, True, True],
        aligned_member_index=0,
        catalog=catalog,
    )
    assert no_call.calls.tolist() == [-1, 0, -1, -1]
    assert no_call.callable_site_count == 0
    assert no_call.no_call_count == 1


@pytest.mark.parametrize(
    ("calls", "expected_type", "expected_breakpoints"),
    [
        ([-1, -1, -1, -1, -1], "no_segment_mismatch", ()),
        ([2, -1, -1, -1, -1], "left_segment_mismatch", ()),
        ([1, -1, 2, -1, -1], "right_segment_mismatch", (1,)),
        ([1, 2, 1, -1, -1], "middle_segment_mismatch", (0.5, 1.5)),
        ([1, 2, 1, 2, 1], "multi_segment_mismatch", (0.5, 1.5, 2.5, 3.5)),
    ],
)
def test_pure_segmentation_edge_middle_and_multi_segment_cases(
    calls,
    expected_type,
    expected_breakpoints,
) -> None:
    result = segment_variant_calls(calls, np.ones(5, dtype=bool), aligned_member_index=0)
    assert result.other_reference_segment_type == expected_type
    assert result.breakpoints == expected_breakpoints


def test_other_reference_flag_is_distinct_from_transition_flag() -> None:
    broad_only = segment_variant_calls(
        [2, -1, -1, -1],
        [True, True, True, True],
        aligned_member_index=0,
    )
    assert broad_only.has_other_reference_segment is True
    assert broad_only.has_breakpoint is False

    transition = segment_variant_calls(
        [1, -1, -1, 2],
        [True, True, True, True],
        aligned_member_index=0,
    )
    assert transition.has_other_reference_segment is True
    assert transition.has_breakpoint is True
