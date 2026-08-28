"""Basecall generation vs. current signal shape classification (`BCS-11`)."""

from __future__ import annotations

import pytest

from smftools.informatics.basecall_mismatch import (
    BasecallSourceShape,
    classify_basecall_source_shape,
    source_identity_ids,
)

pytestmark = pytest.mark.unit


def _ids(*source_ids: str) -> list[str]:
    return ["input-manifest:some-digest", *(f"source:{sid}:deadbeef" for sid in source_ids)]


def test_source_identity_ids_drops_the_manifest_digest_entry() -> None:
    assert source_identity_ids(_ids("a", "b")) == frozenset({"a", "b"})


def test_source_identity_ids_ignores_non_source_entries() -> None:
    entries = ["input-manifest:digest", "source:a:sha", "alignment-reference-bundle:digest"]

    assert source_identity_ids(entries) == frozenset({"a"})


def test_identical_source_sets_classify_as_identical() -> None:
    shape = classify_basecall_source_shape(_ids("a", "b"), _ids("a", "b"))

    assert shape is BasecallSourceShape.IDENTICAL


def test_a_pod5_removed_after_basecalling_classifies_as_signal_pruned() -> None:
    recorded = _ids("a", "b", "c")
    current = _ids("a", "b")  # c no longer present

    assert classify_basecall_source_shape(recorded, current) is BasecallSourceShape.SIGNAL_PRUNED


def test_a_new_pod5_appearing_classifies_as_signal_expanded() -> None:
    recorded = _ids("a", "b")
    current = _ids("a", "b", "c")  # c is new since basecalling

    assert classify_basecall_source_shape(recorded, current) is BasecallSourceShape.SIGNAL_EXPANDED


def test_non_overlapping_source_sets_classify_as_disjoint() -> None:
    recorded = _ids("a", "b")
    current = _ids("c", "d")

    assert classify_basecall_source_shape(recorded, current) is BasecallSourceShape.DISJOINT


def test_partial_overlap_with_both_added_and_removed_classifies_as_disjoint() -> None:
    recorded = _ids("a", "b")
    current = _ids("b", "c")  # a gone, c new -- neither a pure subset nor superset

    assert classify_basecall_source_shape(recorded, current) is BasecallSourceShape.DISJOINT


def test_empty_recorded_and_current_are_identical() -> None:
    assert classify_basecall_source_shape([], []) is BasecallSourceShape.IDENTICAL
