from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smftools.informatics.pod5_identity import (
    Pod5DatasetIndex,
    build_pod5_dataset_index,
    resolve_pod5_identities,
)

pytestmark = pytest.mark.unit


def _index(**sources):
    return Pod5DatasetIndex({read_id: tuple(source_ids) for read_id, source_ids in sources.items()})


def test_durable_pod5_identity_precedes_legacy_read_names():
    observations = pd.DataFrame(
        {
            "read_id": ["child"],
            "source_read_id": ["legacy"],
            "pod5_read_id": ["parent"],
        }
    )

    result = resolve_pod5_identities(
        observations,
        _index(parent=("source-a",), legacy=("source-a",)),
    )

    assert result.rows[0].pod5_read_id == "parent"
    assert result.rows[0].evidence == "pod5_read_id"


def test_split_children_sharing_one_parent_are_resolved_not_ambiguous():
    observations = pd.DataFrame(
        {
            "read_id": ["child-1", "child-2"],
            "basecall_parent_read_id": ["parent", "parent"],
        }
    )

    result = resolve_pod5_identities(
        observations,
        _index(parent=("source-a",)),
    )

    assert result.resolved_count == 2
    assert result.ambiguous_count == 0
    assert result.unique_pod5_read_count == 1
    assert result.duplicate_parent_reference_count == 1


def test_namespaced_observation_resolves_through_source_read_id():
    observations = pd.DataFrame(
        {
            "read_id": ["ns8:lane-one:parent"],
            "source_read_id": ["parent"],
        }
    )

    result = resolve_pod5_identities(
        observations,
        _index(parent=("lane-one",)),
    )

    assert result.rows[0].status == "resolved"
    assert result.rows[0].evidence == "source_read_id"


def test_duplicate_uuid_across_sources_is_ambiguous():
    observations = pd.DataFrame({"read_id": ["parent"]})

    result = resolve_pod5_identities(
        observations,
        _index(parent=("source-a", "source-b")),
    )

    assert result.ambiguous_count == 1
    assert result.rows[0].evidence == "read_id_duplicate_source"


def test_retained_bam_parent_is_last_resort_and_digest_is_stable():
    observations = pd.DataFrame({"read_id": ["split-child"]})

    first = resolve_pod5_identities(
        observations,
        _index(parent=("source-a",)),
        bam_parent_by_observation={"split-child": "parent"},
    )
    second = resolve_pod5_identities(
        observations,
        _index(parent=("source-a",)),
        bam_parent_by_observation={"split-child": "parent"},
    )

    assert first.rows[0].evidence == "bam_pi"
    assert first.digest == second.digest


def test_checked_in_pod5_fixture_builds_deterministic_uuid_index():
    fixture = Path(__file__).parents[2] / "_test_inputs" / "_test_pod5_I.pod5"

    index = build_pod5_dataset_index((("fixture", fixture),))

    assert index.unique_read_count == 4
    assert index.duplicate_read_id_count == 0
    assert index.sources_for("65592224-e412-4b54-8851-356e73cfd0be") == ("fixture",)
