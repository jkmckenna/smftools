"""Single-segment fast path for the molecule spine (`F24`).

Every long read is its own molecule, so on nanopore data every group has size
one -- and the general path spent ~1,060 us per molecule running group
aggregation over a single row (22.6 minutes for 1.28M molecules on a real run).

The risk in adding a fast path is that it diverges from the general one, or
that it swallows a conflict the general path would have raised. These pin both:
the two paths must agree where they overlap, and every conflict check must
still fire for multi-segment molecules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.constants import BARCODE, REFERENCE_STRAND, SAMPLE
from smftools.informatics.raw_store import (
    MOLECULE_UID_COLUMN,
    TEMPLATE_ID_COLUMN,
    _collapse_segment_obs,
)

pytestmark = pytest.mark.unit


def _segments(specs):
    """specs: list of (molecule_uid, template_id, **overrides)."""
    rows = []
    for index, (uid, template, overrides) in enumerate(specs):
        row = {
            MOLECULE_UID_COLUMN: uid,
            TEMPLATE_ID_COLUMN: template,
            SAMPLE: "s1",
            BARCODE: "barcode01",
            "read_group": "rg1",
            "namespace": "ns",
            REFERENCE_STRAND: "6B6_top",
            "paired": False,
            "proper_pair": True,
            "mate_unmapped": False,
            "reference_start": index * 10,
            "reference_end": index * 10 + 100,
            "ragged_shard": "raw/part-00000.parquet",
            "ragged_row": index,
            "canonical_row": index,
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


# --- the fast path itself -----------------------------------------------------


def test_single_segment_molecules_collapse_correctly():
    frame = _segments([("m1", "r1", {}), ("m2", "r2", {})])

    obs = _collapse_segment_obs(frame)

    assert len(obs) == 2
    assert list(obs["segment_count"]) == [1, 1]
    assert list(obs["pair_state"]) == ["single", "single"]
    assert list(obs["read_id"]) == ["r1", "r2"]
    assert list(obs.index) == ["r1", "r2"]


def test_fragment_bounds_come_from_the_single_segment():
    frame = _segments([("m1", "r1", {"reference_start": 55, "reference_end": 4055})])
    obs = _collapse_segment_obs(frame)
    assert obs.loc["r1", "outer_fragment_start"] == 55
    assert obs.loc["r1", "outer_fragment_end"] == 4055


def test_a_paired_read_alone_is_a_singleton_not_single():
    """One segment of a pair means its mate is absent from this molecule."""
    frame = _segments([("m1", "r1", {"paired": True})])
    assert _collapse_segment_obs(frame).loc["r1", "pair_state"] == "singleton"


def test_shard_pointers_are_preserved_for_single_segments():
    """Only multi-segment molecules lose their shard pointer."""
    frame = _segments([("m1", "r1", {"ragged_row": 7})])
    obs = _collapse_segment_obs(frame)
    assert obs.loc["r1", "group_path"] == "raw/part-00000.parquet"
    assert obs.loc["r1", "group_row"] == 7


# --- the general path must still work ----------------------------------------


def test_multi_segment_molecules_still_collapse():
    frame = _segments(
        [
            ("m1", "r1", {"paired": True, "reference_start": 10, "reference_end": 110}),
            ("m1", "r1", {"paired": True, "reference_start": 500, "reference_end": 600}),
        ]
    )

    obs = _collapse_segment_obs(frame)

    assert len(obs) == 1
    assert obs.loc["r1", "segment_count"] == 2
    assert obs.loc["r1", "pair_state"] == "proper_pair"
    assert obs.loc["r1", "outer_fragment_start"] == 10
    assert obs.loc["r1", "outer_fragment_end"] == 600
    assert obs.loc["r1", "group_row"] == -1, "multi-segment molecules drop the shard pointer"


def test_mixed_input_uses_both_paths_and_keeps_order():
    """A split that reorders molecules would silently reorder the spine."""
    frame = _segments(
        [
            ("m1", "r1", {}),
            ("m2", "r2", {"paired": True}),
            ("m2", "r2", {"paired": True}),
            ("m3", "r3", {}),
        ]
    )

    obs = _collapse_segment_obs(frame)

    assert list(obs["read_id"]) == ["r1", "r2", "r3"]
    assert list(obs["segment_count"]) == [1, 2, 1]


def test_mixed_references_are_marked_on_multi_segment_molecules():
    frame = _segments(
        [
            ("m1", "r1", {"paired": True, REFERENCE_STRAND: "6B6_top"}),
            ("m1", "r1", {"paired": True, REFERENCE_STRAND: "6BALB_cJ_top"}),
        ]
    )
    assert _collapse_segment_obs(frame).loc["r1", REFERENCE_STRAND] == "mixed"


# --- checks that must still fire ---------------------------------------------


@pytest.mark.parametrize("column", [SAMPLE, BARCODE, "read_group", "namespace"])
def test_conflicting_metadata_still_raises_for_multi_segment(column):
    """The fast path may only skip checks that cannot fire on one row."""
    frame = _segments([("m1", "r1", {}), ("m1", "r1", {column: "other"})])
    with pytest.raises(ValueError, match=f"conflicting {column}"):
        _collapse_segment_obs(frame)


def test_one_molecule_mapping_to_two_templates_still_raises():
    frame = _segments([("m1", "r1", {}), ("m1", "r2", {})])
    with pytest.raises(ValueError, match="multiple templates"):
        _collapse_segment_obs(frame)


def test_duplicate_template_identities_still_raise():
    """Two distinct molecules claiming one read id must not pass silently."""
    frame = _segments([("m1", "r1", {}), ("m2", "r1", {})])
    with pytest.raises(ValueError, match="unique"):
        _collapse_segment_obs(frame)


def test_empty_input_returns_empty():
    assert _collapse_segment_obs(pd.DataFrame()).empty


def test_absent_optional_columns_do_not_break_the_fast_path():
    """`paired`/`proper_pair`/`mate_unmapped` are optional in the general path."""
    frame = _segments([("m1", "r1", {})]).drop(columns=["paired", "proper_pair", "mate_unmapped"])
    obs = _collapse_segment_obs(frame)
    assert obs.loc["r1", "pair_state"] == "single"


def test_the_two_paths_agree_on_the_same_molecule():
    """Direct comparison: one molecule, collapsed alone vs beside a paired one."""
    alone = _collapse_segment_obs(_segments([("m1", "r1", {})]))
    beside = _collapse_segment_obs(
        _segments(
            [
                ("m1", "r1", {}),
                ("m2", "r2", {"paired": True}),
                ("m2", "r2", {"paired": True}),
            ]
        )
    ).loc[["r1"]]
    common = [c for c in alone.columns if c in beside.columns]
    pd.testing.assert_frame_equal(alone[common], beside[common], check_like=True)


def test_large_input_stays_fast():
    """Guards the regression this fixes: 1,060 us/molecule was 22.6 min at scale."""
    import time

    n = 60_000
    frame = _segments([(f"m{i}", f"r{i}", {}) for i in range(n)])
    start = time.perf_counter()
    obs = _collapse_segment_obs(frame)
    elapsed = time.perf_counter() - start

    assert len(obs) == n
    assert elapsed < 5.0, f"{elapsed:.1f}s for {n:,} molecules suggests the fast path was lost"
