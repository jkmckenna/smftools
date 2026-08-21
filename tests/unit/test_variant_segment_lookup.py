"""Per-read strand lookup for segment-aware variant acceptance (`EGL-20a`)."""

from __future__ import annotations

import pandas as pd
import pytest

from smftools.preprocessing.partitioned_variant import (
    load_deamination_strand_lookup,
    strand_resolver_for_read,
)

pytestmark = pytest.mark.unit


def _write(tmp_path, rows):
    path = tmp_path / "segments.parquet"
    pd.DataFrame(rows, columns=["read_id", "reference", "start", "end", "strand"]).to_parquet(
        path, index=False
    )
    return path


def test_lookup_groups_spans_by_read():
    resolver = strand_resolver_for_read(((0, 10, "top"), (11, 20, "bottom")))
    assert resolver(5) == "top"
    assert resolver(15) == "bottom"


def test_position_outside_every_segment_is_unresolved():
    """Reads are not segmented end to end; the caller supplies a default."""
    assert strand_resolver_for_read(((0, 10, "top"),))(50) is None


def test_boundaries_are_inclusive():
    resolver = strand_resolver_for_read(((0, 10, "top"),))
    assert resolver(0) == "top" and resolver(10) == "top"


def test_lookup_filters_to_the_requested_reference(tmp_path):
    """A shard's task covers one reference; another's segments must not leak in."""
    path = _write(tmp_path, [("r1", "ref_top", 0, 10, "top"), ("r2", "other_top", 0, 10, "bottom")])
    lookup = load_deamination_strand_lookup(path, "ref_top")
    assert set(lookup) == {"r1"}


def test_missing_file_yields_no_lookup(tmp_path):
    """The deamination lane is optional and can be bypassed (`EGL-25`).

    Absence must degrade to per-read acceptance, not fail.
    """
    assert load_deamination_strand_lookup(tmp_path / "absent.parquet", "ref_top") == {}
    assert load_deamination_strand_lookup(None, "ref_top") == {}


def test_empty_frame_yields_no_lookup(tmp_path):
    assert load_deamination_strand_lookup(_write(tmp_path, []), "ref_top") == {}


def test_spans_are_sorted(tmp_path):
    path = _write(tmp_path, [("r1", "ref_top", 50, 60, "bottom"), ("r1", "ref_top", 0, 10, "top")])
    assert load_deamination_strand_lookup(path, "ref_top")["r1"][0][0] == 0
