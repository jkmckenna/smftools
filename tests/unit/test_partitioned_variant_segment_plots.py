"""Variant segment clustermaps for the partitioned pipeline.

The renderer that produces these has existed all along; only the pre-partition
`variant` CLI ever called it, because it reads dense layers and the partitioned
store keeps variant evidence sparse. These tests pin the rasterization that
bridges the two, which is where the picture can silently stop matching the
data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.partitioned_variant_plots import build_variant_segment_layers

pytestmark = pytest.mark.unit


def _segments(rows):
    return pd.DataFrame(rows, columns=["read_id", "start", "end", "state"])


def test_segments_rasterize_onto_the_position_grid():
    positions = np.arange(100, 110, dtype=np.int64)
    segments = _segments([("r1", 100, 105, 1), ("r1", 105, 107, 3), ("r1", 107, 110, 2)])
    calls = pd.DataFrame({"read_id": ["r1"], "position": [104], "call": [1]})

    layer, call_layer, span = build_variant_segment_layers(["r1"], positions, segments, calls)

    assert list(layer[0]) == [1, 1, 1, 1, 1, 3, 3, 2, 2, 2]
    assert call_layer[0, 4] == 1
    assert span[0].all(), "every position a segment covers must count as spanned"


def test_uncovered_positions_stay_no_coverage():
    """A read shorter than the window must not imply data it does not have."""
    positions = np.arange(100, 110, dtype=np.int64)
    segments = _segments([("r1", 103, 106, 1)])

    layer, _calls, span = build_variant_segment_layers(
        ["r1"], positions, segments, pd.DataFrame(columns=["read_id", "position", "call"])
    )

    assert list(layer[0]) == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert list(span[0]) == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_rows_follow_the_requested_read_order():
    """Row order is the caller's, not the frame's.

    The obs frame, the segment frame, and the call frame arrive independently
    sorted; if rasterization followed any one of them the heatmap rows would
    stop corresponding to the annotation strip beside them.
    """
    positions = np.arange(0, 4, dtype=np.int64)
    segments = _segments([("r2", 0, 4, 2), ("r1", 0, 4, 1)])

    layer, _calls, _span = build_variant_segment_layers(
        ["r1", "r2"], positions, segments, pd.DataFrame(columns=["read_id", "position", "call"])
    )

    assert list(layer[0]) == [1, 1, 1, 1]
    assert list(layer[1]) == [2, 2, 2, 2]


def test_unknown_reads_are_ignored():
    """Evidence for reads outside the panel must not land on someone else's row."""
    positions = np.arange(0, 4, dtype=np.int64)
    segments = _segments([("r1", 0, 4, 1), ("ghost", 0, 4, 2)])

    layer, _calls, _span = build_variant_segment_layers(
        ["r1"], positions, segments, pd.DataFrame(columns=["read_id", "position", "call"])
    )

    assert layer.shape == (1, 4)
    assert list(layer[0]) == [1, 1, 1, 1]


def test_no_call_sites_are_not_drawn():
    """Only calls of 1 or 2 are variant calls; 0 means no call."""
    positions = np.arange(0, 4, dtype=np.int64)
    calls = pd.DataFrame({"read_id": ["r1", "r1"], "position": [1, 2], "call": [0, 2]})

    _layer, call_layer, _span = build_variant_segment_layers(
        ["r1"], positions, _segments([("r1", 0, 4, 1)]), calls
    )

    assert call_layer[0, 1] == 0
    assert call_layer[0, 2] == 2


def test_calls_outside_the_window_are_dropped():
    positions = np.arange(10, 14, dtype=np.int64)
    calls = pd.DataFrame({"read_id": ["r1"], "position": [99], "call": [1]})

    _layer, call_layer, _span = build_variant_segment_layers(
        ["r1"], positions, _segments([("r1", 10, 14, 1)]), calls
    )

    assert not call_layer.any()


def test_empty_evidence_yields_an_empty_panel():
    positions = np.arange(0, 4, dtype=np.int64)
    empty_segments = pd.DataFrame(columns=["read_id", "start", "end", "state"])
    empty_calls = pd.DataFrame(columns=["read_id", "position", "call"])

    layer, call_layer, span = build_variant_segment_layers(
        ["r1"], positions, empty_segments, empty_calls
    )

    assert not layer.any() and not call_layer.any() and not span.any()
