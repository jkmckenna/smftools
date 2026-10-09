"""Refreshing duplicate keepers after demux calls change."""

import pandas as pd
import pytest

from smftools.preprocessing.demux_refresh import recompute_duplicate_keepers

pytestmark = pytest.mark.unit


def _obs(demux):
    return pd.DataFrame(
        {
            "read_id": ["a", "b", "c", "d", "e"],
            "duplicate_cluster_id": [0, 0, 0, 1, -1],
            "duplicate_cluster_size": [3, 3, 3, 1, 1],
            "demux_type": demux,
            "read_quality": [30.0, 20.0, 10.0, 25.0, 15.0],
            "passes_qc": [True, True, True, True, False],
        }
    ).set_index("read_id", drop=False)


def test_blind_keeper_is_the_best_metric():
    flags = recompute_duplicate_keepers(
        _obs(["unclassified"] * 5), preferred_demux={"double"}, metric="read_quality"
    )
    assert flags["passes_dedup"].tolist() == [True, False, False, True, False]
    assert flags["is_duplicate_reason"].tolist() == [
        "",
        "sequence_cluster",
        "sequence_cluster",
        "",
        "",
    ]


def test_a_double_member_becomes_the_keeper():
    flags = recompute_duplicate_keepers(
        _obs(["single", "single", "double", "double", "single"]),
        preferred_demux={"double"},
        metric="read_quality",
    )
    # Cluster 0 keeps its only double read although its quality is lowest;
    # singletons and failed-QC reads are untouched.
    assert flags["passes_dedup"].tolist() == [False, False, True, True, False]
    assert flags["is_duplicate"].tolist() == [True, True, False, False, False]


def test_without_a_preferred_member_the_metric_decides():
    flags = recompute_duplicate_keepers(
        _obs(["single", "single", "single", "double", "single"]),
        preferred_demux={"double"},
        metric="read_quality",
    )
    assert flags["passes_dedup"].tolist()[:3] == [True, False, False]
