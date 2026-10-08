"""Per-molecule motif occupancy rules (`MOT-04`)."""

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.motif_occupancy import (
    STATES,
    Instance,
    OccupancyRules,
    classify,
    group_occupancy,
    resolve_instances,
)

pytestmark = pytest.mark.unit
WIDTH = 20  # columns
CORE = Instance(0, slice(8, 12), slice(5, 15))  # a 4-column motif, flank 3


def _reads(rows: list[dict]):
    """Build class and site arrays from per-read descriptions."""
    n = len(rows)
    classes = {
        state: (np.zeros((n, WIDTH), np.float32), np.ones((n, WIDTH), bool))
        for state in ("tf_bound", "medium_bound", "nucleosome", "accessible")
    }
    sites = np.zeros((n, WIDTH), bool)
    for r, row in enumerate(rows):
        for state, columns in row.get("classes", {}).items():
            classes[state][0][r, columns] = 1
        sites[r, row.get("sites", [6, 9, 13])] = True
        for state in classes:
            classes[state][1][r, row.get("unobserved", [])] = False
    return classes, sites


def _state(rows, rules=OccupancyRules(flank=3, min_sites=2, min_cover=0.5)):
    classes, sites = _reads(rows)
    return [STATES[code] for code in classify(classes, sites, [CORE], rules)[:, 0]]


def test_each_state_from_the_class_covering_the_motif():
    rows = [
        {"classes": {"tf_bound": range(6, 14)}},
        {"classes": {"medium_bound": range(0, 20)}},
        {"classes": {"nucleosome": range(0, 20)}},
        {"classes": {"accessible": range(8, 12)}},
        {"classes": {"tf_bound": [8], "accessible": [11]}},  # neither reaches min_cover
    ]
    assert _state(rows) == ["tf_bound", "medium_bound", "nucleosome", "accessible", "other"]


def test_majority_wins_and_ties_follow_precedence():
    rows = [
        {"classes": {"tf_bound": [8], "accessible": [9, 10, 11]}},  # 3 of 4 accessible
        {"classes": {"tf_bound": [8, 9], "accessible": [10, 11]}},  # tie -> tf_bound
    ]
    assert _state(rows) == ["accessible", "tf_bound"]


def test_uninformative_without_sites_or_span():
    rows = [
        {"classes": {"tf_bound": range(8, 12)}, "sites": [9]},  # one site < min_sites
        {"classes": {"tf_bound": range(8, 12)}, "sites": [2, 18]},  # sites outside the flank
        {"classes": {"tf_bound": range(8, 12)}, "unobserved": [11]},  # read ends inside
    ]
    assert _state(rows) == ["uninformative"] * 3
    assert _state(rows[:1], OccupancyRules(flank=3, min_sites=1)) == ["tf_bound"]


def test_instances_outside_the_dataset_positions_are_dropped():
    hits = pd.DataFrame(
        {"motif_id": ["in", "edge", "gap"], "start": [10, 95, 40], "end": [15, 105, 45]}
    )
    coordinates = np.r_[np.arange(0, 42), np.arange(44, 100)]  # a gap at 42-43
    instances, table = resolve_instances(hits, coordinates, flank=2)
    assert table["motif_id"].tolist() == ["in"] and table["instance"].tolist() == [0]
    assert instances[0].core == slice(10, 15) and instances[0].window == slice(8, 17)


def test_group_fractions_count_informative_reads_only():
    codes = {state: code for code, state in enumerate(STATES)}
    states = np.array(
        [
            [codes["tf_bound"]],
            [codes["uninformative"]],
            [codes["accessible"]],
            [codes["nucleosome"]],
        ],
        dtype=np.int8,
    )
    table = group_occupancy(states, ["a", "a", "a", "b"], pd.DataFrame({"instance": [0]}))
    a = table.set_index("group").loc["a"]
    assert (a.n_reads, a.n_informative, a.n_uninformative) == (3, 2, 1)
    assert a.frac_tf_bound == pytest.approx(0.5) and a.frac_accessible == pytest.approx(0.5)
    assert a.frac_bound == pytest.approx(0.5)
    assert 0 <= a.frac_tf_bound_low < 0.5 < a.frac_tf_bound_high <= 1
    b = table.set_index("group").loc["b"]
    assert b.frac_bound == pytest.approx(1.0) and b.frac_tf_bound == 0
