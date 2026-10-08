"""Bulk class tracks and motif-instance contrast (`MOT-03`)."""

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.motif_tracks import (
    TrackCounts,
    filter_hits,
    instance_contrast,
    pack_lanes,
)

pytestmark = pytest.mark.unit


def test_counts_equal_a_direct_count_and_skip_unspanned_positions():
    rng = np.random.default_rng(0)
    values = (rng.random((6, 20, 2)) < 0.4).astype(float)
    observed = rng.random((6, 20, 2)) < 0.8
    values[~observed] = np.nan
    groups = ["a", "a", "b", "a", "b", "b"]
    counts = TrackCounts(np.arange(100, 120), ["t1", "t2"])
    counts.add(groups[:3], values[:3], observed[:3])
    counts.add(groups[3:], values[3:], observed[3:])
    table = counts.table()
    for group in ("a", "b"):
        rows = np.array([g == group for g in groups])
        for t_index, track in enumerate(["t1", "t2"]):
            got = table[(table.group == group) & (table.track == track)]
            span = observed[rows, :, t_index].sum(0)
            hits = (observed[rows, :, t_index] & (values[rows, :, t_index] > 0)).sum(0)
            assert got["spanning"].tolist() == span.tolist()
            assert got["in_class"].tolist() == hits.tolist()
            expected = np.where(span > 0, hits / np.maximum(span, 1), np.nan)
            np.testing.assert_allclose(got["fraction"], expected, equal_nan=True)


def test_merging_shards_equals_one_pass():
    rng = np.random.default_rng(1)
    values = (rng.random((8, 10, 1)) < 0.5).astype(float)
    observed = np.ones_like(values, dtype=bool)
    groups = list("abababab")
    whole = TrackCounts(range(10), ["t"])
    whole.add(groups, values, observed)
    first, second = TrackCounts(range(10), ["t"]), TrackCounts(range(10), ["t"])
    first.add(groups[:5], values[:5], observed[:5])
    second.add(groups[5:], values[5:], observed[5:])
    first.merge(second)
    pd.testing.assert_frame_equal(whole.table(), first.table())


def _tracks(fraction: np.ndarray, start: int = 0, spanning: int = 20) -> pd.DataFrame:
    positions = np.arange(start, start + fraction.size)
    return pd.DataFrame(
        {
            "group": "g",
            "track": "tf",
            "position": positions,
            "spanning": spanning,
            "in_class": (fraction * spanning).astype(int),
            "fraction": fraction,
        }
    )


def _hit(start, end, motif="M", pvalue=1e-5):
    return {
        "motif_id": motif,
        "motif_name": motif,
        "family": "fam",
        "reference": "ref",
        "start": start,
        "end": end,
        "motif_strand": "+",
        "pvalue": pvalue,
    }


def test_contrast_is_high_for_a_footprint_on_the_motif_and_not_around_it():
    fraction = np.full(100, 0.1)
    fraction[40:50] = 0.8  # a footprint exactly over the motif
    hits = pd.DataFrame([_hit(40, 50, "on"), _hit(60, 70, "off"), _hit(35, 55, "wide")])
    contrast = instance_contrast(_tracks(fraction), hits).set_index("motif_id")
    assert contrast.loc["on", "contrast"] == pytest.approx(0.7)
    assert contrast.loc["off", "contrast"] == pytest.approx(0.0)
    assert contrast.loc["on", "contrast"] > contrast.loc["wide", "contrast"]
    assert contrast.loc["on", "min_spanning"] == 20
    narrow = instance_contrast(_tracks(fraction), hits.iloc[[0]], flank=3)
    assert narrow["flanks"].iat[0] == pytest.approx(0.1)


def test_instances_outside_the_tracks_get_nan():
    hits = pd.DataFrame([_hit(500, 510)])
    contrast = instance_contrast(_tracks(np.full(50, 0.2)), hits)
    assert np.isnan(contrast["inside"].iat[0]) and contrast["min_spanning"].iat[0] == 0


def test_lanes_never_overlap():
    rng = np.random.default_rng(2)
    starts = rng.integers(0, 500, 200)
    ends = starts + rng.integers(5, 30, 200)
    lanes = pack_lanes(starts, ends)
    for lane in np.unique(lanes):
        members = np.flatnonzero(lanes == lane)
        order = members[np.argsort(starts[members])]
        assert (starts[order][1:] >= ends[order][:-1]).all()


def test_filters_select_reference_pvalue_family_motif_and_window():
    hits = pd.DataFrame(
        [
            {**_hit(10, 20, "A", 1e-6), "family": "Sox"},
            {**_hit(30, 40, "B", 5e-5), "family": "Homeodomain,POU"},
            {**_hit(50, 60, "C", 1e-6), "reference": "other"},
        ]
    )
    assert filter_hits(hits, reference="ref")["motif_id"].tolist() == ["A", "B"]
    assert filter_hits(hits, max_pvalue=1e-5)["motif_id"].tolist() == ["A", "C"]
    assert filter_hits(hits, families=["pou"])["motif_id"].tolist() == ["B"]
    assert filter_hits(hits, motifs=["A"])["motif_id"].tolist() == ["A"]
    assert filter_hits(hits, window=(15, 35))["motif_id"].tolist() == ["A", "B"]
