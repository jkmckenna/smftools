"""`DLS-03`: HMM feature layers from a posterior."""

import numpy as np
import pytest

from smftools.hmm.features import FeatureSettings, call_hmm_features
from smftools.informatics.partition_query import decode_probability_u8, encode_probability_u8

pytestmark = pytest.mark.unit

SETS = {
    "accessible": {"state": "Modified", "features": {"short_open": (1, 5), "long_open": (5, 1e9)}},
    "footprint": {
        "state": "Non-Modified",
        "features": {"short_bound": (1, 4), "long_bound": (4, 1e9)},
    },
}


def _settings(**kwargs):
    return FeatureSettings(feature_sets=SETS, **kwargs)


def _call(posterior, spans, covered=None, **settings):
    posterior = np.atleast_2d(np.asarray(posterior, dtype=float))
    positions = np.arange(posterior.shape[1])
    return call_hmm_features(
        posterior,
        positions,
        np.atleast_2d(spans),
        prefix="C",
        settings=_settings(**settings),
        covered=covered,
    )


def test_runs_are_measured_inside_the_read_and_read_end_runs_are_flagged():
    # read covers 2..9; the posterior beyond the read would extend both runs
    p = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    out = _call(p, [2, 9])
    accessible = out["C_all_accessible_features"][0]
    assert np.isnan(accessible[[0, 1, 10, 11]]).all()  # outside the read
    # 2..3 open (2 bp, reaches the read start), 4..8 bound (5 bp), 9 open (1 bp, read end)
    np.testing.assert_array_equal(accessible[2:10], [1, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(out["C_short_open"][0][2:10], [1, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(out["C_long_bound"][0][2:10], [0, 0, 1, 1, 1, 1, 1, 0])
    truncated = out["C_all_accessible_features_end_truncated"][0][2:10]
    np.testing.assert_array_equal(truncated, [1, 1, 0, 0, 0, 0, 0, 1])
    assert not out["C_all_footprint_features_end_truncated"][0][2:10].any()
    np.testing.assert_array_equal(
        out["C_all_footprint_features_lengths"][0][2:10], [0, 0, 5, 5, 5, 5, 5, 0]
    )


def test_uncovered_bases_are_bridged_but_masked():
    p = [0.1] * 8
    covered = np.ones((1, 8), bool)
    covered[0, 3] = False
    out = _call(p, [0, 7], covered=covered)
    bound = out["C_long_bound"][0]
    assert np.isnan(bound[3])
    np.testing.assert_array_equal(np.delete(bound, 3), 1)
    lengths = out["C_all_footprint_features_lengths"][0]
    assert set(np.delete(lengths, 3)) == {8}  # one run across the uncovered base


def test_every_position_in_the_read_belongs_to_one_group():
    rng = np.random.default_rng(0)
    p = rng.random((20, 60))
    p[:, ::7] = 0.5  # the threshold itself
    out = _call(p, np.tile([0, 59], (20, 1)))
    a = out["C_all_accessible_features"] > 0
    f = out["C_all_footprint_features"] > 0
    assert not (a & f).any()
    np.testing.assert_array_equal(a | f, np.ones_like(a))
    assert a[:, ::7].all()  # p == 0.5 is accessible


def test_calls_survive_8_bit_storage():
    rng = np.random.default_rng(1)
    p = rng.random((30, 80)).astype(np.float32)
    p[:, ::5] = 0.5 + rng.normal(0, 2e-3, p[:, ::5].shape)  # crowd the threshold
    spans = np.tile([3, 76], (30, 1))
    before = _call(p, spans, merges=(("all_accessible_features", 2),))
    stored = decode_probability_u8(encode_probability_u8(p))
    after = _call(stored, spans, merges=(("all_accessible_features", 2),))
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)


def test_merges_join_short_gaps_and_keep_truncation():
    # open 0..1 (read start), bound 2..3, open 4..6, bound 7..11, open 12
    p = [0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    out = _call(p, [0, 12], merges=(("all_accessible_features", 2),))
    merged = out["C_all_accessible_features_merged"][0]
    np.testing.assert_array_equal(merged, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(out["C_all_accessible_features_merged_lengths"][0][:7], [7] * 7)
    np.testing.assert_array_equal(out["C_long_open_merged"][0][:7], 1)  # 7 bp >= 5
    flags = out["C_all_accessible_features_merged_end_truncated"][0]
    np.testing.assert_array_equal(flags[:7], 1)  # contains the read-start run
    assert flags[12] == 1 and not flags[7:12].any()


def test_settings_record_round_trips_to_json():
    import json

    record = json.loads(json.dumps(_settings(merges=(("all_accessible_features", 60),)).to_dict()))
    assert record["version"] == 2
    assert record["merges"] == [["all_accessible_features", 60]]
    assert record["feature_sets"]["footprint"]["state"] == "Non-Modified"
