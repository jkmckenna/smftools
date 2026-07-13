import numpy as np

from smftools.informatics.signal_features import (
    base_signal_features,
    move_table_to_base_ranges,
    read_signal_features,
)


def test_move_table_ranges_basic():
    # stride=2; 4 blocks; moves start bases at blocks 0 and 2 -> 2 bases
    # base0 = blocks [0,1] -> samples [0,4); base1 = blocks [2,3] -> samples [4,8)
    mv = [2, 1, 0, 1, 0]
    start, end = move_table_to_base_ranges(mv, ts=0)
    assert list(start) == [0, 4]
    assert list(end) == [4, 8]


def test_move_table_ranges_with_trim_and_uneven_dwell():
    # stride=3, ts=5; moves at blocks 0,1,3 -> 3 bases with dwells 1,2,1 blocks
    mv = [3, 1, 1, 0, 1]
    start, end = move_table_to_base_ranges(mv, ts=5)
    # base0: block0 -> [5, 8); base1: blocks1-2 -> [8, 14); base2: block3 -> [14, 17)
    assert list(start) == [5, 8, 14]
    assert list(end) == [8, 14, 17]


def test_base_signal_features_mean_std_dwell():
    signal = np.array([10.0, 10.0, 10.0, 10.0, 0.0, 4.0, 8.0, 12.0], dtype=np.float32)
    start = np.array([0, 4])
    end = np.array([4, 8])
    feats = base_signal_features(signal, start, end)
    assert np.allclose(feats["current_mean"], [10.0, 6.0])
    assert np.isclose(feats["current_std"][0], 0.0)
    assert np.isclose(feats["current_std"][1], np.std([0.0, 4.0, 8.0, 12.0]))
    assert list(feats["dwell"]) == [4.0, 4.0]
    assert list(feats["signal_start"]) == [0.0, 4.0]


def test_base_signal_features_clips_beyond_signal_end():
    signal = np.arange(5, dtype=np.float32)
    feats = base_signal_features(signal, np.array([0, 4]), np.array([4, 12]))
    # second base window [4,12) clipped to [4,5): mean=4, dwell=1
    assert np.isclose(feats["current_mean"][1], 4.0)
    assert feats["dwell"][1] == 1.0


def test_read_signal_features_reverse_is_flipped():
    mv = [2, 1, 0, 1, 0]  # 2 bases
    signal = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    fwd = read_signal_features(mv, 0, False, signal, expected_bases=2)
    rev = read_signal_features(mv, 0, True, signal, expected_bases=2)
    assert np.allclose(fwd["current_mean"], [1.0, 5.0])
    assert np.allclose(rev["current_mean"], [5.0, 1.0])  # flipped to BAM query order


def test_read_signal_features_rejects_base_count_mismatch():
    mv = [2, 1, 0, 1, 0]  # 2 bases
    signal = np.zeros(8, dtype=np.float32)
    assert read_signal_features(mv, 0, False, signal, expected_bases=5) is None


def test_materialize_ragged_densifies_signal_feature_layers():
    import pandas as pd

    from smftools.informatics.ragged_store import materialize_ragged

    frame = pd.DataFrame(
        [
            {
                "read_id": "r1",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "reference_start": 0,
                "cigar": "3M",
                "aligned_length": 3,
                "sequence": [0, 1, 2],
                "quality": [30, 31, 32],
                "mismatch": [4, 4, 4],
                "modification_signal": [0.0, 1.0, 0.0],
                "current_mean": [10.0, 20.0, 30.0],
                "current_std": [1.0, 2.0, 3.0],
                "dwell": [5.0, 6.0, 7.0],
                "signal_start": [0.0, 5.0, 11.0],
            }
        ]
    )
    obs = pd.DataFrame(index=["r1"])

    result = materialize_ragged(frame, obs=obs, reference_lengths={"ref_top": 3})
    assert "current_mean" in result.layers
    assert "current_dwell" in result.layers
    assert "current_signal_start" in result.layers
    assert list(result.layers["current_mean"][0]) == [10.0, 20.0, 30.0]
    assert list(result.layers["current_dwell"][0]) == [5.0, 6.0, 7.0]

    # requesting only a base layer excludes the optional signal layers
    base = materialize_ragged(
        frame, obs=obs, reference_lengths={"ref_top": 3}, layers=["sequence_integer_encoding"]
    )
    assert "current_mean" not in base.layers


def test_materialize_ragged_without_signal_columns_has_only_base_layers():
    import pandas as pd

    from smftools.informatics.ragged_store import materialize_ragged

    frame = pd.DataFrame(
        [
            {
                "read_id": "r1",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "reference_start": 0,
                "cigar": "3M",
                "aligned_length": 3,
                "sequence": [0, 1, 2],
                "quality": [30, 31, 32],
                "mismatch": [4, 4, 4],
                "modification_signal": [0.0, 1.0, 0.0],
            }
        ]
    )
    result = materialize_ragged(frame, obs=pd.DataFrame(index=["r1"]), reference_lengths={"ref_top": 3})
    assert set(result.layers) == {
        "sequence_integer_encoding",
        "mismatch_integer_encoding",
        "base_quality_scores",
        "read_span_mask",
    }
