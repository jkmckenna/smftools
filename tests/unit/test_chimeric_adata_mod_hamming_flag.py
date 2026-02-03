import numpy as np

from smftools.cli.chimeric_adata import (
    _compute_chimeric_by_mod_hamming_distance,
    _max_positive_span_length,
)


def test_max_positive_span_length_finds_longest_run():
    row = np.array([0, 1, 1, 0, 2, 2, 2, 0], dtype=float)
    assert _max_positive_span_length(row) == 3


def test_compute_chimeric_by_mod_hamming_distance_uses_strict_threshold():
    delta_layer = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # max run = 0
            [0, 1, 1, 0, 0, 0],  # max run = 2
            [1, 1, 1, 0, 0, 0],  # max run = 3
        ],
        dtype=float,
    )
    flags = _compute_chimeric_by_mod_hamming_distance(delta_layer, span_threshold=2)
    assert flags.tolist() == [False, False, True]
