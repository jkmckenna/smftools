import numpy as np
import pytest

from smftools.plotting.hmm_plotting import _map_length_matrix_to_subclasses


def test_map_length_matrix_to_subclasses():
    matrix = np.array(
        [
            [0, 5, 25, np.nan],
            [100, 200, 15, 0],
        ],
        dtype=float,
    )
    ranges = [
        (3, 20, "#A5D6A7"),
        (20, 40, "#2E7D32"),
        (100, 200, "#6D4C41"),
    ]

    mapped = _map_length_matrix_to_subclasses(matrix, ranges)

    assert mapped[0, 0] == 0
    assert mapped[0, 1] == 1
    assert mapped[0, 2] == 2
    assert np.isnan(mapped[0, 3])
    assert mapped[1, 0] == 3
    assert mapped[1, 1] == 3
    assert mapped[1, 2] == 1
