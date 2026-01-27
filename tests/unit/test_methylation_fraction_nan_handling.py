import numpy as np

from smftools.plotting.general_plotting import methylation_fraction


def test_methylation_fraction_ignores_nan_values():
    matrix = np.array(
        [
            [1.0, np.nan, 0.0],
            [0.0, 1.0, np.nan],
            [1.0, np.nan, 1.0],
        ]
    )

    result = methylation_fraction(matrix)

    # column 0: valid = [1,0,1] -> 2 methylated out of 2 valid
    # column 1: valid = [1] -> 1 methylated out of 1 valid
    # column 2: valid = [1] -> 1 methylated out of 1 valid
    assert np.allclose(result, np.array([1.0, 1.0, 1.0]))


def test_methylation_fraction_zero_is_valid():
    matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = methylation_fraction(matrix, zero_is_valid=True)

    # column 0: 1 methylated out of 2 valid
    # column 1: 0 methylated out of 2 valid
    # column 2: 2 methylated out of 2 valid
    assert np.allclose(result, np.array([0.5, 0.0, 1.0]))
