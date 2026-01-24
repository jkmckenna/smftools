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
