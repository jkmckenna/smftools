import pytest
import matplotlib


matplotlib.use("Agg")

import numpy as np
from matplotlib import colors

from smftools.plotting.hmm_plotting import _build_hmm_feature_cmap


def test_build_hmm_feature_cmap_sets_zero_and_nan_colors():
    cmap = _build_hmm_feature_cmap("#123456", zero_color="#f5f1e8", nan_color="#ffffff")

    assert np.allclose(cmap(0.0), colors.to_rgba("#f5f1e8"))
    assert np.allclose(cmap(1.0), colors.to_rgba("#123456"))

    masked = np.ma.masked_invalid(np.array([np.nan]))
    assert np.allclose(cmap(masked)[0], colors.to_rgba("#ffffff"))
