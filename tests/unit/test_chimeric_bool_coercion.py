"""String-valued booleans must not be read with Python truthiness (`F13`).

``chimeric_variant_sites`` round-trips through parquet as a *category of the
strings* ``"True"``/``"False"``. The clustermap filters negated it with
``~s.astype(bool)``, and ``bool("False")`` is ``True`` -- so every read was
treated as chimeric, every group filtered to nothing, and every HMM clustermap
silently produced no image while still creating its output directory.

Found by looking at why `241213`'s HMM clustermap folders were empty; it had
been that way in every generation on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.plotting.plotting_utils import coerce_bool_series

pytestmark = pytest.mark.unit


def test_string_false_is_false():
    """The defect, stated at its smallest."""
    series = pd.Series(["True", "False", "True"])
    assert list(coerce_bool_series(series)) == [True, False, True]
    # The behavior being replaced, pinned so the contrast is explicit.
    assert list(series.astype(bool)) == [True, True, True]


def test_categorical_of_strings_is_the_real_shape():
    """This is exactly how the column arrives after a parquet round-trip."""
    series = pd.Series(["True", "False", "False"]).astype("category")
    kept = ~coerce_bool_series(series)
    assert int(kept.sum()) == 2, "non-chimeric reads must survive the filter"


def test_real_booleans_are_unchanged():
    series = pd.Series([True, False, True])
    assert list(coerce_bool_series(series)) == [True, False, True]


def test_numeric_flags_are_unchanged():
    series = pd.Series([1, 0, 1])
    assert list(coerce_bool_series(series)) == [True, False, True]


def test_missing_is_not_evidence_of_chimerism():
    """Unknown status must not exclude a read."""
    series = pd.Series(["True", None, "False"])
    assert list(coerce_bool_series(series)) == [True, False, False]


@pytest.mark.parametrize("truthy", ["true", "TRUE", " True ", "t", "yes", "y", "1", "on"])
def test_truthy_spellings(truthy):
    assert bool(coerce_bool_series(pd.Series([truthy])).iloc[0])


@pytest.mark.parametrize("falsy", ["false", "FALSE", " False ", "f", "no", "n", "0", "off"])
def test_falsy_spellings(falsy):
    assert not bool(coerce_bool_series(pd.Series([falsy])).iloc[0])


def test_clustermap_keeps_non_chimeric_reads_end_to_end():
    """The filter as the plotting code actually builds it.

    Asserting on the mask rather than on a rendered PNG keeps this a unit
    test, but it is the same expression: an all-empty mask here is precisely
    what produced empty clustermap directories.
    """
    obs = pd.DataFrame(
        {
            "chimeric_variant_sites": pd.Series(["False", "True", "False", "False"]).astype(
                "category"
            ),
        }
    )
    chimeric_mask = ~coerce_bool_series(obs["chimeric_variant_sites"])
    assert int(chimeric_mask.sum()) == 3
    assert not bool(chimeric_mask.iloc[1])


def test_all_chimeric_still_filters_everything():
    """The filter must remain capable of excluding; this is not a pass-through."""
    obs = pd.Series(["True", "True"]).astype("category")
    assert int((~coerce_bool_series(obs)).sum()) == 0


def test_unrecognized_strings_are_false():
    """Anything not in the truthy vocabulary is not chimeric.

    Chosen deliberately over raising: an unexpected label should degrade to
    "keep the read and plot it", not abort a pipeline stage.
    """
    assert not bool(coerce_bool_series(pd.Series(["maybe"])).iloc[0])


def test_empty_series_round_trips():
    result = coerce_bool_series(pd.Series([], dtype=object))
    assert len(result) == 0
    assert result.dtype == np.dtype(bool)
