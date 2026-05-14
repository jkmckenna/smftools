from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.filters.position_filters import build_position_mask


def _make_var(coords: list[float], ref: str = "6B6_top") -> pd.DataFrame:
    return pd.DataFrame({f"{ref}_reindexed": coords})


@pytest.mark.unit
def test_build_position_mask_no_span_keeps_finite() -> None:
    var = _make_var([-100.0, 0.0, float("nan"), 100.0])
    keep, coords = build_position_mask(var, "6B6_top")
    assert keep.tolist() == [True, True, False, True]
    assert np.isnan(coords[2])


@pytest.mark.unit
def test_build_position_mask_with_span() -> None:
    var = _make_var([-200.0, -100.0, 0.0, 100.0, 200.0])
    keep, coords = build_position_mask(var, "6B6_top", span=(-150.0, 150.0))
    assert keep.tolist() == [False, True, True, True, False]


@pytest.mark.unit
def test_build_position_mask_all_nan() -> None:
    var = _make_var([float("nan"), float("nan")])
    keep, _ = build_position_mask(var, "6B6_top")
    assert not keep.any()


@pytest.mark.unit
def test_build_position_mask_empty_span_returns_none() -> None:
    var = _make_var([-100.0, 0.0, 100.0])
    keep, _ = build_position_mask(var, "6B6_top", span=(200.0, 300.0))
    assert not keep.any()


@pytest.mark.unit
def test_build_position_mask_missing_column_raises() -> None:
    var = pd.DataFrame({"other_col": [1.0, 2.0]})
    with pytest.raises(KeyError):
        build_position_mask(var, "6B6_top")
