"""Composite chimera classification across methods (`EGL-20`).

Three columns describe chimerism on two orthogonal axes: reference identity
(`chimeric_variant_sites`) and strand chemistry (`deaminase_PCR_chimera`,
`deaminase_segment_chimera`). `omit_chimeric_reads` consumes their union.

The union is materialized rather than computed at each call site so a surprising
exclusion can be decomposed into which method fired, and so a missing method
degrades in one place instead of five.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smftools.plotting.plotting_utils import chimera_filter_column
from smftools.preprocessing.chimera_classes import (
    COMPOSITE_COLUMN,
    append_composite_chimera_column,
)

pytestmark = pytest.mark.unit


def test_union_of_two_methods():
    obs = pd.DataFrame(
        {
            "chimeric_variant_sites": [True, False, False, False],
            "deaminase_segment_chimera": [False, True, False, False],
        }
    )
    assert list(append_composite_chimera_column(obs)[COMPOSITE_COLUMN]) == [
        True,
        True,
        False,
        False,
    ]


def test_string_valued_columns_are_coerced():
    """These round-trip through parquet as `"True"`/`"False"` (`F13`)."""
    obs = pd.DataFrame({"chimeric_variant_sites": ["True", "False"]})
    assert list(append_composite_chimera_column(obs)[COMPOSITE_COLUMN]) == [True, False]


def test_a_method_that_did_not_run_contributes_nothing():
    """Absent is not "not chimeric".

    Treating a disabled detector as a clean bill of health is the failure shape
    of `F13` and `F17`. Only columns that exist participate.
    """
    obs = pd.DataFrame({"deaminase_segment_chimera": [True, False]})
    result = append_composite_chimera_column(obs)
    assert list(result[COMPOSITE_COLUMN]) == [True, False]


def test_no_methods_at_all_adds_no_column():
    """`direct` runs have no chimera detection; nothing should be invented."""
    obs = append_composite_chimera_column(pd.DataFrame({"read_id": ["r1"]}))
    assert COMPOSITE_COLUMN not in obs.columns


def test_all_three_methods_union():
    obs = pd.DataFrame(
        {
            "chimeric_variant_sites": [True, False, False, False],
            "deaminase_PCR_chimera": [False, True, False, False],
            "deaminase_segment_chimera": [False, False, True, False],
        }
    )
    assert list(append_composite_chimera_column(obs)[COMPOSITE_COLUMN]) == [True, True, True, False]


def test_method_columns_survive_the_union():
    """Decomposability: the union must not replace its inputs."""
    obs = append_composite_chimera_column(
        pd.DataFrame({"chimeric_variant_sites": [True], "deaminase_segment_chimera": [False]})
    )
    assert "chimeric_variant_sites" in obs.columns
    assert "deaminase_segment_chimera" in obs.columns


def test_filter_prefers_the_composite():
    assert chimera_filter_column(["is_chimeric_any", "chimeric_variant_sites"]) == (
        "is_chimeric_any"
    )


def test_filter_falls_back_for_older_generations():
    """Generations published before `EGL-20` have no composite column."""
    assert chimera_filter_column(["chimeric_variant_sites"]) == "chimeric_variant_sites"
