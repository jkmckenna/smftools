from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.filters.obs_filters import build_obs_mask, max_cigar_deletion

# ---------------------------------------------------------------------------
# max_cigar_deletion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_max_cigar_deletion_no_deletions() -> None:
    assert max_cigar_deletion("100M") == 0


@pytest.mark.unit
def test_max_cigar_deletion_single() -> None:
    assert max_cigar_deletion("50M200D50M") == 200


@pytest.mark.unit
def test_max_cigar_deletion_multiple_returns_max() -> None:
    assert max_cigar_deletion("10M50D10M100D10M") == 100


@pytest.mark.unit
def test_max_cigar_deletion_non_string_input() -> None:
    assert max_cigar_deletion(None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_obs_mask — basic filters
# ---------------------------------------------------------------------------


def _make_obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Barcode": ["NB01", "NB01", "NB02", "NB01"],
            "Reference_strand": ["6B6_top", "6B6_top", "6B6_top", "6B6_enh_del_top"],
            "demux_type": ["double", "single", "double", "double"],
            "CIGAR": ["100M", "100M", "100M", "100M"],
        }
    )


@pytest.mark.unit
def test_build_obs_mask_barcode_filter() -> None:
    obs = _make_obs()
    mask = build_obs_mask(obs, barcode="NB01")
    assert mask.tolist() == [True, True, False, True]


@pytest.mark.unit
def test_build_obs_mask_ref_strand_filter() -> None:
    obs = _make_obs()
    mask = build_obs_mask(obs, ref_strand="6B6_enh_del_top")
    assert mask.tolist() == [False, False, False, True]


@pytest.mark.unit
def test_build_obs_mask_demux_type_filter() -> None:
    obs = _make_obs()
    mask = build_obs_mask(obs, demux_type="double")
    assert mask.tolist() == [True, False, True, True]


@pytest.mark.unit
def test_build_obs_mask_combined_filters() -> None:
    obs = _make_obs()
    mask = build_obs_mask(obs, barcode="NB01", ref_strand="6B6_top", demux_type="double")
    assert mask.tolist() == [True, False, False, False]


@pytest.mark.unit
def test_build_obs_mask_no_filters_returns_all_true() -> None:
    obs = _make_obs()
    mask = build_obs_mask(obs)
    assert mask.all()


# ---------------------------------------------------------------------------
# build_obs_mask — CIGAR deletion filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_obs_mask_cigar_filter_applied_to_wt_only() -> None:
    obs = pd.DataFrame(
        {
            "Barcode": ["NB01", "NB01"],
            "Reference_strand": ["6B6_top", "6B6_enh_del_top"],
            "demux_type": ["double", "double"],
            "CIGAR": ["100M350D50M", "100M350D50M"],  # both have 350 bp deletion
        }
    )
    # 350 > 200 so WT strand read excluded; enh_del strand read kept (filter not applied)
    mask = build_obs_mask(
        obs,
        ref_strand="6B6_top",
        wt_ref_strands=["6B6_top"],
        max_cigar_del=200,
    )
    assert mask.tolist() == [False, False]

    mask_enh = build_obs_mask(
        obs,
        ref_strand="6B6_enh_del_top",
        wt_ref_strands=["6B6_top"],
        max_cigar_del=200,
    )
    assert mask_enh.tolist() == [False, True]


@pytest.mark.unit
def test_build_obs_mask_cigar_filter_passes_small_deletion() -> None:
    obs = pd.DataFrame(
        {
            "Barcode": ["NB01"],
            "Reference_strand": ["6B6_top"],
            "demux_type": ["double"],
            "CIGAR": ["50M100D50M"],  # 100 <= 200
        }
    )
    mask = build_obs_mask(
        obs,
        barcode="NB01",
        ref_strand="6B6_top",
        demux_type="double",
        wt_ref_strands=["6B6_top"],
        max_cigar_del=200,
    )
    assert mask.tolist() == [True]


@pytest.mark.unit
def test_build_obs_mask_cigar_filter_skipped_when_not_wt() -> None:
    obs = pd.DataFrame(
        {
            "Barcode": ["NB01"],
            "Reference_strand": ["6B6_enh_del_top"],
            "demux_type": ["double"],
            "CIGAR": ["100M500D50M"],  # huge deletion, but not a WT strand
        }
    )
    mask = build_obs_mask(
        obs,
        ref_strand="6B6_enh_del_top",
        wt_ref_strands=["6B6_top"],
        max_cigar_del=200,
    )
    assert mask.tolist() == [True]


# ---------------------------------------------------------------------------
# build_obs_mask — extra_eq
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_obs_mask_extra_eq() -> None:
    obs = pd.DataFrame(
        {
            "Barcode": ["NB01", "NB01"],
            "Reference_strand": ["6B6_top", "6B6_top"],
            "demux_type": ["double", "double"],
            "CIGAR": ["100M", "100M"],
            "Harvest": ["fresh", "cycling"],
        }
    )
    mask = build_obs_mask(obs, extra_eq={"Harvest": "fresh"})
    assert mask.tolist() == [True, False]
