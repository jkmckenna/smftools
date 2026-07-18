import pandas as pd
import pytest

from smftools.informatics.storage_planner import iter_reference_tiles, plan_references


def test_auto_plans_references_independently():
    frame = pd.DataFrame({"Reference_strand": ["locus_top"] * 2 + ["chr1_top"] * 100})
    plans = plan_references(
        frame,
        {"locus_top": 500, "chr1_top": 100_000_000},
        max_full_matrix_gb=0.01,
        tile_size=10_000,
        tile_halo=500,
    )
    by_reference = {plan.reference: plan for plan in plans}

    assert by_reference["locus_top"].analysis_mode == "locus"
    assert by_reference["locus_top"].cache_mode == "full"
    assert by_reference["chr1_top"].analysis_mode == "genome"
    assert by_reference["chr1_top"].cache_mode == "tiled"


def test_tile_plan_clips_halo_to_reference():
    frame = pd.DataFrame({"Reference_strand": ["ref_top"]})
    plan = plan_references(
        frame,
        {"ref_top": 25},
        analysis_mode="genome",
        tile_size=10,
        tile_halo=3,
    )[0]

    assert list(iter_reference_tiles(plan)) == [
        (0, 10, 0, 13),
        (10, 20, 7, 23),
        (20, 25, 17, 25),
    ]


def test_invalid_storage_policy_raises():
    frame = pd.DataFrame({"Reference_strand": ["ref_top"]})
    with pytest.raises(ValueError, match="analysis_mode"):
        plan_references(frame, {"ref_top": 10}, analysis_mode="whole")
