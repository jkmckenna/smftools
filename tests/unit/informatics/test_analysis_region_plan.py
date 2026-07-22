from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from smftools.informatics.analysis_region_plan import (
    ANALYSIS_REGION_PLANNER_VERSION,
    plan_analysis_cores,
)
from smftools.preprocessing.dispatch_plan import plan_preprocess_tasks
from smftools.tools.partitioned_latent import _analysis_units
from smftools.tools.partitioned_spatial import _region_tasks


def _spine(tmp_path, *, reduced: bool = False):
    reference = "chr1:100-200_top" if reduced else "chr1_top"
    length = 100 if reduced else 30
    obs = pd.DataFrame(
        {
            "Reference_strand": [reference, reference],
            "Barcode": ["bc1", "bc1"],
            "reference_start": [0, length - 5],
            "reference_end": [15, length],
            "passes_qc": [True, True],
        },
        index=["left", "right"],
    )
    spine = ad.AnnData(obs=obs)
    spine.uns["reference_plans"] = {
        reference: {
            "analysis_mode": "genome",
            "reference_length": length,
            "tile_size": 10,
            "tile_halo": 2,
        }
    }
    catalog_path = tmp_path / "analysis.parquet"
    mapping_path = tmp_path / "mapping.parquet"
    spine.uns["region_catalogs"] = {"analysis": str(catalog_path)}
    spine.uns["reference_interval_map"] = str(mapping_path)
    return spine, reference, catalog_path, mapping_path


def _write_mapping(path, reference, *, original_start: int, length: int):
    pd.DataFrame(
        {
            "stored_reference": [reference],
            "stored_start": [0],
            "stored_end": [length],
            "original_reference": ["chr1"],
            "original_start": [original_start],
            "original_end": [original_start + length],
            "coordinate_orientation": [1],
        }
    ).to_parquet(path, index=False)


def _write_catalog(path, intervals):
    pd.DataFrame(
        [
            {
                "region_id": region_id,
                "original_reference": "chr1",
                "original_start": start,
                "original_end": end,
            }
            for region_id, start, end in intervals
        ]
    ).to_parquet(path, index=False)


@pytest.mark.unit
def test_overlapping_regions_form_non_overlapping_tile_bounded_cores(tmp_path):
    spine, reference, catalog_path, mapping_path = _spine(tmp_path)
    _write_catalog(catalog_path, [("reg_a", 3, 13), ("reg_b", 8, 18)])
    _write_mapping(mapping_path, reference, original_start=0, length=30)

    cores = plan_analysis_cores(spine)

    assert [(core.core_start, core.core_end) for core in cores] == [(3, 10), (10, 18)]
    assert [core.analysis_region_ids for core in cores] == [
        ("reg_a", "reg_b"),
        ("reg_a", "reg_b"),
    ]
    assert all(core.analysis_core_id.startswith("acore_") for core in cores)
    assert all(core.planner_version == ANALYSIS_REGION_PLANNER_VERSION for core in cores)
    owned_positions = [
        position for core in cores for position in range(core.core_start, core.core_end)
    ]
    assert owned_positions == list(range(3, 18))


@pytest.mark.unit
def test_original_regions_map_to_reduced_stored_coordinates(tmp_path):
    spine, reference, catalog_path, mapping_path = _spine(tmp_path, reduced=True)
    _write_catalog(catalog_path, [("reg_a", 110, 135)])
    _write_mapping(mapping_path, reference, original_start=100, length=100)

    cores = plan_analysis_cores(spine)

    assert [
        (core.core_start, core.core_end, core.original_start, core.original_end) for core in cores
    ] == [(10, 20, 110, 120), (20, 30, 120, 130), (30, 35, 130, 135)]


@pytest.mark.unit
def test_analysis_region_outside_alignment_map_fails_clearly(tmp_path):
    spine, reference, catalog_path, mapping_path = _spine(tmp_path, reduced=True)
    _write_catalog(catalog_path, [("reg_a", 90, 110)])
    _write_mapping(mapping_path, reference, original_start=100, length=100)

    with pytest.raises(ValueError, match="not fully represented"):
        plan_analysis_cores(spine)


@pytest.mark.unit
def test_partitioned_stages_share_core_plan_and_provenance(tmp_path):
    spine, reference, catalog_path, mapping_path = _spine(tmp_path)
    spine.obs.loc["right", ["reference_start", "reference_end"]] = [12, 22]
    _write_catalog(catalog_path, [("reg_a", 3, 13), ("reg_b", 8, 18)])
    _write_mapping(mapping_path, reference, original_start=0, length=30)
    cfg = SimpleNamespace(target_task_memory_mb=1, autocorr_max_lag=4, spatial_regions_bed=None)

    preprocess = plan_preprocess_tasks(spine, target_task_memory_mb=1, filter_mask="passes_qc")
    spatial, regions = _region_tasks(spine, cfg, "passes_qc")
    latent = _analysis_units(spine, "passes_qc")

    expected = [(3, 10), (10, 18)]
    assert sorted({(task.core_start, task.core_end) for task in preprocess}) == expected
    assert sorted({(task.core_start, task.core_end) for task in spatial}) == expected
    assert [(unit["core_start"], unit["core_end"]) for unit in latent] == expected
    assert list(map(tuple, regions[["start", "end"]].to_numpy())) == expected
    assert {task.analysis_planner_version for task in preprocess} == {
        ANALYSIS_REGION_PLANNER_VERSION
    }
    assert all(task.analysis_region_ids for task in preprocess)
    assert all(
        task.load_start <= task.core_start < task.core_end <= task.load_end for task in spatial
    )


@pytest.mark.unit
def test_plot_catalog_pointer_does_not_change_analysis_plan(tmp_path):
    spine, reference, catalog_path, mapping_path = _spine(tmp_path)
    _write_catalog(catalog_path, [("reg_a", 3, 13)])
    _write_mapping(mapping_path, reference, original_start=0, length=30)
    first = plan_analysis_cores(spine)

    spine.uns["region_catalogs"]["plot"] = str(tmp_path / "different_plot.parquet")
    second = plan_analysis_cores(spine)

    assert first == second
