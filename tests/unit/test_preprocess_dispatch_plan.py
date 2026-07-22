from types import SimpleNamespace

import anndata as ad
import pandas as pd

from smftools.preprocessing.dispatch_plan import plan_preprocess_tasks
from smftools.tools.partitioned_spatial import _dense_product_regions, _region_tasks


def _spine():
    obs = pd.DataFrame(
        {
            "Reference_strand": ["locus_top", "locus_top", "chr1_top", "chr1_top"],
            "Barcode": ["bc1", "bc2", "bc1", "bc1"],
            "reference_start": [0, 10, 0, 18],
            "reference_end": [20, 30, 12, 30],
        },
        index=["locus1", "locus2", "genome1", "genome2"],
    )
    spine = ad.AnnData(obs=obs)
    spine.uns["reference_plans"] = {
        "locus_top": {
            "reference_length": 40,
            "analysis_mode": "locus",
            "tile_size": 10,
            "tile_halo": 2,
        },
        "chr1_top": {
            "reference_length": 30,
            "analysis_mode": "genome",
            "tile_size": 10,
            "tile_halo": 2,
        },
    }
    return spine


def test_tasks_partition_by_reference_window_and_barcode():
    tasks = plan_preprocess_tasks(_spine(), target_task_memory_mb=1)

    locus_tasks = [task for task in tasks if task.reference == "locus_top"]
    genome_tasks = [task for task in tasks if task.reference == "chr1_top"]
    assert {(task.barcode, task.core_start, task.core_end) for task in locus_tasks} == {
        ("bc1", 0, 40),
        ("bc2", 0, 40),
    }
    assert {(task.core_start, task.core_end) for task in genome_tasks} == {
        (0, 10),
        (10, 20),
        (20, 30),
    }
    assert genome_tasks[0].load_start == 0
    assert genome_tasks[0].load_end == 12
    assert all(task.estimated_memory_bytes <= 1024**2 for task in tasks)


def test_large_locus_group_is_split_to_memory_budget():
    spine = _spine()
    repeated = pd.concat([spine.obs.iloc[[0]]] * 4000)
    repeated.index = [f"read{i}" for i in range(len(repeated))]
    spine = ad.AnnData(obs=repeated)
    spine.uns["reference_plans"] = {
        "locus_top": {
            "reference_length": 40,
            "analysis_mode": "locus",
            "tile_size": 10,
            "tile_halo": 2,
        }
    }

    tasks = plan_preprocess_tasks(spine, target_task_memory_mb=1)

    assert len(tasks) == 2
    assert sum(task.n_reads for task in tasks) == 4000
    assert max(task.estimated_memory_bytes for task in tasks) <= 1024**2


def test_missing_barcode_is_preserved_as_unclassified():
    spine = _spine()
    spine.obs["Barcode"] = spine.obs["Barcode"].astype(object)
    spine.obs.loc["locus1", "Barcode"] = None

    tasks = plan_preprocess_tasks(spine)

    assert any(task.barcode == "unclassified" and task.read_ids == ("locus1",) for task in tasks)


def test_filter_mask_excludes_failed_reads_before_planning():
    spine = _spine()
    spine.obs["passes_dedup"] = [True, False, True, False]

    tasks = plan_preprocess_tasks(spine, filter_mask="passes_dedup")

    planned_reads = {read_id for task in tasks for read_id in task.read_ids}
    assert planned_reads == {"locus1", "genome1"}


def test_spatial_bed_replaces_genome_tiles_but_preserves_full_locus(tmp_path):
    spine = _spine()
    spine.obs["passes_dedup"] = True
    spine.uns["reference_lengths"] = {"locus_top": 40, "chr1_top": 30}
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t5\t17\tpeak_a\n", encoding="utf-8")
    cfg = SimpleNamespace(
        spatial_regions_bed=str(bed),
        target_task_memory_mb=1,
        autocorr_max_lag=2,
    )

    tasks, bed_regions = _region_tasks(spine, cfg, "passes_dedup")

    locus_tasks = [task for task in tasks if task.analysis_mode == "locus"]
    genome_tasks = [task for task in tasks if task.analysis_mode == "genome"]
    assert {(task.core_start, task.core_end) for task in locus_tasks} == {(0, 40)}
    assert {(task.core_start, task.core_end) for task in genome_tasks} == {(5, 17)}
    assert set(bed_regions["reference"]) == {"chr1_top"}
    dense_regions = _dense_product_regions(spine, bed_regions)
    assert set(map(tuple, dense_regions[["reference", "start", "end"]].to_numpy())) == {
        ("locus_top", 0, 40),
        ("chr1_top", 5, 17),
    }


def test_genome_without_spatial_bed_has_portable_empty_dense_region_catalog(tmp_path):
    spine = _spine()
    spine.uns["reference_plans"].pop("locus_top")
    bed_regions = pd.DataFrame(columns=["reference", "start", "end", "name", "bed_chrom"])

    dense_regions = _dense_product_regions(spine, bed_regions)
    catalog = tmp_path / "regions.parquet"
    dense_regions.to_parquet(catalog, index=False)
    restored = pd.read_parquet(catalog)

    assert restored.empty
    assert list(restored.columns) == ["reference", "start", "end", "name", "source"]
    assert restored.dtypes.astype(str).to_dict() == {
        "reference": "string",
        "start": "int64",
        "end": "int64",
        "name": "string",
        "source": "string",
    }
