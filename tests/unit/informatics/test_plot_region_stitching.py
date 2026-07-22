import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.plot_region_stitching import (
    mask_unanalyzed_gaps,
    resolve_plot_region_plans,
    select_plot_reads,
)


def _spine_with_plot_catalog(tmp_path, *, start=3, end=8):
    catalog = tmp_path / "region_catalogs" / "plot_regions.parquet"
    catalog.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "region_id": ["plot_region"],
            "original_reference": ["original"],
            "original_start": [start],
            "original_end": [end],
            "name": ["requested"],
        }
    ).to_parquet(catalog, index=False)
    mapping = tmp_path / "reference_interval_map.parquet"
    pd.DataFrame(
        {
            "stored_reference": ["ref_top"],
            "stored_start": [0],
            "stored_end": [10],
            "original_reference": ["original"],
            "original_start": [0],
            "original_end": [10],
            "coordinate_orientation": [1],
        }
    ).to_parquet(mapping, index=False)
    spine = ad.AnnData()
    spine.uns["region_catalogs"] = {"plot": "region_catalogs/plot_regions.parquet"}
    spine.uns["reference_interval_map"] = "reference_interval_map.parquet"
    spine_path = tmp_path / "stage_outputs" / "spine.h5ad"
    return spine, spine_path


def _tasks():
    return pd.DataFrame(
        [
            {
                "task_id": "right-b",
                "reference": "ref_top",
                "core_start": 5,
                "core_end": 10,
                "barcode": "bc2",
                "group_path": "right-b.zarr",
                "hmm_model_ids": ["model-b"],
            },
            {
                "task_id": "left-a",
                "reference": "ref_top",
                "core_start": 0,
                "core_end": 5,
                "barcode": "bc1",
                "group_path": "left-a.zarr",
                "hmm_model_ids": ["model-a"],
            },
            {
                "task_id": "right-a",
                "reference": "ref_top",
                "core_start": 5,
                "core_end": 10,
                "barcode": "bc1",
                "group_path": "right-a.zarr",
                "hmm_model_ids": ["model-a"],
            },
            {
                "task_id": "left-b",
                "reference": "ref_top",
                "core_start": 0,
                "core_end": 5,
                "barcode": "bc2",
                "group_path": "left-b.zarr",
                "hmm_model_ids": ["model-b"],
            },
        ]
    )


def test_plot_region_spans_adjacent_cores_once_with_stable_provenance(tmp_path):
    spine, spine_path = _spine_with_plot_catalog(tmp_path)

    forward = resolve_plot_region_plans(spine, _tasks(), spine_path=spine_path)
    reversed_tasks = resolve_plot_region_plans(
        spine, _tasks().iloc[::-1].reset_index(drop=True), spine_path=spine_path
    )

    assert forward == reversed_tasks
    assert len(forward) == 1
    plan = forward[0]
    assert (plan.reference, plan.start, plan.end) == ("ref_top", 3, 8)
    assert plan.task_ids == ("left-a", "left-b", "right-a", "right-b")
    assert plan.model_ids == ("model-a", "model-b")
    assert plan.gaps == ()


def test_plot_region_gap_fails_by_default_and_can_be_labeled(tmp_path):
    spine, spine_path = _spine_with_plot_catalog(tmp_path, start=2, end=8)
    tasks = _tasks().loc[lambda frame: ~frame["task_id"].str.startswith("right")].copy()
    tasks.loc[:, "core_end"] = 4

    with pytest.raises(ValueError, match="unanalyzed gaps: 4-8"):
        resolve_plot_region_plans(spine, tasks, spine_path=spine_path)

    plan = resolve_plot_region_plans(spine, tasks, spine_path=spine_path, allow_gaps=True)[0]
    assert plan.gaps == ((4, 8),)

    adata = ad.AnnData(
        X=np.ones((1, 6), dtype=np.int8),
        layers={"signal": np.ones((1, 6), dtype=np.int8)},
    )
    adata.var_names = list(map(str, range(2, 8)))
    mask_unanalyzed_gaps(adata, plan.gaps)
    assert adata.var["plot_unanalyzed_gap"].tolist() == [False, False, True, True, True, True]
    assert np.isnan(adata.X[0, 2:]).all()
    assert np.isnan(adata.layers["signal"][0, 2:]).all()


def test_explicit_empty_plot_catalog_disables_fallback_regions(tmp_path):
    spine, spine_path = _spine_with_plot_catalog(tmp_path)
    catalog = tmp_path / "region_catalogs" / "plot_regions.parquet"
    pd.read_parquet(catalog).iloc[:0].to_parquet(catalog, index=False)
    fallback = pd.DataFrame([{"reference": "ref_top", "start": 0, "end": 10, "name": "fallback"}])

    plans = resolve_plot_region_plans(
        spine,
        _tasks(),
        spine_path=spine_path,
        fallback_regions=fallback,
    )

    assert plans == []


def test_plot_read_selection_is_invariant_to_task_row_order(tmp_path):
    def write_index(path, tasks):
        path.mkdir()
        rows = []
        for task_id, core_start, core_end in tasks:
            for row, (read_id, barcode) in enumerate(
                (("read3", "bc1"), ("read1", "bc1"), ("read4", "bc2"), ("read2", "bc2"))
            ):
                rows.append(
                    {
                        "read_id": read_id,
                        "molecule_uid": f"molecule-{read_id}",
                        "reference": "ref_top",
                        "barcode": barcode,
                        "core_start": core_start,
                        "core_end": core_end,
                        "group_path": f"{task_id}.zarr",
                        "group_row": row,
                        "task_id": task_id,
                    }
                )
        pd.DataFrame(rows).iloc[::-1].to_parquet(path / "part.parquet", index=False)

    index = tmp_path / "read_index"
    write_index(index, (("left", 0, 5), ("right", 5, 10)))
    rechunked_index = tmp_path / "rechunked_read_index"
    write_index(
        rechunked_index,
        (("left", 0, 3), ("middle", 3, 7), ("right", 7, 10)),
    )
    spine, spine_path = _spine_with_plot_catalog(tmp_path / "catalog")
    plan = resolve_plot_region_plans(spine, _tasks(), spine_path=spine_path)[0]

    selection = select_plot_reads(index, plan, max_reads_per_barcode=1, seed=7)
    rechunked_selection = select_plot_reads(rechunked_index, plan, max_reads_per_barcode=1, seed=7)

    assert len(selection.read_ids) == 2
    assert selection.read_ids == tuple(sorted(selection.read_ids))
    assert selection.selection_sha256
    assert rechunked_selection == selection
