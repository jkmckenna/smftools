import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.artifact_paths import (
    resolve_artifact_path,
    serialize_artifact_path,
)
from smftools.informatics.partition_query import (
    query_batch_rows,
    query_derived_index,
    query_molecule_index,
    read_zarr_subset,
)
from smftools.informatics.partition_read import materialize
from smftools.informatics.physical_layout import (
    portable_matrix_chunks,
    portable_parquet_row_group_rows,
)


def test_query_molecule_index_pushes_all_selection_dimensions(tmp_path):
    index = tmp_path / "molecule_index"
    index.mkdir()
    pd.DataFrame(
        {
            "read_id": ["r1", "r2", "r3"],
            "molecule_uid": ["m1", "m2", "m3"],
            "Reference_strand": ["ref_top", "ref_top", "other_top"],
            "Sample": ["s1", "s2", "s1"],
            "Barcode": ["bc1", "bc2", "bc1"],
            "reference_start": [0, 20, 0],
            "reference_end": [10, 30, 10],
            "canonical_row": [0, 1, 2],
        }
    ).to_parquet(index / "part.parquet", index=False)

    result = query_molecule_index(
        index,
        references="ref_top",
        samples="s1",
        barcodes="bc1",
        read_ids=["r1", "r2"],
        molecule_uids=["m1", "m3"],
        start=5,
        end=15,
    )

    assert result["read_id"].tolist() == ["r1"]


def test_query_derived_index_filters_tasks_and_deduplicates_model_rows(tmp_path):
    index = tmp_path / "read_index"
    index.mkdir()
    pd.DataFrame(
        {
            "read_id": ["r1", "r1", "r2"],
            "molecule_uid": ["m1", "m1", "m2"],
            "reference": ["ref_top"] * 3,
            "barcode": ["bc1", "bc1", "bc2"],
            "core_start": [0, 0, 20],
            "core_end": [10, 10, 30],
            "group_path": ["task.zarr", "task.zarr", "other.zarr"],
            "group_row": [0, 0, 0],
            "task_id": ["task", "task", "other"],
            "model_id": ["a", "b", None],
        }
    ).to_parquet(index / "part.parquet", index=False)

    result = query_derived_index(
        index,
        references="ref_top",
        barcodes="bc1",
        read_ids="r1",
        start=2,
        end=8,
    )

    assert result[["read_id", "group_path", "group_row"]].to_dict("records") == [
        {"read_id": "r1", "group_path": "task.zarr", "group_row": 0}
    ]


def test_read_zarr_subset_projects_before_each_to_memory(tmp_path, monkeypatch):
    path = tmp_path / "part.zarr"
    source = ad.AnnData(
        X=np.arange(60, dtype=np.float32).reshape(6, 10),
        obs=pd.DataFrame(index=[f"r{i}" for i in range(6)]),
        layers={
            "keep": np.ones((6, 10), dtype=np.int8),
            "drop": np.zeros((6, 10), dtype=np.int8),
        },
    )
    source.var_names = list(map(str, range(10)))
    source.write_zarr(path)

    materialized_shapes = []
    original = ad.AnnData.to_memory

    def record_to_memory(self, *args, **kwargs):
        materialized_shapes.append((self.shape, tuple(self.layers)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "to_memory", record_to_memory)
    result = read_zarr_subset(
        path,
        row_indices=[1, 4],
        start=3,
        end=7,
        layers=["keep"],
        lazy=True,
        memory_mb=1,
    )

    assert result.shape == (2, 4)
    assert set(result.layers) == {"keep"}
    assert materialized_shapes == [((2, 4), ("keep",))]


def test_read_zarr_subset_lazy_request_falls_back_without_xarray(tmp_path, monkeypatch):
    path = tmp_path / "part.zarr"
    source = ad.AnnData(
        X=np.arange(20, dtype=np.float32).reshape(4, 5),
        obs=pd.DataFrame(index=[f"r{i}" for i in range(4)]),
    )
    source.var_names = list(map(str, range(5)))
    source.write_zarr(path)

    def missing_xarray(*args, **kwargs):
        raise ImportError("xarray is required to read dataframes lazily")

    monkeypatch.setattr(ad.experimental, "read_lazy", missing_xarray)
    result = read_zarr_subset(
        path,
        read_ids=["r1", "r3"],
        start=1,
        end=4,
        lazy=True,
    )

    assert list(result.obs_names) == ["r1", "r3"]
    assert list(result.var_names) == ["1", "2", "3"]


def test_query_batch_rows_respects_projection_width():
    assert query_batch_rows(1_000, n_arrays=2, memory_mb=1) == 65
    assert query_batch_rows(1_000_000, n_arrays=10, memory_mb=1) == 1


def test_portable_chunks_bound_rows_columns_and_bytes():
    chunks = portable_matrix_chunks((100_000, 1_000_000), np.float32)

    assert chunks == (2_048, 2_048)
    assert np.prod(chunks) * np.dtype(np.float32).itemsize <= 16 * 1024**2


def test_portable_parquet_rows_are_independent_of_logical_frame_size():
    frame = pd.DataFrame({"value": np.arange(1_000_000, dtype=np.int64)})

    rows = portable_parquet_row_group_rows(frame, target_bytes=1024, max_rows=10_000)

    assert rows == 128


def test_artifact_paths_are_relative_and_resolvable(tmp_path):
    anchor = tmp_path / "run"
    artifact = anchor / "stage" / "catalog.parquet"
    stored = serialize_artifact_path(artifact, anchor)

    assert stored == "stage/catalog.parquet"
    assert resolve_artifact_path(stored, anchor) == artifact.resolve()


def test_artifact_path_uses_absolute_cross_volume_fallback(tmp_path, monkeypatch):
    from smftools.informatics import artifact_paths

    artifact = tmp_path / "artifact"

    def cross_volume(*args, **kwargs):
        raise ValueError("different drive")

    monkeypatch.setattr(artifact_paths.os.path, "relpath", cross_volume)

    assert serialize_artifact_path(artifact, tmp_path) == artifact.resolve().as_posix()


def test_empty_index_query_short_circuits_before_spine_load(tmp_path, monkeypatch):
    from smftools.informatics import partition_read

    spine_path = tmp_path / "raw_outputs" / "spine.h5ad"
    spine_path.parent.mkdir()
    (tmp_path / "molecule_index").mkdir()
    monkeypatch.setattr(
        partition_read,
        "query_molecule_index",
        lambda *args, **kwargs: pd.DataFrame({"read_id": pd.Series(dtype=str)}),
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("spine should not be opened for an empty indexed selection")

    monkeypatch.setattr(partition_read, "load_spine", fail_load)

    with pytest.raises(ValueError, match="selection matched no molecules"):
        materialize(spine_path, references="missing")
