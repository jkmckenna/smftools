import shutil

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.informatics.partition_store import write_dense_cache_from_spine
from smftools.informatics.raw_store import write_raw_store
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad


def _frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("read1", "ref1", "bc02", 4),
        ("read2", "ref1", "bc01", 4),
        ("read3", "ref2", "bc02", 6),
        ("read4", "ref2", "bc01", 6),
    ]
    for offset, (read_id, reference, barcode, length) in enumerate(specs):
        rows.append(
            {
                "read_id": read_id,
                "reference": reference,
                "Reference_strand": f"{reference}_top",
                "sample": barcode,
                "barcode": barcode,
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": f"{length}M",
                "aligned_length": length,
                "sequence": [offset % 4] * length,
                "quality": [30 + offset] * length,
                "mismatch": [4] * length,
                "modification_signal": [float(offset)] * length,
            }
        )
    return pd.DataFrame(rows)


def test_write_raw_store_creates_shards_and_thin_spine(tmp_path):
    paths = write_raw_store(
        _frame(),
        tmp_path,
        reference_lengths={"ref1_top": 4, "ref2_top": 6},
        shard_size=2,
        bam_path="/data/aligned.bam",
    )

    assert len(paths["ragged_store"]) == 2
    assert all(path.exists() for path in paths["ragged_store"])
    spine, _ = safe_read_h5ad(paths["spine"])
    assert spine.n_obs == 4
    assert spine.n_vars == 0
    assert not spine.layers
    assert list(spine.obs["ragged_row"]) == [0, 1, 0, 1]
    assert all(str(path).startswith("raw/reference=") for path in spine.obs["ragged_shard"])
    assert all("/start_bin=" in str(path) for path in spine.obs["ragged_shard"])
    assert set(spine.obs["Barcode"].astype(str)) == {"bc01", "bc02"}
    assert list(spine.obs["reference_end"]) == [4, 4, 6, 6]
    assert spine.uns["raw_schema_version"] == 2
    catalog = pd.read_parquet(paths["interval_catalog"])
    assert set(catalog["reference"]) == {"ref1_top", "ref2_top"}
    assert catalog["n_reads"].sum() == 4
    assert set(spine.uns["reference_plans"]) == {"ref1_top", "ref2_top"}


def test_dense_cache_matches_ragged_and_records_barcode_ranges(tmp_path):
    raw_paths = write_raw_store(
        _frame(),
        tmp_path,
        reference_lengths={"ref1_top": 4, "ref2_top": 6},
        shard_size=2,
    )
    ragged = materialize(raw_paths["spine"], references="ref2_top")

    cache_paths = write_dense_cache_from_spine(
        raw_paths["spine"], output_dir=tmp_path / "load_adata_outputs"
    )
    cached = materialize(cache_paths["spine"], references="ref2_top")
    cached = cached[list(ragged.obs_names)]

    np.testing.assert_array_equal(cached.X, ragged.X)
    for layer in ragged.layers:
        np.testing.assert_array_equal(cached.layers[layer], ragged.layers[layer])

    catalog = pd.read_parquet(cache_paths["catalog"])
    assert set(catalog["reference"]) == {"ref1_top", "ref2_top"}
    assert list(catalog.loc[catalog["reference"] == "ref1_top", "barcode"]) == [
        "bc01",
        "bc02",
    ]
    assert list(catalog.loc[catalog["reference"] == "ref1_top", "row_start"]) == [0, 1]
    assert list(catalog.loc[catalog["reference"] == "ref1_top", "row_end"]) == [1, 2]

    spine, _ = safe_read_h5ad(cache_paths["spine"])
    assert "partition" in spine.obs
    assert "partition_row" in spine.obs
    assert list(spine.uns["ragged_store"]) == [
        str(path.resolve()) for path in raw_paths["ragged_store"]
    ]
    assert all(path.exists() for path in raw_paths["ragged_store"])

    shutil.move(cache_paths["store"], tmp_path / "store-disabled")
    fallback = materialize(cache_paths["spine"], references="ref2_top")
    fallback = fallback[list(ragged.obs_names)]
    np.testing.assert_array_equal(fallback.X, ragged.X)


def test_interval_materialization_selects_overlaps_and_genomic_positions(tmp_path):
    frame = _frame()
    frame.loc[frame["read_id"] == "read1", "reference_start"] = 0
    frame.loc[frame["read_id"] == "read2", "reference_start"] = 5
    paths = write_raw_store(
        frame,
        tmp_path,
        reference_lengths={"ref1_top": 12, "ref2_top": 6},
        shard_size=2,
        start_bin_size=4,
    )

    interval = materialize(paths["spine"], references="ref1_top", start=3, end=7)

    assert list(interval.obs_names) == ["read1", "read2"]
    assert list(interval.var_names) == ["3", "4", "5", "6"]
    np.testing.assert_array_equal(
        interval.layers["read_span_mask"],
        np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=np.int8),
    )


def test_genome_plan_writes_and_reads_tiled_dense_cache(tmp_path):
    frame = _frame().loc[lambda value: value["reference"] == "ref1"].copy()
    frame.loc[frame["read_id"] == "read2", "reference_start"] = 5
    raw_paths = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref1_top": 12},
        analysis_mode="genome",
        genome_tile_size=4,
        genome_tile_halo=2,
    )
    cache_paths = write_dense_cache_from_spine(
        raw_paths["spine"], output_dir=tmp_path / "load_adata_outputs"
    )
    catalog = pd.read_parquet(cache_paths["catalog"])

    assert set(catalog["cache_kind"]) == {"tiled"}
    assert list(catalog["core_start"]) == [0, 4, 8]
    spine, _ = safe_read_h5ad(cache_paths["spine"])
    assert (spine.obs["partition"].astype(str) == "").all()

    shutil.rmtree(tmp_path / "raw_outputs" / "raw")
    interval = materialize(cache_paths["spine"], references="ref1_top", start=4, end=7)
    assert list(interval.obs_names) == ["read2"]
    assert list(interval.var_names) == ["4", "5", "6"]


def test_genome_plan_rejects_unbounded_materialization(tmp_path):
    paths = write_raw_store(
        _frame(),
        tmp_path,
        reference_lengths={"ref1_top": 4, "ref2_top": 6},
        analysis_mode="genome",
    )
    with pytest.raises(ValueError, match="require start/end"):
        materialize(paths["spine"], references="ref1_top")
