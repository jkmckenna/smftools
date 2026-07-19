import shutil

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import (
    materialize,
    relative_uns_path,
    resolve_relative_path,
)
from smftools.informatics.partition_store import write_dense_cache_from_spine
from smftools.informatics.ragged_store import validate_ragged_frame
from smftools.informatics.raw_store import (
    write_raw_store,
    write_raw_store_streaming,
)
from smftools.informatics.stage_obs import read_stage_obs
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
    # bam_outputs/ living alongside the raw store, matching real layout
    # (raw_outputs/bam_outputs/...), so the relative bam_path encoding below has
    # a real file to resolve against.
    bam_path = tmp_path / "bam_outputs" / "aligned.bam"
    bam_path.parent.mkdir(parents=True)
    bam_path.touch()
    paths = write_raw_store(
        _frame(),
        tmp_path,
        reference_lengths={"ref1_top": 4, "ref2_top": 6},
        shard_size=2,
        bam_path=bam_path,
    )

    assert len(paths["ragged_store"]) == 2
    assert all(path.exists() for path in paths["ragged_store"])
    spine, _ = safe_read_h5ad(paths["spine"])
    assert spine.n_obs == 4
    assert spine.n_vars == 0
    assert not spine.layers
    # Shards are now Sample-sorted (not just reference_start-sorted), so within
    # each reference's shard the physical write order is bc01 before bc02:
    # ref1_top -> [read2(bc01), read1(bc02)], ref2_top -> [read4(bc01), read3(bc02)].
    # spine.obs is still in the original frame order [read1, read2, read3, read4],
    # so their ragged_row values (position within their own shard) are [1, 0, 1, 0].
    assert list(spine.obs["ragged_row"]) == [1, 0, 1, 0]
    assert all(str(path).startswith("raw/reference=") for path in spine.obs["ragged_shard"])
    assert all("/start_bin=" in str(path) for path in spine.obs["ragged_shard"])
    assert set(spine.obs["Barcode"].astype(str)) == {"bc01", "bc02"}
    assert list(spine.obs["reference_end"]) == [4, 4, 6, 6]
    assert spine.uns["raw_schema_version"] == 2
    catalog = pd.read_parquet(paths["interval_catalog"])
    assert set(catalog["reference"]) == {"ref1_top", "ref2_top"}
    assert catalog["n_reads"].sum() == 4
    assert set(spine.uns["reference_plans"]) == {"ref1_top", "ref2_top"}

    # bam_path is stored relative to the run root (tmp_path.parent here, since
    # write_raw_store's output_dir *is* the raw store dir), not absolute, so it
    # stays resolvable after the containing tree is moved to a different machine.
    run_root = tmp_path.parent
    stored_bam_path = set(spine.obs["bam_path"])
    assert stored_bam_path == {relative_uns_path(bam_path, run_root)}
    resolved = resolve_relative_path(next(iter(stored_bam_path)), run_root)
    assert resolved == bam_path.resolve()
    assert resolved.exists()

    # molecules.parquet: canonical read-id catalog, one row per read, in the exact
    # Sample-sorted physical write order (bc01 before bc02 within each reference).
    assert paths["molecules"] == run_root / "molecules.parquet"
    molecules = pd.read_parquet(paths["molecules"])
    assert list(molecules["read_id"]) == ["read2", "read1", "read4", "read3"]
    assert list(molecules["canonical_row"]) == [0, 1, 2, 3]
    assert list(molecules["Reference_strand"]) == ["ref1_top", "ref1_top", "ref2_top", "ref2_top"]
    assert list(molecules["Sample"]) == ["bc01", "bc02", "bc01", "bc02"]
    assert spine.uns["molecules_catalog"] == relative_uns_path(paths["molecules"], run_root)

    # barcode_index.parquet: contiguous (start_row, end_row) per sample within each
    # reference's shard -- a direct row-slice instead of a scan + boolean mask.
    assert paths["barcode_index"] == tmp_path / "barcode_index.parquet"
    barcode_index = pd.read_parquet(paths["barcode_index"])
    ref1_ranges = barcode_index.loc[barcode_index["reference"] == "ref1_top"].set_index("sample")
    assert (ref1_ranges.loc["bc01", ["start_row", "end_row"]] == [0, 1]).all()
    assert (ref1_ranges.loc["bc02", ["start_row", "end_row"]] == [1, 2]).all()
    ref2_ranges = barcode_index.loc[barcode_index["reference"] == "ref2_top"].set_index("sample")
    assert (ref2_ranges.loc["bc01", ["start_row", "end_row"]] == [0, 1]).all()
    assert (ref2_ranges.loc["bc02", ["start_row", "end_row"]] == [1, 2]).all()
    assert spine.uns["barcode_index"] == "barcode_index.parquet"

    # obs.parquet: full obs (raw has no earlier stage to normalize against),
    # written alongside spine.h5ad -- same content, different format.
    assert paths["obs"] == tmp_path / "obs.parquet"
    obs_parquet = read_stage_obs(tmp_path)
    assert list(obs_parquet.index) == list(spine.obs_names)
    assert set(obs_parquet.columns) == set(spine.obs.columns)
    assert list(obs_parquet["bam_path"]) == list(spine.obs["bam_path"])


def test_write_raw_store_skips_barcode_artifacts_without_sample_column(tmp_path):
    frame = _frame().drop(columns=["sample"])
    paths = write_raw_store(frame, tmp_path, reference_lengths={"ref1_top": 4, "ref2_top": 6})

    assert paths["barcode_index"] is None
    molecules = pd.read_parquet(paths["molecules"])
    assert "Sample" not in molecules.columns
    assert set(molecules["read_id"]) == {"read1", "read2", "read3", "read4"}
    spine, _ = safe_read_h5ad(paths["spine"])
    assert "barcode_index" not in spine.uns


def test_write_raw_store_streaming_matches_whole_frame_writer(tmp_path):
    """write_raw_store_streaming, fed one reference's frame at a time, must
    produce byte-identical shard contents/catalog/molecules/barcode_index to
    write_raw_store fed the exact same rows all at once -- the mathematical
    equivalence _write_raw_shards_streaming's docstring claims (grouping by
    Reference_strand first is equivalent to one global stable sort).
    """
    frame = _frame()
    reference_lengths = {"ref1_top": 4, "ref2_top": 6}

    whole = tmp_path / "whole"
    whole.mkdir()
    whole_paths = write_raw_store(frame, whole, reference_lengths=reference_lengths, shard_size=2)

    streaming = tmp_path / "streaming"
    streaming.mkdir()
    normalized = validate_ragged_frame(frame).reset_index(drop=True)
    reference_groups = (
        (str(name), group, True)
        for name, group in normalized.groupby("Reference_strand", sort=True, observed=True)
    )
    streaming_paths = write_raw_store_streaming(
        reference_groups, streaming, reference_lengths=reference_lengths, shard_size=2
    )

    # Shard contents: same set of shard files, same rows in each (aligned by
    # read_id -- shard *paths* are relative to each store's own root, so only
    # their relative suffix under raw/ needs to match, not the absolute path).
    whole_shards = {p.relative_to(whole).as_posix(): p for p in whole_paths["ragged_store"]}
    streaming_shards = {
        p.relative_to(streaming).as_posix(): p for p in streaming_paths["ragged_store"]
    }
    assert set(whole_shards) == set(streaming_shards)
    for relative_path, whole_shard_path in whole_shards.items():
        whole_shard = pd.read_parquet(whole_shard_path).set_index("read_id")
        streaming_shard = pd.read_parquet(streaming_shards[relative_path]).set_index("read_id")
        pd.testing.assert_frame_equal(
            whole_shard.sort_index(), streaming_shard.sort_index(), check_like=True
        )

    # Catalogs: same content (order can differ trivially by float formatting,
    # so sort before comparing).
    whole_catalog = (
        pd.read_parquet(whole_paths["interval_catalog"])
        .sort_values(["reference", "start_bin"])
        .reset_index(drop=True)
    )
    streaming_catalog = (
        pd.read_parquet(streaming_paths["interval_catalog"])
        .sort_values(["reference", "start_bin"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(whole_catalog, streaming_catalog, check_like=True)

    # molecules.parquet: same (read_id -> canonical_row) bijection and the
    # same Reference_strand/Sample per read -- canonical_row *values* are
    # guaranteed identical by the grouping-equivalence proof.
    whole_molecules = pd.read_parquet(whole_paths["molecules"]).set_index("read_id")
    streaming_molecules = pd.read_parquet(streaming_paths["molecules"]).set_index("read_id")
    pd.testing.assert_frame_equal(
        whole_molecules.sort_index(), streaming_molecules.sort_index(), check_like=True
    )

    # barcode_index.parquet: same contiguous-range rows.
    whole_bidx = (
        pd.read_parquet(whole_paths["barcode_index"])
        .sort_values(["reference", "start_bin", "sample"])
        .reset_index(drop=True)
    )
    streaming_bidx = (
        pd.read_parquet(streaming_paths["barcode_index"])
        .sort_values(["reference", "start_bin", "sample"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(whole_bidx, streaming_bidx, check_like=True)

    # spine.obs: same rows/columns/values, aligned by read_id -- row order is
    # allowed to differ (reference-arrival order for streaming vs. original
    # frame order for the whole-frame writer), documented in
    # write_raw_store_streaming's docstring.
    whole_spine, _ = safe_read_h5ad(whole_paths["spine"])
    streaming_spine, _ = safe_read_h5ad(streaming_paths["spine"])
    assert set(whole_spine.obs_names) == set(streaming_spine.obs_names)
    common = list(whole_spine.obs_names)
    pd.testing.assert_frame_equal(
        whole_spine.obs.loc[common], streaming_spine.obs.loc[common], check_like=True
    )
    assert whole_spine.uns["reference_lengths"] == streaming_spine.uns["reference_lengths"]
    assert set(whole_spine.uns["reference_plans"]) == set(streaming_spine.uns["reference_plans"])


def test_write_raw_store_streaming_handles_reference_split_across_multiple_groups(tmp_path):
    # Regression test for the persistent shard-index fix: a bounded-memory
    # producer (cli/raw_adata.py's _ChromosomeGroupAccumulator flush
    # threshold) can now flush a single reference's rows across more than
    # one group instead of handing the whole reference over at once. Before
    # persistent per-(reference, start_bin) shard-index tracking, each new
    # group for the same reference restarted numbering at shard_index=0 and
    # silently overwrote the previous group's shard file on disk -- this
    # confirms all rows from every flush survive and plan_references still
    # sees the reference's true total read count (not just the last group's).
    frame = _frame()
    normalized = validate_ragged_frame(frame).reset_index(drop=True)
    ref1_rows = normalized[normalized["Reference_strand"] == "ref1_top"]
    ref2_rows = normalized[normalized["Reference_strand"] == "ref2_top"]
    assert len(ref1_rows) == 2

    # ref1_top arrives in two separate, bounded groups (mimicking a flush
    # crossing raw_shard_flush_max_reads mid-reference); ref2_top arrives
    # whole. Both groups' shard writes land in the same start_bin (shard_size
    # is large enough here that each group is its own single shard).
    reference_groups = [
        ("ref1_top", ref1_rows.iloc[[0]], False),
        ("ref1_top", ref1_rows.iloc[[1]], True),
        ("ref2_top", ref2_rows, True),
    ]

    streaming = tmp_path / "streaming"
    streaming.mkdir()
    paths = write_raw_store_streaming(
        reference_groups,
        streaming,
        reference_lengths={"ref1_top": 4, "ref2_top": 6},
        shard_size=100,
    )

    # Both of ref1_top's shard files exist (no overwrite) and together
    # contain both of its reads.
    ref1_shards = [p for p in paths["ragged_store"] if "reference=ref1_top" in p.as_posix()]
    assert len(ref1_shards) == 2
    ref1_read_ids = set()
    for shard_path in ref1_shards:
        ref1_read_ids.update(pd.read_parquet(shard_path)["read_id"])
    assert ref1_read_ids == {"read1", "read2"}

    catalog = pd.read_parquet(paths["interval_catalog"])
    ref1_catalog = catalog[catalog["reference"] == "ref1_top"]
    assert ref1_catalog["n_reads"].sum() == 2

    molecules = pd.read_parquet(paths["molecules"])
    assert set(molecules.loc[molecules["Reference_strand"] == "ref1_top", "read_id"]) == {
        "read1",
        "read2",
    }

    spine, _ = safe_read_h5ad(paths["spine"])
    assert spine.n_obs == 4
    # plan_references saw ref1_top's true total (2 reads across both
    # groups), not just the last group's (1 read).
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
