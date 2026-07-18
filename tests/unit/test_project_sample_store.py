import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_store import write_experiment_store
from smftools.project.sample_store import (
    backfill_per_sample_store,
    list_per_sample_partitions,
    load_per_sample_partition,
)
from smftools.readwrite import safe_write_h5ad


def _modern_spine(tmp_path, *, n_refs=2, n_samples=2, per=5, n_pos=6, seed=1):
    rng = np.random.default_rng(seed)
    n = n_refs * n_samples * per
    x = rng.integers(0, 2, (n, n_pos)).astype(np.float32)
    refs = np.repeat([f"ref{r}_top" for r in range(n_refs)], n_samples * per)
    samples = np.tile(np.repeat([f"bc{s:02d}" for s in range(n_samples)], per), n_refs)
    obs = pd.DataFrame(
        {"Reference_strand": pd.Categorical(refs), "Sample": pd.Categorical(samples)},
        index=[f"read{i:05d}" for i in range(n)],
    )
    a = ad.AnnData(X=x, obs=obs)
    a.var_names = [str(p) for p in range(n_pos)]
    write_experiment_store(a, tmp_path, experiment="e", modality="conversion")
    return tmp_path / "spine.h5ad"


def _legacy_spine(tmp_path, *, filename="legacy.h5ad"):
    obs = pd.DataFrame(
        {"Reference_strand": ["ref0_top"] * 3, "Sample": ["bc00"] * 3},
        index=[f"r{i}" for i in range(3)],
    )
    a = ad.AnnData(X=np.zeros((3, 4), dtype=np.float32), obs=obs)
    path = tmp_path / filename
    safe_write_h5ad(a, path, backup=False, verbose=False)
    assert "is_spine" not in a.uns
    return path


def test_backfill_per_sample_store_catalogs_modern_spine(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_spine(tmp_path / "run")

    written = backfill_per_sample_store(project_dir, "expA", spine_path)

    # 2 refs x 2 samples = 4 partitions, 5 reads each.
    assert len(written) == 4
    partitions = list_per_sample_partitions(project_dir, "expA")
    assert len(partitions) == 4
    by_key = {(p["reference_strand"], p["sample"]): p for p in partitions}
    assert by_key[("ref0_top", "bc00")]["n_reads"] == 5
    assert by_key[("ref1_top", "bc01")]["n_reads"] == 5
    for p in partitions:
        assert p["experiment_id"] == "expA"
        assert p["kind"] == "pointer"


def test_backfill_per_sample_store_caches_legacy_spine(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _legacy_spine(tmp_path / "run")
    before_bytes = spine_path.read_bytes()

    written = backfill_per_sample_store(project_dir, "expLegacy", spine_path)

    assert len(written) == 1
    partitions = list_per_sample_partitions(project_dir, "expLegacy")
    assert partitions == [
        {
            "kind": "cache",
            "experiment_id": "expLegacy",
            "reference_strand": "ref0_top",
            "sample": "bc00",
            "n_reads": 3,
            "cache_path": "cache.h5ad",
        }
    ]

    loaded = load_per_sample_partition(project_dir, "expLegacy", "ref0_top", "bc00")
    assert loaded.n_obs == 3
    assert set(loaded.obs_names) == {"r0", "r1", "r2"}

    # Source legacy file is only ever read, never mutated.
    assert spine_path.read_bytes() == before_bytes


def test_backfill_per_sample_store_legacy_cache_matches_source_data(tmp_path):
    obs = pd.DataFrame(
        {
            "Reference_strand": ["ref0_top", "ref0_top", "ref1_top"],
            "Sample": ["bc00", "bc00", "bc01"],
        },
        index=["r0", "r1", "r2"],
    )
    a = ad.AnnData(X=np.arange(6, dtype=np.float32).reshape(3, 2), obs=obs)
    path = tmp_path / "run" / "legacy.h5ad"
    path.parent.mkdir(parents=True)
    safe_write_h5ad(a, path, backup=False, verbose=False)

    project_dir = tmp_path / "project"
    backfill_per_sample_store(project_dir, "expLegacy", path)

    loaded = load_per_sample_partition(project_dir, "expLegacy", "ref0_top", "bc00")
    expected = a[["r0", "r1"]]
    assert np.array_equal(np.asarray(loaded[["r0", "r1"]].X), np.asarray(expected.X))


def test_load_per_sample_partition_rejects_pointer_kind(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_spine(tmp_path / "run")
    backfill_per_sample_store(project_dir, "expA", spine_path)

    with pytest.raises(ValueError, match="pointer"):
        load_per_sample_partition(project_dir, "expA", "ref0_top", "bc00")


def test_load_per_sample_partition_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_per_sample_partition(tmp_path / "project", "nope", "ref0_top", "bc00")


def test_backfill_per_sample_store_skips_spine_missing_partition_columns(tmp_path):
    project_dir = tmp_path / "project"
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32), obs=pd.DataFrame(index=["a", "b", "c"]))
    a.uns["is_spine"] = True
    path = tmp_path / "run" / "spine.h5ad"
    path.parent.mkdir(parents=True)
    safe_write_h5ad(a, path, backup=False, verbose=False)

    written = backfill_per_sample_store(project_dir, "expNoCols", path)

    assert written == []


def test_list_per_sample_partitions_filters_by_experiment(tmp_path):
    project_dir = tmp_path / "project"
    spine_a = _modern_spine(tmp_path / "runA", seed=1)
    spine_b = _modern_spine(tmp_path / "runB", seed=2)

    backfill_per_sample_store(project_dir, "expA", spine_a)
    backfill_per_sample_store(project_dir, "expB", spine_b)

    assert len(list_per_sample_partitions(project_dir)) == 8
    assert len(list_per_sample_partitions(project_dir, "expA")) == 4
    assert len(list_per_sample_partitions(project_dir, "expB")) == 4
    assert {p["experiment_id"] for p in list_per_sample_partitions(project_dir, "expA")} == {"expA"}


def test_backfill_per_sample_store_overwrites_counts_on_rerun(tmp_path):
    project_dir = tmp_path / "project"
    spine_path = _modern_spine(tmp_path / "run", per=5)
    backfill_per_sample_store(project_dir, "expA", spine_path)

    # Re-backfill from an updated spine with a different read count for the same
    # partitions -- counts should reflect the new state, not accumulate/duplicate.
    updated_path = _modern_spine(tmp_path / "run2", per=9, seed=1)
    written = backfill_per_sample_store(project_dir, "expA", updated_path)

    assert len(written) == 4
    partitions = list_per_sample_partitions(project_dir, "expA")
    assert len(partitions) == 4
    assert all(p["n_reads"] == 9 for p in partitions)
