import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.informatics.partition_store import write_experiment_store
from smftools.informatics.ragged_store import write_ragged_parquet
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

LAYERS = ("sequence_integer_encoding", "read_span_mask")


def _synth(n_refs=3, n_samples=2, per=25, n_pos=18, seed=3):
    rng = np.random.default_rng(seed)
    n = n_refs * n_samples * per
    x = rng.integers(0, 2, (n, n_pos)).astype(np.float32)
    x[rng.random((n, n_pos)) < 0.5] = np.nan
    layers = {
        "sequence_integer_encoding": rng.integers(0, 6, (n, n_pos), dtype=np.int8),
        "read_span_mask": rng.integers(0, 2, (n, n_pos), dtype=np.int8),
    }
    refs = np.repeat([f"ref{r}_top" for r in range(n_refs)], n_samples * per)
    samples = np.tile(np.repeat([f"bc{s:02d}" for s in range(n_samples)], per), n_refs)
    obs = pd.DataFrame(
        {"Reference_strand": pd.Categorical(refs), "Sample": pd.Categorical(samples)},
        index=[f"read{i:05d}" for i in range(n)],
    )
    a = ad.AnnData(X=x, obs=obs, layers=layers)
    a.var_names = [str(p) for p in range(n_pos)]
    a.uns["sequence_integer_encoding_map"] = {"A": 0, "C": 1}
    return a


@pytest.fixture
def store(tmp_path):
    a = _synth()
    write_experiment_store(a, tmp_path, experiment="e", modality="conversion")
    return a, tmp_path / "spine.h5ad"


def _expect(a, mask):
    sub = a[mask]
    return sub


@pytest.mark.parametrize("lazy", [False, True])
def test_materialize_by_reference(store, lazy):
    a, spine_path = store
    got = materialize(spine_path, references="ref1_top", lazy=lazy)
    exp = a[a.obs["Reference_strand"].astype(str) == "ref1_top"]
    assert set(got.obs_names) == set(exp.obs_names)
    # align order and compare
    got = got[list(exp.obs_names)]
    assert np.array_equal(np.asarray(got.X), np.asarray(exp.X), equal_nan=True)
    for layer in LAYERS:
        assert np.array_equal(np.asarray(got.layers[layer]), np.asarray(exp.layers[layer]))


def test_materialize_by_sample_and_mask(store):
    a, spine_path = store
    got = materialize(spine_path, samples="bc00")
    assert set(got.obs_names) == set(a.obs_names[a.obs["Sample"].astype(str) == "bc00"])

    mask = (a.obs["Reference_strand"].astype(str) == "ref0_top").to_numpy()
    got2 = materialize(spine_path, obs_mask=mask)
    assert set(got2.obs_names) == set(a.obs_names[mask])


def test_materialize_by_read_ids_preserves_spine_order(store):
    a, spine_path = store
    ids = [str(a.obs_names[i]) for i in (140, 5, 130, 77)]
    got = materialize(spine_path, read_ids=ids)
    # result is ordered by spine order, not the requested-id order
    assert list(got.obs_names) == sorted(ids)  # spine index is read00000.. sorted


def test_materialize_layers_filter(store):
    a, spine_path = store
    got = materialize(spine_path, references="ref2_top", layers=["read_span_mask"])
    assert set(got.layers.keys()) == {"read_span_mask"}


def test_materialize_lazy_matches_eager(store):
    a, spine_path = store
    eager = materialize(spine_path, references="ref0_top", lazy=False)
    lazy = materialize(spine_path, references="ref0_top", lazy=True)
    lazy = lazy[list(eager.obs_names)]
    assert np.array_equal(np.asarray(eager.X), np.asarray(lazy.X), equal_nan=True)
    for layer in LAYERS:
        assert np.array_equal(np.asarray(eager.layers[layer]), np.asarray(lazy.layers[layer]))


def test_materialize_carries_uns_maps(store):
    a, spine_path = store
    got = materialize(spine_path, samples="bc01")
    assert "sequence_integer_encoding_map" in got.uns


def test_materialize_empty_selection_raises(store):
    a, spine_path = store
    with pytest.raises(ValueError):
        materialize(spine_path, references="does_not_exist")


def test_materialize_falls_back_to_ragged_when_dense_cache_is_missing(tmp_path):
    source = _synth(n_refs=1, n_samples=1, per=5, n_pos=8)
    source.layers["read_span_mask"][:] = 1
    paths = write_experiment_store(source, tmp_path, experiment="e", modality="conversion")

    rows = []
    for read_id in source.obs_names:
        row = source.obs_names.get_loc(read_id)
        rows.append(
            {
                "read_id": str(read_id),
                "reference": "ref0",
                "Reference_strand": "ref0_top",
                "reference_start": 0,
                "cigar": "8M",
                "aligned_length": 8,
                "sequence": source.layers["sequence_integer_encoding"][row].tolist(),
                "modification_signal": np.asarray(source.X[row]).ravel().tolist(),
            }
        )
    ragged_path = write_ragged_parquet(pd.DataFrame(rows), tmp_path / "raw" / "reads.parquet")

    spine, _ = safe_read_h5ad(paths["spine"])
    spine.uns["ragged_store"] = str(ragged_path.relative_to(tmp_path))
    spine.uns["reference_lengths"] = {"ref0_top": 8}
    safe_write_h5ad(spine, paths["spine"], verbose=False)

    dense = materialize(paths["spine"], references="ref0_top")
    shutil.rmtree(paths["store"])
    ragged = materialize(paths["spine"], references="ref0_top")

    assert list(ragged.obs_names) == list(dense.obs_names)
    np.testing.assert_array_equal(ragged.X, dense.X)
    for layer in ("sequence_integer_encoding", "read_span_mask"):
        np.testing.assert_array_equal(ragged.layers[layer], dense.layers[layer])


@pytest.fixture
def legacy_spine_path(tmp_path):
    """A plain monolithic AnnData with no uns["is_spine"] -- the pre-partitioned-
    store shape (X + layers fully resident, no partition/ragged-store pointers)."""
    a = _synth()
    a.obs["reference_start"] = 0
    a.obs["reference_end"] = a.shape[1]
    path = tmp_path / "legacy_preprocessed.h5ad"
    safe_write_h5ad(a, path, backup=False, verbose=False)
    assert "is_spine" not in a.uns
    return a, path


def test_materialize_legacy_spine_detected_and_sliced_by_reference(legacy_spine_path):
    a, path = legacy_spine_path
    got = materialize(path, references="ref1_top")
    exp = a[a.obs["Reference_strand"].astype(str) == "ref1_top"]
    assert set(got.obs_names) == set(exp.obs_names)
    got = got[list(exp.obs_names)]
    assert np.array_equal(np.asarray(got.X), np.asarray(exp.X), equal_nan=True)
    for layer in LAYERS:
        assert np.array_equal(np.asarray(got.layers[layer]), np.asarray(exp.layers[layer]))


def test_materialize_legacy_spine_by_sample_and_read_ids(legacy_spine_path):
    a, path = legacy_spine_path
    got = materialize(path, samples="bc00")
    assert set(got.obs_names) == set(a.obs_names[a.obs["Sample"].astype(str) == "bc00"])

    ids = [str(a.obs_names[i]) for i in (5, 3)]
    got2 = materialize(path, read_ids=ids)
    assert set(got2.obs_names) == set(ids)


def test_materialize_legacy_spine_layers_filter(legacy_spine_path):
    _a, path = legacy_spine_path
    got = materialize(path, references="ref2_top", layers=["read_span_mask"])
    assert set(got.layers.keys()) == {"read_span_mask"}


def test_materialize_legacy_spine_genomic_window(legacy_spine_path):
    a, path = legacy_spine_path
    got = materialize(path, references="ref0_top", start=3, end=7)
    assert list(got.var_names) == ["3", "4", "5", "6"]
    exp = a[a.obs["Reference_strand"].astype(str) == "ref0_top", 3:7]
    assert np.array_equal(np.asarray(got[list(exp.obs_names)].X), np.asarray(exp.X), equal_nan=True)


def test_materialize_legacy_spine_in_memory_needs_no_base_dir(legacy_spine_path):
    """base_dir only matters for partition resolution; a legacy in-memory spine
    (no uns["is_spine"]) has nothing to resolve, so it must not raise even
    though the pre-existing in-memory-spine error path normally requires it."""
    a, _path = legacy_spine_path
    got = materialize(a, references="ref1_top")
    assert set(got.obs_names) == set(
        a.obs_names[a.obs["Reference_strand"].astype(str) == "ref1_top"]
    )


def test_materialize_legacy_spine_empty_selection_raises(legacy_spine_path):
    _a, path = legacy_spine_path
    with pytest.raises(ValueError):
        materialize(path, references="does_not_exist")
