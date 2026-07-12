import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize
from smftools.informatics.partition_store import write_experiment_store

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
