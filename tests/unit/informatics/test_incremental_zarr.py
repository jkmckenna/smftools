import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.incremental_zarr import (
    append_zarr_layer,
    append_zarr_obsm,
    consolidate_zarr_store,
)
from smftools.readwrite import safe_read_zarr, safe_write_zarr


def _skeleton(tmp_path, n_obs=4, n_vars=3):
    path = tmp_path / "store.zarr"
    obs = pd.DataFrame(index=[f"read{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[str(i) for i in range(n_vars)])
    x = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    safe_write_zarr(adata, path, backup=False, verbose=False, zarr_format=3)
    return path, adata


def test_append_zarr_layer_round_trips(tmp_path):
    path, adata = _skeleton(tmp_path)
    layer = (np.arange(adata.n_obs * adata.n_vars, dtype=np.float32) / 2.0).reshape(
        adata.n_obs, adata.n_vars
    )

    append_zarr_layer(path, "nan_half", layer)

    result, _ = safe_read_zarr(path, verbose=False)
    np.testing.assert_array_equal(np.asarray(result.X), np.asarray(adata.X))
    assert set(result.layers) == {"nan_half"}
    np.testing.assert_array_equal(np.asarray(result.layers["nan_half"]), layer)


def test_append_zarr_layer_consolidate_once_at_end(tmp_path):
    path, adata = _skeleton(tmp_path)
    first = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
    second = np.ones((adata.n_obs, adata.n_vars), dtype=np.float32)

    append_zarr_layer(path, "first", first, consolidate=False)
    append_zarr_layer(path, "second", second, consolidate=True)

    result, _ = safe_read_zarr(path, verbose=False)
    assert set(result.layers) == {"first", "second"}
    np.testing.assert_array_equal(np.asarray(result.layers["first"]), first)
    np.testing.assert_array_equal(np.asarray(result.layers["second"]), second)


def test_append_zarr_layer_multiple_layers_accumulate(tmp_path):
    path, adata = _skeleton(tmp_path)
    names = ["nan0_0minus1", "nan1_12", "nan_minus_1", "nan_half"]
    arrays = {
        name: np.full((adata.n_obs, adata.n_vars), float(index), dtype=np.float32)
        for index, name in enumerate(names)
    }

    for index, name in enumerate(names):
        append_zarr_layer(path, name, arrays[name], consolidate=(index == len(names) - 1))

    result, _ = safe_read_zarr(path, verbose=False)
    assert set(result.layers) == set(names)
    for name in names:
        np.testing.assert_array_equal(np.asarray(result.layers[name]), arrays[name])


def test_append_zarr_layer_with_explicit_chunks(tmp_path):
    path, adata = _skeleton(tmp_path, n_obs=10, n_vars=6)
    layer = np.random.default_rng(0).random((adata.n_obs, adata.n_vars)).astype(np.float32)

    append_zarr_layer(path, "chunked", layer, chunks=(4, 6))

    result, _ = safe_read_zarr(path, verbose=False)
    np.testing.assert_array_equal(np.asarray(result.layers["chunked"]), layer)


def test_append_zarr_obsm_round_trips(tmp_path):
    path, adata = _skeleton(tmp_path)
    n_lags = 5
    obsm = np.arange(adata.n_obs * n_lags, dtype=np.float32).reshape(adata.n_obs, n_lags)

    append_zarr_obsm(path, "GpC_spatial_autocorr", obsm)

    result, _ = safe_read_zarr(path, verbose=False)
    assert set(result.obsm) == {"GpC_spatial_autocorr"}
    np.testing.assert_array_equal(np.asarray(result.obsm["GpC_spatial_autocorr"]), obsm)


def test_append_zarr_obsm_multiple_keys_accumulate(tmp_path):
    path, adata = _skeleton(tmp_path)
    names = ["GpC_spatial_autocorr", "GpC_spatial_autocorr_counts", "GpC_lomb_scargle_power"]
    arrays = {
        name: np.full((adata.n_obs, 4), float(index), dtype=np.float32)
        for index, name in enumerate(names)
    }

    for index, name in enumerate(names):
        append_zarr_obsm(path, name, arrays[name], consolidate=(index == len(names) - 1))

    result, _ = safe_read_zarr(path, verbose=False)
    assert set(result.obsm) == set(names)
    for name in names:
        np.testing.assert_array_equal(np.asarray(result.obsm[name]), arrays[name])


def test_consolidate_zarr_store_after_deferred_writes(tmp_path):
    path, adata = _skeleton(tmp_path)
    first = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
    second = np.ones((adata.n_obs, adata.n_vars), dtype=np.float32)

    append_zarr_layer(path, "first", first, consolidate=False)
    append_zarr_layer(path, "second", second, consolidate=False)
    consolidate_zarr_store(path)

    result, _ = safe_read_zarr(path, verbose=False)
    assert set(result.layers) == {"first", "second"}
