import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.clean_NaN import DEFAULT_NAN_LAYERS, VALID_NAN_LAYERS, clean_NaN


def _adata(n_obs=3, n_vars=4):
    x = np.array(
        [
            [0.0, 1.0, np.nan, 0.0],
            [1.0, np.nan, 1.0, 0.0],
            [np.nan, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )[:n_obs, :n_vars]
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"read{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[str(i) for i in range(n_vars)]),
    )


def test_on_layer_fires_once_per_default_variant_with_correct_array():
    adata = _adata()
    calls = {}

    clean_NaN(adata, on_layer=lambda name, array: calls.setdefault(name, array))

    assert set(calls) == set(DEFAULT_NAN_LAYERS)
    for name, array in calls.items():
        np.testing.assert_array_equal(array, adata.layers[name])


def test_on_layer_fires_once_per_configured_variant():
    adata = _adata()
    calls = []

    clean_NaN(
        adata,
        layers_to_build=sorted(VALID_NAN_LAYERS),
        on_layer=lambda name, array: calls.append(name),
    )

    assert sorted(calls) == sorted(VALID_NAN_LAYERS)
    assert len(calls) == len(VALID_NAN_LAYERS)  # exactly once each


def test_keep_in_adata_false_frees_variant_from_adata_layers():
    adata = _adata()
    captured = {}

    clean_NaN(
        adata,
        on_layer=lambda name, array: captured.setdefault(name, array.copy()),
        keep_in_adata=False,
    )

    assert set(adata.layers) == set()  # nothing left resident
    assert set(captured) == set(DEFAULT_NAN_LAYERS)


def test_keep_in_adata_true_still_populates_adata_layers_alongside_callback():
    adata = _adata()
    captured = {}

    clean_NaN(
        adata,
        on_layer=lambda name, array: captured.setdefault(name, array),
        keep_in_adata=True,
    )

    assert set(adata.layers) == set(DEFAULT_NAN_LAYERS)
    for name in DEFAULT_NAN_LAYERS:
        np.testing.assert_array_equal(captured[name], adata.layers[name])


def test_no_callback_behaves_exactly_as_before():
    adata = _adata()

    clean_NaN(adata)

    assert set(adata.layers) == set(DEFAULT_NAN_LAYERS)


def test_callback_derives_from_named_source_layer_not_x():
    adata = _adata()
    adata.layers["binarized_methylation"] = np.where(np.isnan(adata.X), np.nan, adata.X * 10)
    calls = {}

    clean_NaN(
        adata,
        layer="binarized_methylation",
        on_layer=lambda name, array: calls.setdefault(name, array),
    )

    # nan_half fills with 0.5; non-NaN values should be the *10 source, not raw X.
    expected_source = adata.layers["binarized_methylation"]
    nan_mask = np.isnan(expected_source)
    expected_nan_half = np.where(nan_mask, 0.5, expected_source)
    np.testing.assert_array_equal(calls["nan_half"], expected_nan_half)
