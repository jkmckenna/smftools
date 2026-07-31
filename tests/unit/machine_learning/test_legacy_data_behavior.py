"""Characterize legacy ML data behavior and known defects."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pytorch_lightning")

from smftools.machine_learning.data.anndata_data_module import (  # noqa: E402
    AnnDataDataset,
    AnnDataModule,
    build_anndata_loader,
)
from smftools.machine_learning.data.preprocessing import random_fill_nans  # noqa: E402
from smftools.machine_learning.models.transformer import (  # noqa: E402
    DANNTransformerClassifier,
)

pytestmark = pytest.mark.unit


def _binary_adata(*, include_nan: bool = False) -> ad.AnnData:
    """Return a deterministic binary fixture with three biological groups."""
    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    if include_nan:
        values[0, 1] = np.nan
        values[7, 3] = np.nan
    obs = pd.DataFrame(
        {
            "experiment_uid": ["experiment-1"] * 12,
            "Sample": np.repeat(["sample-a", "sample-b", "sample-c"], 4),
            "activity_status": pd.Categorical(
                ["inactive", "active", "inactive", "active"] * 3,
                categories=["inactive", "active"],
            ),
        },
        index=[f"read-{index:02d}" for index in range(12)],
    )
    return ad.AnnData(X=values, obs=obs)


def _loader_features(loader) -> torch.Tensor:
    return torch.cat([batch[0] for batch in loader], dim=0)


def test_binary_fixture_has_three_biological_groups():
    adata = _binary_adata()

    groups = adata.obs[["experiment_uid", "Sample"]].drop_duplicates()

    assert len(groups) == 3
    assert set(adata.obs["activity_status"].astype(str)) == {"inactive", "active"}


@pytest.mark.xfail(
    strict=True,
    reason="B-001: the zero-worker validation loader currently returns the training set",
)
def test_zero_worker_validation_loader_uses_validation_membership():
    datamodule = AnnDataModule(
        _binary_adata(),
        label_col="activity_status",
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        num_workers=0,
    )
    datamodule.setup()
    expected = datamodule.val_set.dataset.X_tensor[datamodule.val_set.indices]

    assert torch.equal(_loader_features(datamodule.val_dataloader()), expected)


@pytest.mark.xfail(
    strict=True,
    reason="B-002: the zero-worker test loader currently returns the training set",
)
def test_zero_worker_test_loader_uses_test_membership():
    datamodule = AnnDataModule(
        _binary_adata(),
        label_col="activity_status",
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        num_workers=0,
    )
    datamodule.setup()
    expected = datamodule.test_set.dataset.X_tensor[datamodule.test_set.indices]

    assert torch.equal(_loader_features(datamodule.test_dataloader()), expected)


@pytest.mark.xfail(
    strict=True,
    reason="B-003: the raw-loader factory swaps split persistence arguments",
)
def test_raw_loader_factory_creates_requested_split_file(tmp_path):
    split_path = tmp_path / "split.csv"

    loaders = build_anndata_loader(
        _binary_adata(),
        label_col="activity_status",
        lightning=False,
        split_save_path=split_path,
        load_existing_split=False,
    )

    assert len(loaders) == 3
    assert split_path.is_file()


@pytest.mark.xfail(
    strict=True,
    reason="B-004: legacy random NaN filling mutates its input array",
)
def test_random_fill_nans_does_not_mutate_source():
    source = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32)
    original = source.copy()

    filled = random_fill_nans(source)

    assert np.array_equal(source, original, equal_nan=True)
    assert np.isfinite(filled).all()


@pytest.mark.xfail(
    strict=True,
    reason="B-007: categorical labels currently depend on pandas category order",
)
def test_categorical_label_values_do_not_depend_on_category_order():
    first = _binary_adata()
    second = _binary_adata()
    second.obs["activity_status"] = second.obs["activity_status"].cat.reorder_categories(
        ["active", "inactive"]
    )

    first_labels = AnnDataDataset(first, label_col="activity_status").y_tensor
    second_labels = AnnDataDataset(second, label_col="activity_status").y_tensor

    assert torch.equal(first_labels, second_labels)


@pytest.mark.xfail(
    strict=True,
    reason="B-008: setup mutates the AnnData source while reconstructing missing-value fills",
)
def test_repeated_setup_preserves_source_missingness():
    adata = _binary_adata(include_nan=True)
    original = np.asarray(adata.X).copy()
    datamodule = AnnDataModule(adata, label_col="activity_status")

    datamodule.setup()
    datamodule.setup()

    assert np.array_equal(np.asarray(adata.X), original, equal_nan=True)


@pytest.mark.xfail(
    strict=True,
    reason="B-010: DANNTransformerClassifier currently forwards incompatible constructor arguments",
)
def test_dann_transformer_classifier_constructs():
    model = DANNTransformerClassifier(
        input_dim=1,
        model_dim=8,
        num_classes=2,
        n_domains=3,
        num_heads=2,
        num_layers=1,
    )

    class_logits, domain_logits = model(torch.zeros(2, 5, 1))

    assert class_logits.shape == (2, 1)
    assert domain_logits.shape == (2, 3)
