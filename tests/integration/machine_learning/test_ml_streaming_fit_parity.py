"""Streamed and materialized fits must be indistinguishable (ML-204).

These run against real partition stores written by ``write_experiment_store``,
so they exercise genuine Zarr projection rather than fabricated arrays. The
unit-level equivalents use stubs and cover the error paths; this file exists to
prove parity where the data actually comes from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smftools.machine_learning.benchmarks.fixtures import FixtureSpec, build_fixture
from smftools.machine_learning.data.balancing import (
    resolve_role_balance,
    resolve_role_balance_from_plan,
)
from smftools.machine_learning.data.partition_dataset import (
    MLMemoryBudgetError,
    PartitionReadPolicy,
)
from smftools.machine_learning.data.streaming_transforms import (
    fit_feature_transform_streaming,
    plan_transform_fit,
)
from smftools.machine_learning.data.transforms import (
    FeatureTransformSpec,
    fit_feature_transform,
)
from smftools.machine_learning.models.registry import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.plan import BalanceRoleSpec, BalancingSpec
from smftools.machine_learning.training.sklearn_backend import (
    SklearnTrainingError,
    fit_sklearn_partition_model,
    fit_sklearn_partition_model_streaming,
)
from smftools.machine_learning.training.torch_backend import (
    TorchTrainingConfig,
    TorchTrainingError,
    fit_torch_partition_model,
    fit_torch_partition_model_streaming,
)

pytestmark = pytest.mark.integration

STREAMABLE_SPECS = [
    FeatureTransformSpec(),
    FeatureTransformSpec(imputation="constant", scaling="standard"),
    FeatureTransformSpec(imputation="mean", scaling="none"),
    FeatureTransformSpec(imputation="mean", scaling="standard"),
    FeatureTransformSpec(imputation="most_frequent", scaling="standard"),
]

BALANCE_METHODS = ("natural", "class_weight", "weighted_sampler", "downsample", "upsample")


def _fixture(tmp_path: Path, **overrides):
    spec = FixtureSpec(n_rows=120, n_positions=24, n_channels=2, **overrides)
    return build_fixture(tmp_path, spec)


@pytest.mark.parametrize("spec", STREAMABLE_SPECS, ids=lambda s: f"{s.imputation}-{s.scaling}")
def test_streamed_transform_matches_materialized_on_real_stores(
    tmp_path: Path, spec: FeatureTransformSpec
) -> None:
    built = _fixture(tmp_path)
    materialized = built.dataset.materialize("train")

    reference = fit_feature_transform(
        materialized,
        spec,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
    )
    streamed = fit_feature_transform_streaming(
        built.dataset,
        spec,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
        coordinates=tuple(int(value) for value in materialized.coordinates),
        channel_names=materialized.channel_names,
        molecule_uids=materialized.molecule_uids,
    )

    assert streamed.transform_id == reference.transform_id
    np.testing.assert_allclose(streamed.fill_values, reference.fill_values)
    np.testing.assert_allclose(streamed.centers, reference.centers)
    np.testing.assert_allclose(streamed.scales, reference.scales)


def test_streamed_transform_is_stable_across_batch_sizes(tmp_path: Path) -> None:
    spec = FeatureTransformSpec(imputation="mean", scaling="standard")

    # Regression guard: transform_id used to hash unrounded float64 moments, so
    # summation order leaked into it and batch_size -- a pure performance knob --
    # rewrote the identity of every model fitted from these rows.
    fixture_spec = FixtureSpec(n_rows=120, n_positions=24, n_channels=2)

    identities = set()
    for batch_size in (4, 16, 64):
        built = build_fixture(
            tmp_path / f"batch-{batch_size}",
            fixture_spec,
            policy=PartitionReadPolicy(batch_size=batch_size),
        )
        materialized = built.dataset.materialize("train")
        identities.add(
            fit_feature_transform_streaming(
                built.dataset,
                spec,
                dataset_snapshot_id=built.plan.dataset.snapshot_id,
                split_id=built.plan.split.split_id,
                coordinates=tuple(int(value) for value in materialized.coordinates),
                channel_names=materialized.channel_names,
                molecule_uids=materialized.molecule_uids,
            ).transform_id
        )

    assert len(identities) == 1


@pytest.mark.parametrize("method", BALANCE_METHODS)
def test_metadata_balance_matches_materialized_balance_on_real_stores(
    tmp_path: Path, method: str
) -> None:
    built = _fixture(tmp_path)
    balancing = BalancingSpec(train=BalanceRoleSpec(method=method))
    schema = built.plan.dataset.label_schema

    reference = resolve_role_balance(
        built.dataset.materialize("train"),
        schema,
        balancing,
        seed=11,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
    )
    from_plan = resolve_role_balance_from_plan(
        built.plan,
        schema,
        balancing,
        seed=11,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
    )

    assert from_plan.resolution_id == reference.resolution_id
    np.testing.assert_array_equal(from_plan.selected_indices, reference.selected_indices)
    assert from_plan.selected_molecule_digest == reference.selected_molecule_digest


SKLEARN_BALANCE_METHODS = ("natural", "class_weight", "downsample", "upsample")


@pytest.mark.parametrize("method", SKLEARN_BALANCE_METHODS)
def test_streamed_sklearn_fit_matches_materialized_fit(tmp_path: Path, method: str) -> None:
    # Imbalanced labels so downsample and upsample actually move rows; with a
    # balanced fixture both are no-ops and the parity claim is vacuous.
    built = build_fixture(
        tmp_path,
        FixtureSpec(n_rows=120, n_positions=16, n_channels=1, imbalanced=True),
    )
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "bernoulli_nb", input_schema=built.plan.dataset.input_schema
    )
    balancing = BalancingSpec(train=BalanceRoleSpec(method=method))

    reference = fit_sklearn_partition_model(built.dataset, resolved, balancing=balancing, seed=5)
    streamed = fit_sklearn_partition_model_streaming(
        built.dataset, resolved, balancing=balancing, seed=5
    )

    assert streamed.n_training_observations == reference.n_training_observations
    assert streamed.balance.resolution_id == reference.balance.resolution_id
    assert streamed.model.transform.transform_id == reference.model.transform.transform_id
    np.testing.assert_allclose(
        streamed.model.estimator.feature_log_prob_,
        reference.model.estimator.feature_log_prob_,
    )
    np.testing.assert_allclose(
        streamed.model.estimator.class_log_prior_,
        reference.model.estimator.class_log_prior_,
    )
    features = reference.model.transform.transform(built.dataset.materialize("train"))
    np.testing.assert_array_equal(
        streamed.model.estimator.predict(features),
        reference.model.estimator.predict(features),
    )


def test_streaming_fit_succeeds_where_materialization_is_refused(tmp_path: Path) -> None:
    # The reason ML-204 exists. Below the materialization budget the ordinary
    # fit cannot run at all -- including with incremental=True, which chunks an
    # array it has already materialized.
    spec = FixtureSpec(n_rows=120, n_positions=16, n_channels=1, imbalanced=True)
    probe = build_fixture(tmp_path / "probe", spec)
    estimate = probe.plan.estimate_materialization_bytes("train")
    built = build_fixture(
        tmp_path / "tight",
        spec,
        policy=PartitionReadPolicy(max_materialization_bytes=estimate - 1),
    )
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "bernoulli_nb", input_schema=built.plan.dataset.input_schema
    )

    with pytest.raises(MLMemoryBudgetError):
        fit_sklearn_partition_model(built.dataset, resolved, incremental=True)

    streamed = fit_sklearn_partition_model_streaming(built.dataset, resolved)

    assert streamed.model.fit_mode == "partial_fit"
    assert streamed.n_training_observations == len(built.plan.entries_for("train"))


def test_non_incremental_families_are_refused_with_a_named_alternative(tmp_path: Path) -> None:
    built = build_fixture(tmp_path, FixtureSpec(n_rows=60, n_positions=8, n_channels=1))
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "random_forest",
        input_schema=built.plan.dataset.input_schema,
        parameters={"n_estimators": 3},
    )

    with pytest.raises(SklearnTrainingError) as error:
        fit_sklearn_partition_model_streaming(built.dataset, resolved)

    message = str(error.value)
    assert "random_forest" in message
    assert "bernoulli_nb" in message
    assert "max_materialization_bytes" in message


class _ReadRecordingDataset:
    """Delegates to a real dataset while recording which split each read hits."""

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self.reads: list[str] = []

    @property
    def plan(self):
        return self._dataset.plan

    def iter_batches(self, split: str, **kwargs):
        self.reads.append(split)
        yield from self._dataset.iter_batches(split, **kwargs)

    def materialize(self, split: str):
        self.reads.append(f"materialize:{split}")
        return self._dataset.materialize(split)


def _torch_fixture(tmp_path: Path, **policy_kwargs):
    spec = FixtureSpec(n_rows=120, n_positions=16, n_channels=1, imbalanced=True)
    policy = PartitionReadPolicy(**policy_kwargs) if policy_kwargs else None
    return build_fixture(tmp_path, spec, policy=policy)


def test_streaming_torch_trains_where_materialization_is_refused(tmp_path: Path) -> None:
    spec = FixtureSpec(n_rows=120, n_positions=16, n_channels=1, imbalanced=True)
    probe = build_fixture(tmp_path / "probe", spec)
    estimate = probe.plan.estimate_materialization_bytes("train")
    built = build_fixture(
        tmp_path / "tight",
        spec,
        policy=PartitionReadPolicy(max_materialization_bytes=estimate - 1),
    )
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )
    config = TorchTrainingConfig(max_epochs=3, device="cpu", batch_size=16)

    with pytest.raises(MLMemoryBudgetError):
        fit_torch_partition_model(built.dataset, resolved, training_config=config)

    streamed = fit_torch_partition_model_streaming(built.dataset, resolved, training_config=config)

    assert len(streamed.model.history) == 3
    assert np.isfinite(streamed.model.validation_loss)
    assert np.isfinite(streamed.model.test_loss)
    assert streamed.n_training_observations == len(built.plan.entries_for("train"))


def test_streaming_torch_leaves_the_test_role_unread_until_selection(tmp_path: Path) -> None:
    # The locked-test contract: no read of "test" may occur before the last
    # train/validation read, or early stopping would have seen it.
    built = _torch_fixture(tmp_path)
    recorder = _ReadRecordingDataset(built.dataset)
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )

    fit_torch_partition_model_streaming(
        recorder,
        resolved,
        training_config=TorchTrainingConfig(max_epochs=3, device="cpu", batch_size=16),
    )

    assert "test" in recorder.reads
    first_test = recorder.reads.index("test")
    last_fit_read = max(
        index for index, split in enumerate(recorder.reads) if split in {"train", "validation"}
    )
    assert first_test > last_fit_read
    assert not any(read.startswith("materialize:") for read in recorder.reads)


def test_streaming_torch_preserves_balance_and_transform_provenance(tmp_path: Path) -> None:
    # Weights differ from the materialized path by design (buffered shuffle),
    # but provenance must not.
    built = _torch_fixture(tmp_path)
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )
    config = TorchTrainingConfig(max_epochs=2, device="cpu", batch_size=16)

    reference = fit_torch_partition_model(built.dataset, resolved, training_config=config)
    streamed = fit_torch_partition_model_streaming(built.dataset, resolved, training_config=config)

    assert streamed.balance.resolution_id == reference.balance.resolution_id
    assert streamed.model.transform.transform_id == reference.model.transform.transform_id
    assert streamed.n_training_observations == reference.n_training_observations
    assert streamed.class_counts == reference.class_counts


def test_streaming_torch_is_reproducible_at_a_fixed_seed(tmp_path: Path) -> None:
    built = _torch_fixture(tmp_path)
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )
    config = TorchTrainingConfig(max_epochs=3, device="cpu", batch_size=16, seed=13)

    first = fit_torch_partition_model_streaming(built.dataset, resolved, training_config=config)
    second = fit_torch_partition_model_streaming(built.dataset, resolved, training_config=config)

    assert [record.to_dict() for record in first.model.history] == [
        record.to_dict() for record in second.model.history
    ]


def test_shuffle_buffer_is_recorded_because_it_changes_the_fitted_model(
    tmp_path: Path,
) -> None:
    # shuffle_buffer_batches lives in TorchTrainingConfig, not in a call
    # argument, because it changes fitted weights. This asserts both halves:
    # the buffer really does change the result, and the value that produced a
    # model is carried in its persisted training config.
    built = _torch_fixture(tmp_path)
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )

    narrow = fit_torch_partition_model_streaming(
        built.dataset,
        resolved,
        training_config=TorchTrainingConfig(
            max_epochs=3, device="cpu", batch_size=8, seed=3, shuffle_buffer_batches=1
        ),
    )
    wide = fit_torch_partition_model_streaming(
        built.dataset,
        resolved,
        training_config=TorchTrainingConfig(
            max_epochs=3, device="cpu", batch_size=8, seed=3, shuffle_buffer_batches=16
        ),
    )

    assert narrow.model.training_config.shuffle_buffer_batches == 1
    assert wide.model.training_config.shuffle_buffer_batches == 16
    assert narrow.model.training_config.to_dict()["shuffle_buffer_batches"] == 1
    narrow_history = [record.train_loss for record in narrow.model.history]
    wide_history = [record.train_loss for record in wide.model.history]
    assert narrow_history != wide_history


def test_streaming_torch_refuses_weighted_sampler_with_a_named_alternative(
    tmp_path: Path,
) -> None:
    built = _torch_fixture(tmp_path)
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
    )

    with pytest.raises(TorchTrainingError) as error:
        fit_torch_partition_model_streaming(
            built.dataset,
            resolved,
            training_config=TorchTrainingConfig(max_epochs=1, device="cpu"),
            balancing=BalancingSpec(train=BalanceRoleSpec(method="weighted_sampler")),
        )

    message = str(error.value)
    assert "weighted_sampler" in message
    assert "class_weight" in message


def test_default_spec_fit_reads_no_batches(tmp_path: Path) -> None:
    # The zero-pass claim, enforced: with the default spec the fit must not
    # touch the reader at all.
    built = _fixture(tmp_path)
    materialized = built.dataset.materialize("train")
    spec = FeatureTransformSpec()
    assert plan_transform_fit(spec).passes == 0

    class _RefusingSource:
        def iter_batches(self, split: str):
            raise AssertionError("the default spec must not read any batches")

    streamed = fit_feature_transform_streaming(
        _RefusingSource(),
        spec,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
        coordinates=tuple(int(value) for value in materialized.coordinates),
        channel_names=materialized.channel_names,
        molecule_uids=materialized.molecule_uids,
    )

    reference = fit_feature_transform(
        materialized,
        spec,
        dataset_snapshot_id=built.plan.dataset.snapshot_id,
        split_id=built.plan.split.split_id,
    )
    assert streamed.transform_id == reference.transform_id
