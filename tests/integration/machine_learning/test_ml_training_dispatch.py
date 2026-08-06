"""Streaming reaches the orchestration façade (ML-204 follow-up).

ML-204 added streaming fit engines but left ``train_partition_model`` calling
the materializing ones, so a user on the approved orchestration surface still
hit the memory ceiling the package existed to remove. These tests pin the
dispatch behaviour and the deliberately asymmetric defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smftools.machine_learning.benchmarks.fixtures import FixtureSpec, build_fixture
from smftools.machine_learning.data.partition_dataset import PartitionReadPolicy
from smftools.machine_learning.models.registry import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.orchestration import (
    MLJobServiceError,
    SklearnTrainOptions,
    TorchTrainOptions,
    train_partition_model,
)
from smftools.machine_learning.training.torch_backend import TorchTrainingConfig

pytestmark = pytest.mark.integration

SPEC = FixtureSpec(n_rows=120, n_positions=16, n_channels=1, imbalanced=True)


def _roomy(tmp_path: Path):
    return build_fixture(tmp_path, SPEC)


def _tight(tmp_path: Path):
    """A dataset whose train split is one byte over the materialization budget."""
    probe = build_fixture(tmp_path / "probe", SPEC)
    estimate = probe.plan.estimate_materialization_bytes("train")
    return build_fixture(
        tmp_path / "tight",
        SPEC,
        policy=PartitionReadPolicy(max_materialization_bytes=estimate - 1),
    )


def _resolved(built, family: str):
    parameters = {"n_estimators": 3} if family == "random_forest" else None
    return BUILTIN_MODEL_REGISTRY.resolve(
        family, input_schema=built.plan.dataset.input_schema, parameters=parameters
    )


def test_incremental_sklearn_family_streams_by_default(tmp_path: Path) -> None:
    # The point of the fix: the approved entry point now works above the
    # materialization ceiling without the caller asking for anything.
    built = _tight(tmp_path)

    result = train_partition_model(built.dataset, _resolved(built, "bernoulli_nb"))

    assert result.model.fit_mode == "partial_fit"
    assert result.n_training_observations == len(built.plan.entries_for("train"))


def test_default_streaming_reproduces_the_materialized_sklearn_fit(tmp_path: Path) -> None:
    # The default only flipped because streamed and materialized sklearn fits
    # are the same model. If that ever stops being true, the default is wrong.
    built = _roomy(tmp_path)
    resolved = _resolved(built, "bernoulli_nb")

    streamed = train_partition_model(built.dataset, resolved)
    materialized = train_partition_model(
        built.dataset, resolved, sklearn_options=SklearnTrainOptions(streaming=False)
    )

    np.testing.assert_allclose(
        streamed.model.estimator.feature_log_prob_,
        materialized.model.estimator.feature_log_prob_,
    )
    assert streamed.balance.resolution_id == materialized.balance.resolution_id
    assert streamed.model.transform.transform_id == materialized.model.transform.transform_id


def test_forcing_materialization_still_hits_the_ceiling_with_a_named_remedy(
    tmp_path: Path,
) -> None:
    built = _tight(tmp_path)

    with pytest.raises(MLJobServiceError) as error:
        train_partition_model(
            built.dataset,
            _resolved(built, "bernoulli_nb"),
            sklearn_options=SklearnTrainOptions(streaming=False),
        )

    message = str(error.value)
    assert "max_materialization_bytes" in message
    assert "SklearnTrainOptions(streaming=True)" in message


def test_non_incremental_family_cannot_stream_and_says_why(tmp_path: Path) -> None:
    built = _roomy(tmp_path)

    with pytest.raises(MLJobServiceError) as error:
        train_partition_model(
            built.dataset,
            _resolved(built, "random_forest"),
            sklearn_options=SklearnTrainOptions(streaming=True),
        )

    message = str(error.value)
    assert "random_forest" in message
    assert "partial_fit" in message


def test_non_incremental_family_refusal_names_a_streaming_capable_family(
    tmp_path: Path,
) -> None:
    built = _tight(tmp_path)

    with pytest.raises(MLJobServiceError) as error:
        train_partition_model(built.dataset, _resolved(built, "random_forest"))

    message = str(error.value)
    assert "streaming-capable family" in message
    assert "max_materialization_bytes" in message


def test_contradictory_streaming_and_incremental_options_are_rejected(tmp_path: Path) -> None:
    built = _roomy(tmp_path)

    with pytest.raises(MLJobServiceError, match="contradictory"):
        train_partition_model(
            built.dataset,
            _resolved(built, "bernoulli_nb"),
            sklearn_options=SklearnTrainOptions(streaming=True, incremental=False),
        )


def test_torch_does_not_stream_by_default_because_weights_would_change(
    tmp_path: Path,
) -> None:
    # Asymmetry with sklearn is deliberate: a streamed Torch fit shuffles within
    # a buffer, so switching silently would hand the user a different model.
    built = _tight(tmp_path)

    with pytest.raises(MLJobServiceError) as error:
        train_partition_model(
            built.dataset,
            _resolved(built, "residual_dilated_cnn"),
            torch_options=TorchTrainOptions(
                training_config=TorchTrainingConfig(max_epochs=1, device="cpu", batch_size=16)
            ),
        )

    message = str(error.value)
    assert "TorchTrainOptions(streaming=True)" in message
    assert "different weights" in message


def test_torch_streams_when_asked(tmp_path: Path) -> None:
    built = _tight(tmp_path)

    result = train_partition_model(
        built.dataset,
        _resolved(built, "residual_dilated_cnn"),
        torch_options=TorchTrainOptions(
            streaming=True,
            training_config=TorchTrainingConfig(max_epochs=2, device="cpu", batch_size=16),
        ),
    )

    assert len(result.model.history) == 2
    assert np.isfinite(result.model.test_loss)
