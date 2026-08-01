from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smftools.machine_learning.artifacts import EnvironmentRecord
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.data.transforms import FeatureTransformSpec
from smftools.machine_learning.inference import apply_torch_partition_model
from smftools.machine_learning.models import (
    BUILTIN_MODEL_REGISTRY,
    TorchArtifactError,
    load_published_torch_model,
    publish_torch_model,
)
from smftools.machine_learning.plan import BalanceRoleSpec, BalancingSpec, parse_ml_plan
from smftools.machine_learning.training import (
    ClassificationTask,
    TorchTrainingConfig,
    TorchTrainingError,
    fit_torch_partition_model,
)
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64
RUN_ID = "12345678-1234-5678-1234-567812345678"
DONE = "2026-08-01T12:02:00+00:00"


def _schemas() -> tuple[InputSchema, LabelSchema]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["sample-a"],
                    "validation_groups": ["sample-b"],
                    "test_groups": ["sample-c"],
                }
            },
            "models": {"cnn": {"backend": "torch", "recipe": "residual_dilated_cnn_v1"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["cnn"],
                }
            },
        }
    )
    dataset = plan.datasets["reads"]
    return (
        InputSchema.from_dataset(dataset, reference="locus", n_positions=8),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _role_data(split: str, *, offset: int) -> MLMaterializedPartitionData:
    negative = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    positive = 1.0 - negative
    values = np.concatenate([negative, positive])[:, :, None]
    labels = np.asarray([0] * 4 + [1] * 4, dtype=np.int64)
    observed = np.ones_like(values, dtype=bool)
    values[0, 2, 0] = np.nan
    observed[0, 2, 0] = False
    n_rows, n_positions, _channels = values.shape
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"{split}-molecule-{offset + index}" for index in range(n_rows)),
        read_ids=tuple(f"{split}-read-{offset + index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.arange(n_positions, dtype=np.int64),
        channel_names=("accessibility",),
        values=values,
        labels=labels,
        observed_mask=observed,
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=np.ones((n_positions, 1), dtype=bool),
        padding_mask=np.zeros((n_rows, n_positions), dtype=bool),
    )


class _Dataset:
    def __init__(self) -> None:
        input_schema, label_schema = _schemas()
        self.plan = SimpleNamespace(
            dataset=SimpleNamespace(
                snapshot_id=DATASET_ID,
                input_schema=input_schema,
                label_schema=label_schema,
            ),
            split=SimpleNamespace(split_id=SPLIT_ID),
        )
        self.roles = {
            "train": _role_data("train", offset=0),
            "validation": _role_data("validation", offset=100),
            "test": _role_data("test", offset=200),
        }
        self.materialized: list[str] = []

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        self.materialized.append(split)
        return self.roles[split]


def _resolved(dataset: _Dataset):
    return BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=dataset.plan.dataset.input_schema,
        parameters={
            "stem_channels": 4,
            "block_channels": [4],
            "dilations": [1],
            "stem_kernel_size": 3,
            "kernel_size": 3,
            "dropout": 0.0,
            "hidden_dim": 4,
            "use_se": False,
            "use_attention_pool": False,
        },
    )


def _training_config(**overrides) -> TorchTrainingConfig:
    values = {
        "max_epochs": 4,
        "batch_size": 4,
        "learning_rate": 0.03,
        "patience": 2,
        "seed": 7,
        "device": "cpu",
    }
    values.update(overrides)
    return TorchTrainingConfig(**values)


def test_training_config_and_task_reject_unsupported_values() -> None:
    config = _training_config()
    assert TorchTrainingConfig.from_dict(config.to_dict()) == config

    payload = config.to_dict()
    payload["unknown"] = True
    with pytest.raises(TorchTrainingError, match="fields must be exactly"):
        TorchTrainingConfig.from_dict(payload)
    with pytest.raises(TorchTrainingError, match="learning_rate must be positive"):
        _training_config(learning_rate=0.0)
    _input_schema, labels = _schemas()
    with pytest.raises(TorchTrainingError, match="incompatible"):
        ClassificationTask(labels, output_dim=3)


def test_torch_training_rejects_appended_mask_indicator_features() -> None:
    dataset = _Dataset()

    with pytest.raises(TorchTrainingError, match="masks remain separate"):
        fit_torch_partition_model(
            dataset,
            _resolved(dataset),
            training_config=_training_config(max_epochs=1, patience=1),
            transform_spec=FeatureTransformSpec(indicators=("observed",)),
        )


def test_short_fit_restores_best_state_and_reads_locked_test_last() -> None:
    dataset = _Dataset()
    result = fit_torch_partition_model(
        dataset,
        _resolved(dataset),
        training_config=_training_config(min_delta=1_000_000.0),
    )

    predictions = apply_torch_partition_model(result.model, dataset.roles["test"])

    assert dataset.materialized == ["train", "validation", "test"]
    assert result.model.best_epoch == 1
    assert result.stopped_early
    best_record = next(row for row in result.model.history if row.epoch == result.model.best_epoch)
    assert result.model.validation_loss == pytest.approx(best_record.validation_loss)
    assert result.class_counts == (4, 4)
    assert predictions.class_order == ("inactive", "active")
    assert predictions.scores.shape == (8, 2)
    assert predictions.probabilities.shape == (8, 2)
    np.testing.assert_allclose(predictions.probabilities.sum(axis=1), 1.0)
    assert np.isfinite([row.train_loss for row in result.model.history]).all()


@pytest.mark.parametrize("method", ["class_weight", "weighted_sampler", "downsample"])
def test_torch_training_uses_shared_balancing_contract(method: str) -> None:
    dataset = _Dataset()
    balancing = BalancingSpec(train=BalanceRoleSpec(method))

    result = fit_torch_partition_model(
        dataset,
        _resolved(dataset),
        training_config=_training_config(max_epochs=1, patience=1),
        balancing=balancing,
    )

    assert result.balance.method == method
    assert result.n_training_observations == 8


def test_same_seed_reproduces_state_and_predictions() -> None:
    first_dataset = _Dataset()
    second_dataset = _Dataset()
    first = fit_torch_partition_model(
        first_dataset,
        _resolved(first_dataset),
        training_config=_training_config(max_epochs=2),
    ).model
    second = fit_torch_partition_model(
        second_dataset,
        _resolved(second_dataset),
        training_config=_training_config(max_epochs=2),
    ).model

    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])
    first_predictions = apply_torch_partition_model(first, first_dataset.roles["test"])
    second_predictions = apply_torch_partition_model(second, second_dataset.roles["test"])
    np.testing.assert_allclose(first_predictions.scores, second_predictions.scores)


def _workspace(tmp_path: Path) -> MLWorkspace:
    owner = tmp_path / "experiment"
    return MLWorkspace(
        scope_kind="experiment",
        scope_id="experiment-1",
        owner_root=owner,
        root=owner / "ml_outputs",
    )


def _environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        smftools_version="2.19.0.dev0",
        python_version="3.12.4",
        platform="test",
        code_revision="abc123",
        dirty_tree=False,
        dependencies={"numpy": np.__version__, "torch": torch.__version__},
    )


def test_state_dict_artifact_round_trip_preserves_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset()
    fitted = fit_torch_partition_model(
        dataset,
        _resolved(dataset),
        training_config=_training_config(max_epochs=2),
    ).model
    before = apply_torch_partition_model(fitted, dataset.roles["test"])
    workspace = _workspace(tmp_path)
    published = publish_torch_model(
        fitted,
        workspace,
        model_key="cnn",
        originating_run_id=RUN_ID,
        environment=_environment(),
        created_at=DONE,
    )
    import smftools.machine_learning.models.torch_artifacts as artifacts

    original_load = artifacts.require("torch", extra="ml-base", purpose="test").load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed["weights_only"] = kwargs.get("weights_only")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded = load_published_torch_model(workspace, published.manifest.model_id)
    after = apply_torch_partition_model(loaded, dataset.roles["test"])

    assert observed["weights_only"] is True
    assert published.manifest.serialization.format == "torch-state-dict"
    assert published.manifest.serialization.requires_unsafe_load is False
    assert published.manifest.serialization.allowed_types == ()
    assert loaded.training_config == fitted.training_config
    assert loaded.history == fitted.history
    np.testing.assert_array_equal(after.class_ids, before.class_ids)
    np.testing.assert_allclose(after.scores, before.scores)
    np.testing.assert_allclose(after.probabilities, before.probabilities)


def test_loader_rejects_dependency_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset()
    fitted = fit_torch_partition_model(
        dataset,
        _resolved(dataset),
        training_config=_training_config(max_epochs=1, patience=1),
    ).model
    workspace = _workspace(tmp_path)
    published = publish_torch_model(
        fitted,
        workspace,
        model_key="cnn",
        originating_run_id=RUN_ID,
        environment=_environment(),
        created_at=DONE,
    )
    import smftools.machine_learning.models.torch_artifacts as artifacts

    monkeypatch.setattr(
        artifacts,
        "_package_versions",
        lambda: {"numpy": "0.0.0", "torch": torch.__version__},
    )
    with pytest.raises(TorchArtifactError, match="dependency versions"):
        load_published_torch_model(workspace, published.manifest.model_id)


def test_plain_torch_vertical_does_not_import_lightning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from smftools.machine_learning.training import fit_torch_partition_model; "
                "from smftools.machine_learning.inference import apply_torch_partition_model; "
                "from smftools.machine_learning.models import publish_torch_model; "
                "assert 'pytorch_lightning' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
