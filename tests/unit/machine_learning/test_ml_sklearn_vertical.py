from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smftools.machine_learning.artifacts import (
    ArtifactReference,
    EnvironmentRecord,
    ModelManifest,
    RunManifest,
    SerializationPolicy,
    file_sha256,
    publish_bundle,
    validate_published_bundle,
)
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import (
    MLMaterializedPartitionData,
    MLMemoryBudgetError,
)
from smftools.machine_learning.inference import apply_sklearn_partition_model
from smftools.machine_learning.models import (
    BUILTIN_MODEL_REGISTRY,
    SklearnArtifactError,
    load_published_sklearn_model,
    publish_sklearn_model,
)
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.training import (
    SklearnTrainingError,
    fit_sklearn_partition_model,
)
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

DATASET_ID = "a" * 64
SPLIT_ID = "b" * 64
PLAN_HASH = "c" * 64
RUN_ID = "12345678-1234-5678-1234-567812345678"
NOW = "2026-08-01T12:00:00+00:00"
STARTED = "2026-08-01T12:01:00+00:00"
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
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                }
            },
        }
    )
    dataset = plan.datasets["reads"]
    return (
        InputSchema.from_dataset(dataset, reference="locus", n_positions=4),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _training_data() -> MLMaterializedPartitionData:
    negative = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    positive = 1.0 - negative
    matrix = np.concatenate([negative, positive, negative, positive])
    labels = np.asarray([0] * 3 + [1] * 3 + [0] * 3 + [1] * 3, dtype=np.int64)
    values = matrix[:, :, np.newaxis]
    n_rows, n_positions, _ = values.shape
    return MLMaterializedPartitionData(
        split="train",
        molecule_uids=tuple(f"molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("deaminase",) * n_rows,
        coordinates=np.arange(n_positions, dtype=np.int64),
        channel_names=("accessibility",),
        values=values,
        labels=labels,
        observed_mask=np.ones_like(values, dtype=bool),
        availability_mask=np.ones((n_rows, 1), dtype=bool),
        design_mask=np.ones((n_positions, 1), dtype=bool),
        padding_mask=np.zeros((n_rows, n_positions), dtype=bool),
    )


class _Dataset:
    def __init__(self, *, fail_materialization: bool = False) -> None:
        input_schema, label_schema = _schemas()
        self.plan = SimpleNamespace(
            dataset=SimpleNamespace(
                snapshot_id=DATASET_ID,
                input_schema=input_schema,
                label_schema=label_schema,
            ),
            split=SimpleNamespace(split_id=SPLIT_ID),
            effective_batch_size=3,
        )
        self.data = _training_data()
        self.fail_materialization = fail_materialization

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        assert split == "train"
        if self.fail_materialization:
            raise MLMemoryBudgetError("unsafe materialization estimate")
        return self.data


def _resolved(dataset: _Dataset, family: str):
    parameters = {"n_estimators": 12} if family == "random_forest" else None
    return BUILTIN_MODEL_REGISTRY.resolve(
        family,
        input_schema=dataset.plan.dataset.input_schema,
        parameters=parameters,
    )


@pytest.mark.parametrize(
    ("family", "expected_mode"),
    [
        ("bernoulli_nb", "partial_fit"),
        ("logistic_regression", "fit"),
        ("random_forest", "fit"),
    ],
)
def test_registered_models_share_training_and_application_contracts(
    family: str,
    expected_mode: str,
) -> None:
    dataset = _Dataset()
    result = fit_sklearn_partition_model(dataset, _resolved(dataset, family))

    predictions = apply_sklearn_partition_model(result.model, dataset.data)

    assert result.model.fit_mode == expected_mode
    assert result.model.estimator.classes_.tolist() == [0, 1]
    assert result.class_counts == (6, 6)
    assert predictions.class_order == ("inactive", "active")
    assert predictions.probabilities.shape == (12, 2)
    np.testing.assert_allclose(predictions.probabilities.sum(axis=1), 1.0)
    assert np.mean(predictions.class_ids == dataset.data.labels) >= 0.75


def test_nonincremental_model_refuses_dataset_materialization_budget() -> None:
    dataset = _Dataset(fail_materialization=True)

    with pytest.raises(MLMemoryBudgetError, match="unsafe materialization"):
        fit_sklearn_partition_model(dataset, _resolved(dataset, "random_forest"))


def test_forced_incremental_fit_is_rejected_before_materialization() -> None:
    dataset = _Dataset(fail_materialization=True)

    with pytest.raises(SklearnTrainingError, match="does not support partial_fit"):
        fit_sklearn_partition_model(
            dataset,
            _resolved(dataset, "logistic_regression"),
            incremental=True,
        )


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
        dependencies={"numpy": np.__version__},
    )


def _reference(role: str, relative_path: str, path: Path) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=relative_path,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        media_type="application/json",
    )


def _publish_completed_run(tmp_path: Path, workspace: MLWorkspace) -> RunManifest:
    plan_path = tmp_path / "resolved_plan.json"
    config_path = tmp_path / "resolved_config.json"
    plan_path.write_text('{"plan": 1}\n', encoding="utf-8")
    config_path.write_text('{"config": 1}\n', encoding="utf-8")
    plan_reference = _reference("resolved_plan", "resolved_plan.json", plan_path)
    config_reference = _reference("resolved_config", "resolved_config.json", config_path)
    manifest = RunManifest.create(
        run_id=RUN_ID,
        workspace_id=workspace.workspace_id,
        action="train",
        job_name="train-nb",
        plan_hash=PLAN_HASH,
        resolved_plan=plan_reference,
        resolved_config=config_reference,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        model_keys=("nb",),
        environment=_environment(),
        seeds={"model": 0},
        device="cpu",
        created_at=NOW,
    ).transition("running", at=STARTED)
    manifest = manifest.transition("completed", at=DONE)
    publish_bundle(
        workspace,
        manifest,
        sources={
            plan_reference.relative_path: plan_path,
            config_reference.relative_path: config_path,
        },
    )
    return manifest


def test_skops_round_trip_and_complete_run_model_publication(tmp_path: Path) -> None:
    dataset = _Dataset()
    trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "bernoulli_nb"))
    before = apply_sklearn_partition_model(trained.model, dataset.data)
    workspace = _workspace(tmp_path)
    run_manifest = _publish_completed_run(tmp_path, workspace)

    published = publish_sklearn_model(
        trained.model,
        workspace,
        model_key="nb",
        originating_run_id=run_manifest.run_id,
        environment=_environment(),
        created_at=DONE,
    )
    loaded = load_published_sklearn_model(workspace, published.manifest.model_id)
    after = apply_sklearn_partition_model(loaded, dataset.data)

    np.testing.assert_array_equal(after.class_ids, before.class_ids)
    np.testing.assert_allclose(after.scores, before.scores)
    np.testing.assert_allclose(after.probabilities, before.probabilities)
    assert published.manifest.serialization.format == "skops"
    assert published.manifest.serialization.requires_unsafe_load is False
    assert set(published.manifest.serialization.package_versions) == {
        "numpy",
        "scipy",
        "scikit-learn",
        "skops",
    }
    assert validate_published_bundle(
        workspace,
        workspace.run_paths(RUN_ID).root,
        kind="run",
        expected_id=RUN_ID,
    )
    assert validate_published_bundle(
        workspace,
        workspace.model_dir(published.manifest.model_id),
        kind="model",
        expected_id=published.manifest.model_id,
    )


def test_loader_rejects_dependency_version_drift_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset()
    trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "bernoulli_nb"))
    workspace = _workspace(tmp_path)
    _publish_completed_run(tmp_path, workspace)
    published = publish_sklearn_model(
        trained.model,
        workspace,
        model_key="nb",
        originating_run_id=RUN_ID,
        environment=_environment(),
        created_at=DONE,
    )
    import smftools.machine_learning.models.sklearn_artifacts as artifacts

    observed = artifacts._package_versions()
    monkeypatch.setattr(
        artifacts,
        "_package_versions",
        lambda: {**observed, "scikit-learn": "0.0.0"},
    )

    with pytest.raises(SklearnArtifactError, match="dependency versions"):
        load_published_sklearn_model(workspace, published.manifest.model_id)


def test_loader_rejects_manifest_that_self_authorizes_extra_types(tmp_path: Path) -> None:
    dataset = _Dataset()
    trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "bernoulli_nb"))
    workspace = _workspace(tmp_path)
    _publish_completed_run(tmp_path, workspace)
    published = publish_sklearn_model(
        trained.model,
        workspace,
        model_key="nb",
        originating_run_id=RUN_ID,
        environment=_environment(),
        created_at=DONE,
    )
    original = published.manifest
    permissive = ModelManifest.create(
        model_key=original.model_key,
        backend=original.backend,
        family=original.family,
        task_type=original.task_type,
        originating_run_id=original.originating_run_id,
        workspace_id=original.workspace_id,
        dataset_snapshot_id=original.dataset_snapshot_id,
        split_id=original.split_id,
        input_schema_hash=original.input_schema_hash,
        label_schema_hash=original.label_schema_hash,
        architecture=original.architecture,
        lineage=original.lineage,
        artifact=original.artifact,
        serialization=SerializationPolicy(
            format="skops",
            loader="skops.io.load",
            requires_unsafe_load=False,
            allowed_types=(*original.serialization.allowed_types, "unreviewed.ArbitraryType"),
            package_versions=original.serialization.package_versions,
        ),
        environment=original.environment,
        created_at=original.created_at,
    )
    source = published.bundle.path / original.artifact.relative_path
    publish_bundle(
        workspace,
        permissive,
        sources={permissive.artifact.relative_path: source},
    )

    with pytest.raises(SklearnArtifactError, match="reviewed model type"):
        load_published_sklearn_model(workspace, permissive.model_id)


def test_publisher_rejects_estimator_outside_registered_family(tmp_path: Path) -> None:
    dataset = _Dataset()
    trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "bernoulli_nb"))
    workspace = _workspace(tmp_path)
    alien = SimpleNamespace(**{**trained.model.__dict__, "estimator": object()})

    with pytest.raises(SklearnArtifactError, match="registered family"):
        publish_sklearn_model(
            alien,
            workspace,
            model_key="nb",
            originating_run_id=RUN_ID,
            environment=_environment(),
            created_at=DONE,
        )
