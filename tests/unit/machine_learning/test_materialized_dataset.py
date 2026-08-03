from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data import (
    FeatureTransformSpec,
    MaterializedDataset,
    MLMaterializedPartitionData,
    MLPartitionDataError,
)
from smftools.machine_learning.inference import apply_sklearn_partition_model
from smftools.machine_learning.manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    GenomicInterval,
    SourceArtifactReference,
    SplitManifest,
)
from smftools.machine_learning.models import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.training import fit_sklearn_partition_model

pytestmark = pytest.mark.unit

EXPERIMENT_UID = "12345678-1234-5678-1234-567812345678"


def _manifests() -> tuple[DatasetSnapshotManifest, SplitManifest]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "activity": {
                    "modalities": ["deaminase"],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "replicates": {
                    "strategy": "explicit_groups",
                    "group_by": ["replicate_id"],
                    "train_groups": ["replicate-a"],
                    "validation_groups": ["replicate-b"],
                    "test_groups": ["replicate-c"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "activity",
                    "split": "replicates",
                    "models": ["nb"],
                }
            },
        }
    )
    spec = plan.datasets["activity"]
    observations = []
    assignments = {}
    for role_index, (role, replicate) in enumerate(
        (("train", "replicate-a"), ("validation", "replicate-b"), ("test", "replicate-c"))
    ):
        for class_id in (0, 1):
            read_id = f"read-{role_index}-{class_id}"
            uid = molecule_uid(EXPERIMENT_UID, read_id)
            observations.append(
                DatasetObservation(
                    molecule_uid=uid,
                    experiment_uid=EXPERIMENT_UID,
                    read_id=read_id,
                    sample_id=replicate,
                    reference="6B6_top",
                    modality="deaminase",
                    class_id=class_id,
                    group_values={"replicate_id": replicate},
                )
            )
            assignments[uid] = role
    snapshot = DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="Nkg2a_DAFseq_merged_v2",
            set_name=None,
            dataset_name="activity",
            plan_hash=plan.plan_hash,
            samples=("replicate-a", "replicate-b", "replicate-c"),
            references=("6B6_top",),
            intervals=(GenomicInterval("6B6_top", 100, 104),),
            filters={"feature_mask": "emseq_enhancer_masked"},
        ),
        input_schema=InputSchema.from_dataset(spec, reference="6B6_top", n_positions=4),
        label_schema=LabelSchema.from_plan_label(spec.labels),
        sources=(
            ExperimentSource(
                experiment_id="legacy-selected-input",
                experiment_uid=EXPERIMENT_UID,
                modality="deaminase",
                stage="preprocess",
                stage_generation_id="legacy-generation",
                membership_fingerprint="membership-fingerprint",
                feature_fingerprint="feature-fingerprint",
                artifacts=(
                    SourceArtifactReference(
                        artifact_id="legacy-spine",
                        kind="spine",
                        relative_path="preprocess_outputs/selected.h5ad.gz",
                        sha256="a" * 64,
                    ),
                ),
            ),
        ),
        observations=observations,
    )
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("replicate_id",),
        assignments=assignments,
    )
    return snapshot, split


def _role_data(
    snapshot: DatasetSnapshotManifest,
    split: SplitManifest,
    role: str,
) -> MLMaterializedPartitionData:
    assignments = {member.molecule_uid: member.split for member in split.members}
    observations = tuple(
        item for item in snapshot.observations if assignments[item.molecule_uid] == role
    )
    labels = np.asarray([item.class_id for item in observations], dtype=np.int64)
    matrix = np.stack(
        [
            np.full(snapshot.input_schema.n_positions, class_id, dtype=np.float32)
            for class_id in labels
        ]
    )
    values = matrix[:, :, np.newaxis]
    return MLMaterializedPartitionData(
        split=role,
        molecule_uids=tuple(item.molecule_uid for item in observations),
        read_ids=tuple(item.read_id for item in observations),
        experiment_uids=tuple(item.experiment_uid for item in observations),
        modalities=tuple(item.modality for item in observations),
        coordinates=np.arange(100, 104, dtype=np.int64),
        channel_names=tuple(channel.name for channel in snapshot.input_schema.channels),
        values=values,
        labels=labels,
        observed_mask=np.ones_like(values, dtype=bool),
        availability_mask=np.ones((len(observations), 1), dtype=bool),
        design_mask=np.ones((snapshot.input_schema.n_positions, 1), dtype=bool),
        padding_mask=np.zeros((len(observations), snapshot.input_schema.n_positions), dtype=bool),
    )


def _dataset() -> MaterializedDataset:
    snapshot, split = _manifests()
    return MaterializedDataset(
        snapshot,
        split,
        {role: _role_data(snapshot, split, role) for role in ("train", "validation", "test")},
        effective_batch_size=2,
    )


def test_materialized_dataset_uses_canonical_training_contract() -> None:
    dataset = _dataset()
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "bernoulli_nb",
        input_schema=dataset.plan.dataset.input_schema,
    )

    trained = fit_sklearn_partition_model(dataset, resolved)

    assert trained.model.dataset_snapshot_id == dataset.plan.dataset.snapshot_id
    assert trained.model.split_id == dataset.plan.split.split_id
    assert trained.model.estimator.classes_.tolist() == [0, 1]
    assert dataset.materialize("test").split == "test"


def test_materialized_random_forest_matches_direct_legacy_pipeline() -> None:
    dataset = _dataset()
    parameters = {
        "n_estimators": 12,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
    }
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "random_forest",
        input_schema=dataset.plan.dataset.input_schema,
        parameters=parameters,
    )
    transform = FeatureTransformSpec(
        imputation="most_frequent",
        scaling="none",
        indicators=(),
    )

    canonical = fit_sklearn_partition_model(
        dataset,
        resolved,
        transform_spec=transform,
    ).model
    test = dataset.materialize("test")
    canonical_predictions = apply_sklearn_partition_model(canonical, test)
    direct = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    train = dataset.materialize("train")
    direct.fit(train.X, train.y)

    np.testing.assert_array_equal(
        canonical_predictions.class_ids,
        direct.predict(test.X),
    )
    np.testing.assert_allclose(
        canonical_predictions.probabilities,
        direct.predict_proba(test.X),
    )


def test_materialized_dataset_rejects_manifest_order_drift() -> None:
    snapshot, split = _manifests()
    data = {role: _role_data(snapshot, split, role) for role in ("train", "validation", "test")}
    train = data["train"]
    data["train"] = replace(train, read_ids=tuple(reversed(train.read_ids)))

    with pytest.raises(MLPartitionDataError, match="read_ids do not match"):
        MaterializedDataset(snapshot, split, data)


def test_materialized_dataset_rejects_label_or_mask_drift() -> None:
    snapshot, split = _manifests()
    data = {role: _role_data(snapshot, split, role) for role in ("train", "validation", "test")}
    train = data["train"]
    data["train"] = replace(
        train,
        labels=1 - train.labels,
        availability_mask=np.zeros_like(train.availability_mask),
    )

    with pytest.raises(MLPartitionDataError, match="labels do not match"):
        MaterializedDataset(snapshot, split, data)


def test_materialized_dataset_requires_every_locked_role() -> None:
    snapshot, split = _manifests()
    data = {role: _role_data(snapshot, split, role) for role in ("train", "validation")}

    with pytest.raises(MLPartitionDataError, match="exactly match"):
        MaterializedDataset(snapshot, split, data)


def test_materialized_dataset_detaches_and_freezes_caller_arrays() -> None:
    snapshot, split = _manifests()
    data = {role: _role_data(snapshot, split, role) for role in ("train", "validation", "test")}
    original = data["train"]
    dataset = MaterializedDataset(snapshot, split, data)

    original.values[0, 0, 0] = 99
    frozen = dataset.materialize("train")

    assert frozen.values[0, 0, 0] != 99
    with pytest.raises(ValueError, match="read-only"):
        frozen.values[0, 0, 0] = 7
