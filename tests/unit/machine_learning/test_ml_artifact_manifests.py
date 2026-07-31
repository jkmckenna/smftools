from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from smftools.machine_learning.artifacts import (
    ArtifactReference,
    CheckpointManifest,
    EnvironmentRecord,
    ExplanationBaseline,
    ExplanationManifest,
    ExplanationMaskPolicy,
    ExplanationTarget,
    FailureRecord,
    MLArtifactManifestError,
    ModelLineage,
    ModelManifest,
    PredictionManifest,
    ResolvedDefinition,
    RunManifest,
    SerializationPolicy,
)
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

NOW = "2026-07-30T12:00:00+00:00"
LATER = "2026-07-30T12:01:00+00:00"
DONE = "2026-07-30T12:02:00+00:00"
RUN_ID = "12345678-1234-5678-1234-567812345678"
OTHER_RUN_ID = "87654321-4321-6789-4321-678987654321"
WORKSPACE_ID = "1" * 64
PLAN_HASH = "2" * 64
DATASET_ID = "3" * 64
SPLIT_ID = "4" * 64
INPUT_HASH = "5" * 64
LABEL_HASH = "6" * 64


def _artifact(
    role: str,
    *,
    sha: str = "a" * 64,
    path: str | None = None,
    media_type: str = "application/octet-stream",
) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=path or f"runs/{RUN_ID}/{role}.bin",
        sha256=sha,
        size_bytes=128,
        media_type=media_type,
    )


def _environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        smftools_version="2.19.0.dev0",
        python_version="3.12.4",
        platform="macOS-arm64",
        code_revision="abcdef123",
        dirty_tree=False,
        dependencies={
            "numpy": "2.1.0",
            "scikit-learn": "1.6.1",
            "skops": "0.12.0",
        },
    )


def _train_run(*, run_id: str | None = RUN_ID) -> RunManifest:
    return RunManifest.create(
        run_id=run_id,
        workspace_id=WORKSPACE_ID,
        action="train",
        job_name="train-activity",
        plan_hash=PLAN_HASH,
        resolved_plan=_artifact(
            "resolved_plan",
            path=f"runs/{RUN_ID}/resolved_plan.json",
            media_type="application/json",
        ),
        resolved_config=_artifact(
            "resolved_config",
            path=f"runs/{RUN_ID}/resolved_config.json",
            media_type="application/json",
        ),
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        model_keys=("rf", "nb"),
        environment=_environment(),
        seeds={"split": 42, "model": 7},
        device="cpu",
        created_at=NOW,
    )


def _architecture(*, width: int = 64) -> ResolvedDefinition:
    return ResolvedDefinition.create(
        name="residual_dilated_cnn",
        version="1",
        parameters={"width": width, "layers": [2, 2, 2]},
    )


def _skops_policy() -> SerializationPolicy:
    return SerializationPolicy(
        format="skops",
        loader="skops.io.load",
        requires_unsafe_load=False,
        allowed_types=(
            "sklearn.naive_bayes.BernoulliNB",
            "sklearn.pipeline.Pipeline",
        ),
        package_versions={
            "numpy": "2.1.0",
            "scikit-learn": "1.6.1",
            "skops": "0.12.0",
        },
    )


def _model(
    *,
    artifact_sha: str = "b" * 64,
    architecture: ResolvedDefinition | None = None,
    lineage: ModelLineage | None = None,
) -> ModelManifest:
    return ModelManifest.create(
        model_key="nb",
        backend="sklearn",
        family="bernoulli_nb",
        task_type="binary_classification",
        originating_run_id=RUN_ID,
        workspace_id=WORKSPACE_ID,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        input_schema_hash=INPUT_HASH,
        label_schema_hash=LABEL_HASH,
        architecture=architecture or _architecture(),
        lineage=lineage
        or ModelLineage(
            kind="from_scratch",
            parent_model_ids=(),
            parent_roles=(),
        ),
        artifact=_artifact(
            "model",
            sha=artifact_sha,
            path="models/model-id/pipeline.skops",
            media_type="application/x-skops",
        ),
        serialization=_skops_policy(),
        environment=_environment(),
        created_at=DONE,
    )


def test_identical_training_attempts_receive_distinct_run_ids() -> None:
    first = _train_run(run_id=None)
    second = _train_run(run_id=None)

    assert first.run_id != second.run_id
    assert first.to_dict() | {"run_id": second.run_id} == second.to_dict()


def test_run_lifecycle_is_immutable_and_round_trips() -> None:
    planned = _train_run()
    running = planned.transition("running", at=LATER)
    completed = running.transition(
        "completed",
        at=DONE,
        artifacts=(
            _artifact(
                "metrics",
                path=f"runs/{RUN_ID}/metrics.parquet",
                media_type="application/vnd.apache.parquet",
            ),
        ),
    )

    assert planned.state == "planned"
    assert running.started_at == LATER
    assert completed.finished_at == DONE
    assert RunManifest.from_dict(completed.to_dict()) == completed
    assert RunManifest.from_dict(completed.to_dict()).canonical_json() == (
        completed.canonical_json()
    )


def test_run_rejects_invalid_transition_and_incomplete_training_provenance() -> None:
    with pytest.raises(MLArtifactManifestError, match="cannot transition"):
        _train_run().transition("completed", at=DONE)

    raw = _train_run().to_dict()
    raw["split_id"] = None
    with pytest.raises(MLArtifactManifestError, match="required for training"):
        RunManifest.from_dict(raw)

    running = _train_run().transition("running", at=LATER)
    raw = running.to_dict()
    raw["started_at"] = "2026-07-29T12:00:00+00:00"
    with pytest.raises(MLArtifactManifestError, match="cannot precede created_at"):
        RunManifest.from_dict(raw)


def test_failed_run_retains_sanitized_diagnostic_context() -> None:
    failure = FailureRecord(
        error_type="RuntimeError",
        message="worker exited before checkpoint publication",
        phase="training",
        traceback_artifact=_artifact(
            "traceback",
            path=f"runs/{RUN_ID}/logs/traceback.txt",
            media_type="text/plain",
        ),
    )
    failed = _train_run().transition("failed", at=LATER, failure=failure)

    assert failed.failure == failure
    assert failed.finished_at == LATER
    assert RunManifest.from_dict(failed.to_dict()) == failed

    raw = failed.to_dict()
    raw["failure"] = None
    with pytest.raises(MLArtifactManifestError, match="required for a failed run"):
        RunManifest.from_dict(raw)


def test_model_identity_includes_content_and_scientific_provenance() -> None:
    base = _model()
    changed_bytes = _model(artifact_sha="c" * 64)
    changed_architecture = _model(architecture=_architecture(width=128))
    parent = "d" * 64
    fine_tuned = _model(
        lineage=ModelLineage(
            kind="fine_tuned",
            parent_model_ids=(parent,),
            parent_roles=("encoder",),
        )
    )

    assert (
        len(
            {
                base.model_id,
                changed_bytes.model_id,
                changed_architecture.model_id,
                fine_tuned.model_id,
            }
        )
        == 4
    )
    assert ModelManifest.from_dict(base.to_dict()) == base


def test_semantic_model_key_is_distinct_from_content_model_identity() -> None:
    model = _model()
    renamed = ModelManifest.create(
        **{
            **{
                key: value
                for key, value in model.__dict__.items()
                if key not in {"schema_version", "model_id", "model_key"}
            },
            "model_key": "renamed-user-alias",
        }
    )

    assert renamed.model_key != model.model_key
    assert renamed.model_id == model.model_id


def test_model_manifest_rejects_tampered_identity() -> None:
    raw = _model().to_dict()
    raw["dataset_snapshot_id"] = "e" * 64

    with pytest.raises(MLArtifactManifestError, match="model_id"):
        ModelManifest.from_dict(raw)


def test_model_lineage_requires_parent_for_fine_tuning() -> None:
    with pytest.raises(MLArtifactManifestError, match="require a parent"):
        ModelLineage(
            kind="fine_tuned",
            parent_model_ids=(),
            parent_roles=(),
        )

    with pytest.raises(MLArtifactManifestError, match="cannot have parents"):
        ModelLineage(
            kind="from_scratch",
            parent_model_ids=("d" * 64,),
            parent_roles=("encoder",),
        )


def test_sklearn_serialization_policy_records_safe_loader_and_versions() -> None:
    model = _model()

    assert model.serialization.format == "skops"
    assert model.serialization.requires_unsafe_load is False
    assert "sklearn.pipeline.Pipeline" in model.serialization.allowed_types
    assert model.serialization.package_versions["scikit-learn"] == "1.6.1"


def test_pickle_and_joblib_must_be_explicitly_unsafe() -> None:
    for format_name in ("pickle", "joblib"):
        with pytest.raises(MLArtifactManifestError, match="explicitly unsafe"):
            SerializationPolicy(
                format=format_name,
                loader=f"{format_name}.load",
                requires_unsafe_load=False,
                allowed_types=(),
                package_versions={"scikit-learn": "1.6.1"},
            )


def test_checkpoint_identity_includes_actual_payload_digest() -> None:
    checkpoint = CheckpointManifest.create(
        run_id=RUN_ID,
        model_key="cnn",
        backend="torch",
        kind="best",
        epoch=3,
        step=120,
        input_schema_hash=INPUT_HASH,
        architecture_hash=_architecture().definition_hash,
        artifact=_artifact(
            "checkpoint",
            sha="7" * 64,
            path=f"runs/{RUN_ID}/checkpoints/best.ckpt",
        ),
        created_at=DONE,
    )
    changed = CheckpointManifest.create(
        run_id=RUN_ID,
        model_key="cnn",
        backend="torch",
        kind="best",
        epoch=3,
        step=120,
        input_schema_hash=INPUT_HASH,
        architecture_hash=_architecture().definition_hash,
        artifact=_artifact(
            "checkpoint",
            sha="8" * 64,
            path=f"runs/{RUN_ID}/checkpoints/best.ckpt",
        ),
        created_at=DONE,
    )

    assert checkpoint.checkpoint_id != changed.checkpoint_id
    assert CheckpointManifest.from_dict(checkpoint.to_dict()) == checkpoint


def _prediction() -> PredictionManifest:
    return PredictionManifest.create(
        run_id=OTHER_RUN_ID,
        workspace_id=WORKSPACE_ID,
        model_id=_model().model_id,
        dataset_snapshot_id=DATASET_ID,
        input_schema_hash=INPUT_HASH,
        label_schema_hash=LABEL_HASH,
        cohort="validation",
        split_role="validation",
        n_observations=20,
        identity_columns=("molecule_uid", "experiment_uid", "read_id"),
        prediction_columns=("predicted_class", "probability_active"),
        table=_artifact(
            "predictions",
            sha="9" * 64,
            path=f"runs/{OTHER_RUN_ID}/predictions/validation.parquet",
            media_type="application/vnd.apache.parquet",
        ),
        created_at=DONE,
    )


def test_prediction_manifest_round_trip_records_stable_row_identity() -> None:
    prediction = _prediction()

    assert "molecule_uid" in prediction.identity_columns
    assert PredictionManifest.from_dict(prediction.to_dict()) == prediction


def test_prediction_manifest_rejects_missing_row_identity_and_tampering() -> None:
    raw = _prediction().to_dict()
    raw["identity_columns"] = ["read_id"]
    with pytest.raises(MLArtifactManifestError, match="molecule_uid"):
        PredictionManifest.from_dict(raw)

    raw = _prediction().to_dict()
    raw["n_observations"] = 21
    with pytest.raises(MLArtifactManifestError, match="prediction_id"):
        PredictionManifest.from_dict(raw)


def _explanation() -> ExplanationManifest:
    return ExplanationManifest.create(
        run_id=OTHER_RUN_ID,
        workspace_id=WORKSPACE_ID,
        model_id=_model().model_id,
        dataset_snapshot_id=DATASET_ID,
        cohort="validation",
        n_observations=20,
        method=ResolvedDefinition.create(
            name="integrated_gradients",
            version="captum-0.7",
            parameters={"steps": 64},
        ),
        target=ExplanationTarget(
            output_name="activity_logits",
            class_id=1,
            class_name="active",
        ),
        baseline=ExplanationBaseline(
            kind="training_cohort_mean",
            description="Mean observed training signal by channel",
            baseline_hash="f" * 64,
            dataset_snapshot_id=DATASET_ID,
            cohort="train",
        ),
        mask_policy=ExplanationMaskPolicy.create(
            mask_kinds=("availability", "observed"),
            handling="zero attribution where either mask is false",
        ),
        feature_axes=("position", "channel"),
        values=_artifact(
            "explanation_values",
            sha="1" * 64,
            path=f"runs/{OTHER_RUN_ID}/explanations/explanation-id/values.zarr",
            media_type="application/vnd.zarr",
        ),
        summary=_artifact(
            "explanation_summary",
            sha="2" * 64,
            path=(f"runs/{OTHER_RUN_ID}/explanations/explanation-id/feature_summary.parquet"),
            media_type="application/vnd.apache.parquet",
        ),
        created_at=DONE,
    )


def test_explanation_manifest_round_trip_preserves_scientific_context() -> None:
    explanation = _explanation()
    restored = ExplanationManifest.from_dict(explanation.to_dict())

    assert restored.method.name == "integrated_gradients"
    assert restored.target.class_name == "active"
    assert restored.baseline.cohort == "train"
    assert restored.mask_policy.mask_kinds == ("availability", "observed")
    assert restored.feature_axes == ("position", "channel")
    assert restored == explanation


def test_explanation_identity_changes_with_baseline_or_mask_policy() -> None:
    explanation = _explanation()
    other_baseline = ExplanationManifest.create(
        **{
            **{
                key: value
                for key, value in explanation.__dict__.items()
                if key not in {"schema_version", "explanation_id"}
            },
            "baseline": replace(
                explanation.baseline,
                baseline_hash="e" * 64,
            ),
        }
    )
    other_mask = ExplanationManifest.create(
        **{
            **{
                key: value
                for key, value in explanation.__dict__.items()
                if key not in {"schema_version", "explanation_id"}
            },
            "mask_policy": ExplanationMaskPolicy.create(
                mask_kinds=("observed",),
                handling="exclude unobserved values",
            ),
        }
    )

    assert explanation.explanation_id != other_baseline.explanation_id
    assert explanation.explanation_id != other_mask.explanation_id


def test_explanation_rejects_unknown_mask_and_tampered_target() -> None:
    with pytest.raises(MLArtifactManifestError, match="unknown masks"):
        ExplanationMaskPolicy.create(
            mask_kinds=("mystery",),
            handling="unknown",
        )

    raw = copy.deepcopy(_explanation().to_dict())
    raw["target"]["class_name"] = "inactive"
    with pytest.raises(MLArtifactManifestError, match="explanation_id"):
        ExplanationManifest.from_dict(raw)


def test_explanation_baseline_cannot_use_locked_test_cohort() -> None:
    with pytest.raises(MLArtifactManifestError, match="locked test data"):
        ExplanationBaseline(
            kind="cohort_mean",
            description="Invalid test-derived baseline",
            baseline_hash="f" * 64,
            dataset_snapshot_id=DATASET_ID,
            cohort="test",
        )


def test_manifest_readers_reject_missing_provenance_and_unknown_versions() -> None:
    raw = _model().to_dict()
    raw.pop("environment")
    with pytest.raises(MLArtifactManifestError, match="missing required"):
        ModelManifest.from_dict(raw)

    raw = _prediction().to_dict()
    raw["schema_version"] = 99
    with pytest.raises(MLArtifactManifestError, match="unsupported version"):
        PredictionManifest.from_dict(raw)


def test_artifact_references_are_portable_and_workspace_compatible(tmp_path) -> None:
    workspace = MLWorkspace(
        scope_kind="experiment",
        scope_id="experiment-a",
        owner_root=tmp_path / "output",
        root=tmp_path / "output/ml_outputs",
    )
    reference = _model().artifact.relative_path

    assert workspace.resolve_reference(reference).is_relative_to(workspace.root)

    with pytest.raises(MLArtifactManifestError, match="portable relative"):
        _artifact("model", path="/absolute/model.skops")
    with pytest.raises(MLArtifactManifestError, match="portable relative"):
        _artifact("model", path="models/model:unsafe.skops")
