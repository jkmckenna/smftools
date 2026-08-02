"""Tests for backend-neutral explanation requests, backgrounds, and results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smftools.machine_learning.artifacts import (
    ArtifactReference,
    ExplanationBaseline,
    ExplanationMaskPolicy,
    ExplanationTarget,
)
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.interpretability import (
    METHOD_CONTRACTS,
    AttributionAggregation,
    AttributionResult,
    ExplanationArtifactLayout,
    ExplanationDecisionProvenance,
    InterpretabilityContractError,
    InterpretabilityRequest,
    create_explanation_manifest,
    sample_training_background,
    validate_interpretability_request,
)
from smftools.machine_learning.models import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.workspace import MLWorkspace

pytestmark = pytest.mark.unit

MODEL_ID = "1" * 64
DATASET_ID = "2" * 64
SPLIT_ID = "3" * 64
WORKSPACE_ID = "4" * 64
RUN_ID = "12345678-1234-5678-1234-567812345678"


def _schemas() -> tuple[InputSchema, LabelSchema]:
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {
                "reads": {
                    "modalities": ["conversion"],
                    "channels": [
                        {
                            "name": "gpc_accessibility",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "conversion",
                                    "stage": "preprocessed",
                                    "layer": "GpC_site_binary",
                                    "site_context": "GpC",
                                }
                            ],
                        },
                        {
                            "name": "cpg_methylation",
                            "biological_role": "endogenous_methylation",
                            "sources": [
                                {
                                    "modality": "conversion",
                                    "stage": "preprocessed",
                                    "layer": "CpG_site_binary",
                                    "site_context": "CpG",
                                }
                            ],
                        },
                    ],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample"],
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
        InputSchema.from_dataset(dataset, reference="locus", n_positions=3),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _data(split: str = "train") -> MLMaterializedPartitionData:
    n_rows = 10
    values = np.arange(n_rows * 3 * 2, dtype=np.float32).reshape(n_rows, 3, 2) / 10
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"read-{index}" for index in range(n_rows)),
        experiment_uids=("experiment",) * n_rows,
        modalities=("conversion",) * n_rows,
        coordinates=np.asarray([100, 101, 102]),
        channel_names=("gpc_accessibility", "cpg_methylation"),
        values=values,
        labels=np.asarray([0, 1] * 5),
        observed_mask=np.ones_like(values, dtype=bool),
        availability_mask=np.ones((n_rows, 2), dtype=bool),
        design_mask=np.ones((3, 2), dtype=bool),
        padding_mask=np.zeros((n_rows, 3), dtype=bool),
    )


def _request(
    *,
    method: str = "IntegratedGradients",
    split_role: str = "test",
    baseline=None,
    layer: str | None = None,
) -> InterpretabilityRequest:
    input_schema, _labels = _schemas()
    if baseline is None and METHOD_CONTRACTS.get(method, None) is not None:
        if METHOD_CONTRACTS[method].baseline_policy == "required":
            baseline = sample_training_background(
                _data(),
                input_schema,
                dataset_snapshot_id=DATASET_ID,
                split_id=SPLIT_ID,
                max_observations=4,
                random_seed=7,
            ).to_baseline()
    return InterpretabilityRequest.create(
        method=method,
        model_id=MODEL_ID,
        dataset_snapshot_id=DATASET_ID,
        input_schema_hash=input_schema.schema_hash,
        split_role=split_role,
        cohort=f"{split_role}-cohort",
        observation_uids=tuple(f"molecule-{index}" for index in range(4)),
        target=ExplanationTarget(
            output_name="activity_probability",
            class_id=1,
            class_name="active",
        ),
        baseline=baseline,
        layer=layer,
        mask_policy=ExplanationMaskPolicy.create(
            mask_kinds=("observed", "availability", "design"),
            handling="zero attribution where any declared validity mask is false",
        ),
        aggregation=AttributionAggregation(
            reduction="mean_absolute",
            axes=("observation",),
            group_by=("modality", "class"),
        ),
        decision=ExplanationDecisionProvenance(
            kind="selected",
            split_role="validation",
            cohort="validation-method-selection",
        ),
        parameters={"n_steps": 32, "internal_batch_size": 8},
        random_seed=11,
    )


def test_canonical_names_and_non_test_decision_provenance_are_strict() -> None:
    assert "GradientSHAP" in METHOD_CONTRACTS
    assert "GradSHAP" not in METHOD_CONTRACTS
    assert "AttentionCAM" not in METHOD_CONTRACTS

    with pytest.raises(InterpretabilityContractError, match="aliases are not accepted"):
        _request(method="GradSHAP")
    with pytest.raises(InterpretabilityContractError, match="train or validation"):
        ExplanationDecisionProvenance(
            kind="selected",
            split_role="test",
            cohort="test",
        )


def test_contract_import_does_not_load_explanation_frameworks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import smftools.machine_learning.interpretability; "
                "assert not any(name == 'captum' or name.startswith('captum.') "
                "for name in sys.modules); "
                "assert not any(name == 'shap' or name.startswith('shap.') "
                "for name in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_capability_dispatch_fails_before_explainer_execution() -> None:
    input_schema, label_schema = _schemas()
    torch_model = BUILTIN_MODEL_REGISTRY.resolve(
        "residual_dilated_cnn",
        input_schema=input_schema,
        parameters={"in_channels": 2},
    )
    sklearn_model = BUILTIN_MODEL_REGISTRY.resolve(
        "random_forest",
        input_schema=input_schema,
        parameters={"n_estimators": 10},
    )
    request = _request()
    assert InterpretabilityRequest.from_dict(request.to_dict()) == request

    resolved = validate_interpretability_request(
        request,
        family=torch_model.family,
        capabilities=torch_model.capabilities,
        input_schema=input_schema,
        label_schema=label_schema,
    )
    assert resolved.name == "IntegratedGradients"

    with pytest.raises(InterpretabilityContractError, match="does not support backend"):
        validate_interpretability_request(
            request,
            family=sklearn_model.family,
            capabilities=sklearn_model.capabilities,
            input_schema=input_schema,
            label_schema=label_schema,
        )
    with pytest.raises(InterpretabilityContractError, match="requires a target layer"):
        validate_interpretability_request(
            _request(method="LayerGradCam", baseline=None),
            family=torch_model.family,
            capabilities=torch_model.capabilities,
            input_schema=input_schema,
            label_schema=label_schema,
        )
    with pytest.raises(InterpretabilityContractError, match="not exposed"):
        validate_interpretability_request(
            _request(method="LayerGradCam", baseline=None, layer="encoder.block"),
            family=torch_model.family,
            capabilities=torch_model.capabilities,
            input_schema=input_schema,
            label_schema=label_schema,
            available_layers=("stem",),
        )
    with pytest.raises(InterpretabilityContractError, match="capabilities"):
        validate_interpretability_request(
            _request(method="AttentionRollout", baseline=None),
            family=torch_model.family,
            capabilities=torch_model.capabilities,
            input_schema=input_schema,
            label_schema=label_schema,
        )
    validation_baseline = ExplanationBaseline(
        kind="validation_background",
        description="invalid validation-derived background",
        baseline_hash="9" * 64,
        dataset_snapshot_id=DATASET_ID,
        cohort="validation",
    )
    with pytest.raises(InterpretabilityContractError, match="train split"):
        validate_interpretability_request(
            _request(baseline=validation_baseline),
            family=torch_model.family,
            capabilities=torch_model.capabilities,
            input_schema=input_schema,
            label_schema=label_schema,
        )


def test_training_background_is_bounded_deterministic_and_never_test_derived() -> None:
    input_schema, _labels = _schemas()
    first = sample_training_background(
        _data(),
        input_schema,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        max_observations=3,
        random_seed=19,
    )
    second = sample_training_background(
        _data(),
        input_schema,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        max_observations=3,
        random_seed=19,
    )

    assert first.background_hash == second.background_hash
    assert first.molecule_uids == second.molecule_uids
    assert first.modalities == ("conversion",) * 3
    assert first.experiment_uids == ("experiment",) * 3
    assert len(first.molecule_uids) == 3
    assert not first.values.flags.writeable
    assert first.to_baseline().cohort == "train"
    assert first.to_baseline().baseline_hash == first.background_hash
    first.validate_against(input_schema)
    with pytest.raises(InterpretabilityContractError, match="only accepts.*train"):
        sample_training_background(
            _data("test"),
            input_schema,
            dataset_snapshot_id=DATASET_ID,
            split_id=SPLIT_ID,
            max_observations=3,
        )


def test_attribution_result_retains_biological_channel_and_site_context() -> None:
    input_schema, _labels = _schemas()
    request = _request()
    result = AttributionResult.create(
        request=request,
        axes=("observation", "position", "channel"),
        values=np.ones((4, 3, 2)),
        observation_uids=request.observation_uids,
        coordinates=np.asarray([100, 101, 102]),
        channels=input_schema.channels,
        convergence_delta=np.asarray([0.01, 0.02, 0.01, 0.03]),
        metadata={"implementation": "captum", "implementation_version": "test"},
    )

    result.validate_against(input_schema)
    assert result.channels[0].biological_role == "accessibility"
    assert result.channels[0].sources[0].site_context == "GpC"
    assert result.channels[1].biological_role == "endogenous_methylation"
    assert result.channels[1].sources[0].site_context == "CpG"
    assert not result.values.flags.writeable
    altered = AttributionResult.create(
        request=request,
        axes=result.axes,
        values=result.values,
        observation_uids=result.observation_uids,
        coordinates=result.coordinates,
        channels=result.channels[::-1],
    )
    with pytest.raises(InterpretabilityContractError, match="channel names.*site contexts"):
        altered.validate_against(input_schema)

    position_only = AttributionResult.create(
        request=request,
        axes=("observation", "position"),
        values=np.ones((4, 3)),
        observation_uids=request.observation_uids,
        coordinates=result.coordinates,
        channels=(),
    )
    position_only.validate_against(input_schema)
    assert position_only.channels == ()


def _artifact(role: str, path: str, sha: str) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        relative_path=path,
        sha256=sha,
        size_bytes=128,
        media_type=(
            "application/vnd.zarr"
            if role == "explanation_values"
            else "application/vnd.apache.parquet"
        ),
    )


def test_runtime_result_maps_to_existing_manifest_and_zarr_layout(tmp_path: Path) -> None:
    input_schema, _labels = _schemas()
    request = _request()
    result = AttributionResult.create(
        request=request,
        axes=("observation", "position", "channel"),
        values=np.ones((4, 3, 2)),
        observation_uids=request.observation_uids,
        coordinates=np.asarray([100, 101, 102]),
        channels=input_schema.channels,
    )
    manifest = create_explanation_manifest(
        result,
        input_schema=input_schema,
        run_id=RUN_ID,
        workspace_id=WORKSPACE_ID,
        values=_artifact(
            "explanation_values",
            f"runs/{RUN_ID}/explanations/{result.result_id}/values.zarr",
            "5" * 64,
        ),
        summary=_artifact(
            "explanation_summary",
            f"runs/{RUN_ID}/explanations/{result.result_id}/feature_summary.parquet",
            "6" * 64,
        ),
        created_at="2026-08-01T12:00:00+00:00",
    )
    owner = tmp_path / "experiment"
    workspace = MLWorkspace(
        scope_kind="experiment",
        scope_id="experiment",
        owner_root=owner,
        root=owner / "ml_outputs",
    )
    layout = ExplanationArtifactLayout.resolve(workspace.run_paths(RUN_ID), result.result_id)

    assert manifest.method.name == "IntegratedGradients"
    assert manifest.method.parameters["layer"] is None
    assert manifest.method.parameters["aggregation"]["reduction"] == "mean_absolute"
    assert manifest.method.parameters["request_id"] == request.request_id
    assert manifest.feature_axes == ("position", "channel")
    assert layout.values.name == "values.zarr"
    assert layout.feature_summary.name == "feature_summary.parquet"
    assert layout.group_summary.name == "group_summary.parquet"
    assert layout.result_id == result.result_id
    assert not layout.root.exists()
