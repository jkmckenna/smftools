"""Tests for canonical sklearn explanation adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smftools.machine_learning.artifacts import ExplanationMaskPolicy, ExplanationTarget
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import MLMaterializedPartitionData
from smftools.machine_learning.data.transforms import FeatureTransformSpec
from smftools.machine_learning.interpretability import (
    ExplanationDecisionProvenance,
    InterpretabilityContractError,
    InterpretabilityRequest,
    explain_sklearn_model,
    sample_training_background,
)
from smftools.machine_learning.models import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.training import fit_sklearn_partition_model

pytestmark = pytest.mark.unit

MODEL_ID = "1" * 64
DATASET_ID = "2" * 64
SPLIT_ID = "3" * 64


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
        InputSchema.from_dataset(dataset, reference="locus", n_positions=3),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _data(split: str, *, repeats: int = 3) -> MLMaterializedPartitionData:
    negative = np.asarray(
        [
            [[0, 0], [0, 1], [0, 0]],
            [[0, 0], [1, 0], [0, 0]],
            [[0, 1], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [1, 0]],
        ],
        dtype=np.float32,
    )
    positive = 1.0 - negative
    values = np.concatenate([negative, positive] * repeats)
    labels = np.asarray(([0] * len(negative) + [1] * len(positive)) * repeats)
    n_rows, n_positions, n_channels = values.shape
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=tuple(f"{split}-molecule-{index}" for index in range(n_rows)),
        read_ids=tuple(f"{split}-read-{index}" for index in range(n_rows)),
        experiment_uids=(f"{split}-experiment",) * n_rows,
        modalities=("conversion",) * n_rows,
        coordinates=np.asarray([100, 101, 102]),
        channel_names=("gpc_accessibility", "cpg_methylation"),
        values=values,
        labels=labels,
        observed_mask=np.ones_like(values, dtype=bool),
        availability_mask=np.ones((n_rows, n_channels), dtype=bool),
        design_mask=np.ones((n_positions, n_channels), dtype=bool),
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
            effective_batch_size=8,
        )
        self.train = _data("train")

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        assert split == "train"
        return self.train


def _fit(
    family: str,
    *,
    transform_spec: FeatureTransformSpec | None = None,
):
    dataset = _Dataset()
    parameters = {"n_estimators": 12, "max_depth": 3} if family == "random_forest" else None
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        family,
        input_schema=dataset.plan.dataset.input_schema,
        parameters=parameters,
    )
    trained = fit_sklearn_partition_model(
        dataset,
        resolved,
        transform_spec=transform_spec,
        incremental=False if family == "bernoulli_nb" else None,
    )
    return dataset, trained.model


def _request(
    model,
    data: MLMaterializedPartitionData,
    method: str,
    parameters: dict,
    *,
    baseline=None,
) -> InterpretabilityRequest:
    return InterpretabilityRequest.create(
        method=method,
        model_id=MODEL_ID,
        dataset_snapshot_id=DATASET_ID,
        input_schema_hash=model.input_schema.schema_hash,
        split_role=data.split,
        cohort=f"{data.split}-natural",
        observation_uids=data.molecule_uids,
        target=ExplanationTarget(
            output_name="activity_probability",
            class_id=1,
            class_name="active",
        ),
        baseline=baseline,
        mask_policy=ExplanationMaskPolicy.create(
            mask_kinds=(),
            handling="validity is represented by fitted transformed indicator features",
        ),
        decision=ExplanationDecisionProvenance("fixed"),
        parameters=parameters,
        random_seed=17,
    )


def test_naive_bayes_log_odds_reconstruct_and_retain_indicator_semantics() -> None:
    dataset, model = _fit(
        "bernoulli_nb",
        transform_spec=FeatureTransformSpec(indicators=("observed", "design")),
    )
    validation = _data("validation", repeats=1)
    request = _request(model, validation, "NaiveBayesLogOdds", {})

    result = explain_sklearn_model(model, validation, request)
    transformed = model.transform.transform(validation)
    posterior_log_odds = (
        model.estimator.predict_log_proba(transformed)[:, 1]
        - model.estimator.predict_log_proba(transformed)[:, 0]
    )

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.metadata["prior_log_odds"],
        posterior_log_odds,
    )
    assert result.axes == ("observation", "feature")
    assert result.features[0].kind == "signal"
    assert result.features[0].channel.biological_role == "accessibility"
    assert result.features[0].channel.sources[0].site_context == "GpC"
    assert any(feature.kind == "observed" for feature in result.features)
    assert any(feature.kind == "design" for feature in result.features)
    result.validate_against(dataset.plan.dataset.input_schema)


def test_linear_coefficients_and_odds_ratios_use_fitted_feature_units() -> None:
    _dataset, model = _fit(
        "logistic_regression",
        transform_spec=FeatureTransformSpec(scaling="standard", indicators=()),
    )
    validation = _data("validation", repeats=1)
    coefficient_request = _request(
        model,
        validation,
        "LinearCoefficients",
        {"statistic": "coefficient"},
    )
    odds_request = _request(
        model,
        validation,
        "LinearCoefficients",
        {"statistic": "odds_ratio"},
    )

    coefficients = explain_sklearn_model(model, validation, coefficient_request)
    odds_ratios = explain_sklearn_model(model, validation, odds_request)

    np.testing.assert_allclose(coefficients.values, model.estimator.coef_[0])
    np.testing.assert_allclose(odds_ratios.values, np.exp(coefficients.values))
    assert coefficients.axes == ("feature",)
    assert coefficients.metadata["standardized_signal_features"] is True
    assert coefficients.metadata["statistic"] == "coefficient"


def test_permutation_importance_is_held_out_deterministic_and_auditable() -> None:
    _dataset, model = _fit("random_forest", transform_spec=FeatureTransformSpec(indicators=()))
    validation = _data("validation")
    request = _request(
        model,
        validation,
        "PermutationImportance",
        {"metric": "roc_auc", "n_repeats": 4},
    )

    first = explain_sklearn_model(model, validation, request)
    second = explain_sklearn_model(model, validation, request)

    assert first.result_id == second.result_id
    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata["held_out_split"] == "validation"
    assert first.metadata["held_out_cohort"] == "validation-natural"
    assert first.metadata["metric"] == "roc_auc"
    assert first.metadata["n_repeats"] == 4
    assert first.metadata["random_seed"] == 17
    assert len(first.metadata["importance_standard_deviation"]) == len(first.features)

    train_request = _request(
        model,
        _data("train", repeats=1),
        "PermutationImportance",
        {"metric": "roc_auc", "n_repeats": 2},
    )
    with pytest.raises(InterpretabilityContractError, match="held-out validation or test"):
        explain_sklearn_model(model, _data("train", repeats=1), train_request)


def test_tree_shap_records_output_perturbation_and_training_background() -> None:
    dataset, model = _fit("random_forest", transform_spec=FeatureTransformSpec(indicators=()))
    validation = _data("validation", repeats=1)
    background = sample_training_background(
        dataset.train,
        model.input_schema,
        dataset_snapshot_id=DATASET_ID,
        split_id=SPLIT_ID,
        max_observations=6,
        random_seed=5,
    )
    request = _request(
        model,
        validation,
        "TreeSHAP",
        {
            "model_output": "probability",
            "feature_perturbation": "interventional",
            "check_additivity": False,
        },
        baseline=background.to_baseline(),
    )

    result = explain_sklearn_model(model, validation, request, background=background)

    assert result.values.shape == (
        len(validation.molecule_uids),
        len(model.transform.feature_names),
    )
    assert result.metadata["implementation"] == "shap.TreeExplainer"
    assert result.metadata["model_output"] == "probability"
    assert result.metadata["feature_perturbation"] == "interventional"
    assert result.metadata["background_hash"] == background.background_hash

    with pytest.raises(InterpretabilityContractError, match="requires its checksummed background"):
        explain_sklearn_model(model, validation, request)


def test_tree_shap_parameter_validation_happens_before_optional_import(monkeypatch) -> None:
    _dataset, model = _fit("random_forest", transform_spec=FeatureTransformSpec(indicators=()))
    validation = _data("validation", repeats=1)
    request = _request(
        model,
        validation,
        "TreeSHAP",
        {
            "model_output": "probability",
            "feature_perturbation": "tree_path_dependent",
            "check_additivity": False,
        },
    )

    def fail_require(*args, **kwargs):
        raise AssertionError("optional dependency should not be imported")

    monkeypatch.setattr(
        "smftools.machine_learning.interpretability.classical.require",
        fail_require,
    )
    with pytest.raises(InterpretabilityContractError, match="require model_output='raw'"):
        explain_sklearn_model(model, validation, request)
