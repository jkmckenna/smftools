"""Canonical explanations for fitted sklearn classifiers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from smftools.optional_imports import require

from ..contracts import InputSchema
from ..data.partition_dataset import MLMaterializedPartitionData, MLPartitionBatch
from ..data.transforms import FittedFeatureTransform
from ..training.sklearn_backend import FittedSklearnModel
from .background import BackgroundReference
from .contracts import (
    AttributionFeature,
    AttributionResult,
    InterpretabilityContractError,
    InterpretabilityRequest,
    _integer,
    _string,
    _thaw_json,
    validate_interpretability_request,
)

_ClassicalData = MLMaterializedPartitionData | MLPartitionBatch
_PERMUTATION_METRICS = frozenset({"accuracy", "average_precision", "negative_log_loss", "roc_auc"})


def _parameters(request: InterpretabilityRequest, allowed: set[str]) -> dict[str, Any]:
    values = _thaw_json(request.parameters)
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise InterpretabilityContractError(
            f"{request.method} received unsupported parameters: {unknown}"
        )
    return values


def _validate_data(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
) -> np.ndarray:
    validate_interpretability_request(
        request,
        family=model.family,
        capabilities=model.architecture.capabilities,
        input_schema=model.input_schema,
        label_schema=model.label_schema,
    )
    if request.dataset_snapshot_id != model.dataset_snapshot_id:
        raise InterpretabilityContractError(
            "explanation request dataset differs from the fitted sklearn model"
        )
    if request.split_role != data.split:
        raise InterpretabilityContractError(
            "explanation request split differs from the supplied data"
        )
    if request.observation_uids != tuple(data.molecule_uids):
        raise InterpretabilityContractError(
            "explanation request observations differ from the supplied data order"
        )
    if tuple(data.channel_names) != tuple(channel.name for channel in model.input_schema.channels):
        raise InterpretabilityContractError(
            "explanation data channel order differs from the fitted model schema"
        )
    if tuple(map(int, data.coordinates)) != model.transform.coordinates:
        raise InterpretabilityContractError(
            "explanation data coordinates differ from the fitted feature transform"
        )
    features = np.asarray(model.transform.transform(data), dtype=np.float64)
    expected = (len(data.molecule_uids), len(model.transform.feature_names))
    if features.shape != expected or not np.isfinite(features).all():
        raise InterpretabilityContractError(
            f"transformed explanation data must be finite with shape {expected}"
        )
    return features


def _feature_metadata(
    transform: FittedFeatureTransform,
    input_schema: InputSchema,
) -> tuple[AttributionFeature, ...]:
    if transform.channel_names != tuple(channel.name for channel in input_schema.channels):
        raise InterpretabilityContractError(
            "fitted feature transform channel order differs from the input schema"
        )
    features = tuple(
        AttributionFeature(
            name=f"{kind}:{channel.name}@{coordinate}",
            kind=kind,
            coordinate=int(coordinate),
            channel=channel,
        )
        for kind in ("signal", *transform.spec.indicators)
        for coordinate in transform.coordinates
        for channel in input_schema.channels
    )
    if tuple(feature.name for feature in features) != transform.feature_names:
        raise InterpretabilityContractError(
            "fitted transform feature names differ from canonical transformed-feature metadata"
        )
    return features


def _reference_class(
    request: InterpretabilityRequest,
    parameters: Mapping[str, Any],
    *,
    n_classes: int,
) -> int:
    target = request.target.class_id
    assert target is not None  # validated by validate_interpretability_request
    reference = parameters.get("reference_class_id")
    if reference is None:
        if n_classes != 2:
            raise InterpretabilityContractError(
                f"{request.method} requires reference_class_id for multiclass models"
            )
        return 1 - target
    reference = _integer(reference, "reference_class_id")
    if reference >= n_classes or reference == target:
        raise InterpretabilityContractError(
            "reference_class_id must identify a non-target class in the label schema"
        )
    return reference


def _result(
    request: InterpretabilityRequest,
    model: FittedSklearnModel,
    data: _ClassicalData,
    values: Any,
    *,
    per_observation: bool,
    metadata: Mapping[str, Any],
) -> AttributionResult:
    result = AttributionResult.create(
        request=request,
        axes=("observation", "feature") if per_observation else ("feature",),
        values=values,
        observation_uids=request.observation_uids if per_observation else (),
        coordinates=data.coordinates,
        channels=(),
        features=_feature_metadata(model.transform, model.input_schema),
        metadata={
            "backend": "sklearn",
            "family": model.family,
            "transform_id": model.transform.transform_id,
            "transform_scaling": model.transform.spec.scaling,
            "indicator_kinds": list(model.transform.spec.indicators),
            **metadata,
        },
    )
    result.validate_against(model.input_schema)
    return result


def _naive_bayes_log_odds(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
    features: np.ndarray,
) -> AttributionResult:
    parameters = _parameters(request, {"reference_class_id"})
    estimator = model.estimator
    log_probabilities = np.asarray(getattr(estimator, "feature_log_prob_", None), dtype=np.float64)
    priors = np.asarray(getattr(estimator, "class_log_prior_", None), dtype=np.float64)
    n_classes = len(model.label_schema.class_order)
    if log_probabilities.shape != (n_classes, features.shape[1]) or priors.shape != (n_classes,):
        raise InterpretabilityContractError(
            "fitted BernoulliNB parameters do not match the transformed feature schema"
        )
    threshold = getattr(estimator, "binarize", None)
    if threshold is None:
        if not np.all((features == 0.0) | (features == 1.0)):
            raise InterpretabilityContractError(
                "BernoulliNB with binarize=None requires binary transformed features"
            )
        binary = features
    else:
        binary = (features > float(threshold)).astype(np.float64)
    target = request.target.class_id
    assert target is not None
    reference = _reference_class(request, parameters, n_classes=n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        negative_log_probabilities = np.log1p(-np.exp(log_probabilities))
    target_values = (
        binary * log_probabilities[target] + (1.0 - binary) * negative_log_probabilities[target]
    )
    reference_values = (
        binary * log_probabilities[reference]
        + (1.0 - binary) * negative_log_probabilities[reference]
    )
    contributions = target_values - reference_values
    if not np.isfinite(contributions).all():
        raise InterpretabilityContractError(
            "BernoulliNB log-odds contributions are non-finite; increase smoothing"
        )
    return _result(
        request,
        model,
        data,
        contributions,
        per_observation=True,
        metadata={
            "implementation": "sklearn_native_parameters",
            "effect_scale": "target_vs_reference_log_odds",
            "reference_class_id": reference,
            "reference_class_name": model.label_schema.class_order[reference],
            "prior_log_odds": float(priors[target] - priors[reference]),
            "binarize": None if threshold is None else float(threshold),
        },
    )


def _linear_coefficients(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
    features: np.ndarray,
) -> AttributionResult:
    del features
    parameters = _parameters(request, {"reference_class_id", "statistic"})
    statistic = _string(parameters.get("statistic", "coefficient"), "statistic")
    if statistic not in {"coefficient", "odds_ratio"}:
        raise InterpretabilityContractError(
            "LinearCoefficients statistic must be 'coefficient' or 'odds_ratio'"
        )
    estimator = model.estimator
    coefficients = np.asarray(getattr(estimator, "coef_", None), dtype=np.float64)
    intercepts = np.asarray(getattr(estimator, "intercept_", None), dtype=np.float64)
    n_classes = len(model.label_schema.class_order)
    target = request.target.class_id
    assert target is not None
    reference = _reference_class(request, parameters, n_classes=n_classes)
    if n_classes == 2 and coefficients.shape == (1, len(model.transform.feature_names)):
        direction = 1.0 if target == 1 else -1.0
        values = direction * coefficients[0]
        intercept = direction * float(intercepts[0])
    elif coefficients.shape == (n_classes, len(model.transform.feature_names)):
        values = coefficients[target] - coefficients[reference]
        intercept = float(intercepts[target] - intercepts[reference])
    else:
        raise InterpretabilityContractError(
            "fitted logistic-regression coefficients do not match the feature and class schemas"
        )
    if statistic == "odds_ratio":
        values = np.exp(values)
    if not np.isfinite(values).all():
        raise InterpretabilityContractError("linear explanation values are non-finite")
    return _result(
        request,
        model,
        data,
        values,
        per_observation=False,
        metadata={
            "implementation": "sklearn_native_parameters",
            "statistic": statistic,
            "effect_scale": (
                "target_vs_reference_log_odds_per_transformed_unit"
                if statistic == "coefficient"
                else "target_vs_reference_odds_ratio_per_transformed_unit"
            ),
            "reference_class_id": reference,
            "reference_class_name": model.label_schema.class_order[reference],
            "intercept_log_odds": intercept,
            "standardized_signal_features": model.transform.spec.scaling == "standard",
        },
    )


def _permutation_score(
    model: FittedSklearnModel,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    phase: str,
    target_class: int,
) -> float:
    predictor = model.predictor
    if metric == "accuracy":
        return float(np.mean(predictor.predict(features, phase=phase) == labels))
    truth = (labels == target_class).astype(np.int64)
    if len(np.unique(truth)) != 2:
        raise InterpretabilityContractError(
            f"permutation metric {metric!r} requires both target and non-target observations"
        )
    probability = predictor.predict_probabilities(features, phase=phase)[:, target_class]
    sklearn_metrics = require(
        "sklearn.metrics",
        extra="ml-base",
        purpose="held-out permutation importance",
    )
    if metric == "roc_auc":
        return float(sklearn_metrics.roc_auc_score(truth, probability))
    if metric == "average_precision":
        return float(sklearn_metrics.average_precision_score(truth, probability))
    clipped = np.clip(probability, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return float(np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)))


def _permutation_importance(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
    features: np.ndarray,
) -> AttributionResult:
    if data.split not in {"validation", "test"}:
        raise InterpretabilityContractError(
            "permutation importance requires a held-out validation or test cohort"
        )
    if data.labels is None:
        raise InterpretabilityContractError("permutation importance requires held-out labels")
    parameters = _parameters(request, {"metric", "n_repeats"})
    metric = _string(parameters.get("metric"), "metric")
    if metric not in _PERMUTATION_METRICS:
        raise InterpretabilityContractError(
            f"permutation metric must be one of {sorted(_PERMUTATION_METRICS)}"
        )
    repeats = _integer(parameters.get("n_repeats"), "n_repeats", minimum=1)
    labels = np.asarray(data.labels, dtype=np.int64)
    target = request.target.class_id
    assert target is not None
    baseline = _permutation_score(
        model,
        features,
        labels,
        metric=metric,
        phase=data.split,
        target_class=target,
    )
    generator = np.random.default_rng(request.random_seed)
    importance = np.empty((features.shape[1], repeats), dtype=np.float64)
    permuted = features.copy()
    for feature_index in range(features.shape[1]):
        original = features[:, feature_index]
        for repeat in range(repeats):
            permuted[:, feature_index] = original[generator.permutation(len(original))]
            importance[feature_index, repeat] = baseline - _permutation_score(
                model,
                permuted,
                labels,
                metric=metric,
                phase=data.split,
                target_class=target,
            )
        permuted[:, feature_index] = original
    return _result(
        request,
        model,
        data,
        importance.mean(axis=1),
        per_observation=False,
        metadata={
            "implementation": "smftools_deterministic_permutation",
            "metric": metric,
            "metric_scope": "all_classes" if metric == "accuracy" else "target_vs_rest",
            "baseline_score": baseline,
            "n_repeats": repeats,
            "random_seed": request.random_seed,
            "importance_standard_deviation": importance.std(axis=1).tolist(),
            "held_out_split": data.split,
            "held_out_cohort": request.cohort,
        },
    )


def _tree_shap(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
    features: np.ndarray,
    background: BackgroundReference | None,
) -> AttributionResult:
    parameters = _parameters(
        request,
        {"check_additivity", "feature_perturbation", "model_output"},
    )
    perturbation = _string(parameters.get("feature_perturbation"), "feature_perturbation")
    model_output = _string(parameters.get("model_output"), "model_output")
    check_additivity = parameters.get("check_additivity")
    if perturbation not in {"interventional", "tree_path_dependent"}:
        raise InterpretabilityContractError(
            "TreeSHAP feature_perturbation must be 'interventional' or 'tree_path_dependent'"
        )
    if model_output not in {"raw", "probability"}:
        raise InterpretabilityContractError("TreeSHAP model_output must be 'raw' or 'probability'")
    if not isinstance(check_additivity, bool):
        raise InterpretabilityContractError("TreeSHAP check_additivity must be boolean")
    if perturbation == "tree_path_dependent" and model_output != "raw":
        raise InterpretabilityContractError(
            "TreeSHAP tree_path_dependent explanations require model_output='raw'"
        )
    background_features = None
    if background is not None:
        background.validate_against(model.input_schema)
        if request.baseline is None or request.baseline.baseline_hash != background.background_hash:
            raise InterpretabilityContractError(
                "TreeSHAP background differs from the explanation request baseline"
            )
        background_features = model.transform.transform(
            background.as_materialized(model.input_schema)
        )
    elif request.baseline is not None:
        raise InterpretabilityContractError(
            "TreeSHAP request baseline requires its checksummed background values"
        )
    if perturbation == "interventional" and background_features is None:
        raise InterpretabilityContractError(
            "TreeSHAP interventional explanations require a training background"
        )
    if perturbation == "tree_path_dependent" and background_features is not None:
        raise InterpretabilityContractError(
            "TreeSHAP tree_path_dependent explanations do not consume a background"
        )
    shap = require("shap", extra="ml-extended", purpose="TreeSHAP explanations")
    explainer = shap.TreeExplainer(
        model.estimator,
        data=background_features,
        model_output=model_output,
        feature_perturbation=perturbation,
    )
    raw_values = explainer.shap_values(features, check_additivity=check_additivity)
    target = request.target.class_id
    assert target is not None
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[target], dtype=np.float64)
    else:
        array = np.asarray(raw_values, dtype=np.float64)
        if array.ndim == 3:
            if target >= array.shape[2]:
                raise InterpretabilityContractError(
                    "TreeSHAP output class axis differs from the persisted label schema"
                )
            values = array[:, :, target]
        elif array.ndim == 2 and len(model.label_schema.class_order) == 2:
            values = array if target == 1 else -array
        else:
            raise InterpretabilityContractError(
                f"TreeSHAP returned unsupported attribution shape {array.shape}"
            )
    if values.shape != features.shape or not np.isfinite(values).all():
        raise InterpretabilityContractError(
            "TreeSHAP values are non-finite or differ from transformed feature shape"
        )
    return _result(
        request,
        model,
        data,
        values,
        per_observation=True,
        metadata={
            "implementation": "shap.TreeExplainer",
            "implementation_version": str(getattr(shap, "__version__", "unknown")),
            "model_output": model_output,
            "feature_perturbation": perturbation,
            "check_additivity": check_additivity,
            "background_hash": None if background is None else background.background_hash,
        },
    )


def explain_sklearn_model(
    model: FittedSklearnModel,
    data: _ClassicalData,
    request: InterpretabilityRequest,
    *,
    background: BackgroundReference | None = None,
) -> AttributionResult:
    """Execute one validated classical explanation without loading model artifacts.

    Model loading remains governed by the existing safe ``skops`` artifact policy. This
    function accepts only an already-loaded :class:`FittedSklearnModel` and imports SHAP
    lazily only when ``TreeSHAP`` is requested.

    Args:
        model: Trusted, fitted canonical sklearn model and preprocessing state.
        data: Materialized or bounded-batch observations in request order.
        request: Validated explanation intent and method parameters.
        background: Optional checksummed training background required by interventional
            TreeSHAP.

    Returns:
        Immutable schema-aligned attribution result.
    """
    if not isinstance(model, FittedSklearnModel):
        raise InterpretabilityContractError(
            "classical explanations require a canonical FittedSklearnModel"
        )
    features = _validate_data(model, data, request)
    if request.method != "TreeSHAP" and background is not None:
        raise InterpretabilityContractError(
            f"{request.method} does not consume a background reference"
        )
    if request.method == "NaiveBayesLogOdds":
        return _naive_bayes_log_odds(model, data, request, features)
    if request.method == "LinearCoefficients":
        return _linear_coefficients(model, data, request, features)
    if request.method == "PermutationImportance":
        return _permutation_importance(model, data, request, features)
    if request.method == "TreeSHAP":
        return _tree_shap(model, data, request, features, background)
    raise InterpretabilityContractError(
        f"method {request.method!r} is not implemented by the sklearn explanation adapter"
    )
