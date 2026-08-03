"""Manifest-bound training for supported scikit-learn model families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..contracts import InputSchema, LabelSchema
from ..data.balancing import BalanceResolution, resolve_role_balance
from ..data.materialized_dataset import MLDatasetProtocol
from ..data.transforms import (
    FeatureTransformSpec,
    FittedFeatureTransform,
    fit_feature_transform,
)
from ..models.protocols import SklearnPredictor
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry, ResolvedModelDefinition
from ..plan import BalancingSpec


class SklearnTrainingError(ValueError):
    """Raised when sklearn training violates a resolved ML contract."""


@dataclass(frozen=True)
class FittedSklearnModel:
    """Reusable fitted sklearn estimator plus its immutable preprocessing state."""

    family: str
    architecture: ResolvedModelDefinition
    estimator: Any
    transform: FittedFeatureTransform
    input_schema: InputSchema
    label_schema: LabelSchema
    dataset_snapshot_id: str
    split_id: str
    fit_mode: str
    native_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.architecture.backend != "sklearn" or self.family != self.architecture.family:
            raise SklearnTrainingError("fitted model must use its resolved sklearn family")
        if self.fit_mode not in {"fit", "partial_fit"}:
            raise SklearnTrainingError("fit_mode must be 'fit' or 'partial_fit'")
        if self.transform.dataset_snapshot_id != self.dataset_snapshot_id:
            raise SklearnTrainingError("transform dataset snapshot differs from fitted model")
        if self.transform.split_id != self.split_id:
            raise SklearnTrainingError("transform split differs from fitted model")
        expected_classes = np.arange(len(self.label_schema.class_order), dtype=np.int64)
        if not np.array_equal(getattr(self.estimator, "classes_", None), expected_classes):
            raise SklearnTrainingError("fitted estimator classes_ differ from label schema")
        object.__setattr__(
            self,
            "native_parameters",
            MappingProxyType(dict(self.native_parameters)),
        )

    @property
    def predictor(self) -> SklearnPredictor:
        """Return the backend-neutral adapter for transformed feature matrices."""
        return SklearnPredictor(
            model=self.estimator,
            input_schema=self.input_schema,
            label_schema=self.label_schema,
            capabilities=self.architecture.capabilities,
        )


@dataclass(frozen=True)
class SklearnTrainingResult:
    """Fitted model and auditable train-role balancing outcome."""

    model: FittedSklearnModel
    balance: BalanceResolution
    n_training_observations: int
    class_counts: tuple[int, ...]


def _fit_parameters(
    estimator: Any,
    labels: np.ndarray,
    balance: BalanceResolution,
) -> dict[str, np.ndarray]:
    if balance.method == "class_weight":
        if balance.class_weights is None:
            raise SklearnTrainingError("class-weight balancing did not resolve class weights")
        return {"sample_weight": balance.class_weights[labels]}
    if balance.method == "weighted_sampler":
        raise SklearnTrainingError("weighted_sampler is a Torch-only balancing method")
    if balance.method in {"natural", "downsample", "upsample"}:
        return {}
    raise SklearnTrainingError(f"unsupported sklearn balancing method {balance.method!r}")


def _json_parameters(estimator: Any) -> Mapping[str, Any]:
    parameters = estimator.get_params(deep=False)
    result: dict[str, Any] = {}
    for name, value in sorted(parameters.items()):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
        else:
            result[name] = repr(value)
    return result


def fit_sklearn_partition_model(
    dataset: MLDatasetProtocol,
    resolved_model: ResolvedModelDefinition,
    *,
    transform_spec: FeatureTransformSpec | None = None,
    balancing: BalancingSpec | None = None,
    seed: int = 0,
    incremental: bool | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> SklearnTrainingResult:
    """Fit one registered sklearn estimator from the manifest train role.

    Partition-backed inputs run their conservative memory preflight before
    reads; validated pre-materialized inputs are consumed through the same
    manifest-bound interface. Estimators with ``partial_fit`` are updated in
    deterministic chunks after the same train-only preprocessing and balancing
    contracts have been resolved.
    """
    if resolved_model.backend != "sklearn":
        raise SklearnTrainingError("resolved_model must use the sklearn backend")
    if incremental is not None and not isinstance(incremental, bool):
        raise SklearnTrainingError("incremental must be boolean or null")
    use_incremental = (
        resolved_model.capabilities.incremental_fit if incremental is None else incremental
    )
    if use_incremental and not resolved_model.capabilities.incremental_fit:
        raise SklearnTrainingError(
            f"model family {resolved_model.family!r} does not support partial_fit"
        )
    input_schema = dataset.plan.dataset.input_schema
    label_schema = dataset.plan.dataset.label_schema
    if label_schema is None:
        raise SklearnTrainingError("sklearn classification requires a label schema")
    train = dataset.materialize("train")
    if train.labels is None:
        raise SklearnTrainingError("training data does not contain labels")
    fitted_transform = fit_feature_transform(
        train,
        transform_spec or FeatureTransformSpec(),
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )
    balance = resolve_role_balance(
        train,
        label_schema,
        balancing or BalancingSpec(),
        seed=seed,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )
    estimator = registry.build(resolved_model)

    labels = np.asarray(train.labels, dtype=np.int64)
    selected = balance.selected_indices
    selected_labels = labels[selected]
    features = fitted_transform.transform(train)[selected]
    fit_parameters = _fit_parameters(estimator, selected_labels, balance)
    if use_incremental:
        partial_fit = getattr(estimator, "partial_fit", None)
        if not callable(partial_fit):
            raise SklearnTrainingError("incremental capability declared without partial_fit")
        chunk_size = dataset.plan.effective_batch_size
        classes = np.arange(len(label_schema.class_order), dtype=np.int64)
        for offset in range(0, len(selected_labels), chunk_size):
            stop = offset + chunk_size
            chunk_parameters = {
                name: values[offset:stop] for name, values in fit_parameters.items()
            }
            partial_fit(
                features[offset:stop],
                selected_labels[offset:stop],
                classes=classes,
                **chunk_parameters,
            )
        fit_mode = "partial_fit"
    else:
        estimator.fit(features, selected_labels, **fit_parameters)
        fit_mode = "fit"

    fitted = FittedSklearnModel(
        family=resolved_model.family,
        architecture=resolved_model,
        estimator=estimator,
        transform=fitted_transform,
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        fit_mode=fit_mode,
        native_parameters=_json_parameters(estimator),
    )
    return SklearnTrainingResult(
        model=fitted,
        balance=balance,
        n_training_observations=len(selected),
        class_counts=balance.result_counts,
    )
