"""Manifest-bound training for supported scikit-learn model families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..contracts import InputSchema, LabelSchema
from ..data.balancing import (
    BalanceResolution,
    resolve_role_balance,
    resolve_role_balance_from_plan,
)
from ..data.materialized_dataset import MLDatasetProtocol
from ..data.streaming_transforms import fit_feature_transform_streaming, plan_transform_fit
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


def _streaming_capable_families(registry: ModelRegistry, input_schema: InputSchema) -> list[str]:
    """Names of registered sklearn families whose fit can be streamed."""
    capable = []
    for name in registry.names:
        try:
            candidate = registry.resolve(name, input_schema=input_schema)
        except Exception:  # noqa: BLE001 - a family that cannot resolve here is not a suggestion
            continue
        if candidate.backend == "sklearn" and candidate.capabilities.incremental_fit:
            capable.append(name)
    return capable


def fit_sklearn_partition_model_streaming(
    dataset: MLDatasetProtocol,
    resolved_model: ResolvedModelDefinition,
    *,
    transform_spec: FeatureTransformSpec | None = None,
    balancing: BalancingSpec | None = None,
    seed: int = 0,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> SklearnTrainingResult:
    """Fit an incremental sklearn estimator without materializing the train split.

    :func:`fit_sklearn_partition_model` materializes ``train`` before its
    ``partial_fit`` branch runs, so the incremental capability bounds memory
    inside the estimator and nothing at the data boundary -- the materialization
    preflight refuses the job first. This path removes that ceiling: balancing
    resolves from read-plan metadata, the transform fits from streamed batches,
    and rows reach ``partial_fit`` one batch at a time.

    **Row order differs from the materialized path by design.** The materialized
    fit feeds rows in the balance's permuted ``selected_indices`` order; a
    streaming fit must feed them in canonical batch order, applying each row's
    selection multiplicity as it passes. That is equivalent only for estimators
    whose incremental update is order-independent, which is why this path is
    restricted to families declaring ``incremental_fit`` and why parity against
    the materialized fit is asserted in the tests rather than assumed.
    """
    if resolved_model.backend != "sklearn":
        raise SklearnTrainingError("resolved_model must use the sklearn backend")
    input_schema = dataset.plan.dataset.input_schema
    if not resolved_model.capabilities.incremental_fit:
        capable = _streaming_capable_families(registry, input_schema)
        raise SklearnTrainingError(
            f"model family {resolved_model.family!r} cannot be fitted from streamed batches "
            "because it has no partial_fit; it requires the train split to be materialized and "
            "is therefore bounded by max_materialization_bytes. Streaming-capable families: "
            f"{capable}. Use fit_sklearn_partition_model for this family, raising "
            "max_materialization_bytes only if the split genuinely fits in memory."
        )
    label_schema = dataset.plan.dataset.label_schema
    if label_schema is None:
        raise SklearnTrainingError("sklearn classification requires a label schema")

    spec = transform_spec or FeatureTransformSpec()
    # Refuses non-streamable specs (imputation="median") before any read.
    plan_transform_fit(spec)

    balance = resolve_role_balance_from_plan(
        dataset.plan,
        label_schema,
        balancing or BalancingSpec(),
        role="train",
        seed=seed,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )
    entries = dataset.plan.entries_for("train")
    fitted_transform = fit_feature_transform_streaming(
        dataset,
        spec,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        coordinates=tuple(int(value) for value in dataset.plan.coordinates),
        channel_names=tuple(channel.name for channel in input_schema.channels),
        molecule_uids=tuple(entry.molecule_uid for entry in entries),
    )

    estimator = registry.build(resolved_model)
    partial_fit = getattr(estimator, "partial_fit", None)
    if not callable(partial_fit):
        raise SklearnTrainingError("incremental capability declared without partial_fit")

    # How many times each split-relative row is selected. Upsampling repeats
    # rows, downsampling drops them, and natural selection keeps each once.
    multiplicity = np.bincount(balance.selected_indices, minlength=len(entries))
    classes = np.arange(len(label_schema.class_order), dtype=np.int64)
    class_weights = balance.class_weights if balance.method == "class_weight" else None
    if balance.method == "weighted_sampler":
        raise SklearnTrainingError("weighted_sampler is a Torch-only balancing method")

    offset = 0
    fitted_any = False
    for batch in dataset.iter_batches("train"):
        rows = len(batch.molecule_uids)
        counts = multiplicity[offset : offset + rows]
        offset += rows
        if not counts.any():
            continue
        local = np.repeat(np.flatnonzero(counts), counts[counts > 0])
        features = fitted_transform.transform(batch)[local]
        labels = np.asarray(batch.labels, dtype=np.int64)[local]
        parameters = {}
        if class_weights is not None:
            parameters["sample_weight"] = class_weights[labels]
        partial_fit(features, labels, classes=classes, **parameters)
        fitted_any = True

    if offset != len(entries):
        raise SklearnTrainingError(
            f"streamed {offset} train rows but the read plan declares {len(entries)}"
        )
    if not fitted_any:
        raise SklearnTrainingError("balancing selected no training rows")

    fitted = FittedSklearnModel(
        family=resolved_model.family,
        architecture=resolved_model,
        estimator=estimator,
        transform=fitted_transform,
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        fit_mode="partial_fit",
        native_parameters=_json_parameters(estimator),
    )
    return SklearnTrainingResult(
        model=fitted,
        balance=balance,
        n_training_observations=len(balance.selected_indices),
        class_counts=balance.result_counts,
    )


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
