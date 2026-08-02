"""Backend-neutral scientific dispatch used by higher-level ML job services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..data.partition_dataset import (
    MLMaterializedPartitionData,
    MLPartitionBatch,
    PartitionDataset,
)
from ..data.transforms import FeatureTransformSpec
from ..evaluation import (
    CalibrationProvenance,
    EvaluationResult,
    PredictionResult,
    ThresholdProvenance,
    evaluate_predictions,
)
from ..inference import apply_sklearn_partition_model, apply_torch_partition_model
from ..interpretability import (
    AttributionResult,
    BackgroundReference,
    InterpretabilityRequest,
    explain_sklearn_model,
    explain_torch_model,
)
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry, ResolvedModelDefinition
from ..plan import BalancingSpec
from ..training import (
    FittedSklearnModel,
    FittedTorchModel,
    SklearnTrainingResult,
    TorchTrainingConfig,
    TorchTrainingResult,
    fit_sklearn_partition_model,
    fit_torch_partition_model,
)
from .contracts import MLJobServiceError

_FittedModel = FittedSklearnModel | FittedTorchModel
_Data = MLMaterializedPartitionData | MLPartitionBatch


@dataclass(frozen=True)
class SklearnTrainOptions:
    """Explicit sklearn-only options for backend-neutral training dispatch."""

    transform_spec: FeatureTransformSpec | None = None
    balancing: BalancingSpec | None = None
    seed: int = 0
    incremental: bool | None = None


@dataclass(frozen=True)
class TorchTrainOptions:
    """Explicit plain-Torch-only options for backend-neutral training dispatch."""

    training_config: TorchTrainingConfig | None = None
    transform_spec: FeatureTransformSpec | None = None
    balancing: BalancingSpec | None = None


def train_partition_model(
    dataset: PartitionDataset,
    resolved_model: ResolvedModelDefinition,
    *,
    sklearn_options: SklearnTrainOptions | None = None,
    torch_options: TorchTrainOptions | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> SklearnTrainingResult | TorchTrainingResult:
    """Train one resolved model through its canonical backend engine."""
    if resolved_model.backend == "sklearn":
        if torch_options is not None:
            raise MLJobServiceError("Torch options cannot be supplied to an sklearn model")
        options = sklearn_options or SklearnTrainOptions()
        return fit_sklearn_partition_model(
            dataset,
            resolved_model,
            transform_spec=options.transform_spec,
            balancing=options.balancing,
            seed=options.seed,
            incremental=options.incremental,
            registry=registry,
        )
    if resolved_model.backend == "torch":
        if sklearn_options is not None:
            raise MLJobServiceError("sklearn options cannot be supplied to a Torch model")
        options = torch_options or TorchTrainOptions()
        return fit_torch_partition_model(
            dataset,
            resolved_model,
            training_config=options.training_config,
            transform_spec=options.transform_spec,
            balancing=options.balancing,
            registry=registry,
        )
    raise MLJobServiceError(f"unsupported training backend {resolved_model.backend!r}")


def apply_partition_model(
    model: _FittedModel,
    data: _Data,
    *,
    phase: str | None = None,
    cohort: str | None = None,
    groups: Sequence[str | None] | None = None,
    model_id: str,
) -> PredictionResult:
    """Apply an already-fitted canonical model without invoking training."""
    if isinstance(model, FittedSklearnModel):
        return apply_sklearn_partition_model(
            model,
            data,
            phase=phase,
            cohort=cohort,
            groups=groups,
            model_id=model_id,
        )
    if isinstance(model, FittedTorchModel):
        return apply_torch_partition_model(
            model,
            data,
            phase=phase,
            cohort=cohort,
            groups=groups,
            model_id=model_id,
        )
    raise MLJobServiceError("application requires a canonical fitted sklearn or Torch model")


def evaluate_prediction_result(
    predictions: PredictionResult,
    *,
    threshold: ThresholdProvenance | None = None,
    calibration: CalibrationProvenance | None = None,
    by_modality: bool = True,
) -> EvaluationResult:
    """Evaluate immutable stored predictions without fitting or applying a model."""
    return evaluate_predictions(
        predictions,
        threshold=threshold,
        calibration=calibration,
        by_modality=by_modality,
    )


def explain_partition_model(
    model: _FittedModel,
    data: _Data,
    request: InterpretabilityRequest,
    *,
    background: BackgroundReference | None = None,
) -> AttributionResult:
    """Explain an already-fitted canonical model without invoking training."""
    if isinstance(model, FittedSklearnModel):
        return explain_sklearn_model(model, data, request, background=background)
    if isinstance(model, FittedTorchModel):
        return explain_torch_model(model, data, request, background=background)
    raise MLJobServiceError("explanation requires a canonical fitted sklearn or Torch model")
