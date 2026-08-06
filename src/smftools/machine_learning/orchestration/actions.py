"""Backend-neutral scientific dispatch used by higher-level ML job services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..data.materialized_dataset import MLDatasetProtocol
from ..data.partition_dataset import (
    MLMaterializedPartitionData,
    MLMemoryBudgetError,
    MLPartitionBatch,
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
    fit_sklearn_partition_model_streaming,
    fit_torch_partition_model,
    fit_torch_partition_model_streaming,
)
from .contracts import MLJobServiceError

_FittedModel = FittedSklearnModel | FittedTorchModel
_Data = MLMaterializedPartitionData | MLPartitionBatch


@dataclass(frozen=True)
class SklearnTrainOptions:
    """Explicit sklearn-only options for backend-neutral training dispatch.

    Fields:
        streaming: Whether to fit from streamed batches instead of a
            materialized split. ``None`` streams whenever the family declares
            ``incremental_fit``, because a streamed sklearn fit is
            *numerically identical* to the materialized one -- same
            ``feature_log_prob_``, same predictions, same balance and transform
            identities. There is no reason to make the user ask for it. ``True``
            demands streaming and raises for a family without ``partial_fit``;
            ``False`` forces the materialized path and its memory ceiling.
    """

    transform_spec: FeatureTransformSpec | None = None
    balancing: BalancingSpec | None = None
    seed: int = 0
    incremental: bool | None = None
    streaming: bool | None = None


@dataclass(frozen=True)
class TorchTrainOptions:
    """Explicit plain-Torch-only options for backend-neutral training dispatch.

    Fields:
        streaming: Whether to train from streamed batches. Defaults to
            ``False``, unlike the sklearn option, and the asymmetry is
            deliberate: a streamed Torch fit shuffles within a buffer rather
            than globally, so it produces **different weights** from a
            materialized fit at the same seed. Switching strategy silently
            would change a user's model without them asking. Opting in is one
            argument, and the materialization refusal names it.
    """

    training_config: TorchTrainingConfig | None = None
    transform_spec: FeatureTransformSpec | None = None
    balancing: BalancingSpec | None = None
    streaming: bool = False


def _materialized_or_guided(operation, *, remedy: str):
    """Run a materializing fit, naming the streaming remedy if it is refused.

    ``MLMemoryBudgetError`` reports the estimate and the budget, which is the
    right message for a bare read. At the training boundary the caller also
    needs to know a streaming engine exists, so the refusal is re-raised with
    that named. The original is preserved as the cause.
    """
    try:
        return operation()
    except MLMemoryBudgetError as exc:
        raise MLJobServiceError(
            f"training refused: the train split exceeds max_materialization_bytes ({exc}). "
            f"Retry with {remedy}."
        ) from exc


def train_partition_model(
    dataset: MLDatasetProtocol,
    resolved_model: ResolvedModelDefinition,
    *,
    sklearn_options: SklearnTrainOptions | None = None,
    torch_options: TorchTrainOptions | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> SklearnTrainingResult | TorchTrainingResult:
    """Train one resolved model through its canonical backend engine.

    Dispatches to a streaming or materializing engine per backend. See
    :class:`SklearnTrainOptions` and :class:`TorchTrainOptions` for why the
    defaults differ: a streamed sklearn fit reproduces the materialized one
    exactly, while a streamed Torch fit does not.
    """
    if resolved_model.backend == "sklearn":
        if torch_options is not None:
            raise MLJobServiceError("Torch options cannot be supplied to an sklearn model")
        options = sklearn_options or SklearnTrainOptions()
        incremental_capable = bool(resolved_model.capabilities.incremental_fit)
        use_streaming = incremental_capable if options.streaming is None else options.streaming
        if use_streaming and not incremental_capable:
            raise MLJobServiceError(
                f"model family {resolved_model.family!r} cannot stream: it has no partial_fit. "
                "Use SklearnTrainOptions(streaming=False) to materialize the train split, which "
                "is bounded by max_materialization_bytes."
            )
        if use_streaming:
            if options.incremental is False:
                raise MLJobServiceError(
                    "streaming=True and incremental=False are contradictory; a streamed sklearn "
                    "fit is always incremental"
                )
            return fit_sklearn_partition_model_streaming(
                dataset,
                resolved_model,
                transform_spec=options.transform_spec,
                balancing=options.balancing,
                seed=options.seed,
                registry=registry,
            )
        return _materialized_or_guided(
            lambda: fit_sklearn_partition_model(
                dataset,
                resolved_model,
                transform_spec=options.transform_spec,
                balancing=options.balancing,
                seed=options.seed,
                incremental=options.incremental,
                registry=registry,
            ),
            remedy=(
                "SklearnTrainOptions(streaming=True)"
                if incremental_capable
                else (
                    f"a streaming-capable family (model family {resolved_model.family!r} has no "
                    "partial_fit), or raise max_materialization_bytes if the split genuinely fits"
                )
            ),
        )
    if resolved_model.backend == "torch":
        if sklearn_options is not None:
            raise MLJobServiceError("sklearn options cannot be supplied to a Torch model")
        options = torch_options or TorchTrainOptions()
        if options.streaming:
            return fit_torch_partition_model_streaming(
                dataset,
                resolved_model,
                training_config=options.training_config,
                transform_spec=options.transform_spec,
                balancing=options.balancing,
                registry=registry,
            )
        return _materialized_or_guided(
            lambda: fit_torch_partition_model(
                dataset,
                resolved_model,
                training_config=options.training_config,
                transform_spec=options.transform_spec,
                balancing=options.balancing,
                registry=registry,
            ),
            remedy=(
                "TorchTrainOptions(streaming=True), noting that a streamed fit shuffles within a "
                "buffer rather than globally and therefore produces different weights"
            ),
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
