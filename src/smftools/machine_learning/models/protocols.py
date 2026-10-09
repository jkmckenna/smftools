"""Backend-neutral predictor protocol and composition-based backend adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from smftools.optional_imports import require

from ..contracts import (
    InputSchema,
    LabelSchema,
    PredictorCapabilities,
    assert_input_compatible,
    validate_mask_arrays,
    validate_predictor_masks,
)

_CAPABILITY_FLAGS = frozenset(
    {
        "probability_output",
        "incremental_fit",
        "sample_weights",
        "position_masks",
        "gradients",
        "convolutional_layers",
        "attention_data",
    }
)


class PredictorError(ValueError):
    """Raised when a predictor request violates its schemas or capabilities."""


@runtime_checkable
class PredictorProtocol(Protocol):
    """Small interface used by application and evaluation services."""

    backend: str
    input_schema: InputSchema
    label_schema: LabelSchema
    capabilities: PredictorCapabilities

    def predict(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return integer class IDs in persisted label order."""

    def predict_scores(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return one continuous score column per persisted class."""

    def predict_probabilities(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return one probability column per persisted class."""

    def require_capabilities(self, *names: str) -> None:
        """Reject unsupported prediction or explanation capabilities."""


@runtime_checkable
class PredictorLoaderProtocol(Protocol):
    """Adapter factory for a backend object loaded under an artifact trust policy."""

    def __call__(
        self,
        model: Any,
        *,
        input_schema: InputSchema,
        label_schema: LabelSchema,
        capabilities: PredictorCapabilities,
    ) -> PredictorProtocol:
        """Wrap an already loaded backend model without changing artifact policy."""


def require_capabilities(capabilities: PredictorCapabilities, *names: str) -> None:
    """Raise before execution when requested capability flags are unavailable."""
    unknown = sorted(set(names).difference(_CAPABILITY_FLAGS))
    if unknown:
        raise PredictorError(f"unknown predictor capabilities: {unknown}")
    unsupported = [name for name in names if not getattr(capabilities, name)]
    if unsupported:
        raise PredictorError(
            f"backend {capabilities.backend!r} does not support capabilities: {unsupported}"
        )


def _n_rows(values: Any) -> int:
    shape = tuple(getattr(values, "shape", ()))
    if not shape or shape[0] <= 0:
        raise PredictorError("prediction values must contain at least one observation")
    return int(shape[0])


def _validate_request(
    *,
    values: Any,
    masks: Mapping[str, Any] | None,
    expected_schema: InputSchema,
    actual_schema: InputSchema | None,
    capabilities: PredictorCapabilities,
    phase: str,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    if actual_schema is not None:
        assert_input_compatible(expected_schema, actual_schema)
    supplied = {} if masks is None else dict(masks)
    selected = validate_predictor_masks(
        expected_schema,
        capabilities,
        supplied,
        phase=phase,
    )
    if supplied:
        validate_mask_arrays(
            expected_schema,
            supplied,
            batch_size=_n_rows(values),
            require_all=False,
        )
    return selected, supplied


def _class_ids(label_schema: LabelSchema) -> np.ndarray:
    return np.arange(len(label_schema.class_order), dtype=np.int64)


def _validate_predictions(predictions: Any, *, n_rows: int, n_classes: int) -> np.ndarray:
    values = np.asarray(predictions)
    if values.shape != (n_rows,):
        raise PredictorError(f"predict returned shape {values.shape}; expected ({n_rows},)")
    try:
        integers = values.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise PredictorError("predict must return integer class IDs") from exc
    if not np.array_equal(values, integers):
        raise PredictorError("predict must return integer class IDs")
    if np.any(integers < 0) or np.any(integers >= n_classes):
        raise PredictorError("predict returned class IDs outside persisted label order")
    return integers


def _score_matrix(values: Any, *, n_rows: int, n_classes: int, source: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim == 1 and n_classes == 2 and scores.shape == (n_rows,):
        scores = np.column_stack((np.zeros(n_rows, dtype=np.float64), scores))
    elif scores.ndim == 2 and n_classes == 2 and scores.shape == (n_rows, 1):
        scores = np.column_stack((np.zeros(n_rows, dtype=np.float64), scores[:, 0]))
    if scores.shape != (n_rows, n_classes):
        raise PredictorError(
            f"{source} returned shape {scores.shape}; expected ({n_rows}, {n_classes})"
        )
    if not np.isfinite(scores).all():
        raise PredictorError(f"{source} returned non-finite values")
    return scores


def _probability_matrix(
    values: Any,
    *,
    n_rows: int,
    n_classes: int,
    source: str,
) -> np.ndarray:
    probabilities = _score_matrix(
        values,
        n_rows=n_rows,
        n_classes=n_classes,
        source=source,
    )
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise PredictorError(f"{source} returned values outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise PredictorError(f"{source} rows must sum to one")
    return probabilities


# Only infinities are clipped (an underflowed probability); finite log-odds
# keep their order, however large (naive Bayes on long reads).
LOG_ODDS_LIMIT = 1e6


def positive_log_odds(estimator: Any, features: Any) -> np.ndarray:
    """A binary sklearn estimator's positive-class log-odds, from
    ``predict_log_proba`` where it is exact (naive Bayes), else from
    probabilities floored at half a tree's vote (forests) or 1e-12."""
    trees = getattr(estimator, "estimators_", None)
    log_probability = getattr(estimator, "predict_log_proba", None)
    if trees is None and callable(log_probability):
        logs = np.asarray(log_probability(features), dtype=np.float64)
        odds = logs[:, 1] - logs[:, 0]
    else:
        floor = 0.5 / len(trees) if trees is not None else 1e-12
        probability = np.clip(
            np.asarray(estimator.predict_proba(features), dtype=np.float64), floor, 1 - floor
        )
        odds = np.log(probability[:, 1]) - np.log(probability[:, 0])
    return np.clip(np.nan_to_num(odds, nan=0.0), -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT)


def calibrated_probabilities(
    estimator: Any, features: Any, calibration: Mapping[str, Any]
) -> np.ndarray:
    """Class probabilities after Platt scaling of the positive-class log-odds."""
    if calibration.get("method") != "sigmoid":
        raise PredictorError(f"unsupported calibration {calibration.get('method')!r}")
    odds = float(calibration["slope"]) * positive_log_odds(estimator, features) + float(
        calibration["intercept"]
    )
    positive = 1.0 / (1.0 + np.exp(-np.clip(odds, -500.0, 500.0)))
    return np.column_stack([1.0 - positive, positive])


@dataclass(frozen=True)
class SklearnPredictor:
    """Predictor adapter around one fitted sklearn-compatible estimator."""

    model: Any
    input_schema: InputSchema
    label_schema: LabelSchema
    capabilities: PredictorCapabilities
    backend: str = "sklearn"
    # Platt scaling ({"method": "sigmoid", "slope", "intercept"}) of the
    # model's positive-class log-odds; None: the model's own probabilities.
    calibration: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.capabilities.backend != self.backend:
            raise PredictorError("sklearn predictor requires sklearn capabilities")
        if self.calibration is not None and len(self.label_schema.class_order) != 2:
            raise PredictorError("probability calibration is binary-only")
        if not callable(getattr(self.model, "predict", None)):
            raise PredictorError("sklearn predictor model must define predict")
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise PredictorError("sklearn predictor model must be fitted and expose classes_")
        if not np.array_equal(classes, _class_ids(self.label_schema)):
            raise PredictorError("fitted sklearn classes_ differ from persisted label order")

    def require_capabilities(self, *names: str) -> None:
        """Reject unsupported prediction or explanation capabilities."""
        require_capabilities(self.capabilities, *names)

    def _request(
        self,
        values: Any,
        masks: Mapping[str, Any] | None,
        input_schema: InputSchema | None,
        phase: str,
    ) -> np.ndarray:
        features = np.asarray(values)
        if features.ndim != 2:
            raise PredictorError("sklearn prediction values must be a two-dimensional matrix")
        _validate_request(
            values=features,
            masks=masks,
            expected_schema=self.input_schema,
            actual_schema=input_schema,
            capabilities=self.capabilities,
            phase=phase,
        )
        return features

    def predict(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return integer class IDs in persisted label order."""
        features = self._request(values, masks, input_schema, phase)
        if self.calibration is not None:
            calibrated = calibrated_probabilities(self.model, features, self.calibration)
            return np.argmax(calibrated, axis=1).astype(np.int64)
        return _validate_predictions(
            self.model.predict(features),
            n_rows=len(features),
            n_classes=len(self.label_schema.class_order),
        )

    def predict_scores(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return decision values or probabilities as an ordered score matrix."""
        features = self._request(values, masks, input_schema, phase)
        decision = getattr(self.model, "decision_function", None)
        if self.calibration is not None:
            raw = calibrated_probabilities(self.model, features, self.calibration)
            source = "predict_proba"
        elif callable(decision):
            raw = decision(features)
            source = "decision_function"
        else:
            self.require_capabilities("probability_output")
            probability = getattr(self.model, "predict_proba", None)
            if not callable(probability):
                raise PredictorError("probability capability declared without predict_proba")
            raw = probability(features)
            source = "predict_proba"
        return _score_matrix(
            raw,
            n_rows=len(features),
            n_classes=len(self.label_schema.class_order),
            source=source,
        )

    def predict_probabilities(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return probabilities in persisted class order."""
        self.require_capabilities("probability_output")
        features = self._request(values, masks, input_schema, phase)
        probability = getattr(self.model, "predict_proba", None)
        if not callable(probability):
            raise PredictorError("probability capability declared without predict_proba")
        return _probability_matrix(
            probability(features)
            if self.calibration is None
            else calibrated_probabilities(self.model, features, self.calibration),
            n_rows=len(features),
            n_classes=len(self.label_schema.class_order),
            source="predict_proba",
        )


@dataclass(frozen=True)
class TorchPredictor:
    """Predictor adapter around a plain Torch module returning classification logits.

    Contract-layout position/channel masks are transposed before forwarding so
    Torch models receive channel-first masks matching channel-first values.
    """

    model: Any
    input_schema: InputSchema
    label_schema: LabelSchema
    capabilities: PredictorCapabilities
    device: str | None = None
    mask_argument_names: Mapping[str, str] | None = None
    backend: str = "torch"

    def __post_init__(self) -> None:
        if self.capabilities.backend != self.backend:
            raise PredictorError("Torch predictor requires torch capabilities")
        if not callable(getattr(self.model, "forward", None)):
            raise PredictorError("Torch predictor model must define forward")
        if not callable(getattr(self.model, "eval", None)) or not callable(
            getattr(self.model, "train", None)
        ):
            raise PredictorError("Torch predictor model must define eval and train")
        if not isinstance(getattr(self.model, "training", None), bool):
            raise PredictorError("Torch predictor model must expose boolean training state")
        names = {} if self.mask_argument_names is None else dict(self.mask_argument_names)
        unknown = sorted(set(names).difference(self.capabilities.supported_mask_kinds))
        if unknown:
            raise PredictorError(f"mask argument mapping contains unsupported kinds: {unknown}")
        object.__setattr__(self, "mask_argument_names", names)

    def require_capabilities(self, *names: str) -> None:
        """Reject unsupported prediction or explanation capabilities."""
        require_capabilities(self.capabilities, *names)

    def _logits(
        self,
        values: Any,
        masks: Mapping[str, Any] | None,
        input_schema: InputSchema | None,
        phase: str,
    ) -> np.ndarray:
        torch = require("torch", extra="ml-base", purpose="plain Torch prediction")
        n_rows = _n_rows(values)
        shape = tuple(getattr(values, "shape", ()))
        expected_shape = (
            n_rows,
            len(self.input_schema.channels),
            self.input_schema.n_positions,
        )
        if shape != expected_shape:
            raise PredictorError(
                f"Torch prediction values have shape {shape}; expected channel-first "
                f"shape {expected_shape}"
            )
        selected, supplied = _validate_request(
            values=values,
            masks=masks,
            expected_schema=self.input_schema,
            actual_schema=input_schema,
            capabilities=self.capabilities,
            phase=phase,
        )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        by_name = {mask.name: mask for mask in selected}
        kwargs = {}
        for name, array in supplied.items():
            mask = by_name[name]
            kind = mask.kind
            argument = self.mask_argument_names.get(kind, f"{kind}_mask")
            tensor_mask = torch.as_tensor(array, dtype=torch.bool, device=self.device)
            if mask.axes == ("observation", "position", "channel"):
                tensor_mask = tensor_mask.permute(0, 2, 1)
            elif mask.axes == ("position", "channel"):
                tensor_mask = tensor_mask.permute(1, 0)
            kwargs[argument] = tensor_mask
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(tensor, **kwargs)
        finally:
            self.model.train(was_training)
        logits = output.detach().cpu().numpy()
        return _score_matrix(
            logits,
            n_rows=n_rows,
            n_classes=len(self.label_schema.class_order),
            source="Torch model",
        )

    def predict_scores(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return ordered logits without exposing Torch tensors."""
        return self._logits(values, masks, input_schema, phase)

    def predict_probabilities(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return sigmoid/softmax probabilities in persisted class order."""
        self.require_capabilities("probability_output")
        scores = self._logits(values, masks, input_schema, phase)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(
        self,
        values: Any,
        *,
        masks: Mapping[str, Any] | None = None,
        input_schema: InputSchema | None = None,
        phase: str = "inference",
    ) -> np.ndarray:
        """Return integer class IDs from the ordered logit matrix."""
        scores = self._logits(values, masks, input_schema, phase)
        return np.argmax(scores, axis=1).astype(np.int64)


def adapt_loaded_predictor(
    model: Any,
    *,
    input_schema: InputSchema,
    label_schema: LabelSchema,
    capabilities: PredictorCapabilities,
) -> PredictorProtocol:
    """Wrap a trusted, already-loaded backend object using its declared backend."""
    if capabilities.backend == "sklearn":
        return SklearnPredictor(
            model=model,
            input_schema=input_schema,
            label_schema=label_schema,
            capabilities=capabilities,
        )
    if capabilities.backend == "torch":
        return TorchPredictor(
            model=model,
            input_schema=input_schema,
            label_schema=label_schema,
            capabilities=capabilities,
        )
    raise PredictorError(f"unsupported predictor backend {capabilities.backend!r}")
