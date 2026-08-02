"""Immutable backend-neutral records for classification evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

EVALUATION_SPLITS = frozenset({"train", "validation", "test", "inference"})
IDENTITY_POLICIES = frozenset({"include", "omit", "hash"})


class EvaluationContractError(ValueError):
    """Raised when evaluation data would be ambiguous or leak test information."""


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise EvaluationContractError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise EvaluationContractError(f"{name} must be a non-empty string")
    return result


def _optional_nonempty(value: str | None, name: str) -> str | None:
    return None if value is None else _nonempty(value, name)


def _strings(
    values: Sequence[str],
    *,
    name: str,
    length: int,
) -> tuple[str, ...]:
    result = tuple(_nonempty(value, name) for value in values)
    if len(result) != length:
        raise EvaluationContractError(f"{name} has {len(result)} rows; expected {length}")
    return result


def _optional_strings(
    values: Sequence[str | None] | None,
    *,
    name: str,
    length: int,
) -> tuple[str | None, ...]:
    if values is None:
        return (None,) * length
    result = tuple(_optional_nonempty(value, name) for value in values)
    if len(result) != length:
        raise EvaluationContractError(f"{name} has {len(result)} rows; expected {length}")
    return result


def _readonly_array(
    values: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    if array.shape != shape:
        raise EvaluationContractError(f"{name} has shape {array.shape}; expected {shape}")
    array.setflags(write=False)
    return array


def _readonly_class_ids(
    values: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    raw = np.asarray(values)
    try:
        array = raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(f"{name} must contain integer class IDs") from exc
    if raw.shape != shape or not np.array_equal(raw, array):
        raise EvaluationContractError(f"{name} must have shape {shape} and contain integers")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PredictionResult:
    """One ordered prediction table independent of estimator backend.

    The first six fields retain the original sklearn/Torch result interface.
    Application services populate the additional identity and truth columns so
    the same stored rows can be evaluated later without re-running a model.
    """

    molecule_uids: tuple[str, ...]
    class_ids: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    class_order: tuple[str, ...]
    split: str
    experiment_uids: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    groups: tuple[str | None, ...] | None = None
    truth_class_ids: np.ndarray | None = None
    positive_class: str | None = None
    cohort: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        molecule_uids = tuple(self.molecule_uids)
        n_rows = len(molecule_uids)
        if n_rows == 0:
            raise EvaluationContractError("predictions must contain at least one observation")
        if len(set(molecule_uids)) != n_rows:
            raise EvaluationContractError("molecule_uids must be unique within a prediction cohort")
        object.__setattr__(
            self,
            "molecule_uids",
            _strings(molecule_uids, name="molecule_uids", length=n_rows),
        )
        class_order = tuple(_nonempty(value, "class_order") for value in self.class_order)
        if len(class_order) < 2 or len(set(class_order)) != len(class_order):
            raise EvaluationContractError("class_order must contain at least two unique classes")
        object.__setattr__(self, "class_order", class_order)
        split = _nonempty(self.split, "split")
        if split not in EVALUATION_SPLITS:
            raise EvaluationContractError(f"split must be one of {sorted(EVALUATION_SPLITS)}")
        object.__setattr__(self, "split", split)
        n_classes = len(class_order)
        class_ids = _readonly_class_ids(
            self.class_ids,
            name="class_ids",
            shape=(n_rows,),
        )
        if np.any(class_ids < 0) or np.any(class_ids >= n_classes):
            raise EvaluationContractError("class_ids contain values outside class_order")
        object.__setattr__(self, "class_ids", class_ids)
        scores = _readonly_array(
            self.scores,
            name="scores",
            shape=(n_rows, n_classes),
            dtype=np.float64,
        )
        probabilities = _readonly_array(
            self.probabilities,
            name="probabilities",
            shape=(n_rows, n_classes),
            dtype=np.float64,
        )
        if not np.isfinite(scores).all():
            raise EvaluationContractError("scores must be finite")
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise EvaluationContractError(
                "probabilities must be finite, within [0, 1], and sum to one"
            )
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "probabilities", probabilities)
        experiment_uids = self.experiment_uids or ("unknown",) * n_rows
        modalities = self.modalities or ("unknown",) * n_rows
        object.__setattr__(
            self,
            "experiment_uids",
            _strings(experiment_uids, name="experiment_uids", length=n_rows),
        )
        object.__setattr__(
            self,
            "modalities",
            _strings(modalities, name="modalities", length=n_rows),
        )
        object.__setattr__(
            self,
            "groups",
            _optional_strings(self.groups, name="groups", length=n_rows),
        )
        if self.truth_class_ids is not None:
            truth = _readonly_class_ids(
                self.truth_class_ids,
                name="truth_class_ids",
                shape=(n_rows,),
            )
            if np.any(truth < 0) or np.any(truth >= n_classes):
                raise EvaluationContractError("truth_class_ids contain values outside class_order")
            object.__setattr__(self, "truth_class_ids", truth)
        positive_class = _optional_nonempty(self.positive_class, "positive_class")
        if positive_class is not None and positive_class not in class_order:
            raise EvaluationContractError("positive_class must occur in class_order")
        if n_classes == 2 and positive_class is None:
            positive_class = class_order[1]
        object.__setattr__(self, "positive_class", positive_class)
        object.__setattr__(
            self,
            "cohort",
            split if self.cohort is None else _nonempty(self.cohort, "cohort"),
        )
        object.__setattr__(self, "model_id", _optional_nonempty(self.model_id, "model_id"))

    @property
    def n_observations(self) -> int:
        """Return the number of ordered prediction rows."""
        return len(self.molecule_uids)

    def select(self, indices: Any, *, cohort: str | None = None) -> PredictionResult:
        """Return an immutable row subset while preserving class semantics."""
        selected = np.asarray(indices)
        if selected.dtype == bool:
            if selected.shape != (self.n_observations,):
                raise EvaluationContractError("boolean selection must match prediction rows")
            selected = np.flatnonzero(selected)
        selected = np.asarray(selected, dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise EvaluationContractError("prediction selection must contain row indices")
        if np.any(selected < 0) or np.any(selected >= self.n_observations):
            raise EvaluationContractError("prediction selection contains out-of-range rows")
        return PredictionResult(
            molecule_uids=tuple(self.molecule_uids[index] for index in selected),
            class_ids=self.class_ids[selected],
            scores=self.scores[selected],
            probabilities=self.probabilities[selected],
            class_order=self.class_order,
            split=self.split,
            experiment_uids=tuple(self.experiment_uids[index] for index in selected),
            modalities=tuple(self.modalities[index] for index in selected),
            groups=tuple(self.groups[index] for index in selected),
            truth_class_ids=(
                None if self.truth_class_ids is None else self.truth_class_ids[selected]
            ),
            positive_class=self.positive_class,
            cohort=cohort or self.cohort,
            model_id=self.model_id,
        )

    def to_rows(
        self,
        *,
        identity_policy: Literal["include", "omit", "hash"] = "include",
        hash_salt: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return table rows with explicit sensitive-identity export policy."""
        if identity_policy not in IDENTITY_POLICIES:
            raise EvaluationContractError(
                f"identity_policy must be one of {sorted(IDENTITY_POLICIES)}"
            )
        if identity_policy == "hash" and not hash_salt:
            raise EvaluationContractError("hash_salt is required when identity_policy='hash'")

        def identity(value: str | None) -> str | None:
            if identity_policy == "omit" or value is None:
                return None
            if identity_policy == "include":
                return value
            return hashlib.sha256(f"{hash_salt}:{value}".encode()).hexdigest()

        rows: list[dict[str, Any]] = []
        for index in range(self.n_observations):
            row: dict[str, Any] = {
                "modality": self.modalities[index],
                "split": self.split,
                "cohort": self.cohort,
                "truth_class_id": (
                    None if self.truth_class_ids is None else int(self.truth_class_ids[index])
                ),
                "predicted_class_id": int(self.class_ids[index]),
                "predicted_class": self.class_order[int(self.class_ids[index])],
                "model_id": self.model_id,
            }
            for class_index, class_name in enumerate(self.class_order):
                row[f"score_{class_name}"] = float(self.scores[index, class_index])
                row[f"probability_{class_name}"] = float(self.probabilities[index, class_index])
            if identity_policy != "omit":
                row.update(
                    {
                        "molecule_uid": identity(self.molecule_uids[index]),
                        "experiment_uid": identity(self.experiment_uids[index]),
                        "group": identity(self.groups[index]),
                    }
                )
            rows.append(row)
        return tuple(rows)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        class_order: Sequence[str],
        positive_class: str | None = None,
    ) -> PredictionResult:
        """Restore evaluable predictions from rows emitted by :meth:`to_rows`."""
        stored = tuple(dict(row) for row in rows)
        if not stored:
            raise EvaluationContractError("stored prediction rows cannot be empty")
        classes = tuple(class_order)
        required = {
            "molecule_uid",
            "experiment_uid",
            "group",
            "modality",
            "split",
            "cohort",
            "truth_class_id",
            "predicted_class_id",
            "predicted_class",
            "model_id",
        }
        required.update(f"score_{name}" for name in classes)
        required.update(f"probability_{name}" for name in classes)
        for index, row in enumerate(stored):
            missing = sorted(required.difference(row))
            if missing:
                raise EvaluationContractError(
                    f"stored prediction row {index} lacks required columns: {missing}"
                )
        splits = {row["split"] for row in stored}
        cohorts = {row["cohort"] for row in stored}
        model_ids = {row["model_id"] for row in stored}
        if len(splits) != 1 or len(cohorts) != 1 or len(model_ids) != 1:
            raise EvaluationContractError(
                "stored prediction rows must share split, cohort, and model_id"
            )
        class_ids = np.asarray([row["predicted_class_id"] for row in stored])
        for index, (row, class_id) in enumerate(zip(stored, class_ids, strict=True)):
            if (
                isinstance(class_id, bool)
                or not isinstance(class_id, (int, np.integer))
                or class_id < 0
                or class_id >= len(classes)
                or row["predicted_class"] != classes[int(class_id)]
            ):
                raise EvaluationContractError(
                    f"stored prediction row {index} has inconsistent predicted class"
                )
        truth_values = tuple(row["truth_class_id"] for row in stored)
        if any(value is None for value in truth_values) and not all(
            value is None for value in truth_values
        ):
            raise EvaluationContractError(
                "stored prediction truth_class_id must be present for all rows or none"
            )
        truth = None if truth_values[0] is None else np.asarray(truth_values)
        return cls(
            molecule_uids=tuple(row["molecule_uid"] for row in stored),
            class_ids=class_ids,
            scores=np.asarray([[row[f"score_{name}"] for name in classes] for row in stored]),
            probabilities=np.asarray(
                [[row[f"probability_{name}"] for name in classes] for row in stored]
            ),
            class_order=classes,
            split=next(iter(splits)),
            experiment_uids=tuple(row["experiment_uid"] for row in stored),
            modalities=tuple(row["modality"] for row in stored),
            groups=tuple(row["group"] for row in stored),
            truth_class_ids=truth,
            positive_class=positive_class,
            cohort=next(iter(cohorts)),
            model_id=next(iter(model_ids)),
        )


@dataclass(frozen=True)
class ThresholdProvenance:
    """A decision threshold and the non-test cohort on which it was selected."""

    value: float
    positive_class: str
    method: str
    fitted_split: str | None = None
    fitted_cohort: str | None = None
    model_id: str | None = None
    class_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value) or value < 0 or value > 1:
            raise EvaluationContractError("threshold value must be finite and within [0, 1]")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "positive_class", _nonempty(self.positive_class, "positive_class"))
        object.__setattr__(self, "method", _nonempty(self.method, "method"))
        if self.fitted_split is not None and self.fitted_split not in {"train", "validation"}:
            raise EvaluationContractError(
                "thresholds can only be fit on train or validation data, never locked test data"
            )
        object.__setattr__(
            self,
            "fitted_cohort",
            _optional_nonempty(self.fitted_cohort, "fitted_cohort"),
        )
        object.__setattr__(self, "model_id", _optional_nonempty(self.model_id, "model_id"))
        class_order = tuple(_nonempty(value, "class_order") for value in self.class_order)
        if class_order and (
            len(class_order) < 2
            or len(set(class_order)) != len(class_order)
            or self.positive_class not in class_order
        ):
            raise EvaluationContractError(
                "threshold class_order must be unique and contain positive_class"
            )
        object.__setattr__(self, "class_order", class_order)


@dataclass(frozen=True)
class CalibrationProvenance:
    """Calibration method identity without backend-specific fitted objects."""

    method: str
    fitted_split: str
    fitted_cohort: str
    parameters: Mapping[str, Any]
    model_id: str | None = None
    class_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fitted_split not in {"train", "validation"}:
            raise EvaluationContractError(
                "calibration must be fit on a declared train or validation split"
            )
        object.__setattr__(self, "method", _nonempty(self.method, "method"))
        object.__setattr__(self, "fitted_cohort", _nonempty(self.fitted_cohort, "fitted_cohort"))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "model_id", _optional_nonempty(self.model_id, "model_id"))
        class_order = tuple(_nonempty(value, "class_order") for value in self.class_order)
        if class_order and (len(class_order) < 2 or len(set(class_order)) != len(class_order)):
            raise EvaluationContractError("calibration class_order must contain unique classes")
        object.__setattr__(self, "class_order", class_order)


@dataclass(frozen=True)
class MetricRecord:
    """One scalar metric for a precisely identified evaluation slice."""

    name: str
    value: float | None
    n_observations: int
    split: str
    cohort: str
    scope: str = "pooled"
    modality: str | None = None
    class_name: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "metric name"))
        if self.value is not None:
            value = float(self.value)
            if not np.isfinite(value):
                raise EvaluationContractError("metric value must be finite or null")
            object.__setattr__(self, "value", value)
        if isinstance(self.n_observations, bool) or self.n_observations <= 0:
            raise EvaluationContractError("metric n_observations must be positive")
        if self.split not in EVALUATION_SPLITS:
            raise EvaluationContractError("metric split is invalid")
        object.__setattr__(self, "cohort", _nonempty(self.cohort, "cohort"))
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))


@dataclass(frozen=True)
class CurveRecord:
    """One ROC, PR, calibration, or threshold curve for a named class."""

    kind: str
    x: np.ndarray
    y: np.ndarray
    split: str
    cohort: str
    scope: str = "pooled"
    modality: str | None = None
    class_name: str | None = None
    thresholds: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _nonempty(self.kind, "curve kind"))
        x = np.asarray(self.x, dtype=np.float64).copy()
        y = np.asarray(self.y, dtype=np.float64).copy()
        if x.ndim != 1 or y.shape != x.shape or x.size == 0:
            raise EvaluationContractError("curve x and y must be equal non-empty vectors")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise EvaluationContractError("curve values must be finite")
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if self.thresholds is not None:
            thresholds = np.asarray(self.thresholds, dtype=np.float64).copy()
            if thresholds.ndim != 1 or np.isnan(thresholds).any():
                raise EvaluationContractError("curve thresholds must be a vector without NaN")
            thresholds.setflags(write=False)
            object.__setattr__(self, "thresholds", thresholds)


@dataclass(frozen=True)
class ConfusionRecord:
    """Confusion matrix in explicit persisted class order."""

    matrix: np.ndarray
    class_order: tuple[str, ...]
    split: str
    cohort: str
    scope: str = "pooled"
    modality: str | None = None

    def __post_init__(self) -> None:
        class_order = tuple(_nonempty(value, "class_order") for value in self.class_order)
        if len(class_order) < 2 or len(set(class_order)) != len(class_order):
            raise EvaluationContractError("confusion class_order must be unique")
        object.__setattr__(self, "class_order", class_order)
        matrix = _readonly_class_ids(
            self.matrix,
            name="confusion matrix",
            shape=(len(class_order), len(class_order)),
        )
        if np.any(matrix < 0):
            raise EvaluationContractError("confusion counts cannot be negative")
        object.__setattr__(self, "matrix", matrix)
        if self.split not in EVALUATION_SPLITS:
            raise EvaluationContractError("confusion split is invalid")
        object.__setattr__(self, "cohort", _nonempty(self.cohort, "cohort"))
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))


@dataclass(frozen=True)
class ClassBalanceRecord:
    """Natural observed class support for one evaluation slice."""

    class_name: str
    count: int
    fraction: float
    split: str
    cohort: str
    scope: str = "pooled"
    modality: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_name", _nonempty(self.class_name, "class_name"))
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise EvaluationContractError("class-balance count must be non-negative")
        fraction = float(self.fraction)
        if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
            raise EvaluationContractError("class-balance fraction must be within [0, 1]")
        object.__setattr__(self, "fraction", fraction)
        if self.split not in EVALUATION_SPLITS:
            raise EvaluationContractError("class-balance split is invalid")
        object.__setattr__(self, "cohort", _nonempty(self.cohort, "cohort"))
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))


@dataclass(frozen=True)
class TrainingEvent:
    """One training event; epoch and step are optional by design."""

    event_index: int
    event_type: str
    metrics: Mapping[str, float]
    epoch: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise EvaluationContractError("event_index must be a non-negative integer")
        object.__setattr__(self, "event_type", _nonempty(self.event_type, "event_type"))
        clean: dict[str, float] = {}
        for name, value in self.metrics.items():
            number = float(value)
            if not np.isfinite(number):
                raise EvaluationContractError("training metrics must be finite")
            clean[_nonempty(name, "training metric name")] = number
        object.__setattr__(self, "metrics", MappingProxyType(clean))
        for name in ("epoch", "step"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise EvaluationContractError(f"{name} must be a non-negative integer or null")


@dataclass(frozen=True)
class TrainingHistory:
    """Backend-neutral event history, including valid empty one-shot histories."""

    backend: str
    events: tuple[TrainingEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _nonempty(self.backend, "backend"))
        events = tuple(self.events)
        if tuple(event.event_index for event in events) != tuple(range(len(events))):
            raise EvaluationContractError("training event indices must be contiguous from zero")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True)
class FoldMetricSummary:
    """Aggregate of one matched metric across independent folds."""

    name: str
    fold_names: tuple[str, ...]
    values: tuple[float, ...]
    mean: float
    standard_deviation: float | None
    uncertainty: str
    scope: str
    modality: str | None = None
    class_name: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Complete deterministic evaluation derived from prediction rows."""

    predictions: PredictionResult
    metrics: tuple[MetricRecord, ...]
    curves: tuple[CurveRecord, ...]
    confusion: tuple[ConfusionRecord, ...]
    class_balance: tuple[ClassBalanceRecord, ...]
    threshold: ThresholdProvenance | None = None
    calibration: CalibrationProvenance | None = None
