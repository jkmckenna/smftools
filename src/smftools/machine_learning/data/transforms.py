"""Leakage-safe fitted feature transforms for partition-backed ML data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from smftools.optional_imports import require

from .partition_dataset import MLMaterializedPartitionData, MLPartitionBatch

ML_FEATURE_TRANSFORM_VERSION = 1
_IMPUTATION_METHODS = frozenset({"constant", "mean", "median", "most_frequent"})
_SCALING_METHODS = frozenset({"none", "standard"})
_INDICATOR_KINDS = frozenset({"observed", "design", "availability", "padding"})
_INDICATOR_ORDER = ("observed", "design", "availability", "padding")


class MLTransformError(ValueError):
    """Raised when a fitted feature transform is invalid or used unsafely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MLTransformError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise MLTransformError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _molecule_digest(values: Sequence[str]) -> str:
    return _sha256({"molecule_uids": list(values)})


@dataclass(frozen=True)
class FeatureTransformSpec:
    """User-tunable imputation, scaling, and indicator policy."""

    imputation: str = "constant"
    fill_value: float = 0.0
    scaling: str = "none"
    indicators: tuple[str, ...] = _INDICATOR_ORDER

    def __post_init__(self) -> None:
        imputation = str(self.imputation).strip().lower()
        scaling = str(self.scaling).strip().lower()
        indicators = tuple(str(item).strip().lower() for item in self.indicators)
        if imputation not in _IMPUTATION_METHODS:
            raise MLTransformError(f"imputation must be one of {sorted(_IMPUTATION_METHODS)}")
        if scaling not in _SCALING_METHODS:
            raise MLTransformError(f"scaling must be one of {sorted(_SCALING_METHODS)}")
        if not np.isfinite(self.fill_value):
            raise MLTransformError("fill_value must be finite")
        unknown = sorted(set(indicators).difference(_INDICATOR_KINDS))
        if unknown:
            raise MLTransformError(f"unknown indicator kinds: {unknown}")
        if len(indicators) != len(set(indicators)):
            raise MLTransformError("indicators cannot contain duplicates")
        canonical = tuple(kind for kind in _INDICATOR_ORDER if kind in indicators)
        object.__setattr__(self, "imputation", imputation)
        object.__setattr__(self, "fill_value", float(self.fill_value))
        object.__setattr__(self, "scaling", scaling)
        object.__setattr__(self, "indicators", canonical)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable transform declaration."""
        return {
            "imputation": self.imputation,
            "fill_value": self.fill_value,
            "scaling": self.scaling,
            "indicators": list(self.indicators),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureTransformSpec:
        """Validate and restore a transform declaration."""
        expected = {"imputation", "fill_value", "scaling", "indicators"}
        if set(raw) != expected:
            raise MLTransformError(f"transform spec fields must be exactly {sorted(expected)}")
        indicators = raw["indicators"]
        if not isinstance(indicators, Sequence) or isinstance(indicators, (str, bytes)):
            raise MLTransformError("indicators must be a sequence")
        return cls(
            imputation=str(raw["imputation"]),
            fill_value=float(raw["fill_value"]),
            scaling=str(raw["scaling"]),
            indicators=tuple(str(item) for item in indicators),
        )


@runtime_checkable
class FittedFeatureTransformProtocol(Protocol):
    """Minimal backend-neutral interface for reusable fitted transforms."""

    transform_id: str

    def transform(self, data: MLMaterializedPartitionData | MLPartitionBatch) -> np.ndarray:
        """Transform data without changing fitted state."""

    def to_dict(self) -> dict[str, Any]:
        """Return complete serialized fitted state and provenance."""


def _expanded_masks(
    data: MLMaterializedPartitionData | MLPartitionBatch,
) -> dict[str, np.ndarray]:
    values = np.asarray(data.values)
    if values.ndim != 3:
        raise MLTransformError("values must have observation, position, and channel axes")
    n_rows, n_positions, n_channels = values.shape
    observed = np.asarray(data.observed_mask, dtype=bool)
    availability = np.asarray(data.availability_mask, dtype=bool)
    design = np.asarray(data.design_mask, dtype=bool)
    padding = np.asarray(data.padding_mask, dtype=bool)
    if observed.shape != values.shape:
        raise MLTransformError("observed mask shape does not match values")
    if availability.shape != (n_rows, n_channels):
        raise MLTransformError("availability mask shape does not match values")
    if design.shape == (n_positions, n_channels):
        design = np.broadcast_to(design, values.shape)
    elif design.shape != values.shape:
        raise MLTransformError("design mask shape does not match values")
    if padding.shape != (n_rows, n_positions):
        raise MLTransformError("padding mask shape does not match values")
    return {
        "observed": observed,
        "design": design,
        "availability": np.broadcast_to(availability[:, np.newaxis, :], values.shape),
        "padding": np.broadcast_to(padding[:, :, np.newaxis], values.shape),
    }


def _raw_feature_rows(
    data: MLMaterializedPartitionData | MLPartitionBatch,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    values = np.asarray(data.values, dtype=np.float32)
    masks = _expanded_masks(data)
    valid = (
        np.isfinite(values)
        & masks["observed"]
        & masks["design"]
        & masks["availability"]
        & ~masks["padding"]
    )
    raw = values.reshape(values.shape[0], -1)
    flattened = {name: mask.reshape(values.shape[0], -1) for name, mask in masks.items()}
    return raw, flattened, valid.reshape(values.shape[0], -1)


def _fill_values(raw: np.ndarray, valid: np.ndarray, spec: FeatureTransformSpec) -> np.ndarray:
    result = np.full(raw.shape[1], spec.fill_value, dtype=np.float64)
    if spec.imputation == "constant":
        return result
    for column in range(raw.shape[1]):
        values = raw[valid[:, column], column]
        if not len(values):
            continue
        if spec.imputation == "mean":
            result[column] = float(np.mean(values, dtype=np.float64))
        elif spec.imputation == "median":
            result[column] = float(np.median(values))
        else:
            unique, counts = np.unique(values, return_counts=True)
            result[column] = float(unique[np.flatnonzero(counts == counts.max())[0]])
    return result


def _impute(raw: np.ndarray, valid: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    return np.where(valid, raw, fill_values[np.newaxis, :]).astype(np.float64, copy=False)


def _feature_names(
    coordinates: Sequence[int],
    channel_names: Sequence[str],
    indicators: Sequence[str],
) -> tuple[str, ...]:
    signal = [
        f"signal:{channel}@{coordinate}" for coordinate in coordinates for channel in channel_names
    ]
    extra = [
        f"{kind}:{channel}@{coordinate}"
        for kind in indicators
        for coordinate in coordinates
        for channel in channel_names
    ]
    return tuple((*signal, *extra))


@dataclass(frozen=True)
class FittedFeatureTransform:
    """Immutable fitted state shared by sklearn and Torch adapters."""

    schema_version: int
    transform_id: str
    spec: FeatureTransformSpec
    dataset_snapshot_id: str
    split_id: str
    fit_molecule_digest: str
    n_positions: int
    channel_names: tuple[str, ...]
    coordinates: tuple[int, ...]
    fill_values: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ML_FEATURE_TRANSFORM_VERSION:
            raise MLTransformError(
                f"unsupported transform version {self.schema_version}; "
                f"expected {ML_FEATURE_TRANSFORM_VERSION}"
            )
        _digest(self.transform_id, "transform_id")
        _digest(self.dataset_snapshot_id, "dataset_snapshot_id")
        _digest(self.split_id, "split_id")
        _digest(self.fit_molecule_digest, "fit_molecule_digest")
        channels = tuple(str(item) for item in self.channel_names)
        coordinates = tuple(int(item) for item in self.coordinates)
        if not channels or len(channels) != len(set(channels)):
            raise MLTransformError("channel_names must be non-empty and unique")
        if len(coordinates) != self.n_positions or self.n_positions <= 0:
            raise MLTransformError("coordinates must match positive n_positions")
        n_signal = self.n_positions * len(channels)
        arrays = []
        for name in ("fill_values", "centers", "scales"):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if array.shape != (n_signal,) or not np.isfinite(array).all():
                raise MLTransformError(f"{name} must contain one finite value per signal feature")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
            arrays.append(array)
        if np.any(self.scales <= 0):
            raise MLTransformError("scales must be positive")
        expected_names = _feature_names(coordinates, channels, self.spec.indicators)
        if tuple(self.feature_names) != expected_names:
            raise MLTransformError("feature_names do not match coordinates, channels, and spec")
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "feature_names", expected_names)
        identity = self._identity_dict()
        if self.transform_id != _sha256(identity):
            raise MLTransformError("transform_id does not match fitted state and provenance")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
            "fit_molecule_digest": self.fit_molecule_digest,
            "n_positions": self.n_positions,
            "channel_names": list(self.channel_names),
            "coordinates": list(self.coordinates),
            "fill_values": self.fill_values.tolist(),
            "centers": self.centers.tolist(),
            "scales": self.scales.tolist(),
            "feature_names": list(self.feature_names),
        }

    def transform(self, data: MLMaterializedPartitionData | MLPartitionBatch) -> np.ndarray:
        """Apply immutable training-fitted state to any compatible role."""
        signal, masks = self._transform_signal(data)
        matrices = [signal]
        matrices.extend(masks[kind].astype(np.float32) for kind in self.spec.indicators)
        return np.concatenate(matrices, axis=1)

    def _transform_signal(
        self, data: MLMaterializedPartitionData | MLPartitionBatch
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return transformed signals and flattened masks for backend adapters."""
        if tuple(data.channel_names) != self.channel_names:
            raise MLTransformError("data channel order differs from fitted transform")
        if tuple(map(int, data.coordinates)) != self.coordinates:
            raise MLTransformError("data coordinates differ from fitted transform")
        raw, masks, valid = _raw_feature_rows(data)
        signal = _impute(raw, valid, self.fill_values)
        signal = (signal - self.centers[np.newaxis, :]) / self.scales[np.newaxis, :]
        return signal.astype(np.float32), masks

    def to_dict(self) -> dict[str, Any]:
        """Return complete fitted state for safe JSON serialization."""
        return {"transform_id": self.transform_id, **self._identity_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FittedFeatureTransform:
        """Strictly validate and restore serialized fitted state."""
        expected = {
            "schema_version",
            "transform_id",
            "spec",
            "dataset_snapshot_id",
            "split_id",
            "fit_molecule_digest",
            "n_positions",
            "channel_names",
            "coordinates",
            "fill_values",
            "centers",
            "scales",
            "feature_names",
        }
        if set(raw) != expected:
            raise MLTransformError(f"fitted transform fields must be exactly {sorted(expected)}")
        return cls(
            schema_version=int(raw["schema_version"]),
            transform_id=str(raw["transform_id"]),
            spec=FeatureTransformSpec.from_dict(raw["spec"]),
            dataset_snapshot_id=str(raw["dataset_snapshot_id"]),
            split_id=str(raw["split_id"]),
            fit_molecule_digest=str(raw["fit_molecule_digest"]),
            n_positions=int(raw["n_positions"]),
            channel_names=tuple(str(item) for item in raw["channel_names"]),
            coordinates=tuple(int(item) for item in raw["coordinates"]),
            fill_values=np.asarray(raw["fill_values"], dtype=np.float64),
            centers=np.asarray(raw["centers"], dtype=np.float64),
            scales=np.asarray(raw["scales"], dtype=np.float64),
            feature_names=tuple(str(item) for item in raw["feature_names"]),
        )


def fit_feature_transform(
    training_data: MLMaterializedPartitionData,
    spec: FeatureTransformSpec,
    *,
    dataset_snapshot_id: str,
    split_id: str,
) -> FittedFeatureTransform:
    """Fit one transform, refusing any role other than the immutable train split."""
    if training_data.split != "train":
        raise MLTransformError("fitted transforms may only be fit on the 'train' role")
    dataset_snapshot_id = _digest(dataset_snapshot_id, "dataset_snapshot_id")
    split_id = _digest(split_id, "split_id")
    raw, _masks, valid = _raw_feature_rows(training_data)
    fill_values = _fill_values(raw, valid, spec)
    imputed = _impute(raw, valid, fill_values)
    if spec.scaling == "standard":
        centers = np.mean(imputed, axis=0, dtype=np.float64)
        scales = np.std(imputed, axis=0, dtype=np.float64)
        scales[scales == 0] = 1.0
    else:
        centers = np.zeros(raw.shape[1], dtype=np.float64)
        scales = np.ones(raw.shape[1], dtype=np.float64)
    molecule_digest = _molecule_digest(training_data.molecule_uids)
    names = _feature_names(
        training_data.coordinates,
        training_data.channel_names,
        spec.indicators,
    )
    identity = {
        "schema_version": ML_FEATURE_TRANSFORM_VERSION,
        "spec": spec.to_dict(),
        "dataset_snapshot_id": dataset_snapshot_id,
        "split_id": split_id,
        "fit_molecule_digest": molecule_digest,
        "n_positions": len(training_data.coordinates),
        "channel_names": list(training_data.channel_names),
        "coordinates": list(map(int, training_data.coordinates)),
        "fill_values": fill_values.tolist(),
        "centers": centers.tolist(),
        "scales": scales.tolist(),
        "feature_names": list(names),
    }
    return FittedFeatureTransform(
        schema_version=ML_FEATURE_TRANSFORM_VERSION,
        transform_id=_sha256(identity),
        spec=spec,
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        fit_molecule_digest=molecule_digest,
        n_positions=len(training_data.coordinates),
        channel_names=training_data.channel_names,
        coordinates=tuple(map(int, training_data.coordinates)),
        fill_values=fill_values,
        centers=centers,
        scales=scales,
        feature_names=names,
    )


class ManifestFeatureTransformer(TransformerMixin, BaseEstimator):
    """Sklearn-compatible estimator that can fit only manifest-labeled training data."""

    def __init__(
        self,
        spec: FeatureTransformSpec | None = None,
        dataset_snapshot_id: str | None = None,
        split_id: str | None = None,
    ) -> None:
        self.spec = spec
        self.dataset_snapshot_id = dataset_snapshot_id
        self.split_id = split_id

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn clone parameters."""
        return {
            "spec": self.spec,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
        }

    def set_params(self, **parameters: Any) -> ManifestFeatureTransformer:
        """Set sklearn clone parameters."""
        unknown = sorted(set(parameters).difference(self.get_params()))
        if unknown:
            raise ValueError(f"unknown transform parameters: {unknown}")
        for name, value in parameters.items():
            setattr(self, name, value)
        return self

    def fit(
        self,
        data: MLMaterializedPartitionData,
        y: Any = None,
    ) -> ManifestFeatureTransformer:
        """Fit state from training data only; ``y`` is accepted for sklearn compatibility."""
        if self.dataset_snapshot_id is None or self.split_id is None:
            raise MLTransformError("dataset_snapshot_id and split_id are required before fit")
        if y is not None and data.labels is not None:
            supplied = np.asarray(y)
            if supplied.shape != data.labels.shape or not np.array_equal(supplied, data.labels):
                raise MLTransformError("supplied y does not match manifest labels")
        self.fitted_transform_ = fit_feature_transform(
            data,
            self.spec or FeatureTransformSpec(),
            dataset_snapshot_id=self.dataset_snapshot_id,
            split_id=self.split_id,
        )
        return self

    def transform(self, data: MLMaterializedPartitionData | MLPartitionBatch) -> np.ndarray:
        """Apply fitted state without mutation or refitting."""
        if not hasattr(self, "fitted_transform_"):
            raise MLTransformError("transformer is not fitted")
        return self.fitted_transform_.transform(data)

    def fit_transform(
        self,
        data: MLMaterializedPartitionData,
        y: Any = None,
        **fit_parameters: Any,
    ) -> np.ndarray:
        """Fit on training data and return its transformed matrix."""
        if fit_parameters:
            raise MLTransformError(f"unsupported sklearn fit parameters: {sorted(fit_parameters)}")
        return self.fit(data, y).transform(data)


def build_sklearn_preprocessing_pipeline(
    spec: FeatureTransformSpec,
    *,
    dataset_snapshot_id: str,
    split_id: str,
):
    """Build an unfitted sklearn pipeline whose only fit-capable role is train."""
    sklearn_pipeline = require(
        "sklearn.pipeline",
        extra="ml-base",
        purpose="manifest-aware ML feature preprocessing",
    )
    return sklearn_pipeline.Pipeline(
        [
            (
                "features",
                ManifestFeatureTransformer(
                    spec=spec,
                    dataset_snapshot_id=dataset_snapshot_id,
                    split_id=split_id,
                ),
            )
        ]
    )


@dataclass(frozen=True)
class TorchTransformedBatch:
    """Channel-first Torch values with masks kept as separate semantic tensors."""

    values: Any
    labels: Any | None
    observed_mask: Any
    availability_mask: Any
    design_mask: Any
    padding_mask: Any


class TorchFeatureTransform:
    """Torch adapter returning channel-first signals plus distinct scientific masks."""

    def __init__(self, fitted: FittedFeatureTransform, *, device: str | None = None):
        self.fitted = fitted
        self.device = device

    def __call__(
        self,
        data: MLMaterializedPartitionData | MLPartitionBatch,
    ) -> TorchTransformedBatch:
        """Apply fitted state without folding mask meanings into the signal tensor."""
        torch = require(
            "torch",
            extra="ml-base",
            purpose="manifest-aware ML feature preprocessing",
        )
        signal, _masks = self.fitted._transform_signal(data)
        n_rows = signal.shape[0]
        values = signal.reshape(
            n_rows,
            self.fitted.n_positions,
            len(self.fitted.channel_names),
        ).transpose(0, 2, 1)
        labels = None if data.labels is None else np.asarray(data.labels, dtype=np.int64)
        design = np.asarray(data.design_mask, dtype=bool)
        design = design.T if design.ndim == 2 else design.transpose(0, 2, 1)
        return TorchTransformedBatch(
            values=torch.as_tensor(values, dtype=torch.float32, device=self.device),
            labels=(
                None
                if labels is None
                else torch.as_tensor(labels, dtype=torch.long, device=self.device)
            ),
            observed_mask=torch.as_tensor(
                np.asarray(data.observed_mask, dtype=bool).transpose(0, 2, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            availability_mask=torch.as_tensor(
                np.asarray(data.availability_mask, dtype=bool),
                dtype=torch.bool,
                device=self.device,
            ),
            design_mask=torch.as_tensor(
                design,
                dtype=torch.bool,
                device=self.device,
            ),
            padding_mask=torch.as_tensor(
                np.asarray(data.padding_mask, dtype=bool),
                dtype=torch.bool,
                device=self.device,
            ),
        )
