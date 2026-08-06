"""Deterministic training-only class balancing and sensitivity cohorts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from smftools.optional_imports import require

from ..contracts import LabelSchema
from ..plan import BalancingSpec
from .partition_dataset import MLMaterializedPartitionData, MLPartitionDataPlan

ML_BALANCE_RESOLUTION_VERSION = 1
_TRAIN_METHODS = frozenset(
    {"natural", "class_weight", "weighted_sampler", "downsample", "upsample"}
)


class MLBalanceError(ValueError):
    """Raised when class balancing would violate split or label contracts."""


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
        raise MLBalanceError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise MLBalanceError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _validated_labels(
    labels: Any,
    role: str,
    label_schema: LabelSchema,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if labels is None:
        raise MLBalanceError("balancing requires supervised labels")
    labels = np.asarray(labels, dtype=np.int64)
    expected_ids = tuple(range(len(label_schema.class_order)))
    unknown = sorted(set(map(int, labels)).difference(expected_ids))
    if unknown:
        raise MLBalanceError(f"labels contain class IDs outside persisted class order: {unknown}")
    counts = tuple(int(np.count_nonzero(labels == class_id)) for class_id in expected_ids)
    absent = [label_schema.class_order[index] for index, count in enumerate(counts) if count == 0]
    if absent:
        raise MLBalanceError(f"role {role!r} is missing persisted classes: {absent}")
    return labels, counts


def _labels(
    data: MLMaterializedPartitionData,
    label_schema: LabelSchema,
) -> tuple[np.ndarray, tuple[int, ...]]:
    return _validated_labels(data.labels, data.split, label_schema)


def _balanced_class_weights(counts: tuple[int, ...]) -> np.ndarray:
    total = sum(counts)
    return np.asarray(
        [total / (len(counts) * count) for count in counts],
        dtype=np.float64,
    )


def _resampled_indices(labels: np.ndarray, method: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_ids = tuple(sorted(set(map(int, labels))))
    by_class = {class_id: np.flatnonzero(labels == class_id) for class_id in class_ids}
    target = (
        min(len(indices) for indices in by_class.values())
        if method == "downsample"
        else max(len(indices) for indices in by_class.values())
    )
    selected = []
    for class_id in class_ids:
        indices = by_class[class_id]
        selected.append(
            rng.choice(
                indices,
                size=target,
                replace=method == "upsample" and len(indices) < target,
            )
        )
    return rng.permutation(np.concatenate(selected)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class BalanceResolution:
    """Immutable indices/weights and provenance for one split role."""

    schema_version: int
    resolution_id: str
    dataset_snapshot_id: str
    split_id: str
    role: str
    purpose: str
    method: str
    seed: int
    class_order: tuple[str, ...]
    source_counts: tuple[int, ...]
    result_counts: tuple[int, ...]
    source_molecule_digest: str
    selected_molecule_digest: str
    selected_indices: np.ndarray
    class_weights: np.ndarray | None
    sample_weights: np.ndarray | None

    def __post_init__(self) -> None:
        if self.schema_version != ML_BALANCE_RESOLUTION_VERSION:
            raise MLBalanceError(
                f"unsupported balance version {self.schema_version}; "
                f"expected {ML_BALANCE_RESOLUTION_VERSION}"
            )
        _digest(self.resolution_id, "resolution_id")
        _digest(self.dataset_snapshot_id, "dataset_snapshot_id")
        _digest(self.split_id, "split_id")
        _digest(self.source_molecule_digest, "source_molecule_digest")
        _digest(self.selected_molecule_digest, "selected_molecule_digest")
        if self.role not in {"train", "validation", "test"}:
            raise MLBalanceError("role must be train, validation, or test")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise MLBalanceError("purpose must be a non-empty string")
        if self.method not in _TRAIN_METHODS:
            raise MLBalanceError(f"unsupported balance method {self.method!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise MLBalanceError("seed must be a non-negative integer")
        class_order = tuple(str(item) for item in self.class_order)
        if len(class_order) < 2 or len(class_order) != len(set(class_order)):
            raise MLBalanceError("class_order must contain at least two unique classes")
        if len(self.source_counts) != len(class_order) or len(self.result_counts) != len(
            class_order
        ):
            raise MLBalanceError("class counts must follow class_order")
        if any(count <= 0 for count in (*self.source_counts, *self.result_counts)):
            raise MLBalanceError("every persisted class must have a positive count")
        indices = np.asarray(self.selected_indices, dtype=np.int64).copy()
        if indices.ndim != 1 or np.any(indices < 0):
            raise MLBalanceError("selected_indices must be one-dimensional and non-negative")
        indices.setflags(write=False)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "class_order", class_order)
        for name, expected_length in (
            ("class_weights", len(class_order)),
            ("sample_weights", sum(self.source_counts)),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64).copy()
            if array.shape != (expected_length,) or not np.isfinite(array).all():
                raise MLBalanceError(f"{name} has invalid shape or non-finite values")
            if np.any(array <= 0):
                raise MLBalanceError(f"{name} must be positive")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        expected_id = _sha256(self._identity_dict())
        if self.resolution_id != expected_id:
            raise MLBalanceError("resolution_id does not match balancing provenance")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
            "role": self.role,
            "purpose": self.purpose,
            "method": self.method,
            "seed": self.seed,
            "class_order": list(self.class_order),
            "source_counts": list(self.source_counts),
            "result_counts": list(self.result_counts),
            "source_molecule_digest": self.source_molecule_digest,
            "selected_molecule_digest": self.selected_molecule_digest,
            "selected_indices": self.selected_indices.tolist(),
            "class_weights": (None if self.class_weights is None else self.class_weights.tolist()),
            "sample_weights": (
                None if self.sample_weights is None else self.sample_weights.tolist()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return complete indices, weights, class order, and provenance."""
        return {"resolution_id": self.resolution_id, **self._identity_dict()}

    def torch_weighted_sampler(self):
        """Build a deterministic Torch sampler for a weighted-sampler resolution."""
        if self.method != "weighted_sampler" or self.sample_weights is None:
            raise MLBalanceError("this resolution does not define weighted sampling")
        torch = require("torch", extra="ml-base", purpose="weighted ML sampling")
        torch_data = require(
            "torch.utils.data",
            extra="ml-base",
            purpose="weighted ML sampling",
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch_data.WeightedRandomSampler(
            weights=torch.tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
            generator=generator,
        )


def _resolve_from_labels(
    labels: Any,
    molecule_uids: Sequence[str],
    role: str,
    label_schema: LabelSchema,
    *,
    method: str,
    seed: int,
    dataset_snapshot_id: str,
    split_id: str,
    purpose: str,
    allow_evaluation_resampling: bool,
) -> BalanceResolution:
    """Resolve a balance from labels and identities alone.

    Balancing never reads feature values, so this core needs only the label
    vector, the molecule identities, and the role name. That is what lets
    :func:`resolve_role_balance_from_plan` resolve a cohort from read-plan
    metadata without decoding a single batch.
    """
    dataset_snapshot_id = _digest(dataset_snapshot_id, "dataset_snapshot_id")
    split_id = _digest(split_id, "split_id")
    method = str(method).strip().lower()
    if method not in _TRAIN_METHODS:
        raise MLBalanceError(f"unsupported balance method {method!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MLBalanceError("seed must be a non-negative integer")
    if role != "train" and method != "natural" and not allow_evaluation_resampling:
        raise MLBalanceError("validation and test primary cohorts must retain natural prevalence")
    labels, counts = _validated_labels(labels, role, label_schema)
    molecule_uids = tuple(str(item) for item in molecule_uids)
    if len(molecule_uids) != len(labels):
        raise MLBalanceError("molecule identities and labels must have the same length")
    weights = _balanced_class_weights(counts)
    class_weights = weights if method in {"class_weight", "weighted_sampler"} else None
    sample_weights = weights[labels] if method == "weighted_sampler" else None
    if method in {"downsample", "upsample"}:
        indices = _resampled_indices(labels, method, seed)
    else:
        indices = np.arange(len(labels), dtype=np.int64)
    result_counts = tuple(
        int(np.count_nonzero(labels[indices] == item)) for item in range(len(counts))
    )
    source_molecule_digest = _sha256({"molecule_uids": list(molecule_uids)})
    selected_uids = [molecule_uids[index] for index in indices]
    selected_molecule_digest = _sha256({"molecule_uids": selected_uids})
    identity = {
        "schema_version": ML_BALANCE_RESOLUTION_VERSION,
        "dataset_snapshot_id": dataset_snapshot_id,
        "split_id": split_id,
        "role": role,
        "purpose": purpose,
        "method": method,
        "seed": seed,
        "class_order": list(label_schema.class_order),
        "source_counts": list(counts),
        "result_counts": list(result_counts),
        "source_molecule_digest": source_molecule_digest,
        "selected_molecule_digest": selected_molecule_digest,
        "selected_indices": indices.tolist(),
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "sample_weights": None if sample_weights is None else sample_weights.tolist(),
    }
    return BalanceResolution(
        schema_version=ML_BALANCE_RESOLUTION_VERSION,
        resolution_id=_sha256(identity),
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        role=role,
        purpose=purpose,
        method=method,
        seed=seed,
        class_order=label_schema.class_order,
        source_counts=counts,
        result_counts=result_counts,
        source_molecule_digest=source_molecule_digest,
        selected_molecule_digest=selected_molecule_digest,
        selected_indices=indices,
        class_weights=class_weights,
        sample_weights=sample_weights,
    )


def _resolve(
    data: MLMaterializedPartitionData,
    label_schema: LabelSchema,
    *,
    method: str,
    seed: int,
    dataset_snapshot_id: str,
    split_id: str,
    purpose: str,
    allow_evaluation_resampling: bool,
) -> BalanceResolution:
    return _resolve_from_labels(
        data.labels,
        data.molecule_uids,
        data.split,
        label_schema,
        method=method,
        seed=seed,
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        purpose=purpose,
        allow_evaluation_resampling=allow_evaluation_resampling,
    )


def resolve_role_balance_from_plan(
    plan: MLPartitionDataPlan,
    label_schema: LabelSchema,
    balancing: BalancingSpec,
    *,
    role: str = "train",
    seed: int,
    dataset_snapshot_id: str,
    split_id: str,
) -> BalanceResolution:
    """Resolve a primary cohort from read-plan metadata, reading no data.

    Balancing depends only on labels and molecule identities, and
    ``PartitionReadEntry`` already carries both. A cohort can therefore be
    resolved before a single batch is decoded, which is what makes balancing
    free for a streaming fit rather than a reason to materialize.

    Produces a resolution identical to :func:`resolve_role_balance` on the same
    role, because the plan's canonical entry order is the order
    ``materialize`` returns.
    """
    role_specs = {
        "train": balancing.train,
        "validation": balancing.validation,
        "test": balancing.test,
    }
    if role not in role_specs:
        raise MLBalanceError(f"unsupported split role {role!r}")
    entries = plan.entries_for(role)
    if any(entry.class_id is None for entry in entries):
        raise MLBalanceError("balancing requires supervised labels")
    return _resolve_from_labels(
        np.asarray([entry.class_id for entry in entries], dtype=np.int64),
        tuple(entry.molecule_uid for entry in entries),
        role,
        label_schema,
        method=role_specs[role].method,
        seed=seed,
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        purpose="primary",
        allow_evaluation_resampling=False,
    )


def resolve_role_balance(
    data: MLMaterializedPartitionData,
    label_schema: LabelSchema,
    balancing: BalancingSpec,
    *,
    seed: int,
    dataset_snapshot_id: str,
    split_id: str,
) -> BalanceResolution:
    """Resolve one primary role, enforcing natural validation/test prevalence."""
    role_specs = {
        "train": balancing.train,
        "validation": balancing.validation,
        "test": balancing.test,
    }
    if data.split not in role_specs:
        raise MLBalanceError(f"unsupported split role {data.split!r}")
    method = role_specs[data.split].method
    return _resolve(
        data,
        label_schema,
        method=method,
        seed=seed,
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        purpose="primary",
        allow_evaluation_resampling=False,
    )


def resolve_evaluation_sensitivity(
    data: MLMaterializedPartitionData,
    label_schema: LabelSchema,
    *,
    name: str,
    seed: int,
    dataset_snapshot_id: str,
    split_id: str,
) -> BalanceResolution:
    """Create a separately named balanced validation/test sensitivity cohort."""
    name = str(name).strip()
    if not name:
        raise MLBalanceError("evaluation sensitivity name must be non-empty")
    if data.split not in {"validation", "test"}:
        raise MLBalanceError("evaluation sensitivity is only valid for validation or test")
    return _resolve(
        data,
        label_schema,
        method="downsample",
        seed=seed,
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        purpose=f"evaluation_sensitivity:{name}",
        allow_evaluation_resampling=True,
    )


def balance_counts(resolution: BalanceResolution) -> Mapping[str, int]:
    """Return result counts keyed in persisted class order."""
    return MappingProxyType(dict(zip(resolution.class_order, resolution.result_counts)))
