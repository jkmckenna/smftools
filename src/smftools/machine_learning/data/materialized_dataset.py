"""Validated pre-materialized inputs for canonical ML execution.

This adapter is intended for callers that already resolved a fixed-width matrix
from a legacy or external data source. It preserves the immutable dataset and
split manifests as the scientific authority while allowing the same training
engines used by partition-backed datasets to consume the prepared splits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ..contracts import InputSchema, validate_mask_arrays, validate_mask_relationships
from ..manifests import DatasetObservation, DatasetSnapshotManifest, SplitManifest
from .partition_dataset import (
    DEFAULT_BATCH_SIZE,
    MLMaterializedPartitionData,
    MLPartitionDataError,
)


class MLDatasetPlanProtocol(Protocol):
    """Minimum executable-plan surface required by canonical trainers."""

    dataset: DatasetSnapshotManifest
    split: SplitManifest
    effective_batch_size: int


class MLDatasetProtocol(Protocol):
    """Dataset interface shared by partitioned and pre-materialized inputs."""

    plan: MLDatasetPlanProtocol

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        """Return one validated split in immutable manifest order."""


@dataclass(frozen=True)
class MaterializedDatasetPlan:
    """Immutable manifest bindings for an already materialized dataset."""

    dataset: DatasetSnapshotManifest
    split: SplitManifest
    effective_batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        self.split.validate_against(self.dataset)
        if (
            isinstance(self.effective_batch_size, bool)
            or not isinstance(self.effective_batch_size, int)
            or self.effective_batch_size <= 0
        ):
            raise MLPartitionDataError("effective_batch_size must be a positive integer")


def _observations_for_role(
    dataset: DatasetSnapshotManifest,
    split: SplitManifest,
    role: str,
) -> tuple[DatasetObservation, ...]:
    assignments = {member.molecule_uid: member.split for member in split.members}
    return tuple(
        observation
        for observation in dataset.observations
        if assignments[observation.molecule_uid] == role
    )


def _declared_masks(
    data: MLMaterializedPartitionData,
    schema: InputSchema,
) -> Mapping[str, np.ndarray]:
    by_kind = {
        "observed": data.observed_mask,
        "availability": data.availability_mask,
        "design": data.design_mask,
        "padding": data.padding_mask,
    }
    unsupported = sorted({mask.kind for mask in schema.masks}.difference(by_kind))
    if unsupported:
        raise MLPartitionDataError(
            f"pre-materialized data cannot supply declared mask kinds: {unsupported}"
        )
    return MappingProxyType({mask.name: by_kind[mask.kind] for mask in schema.masks})


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _freeze_role_data(data: MLMaterializedPartitionData) -> MLMaterializedPartitionData:
    """Detach validated inputs from caller-owned mutable arrays."""
    return MLMaterializedPartitionData(
        split=str(data.split),
        molecule_uids=tuple(data.molecule_uids),
        read_ids=tuple(data.read_ids),
        experiment_uids=tuple(data.experiment_uids),
        modalities=tuple(data.modalities),
        coordinates=_readonly_array(data.coordinates),
        channel_names=tuple(data.channel_names),
        values=_readonly_array(data.values),
        labels=None if data.labels is None else _readonly_array(data.labels),
        observed_mask=_readonly_array(data.observed_mask),
        availability_mask=_readonly_array(data.availability_mask),
        design_mask=_readonly_array(data.design_mask),
        padding_mask=_readonly_array(data.padding_mask),
    )


def _validate_role_data(
    data: MLMaterializedPartitionData,
    *,
    role: str,
    dataset: DatasetSnapshotManifest,
    split: SplitManifest,
) -> None:
    if data.split != role:
        raise MLPartitionDataError(f"materialized data key {role!r} contains split {data.split!r}")
    observations = _observations_for_role(dataset, split, role)
    if not observations:
        raise MLPartitionDataError(f"split role {role!r} is absent from the split manifest")

    expected_metadata = {
        "molecule_uids": tuple(item.molecule_uid for item in observations),
        "read_ids": tuple(item.read_id for item in observations),
        "experiment_uids": tuple(item.experiment_uid for item in observations),
        "modalities": tuple(item.modality for item in observations),
    }
    for name, expected in expected_metadata.items():
        if tuple(getattr(data, name)) != expected:
            raise MLPartitionDataError(
                f"materialized {role!r} {name} do not match immutable manifest order"
            )

    schema = dataset.input_schema
    n_rows = len(observations)
    expected_shape = (n_rows, schema.n_positions, len(schema.channels))
    values = np.asarray(data.values)
    if values.shape != expected_shape:
        raise MLPartitionDataError(
            f"materialized {role!r} values shape {values.shape} does not match {expected_shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise MLPartitionDataError("materialized values must have a numeric dtype")
    coordinates = np.asarray(data.coordinates)
    if coordinates.shape != (schema.n_positions,) or not np.issubdtype(
        coordinates.dtype, np.integer
    ):
        raise MLPartitionDataError(
            "materialized coordinates must be a one-dimensional integer array "
            "matching input_schema.n_positions"
        )
    if len(coordinates) > 1 and np.any(np.diff(coordinates) <= 0):
        raise MLPartitionDataError("materialized coordinates must be strictly increasing")
    expected_channels = tuple(channel.name for channel in schema.channels)
    if tuple(data.channel_names) != expected_channels:
        raise MLPartitionDataError(
            "materialized channel_names do not match the ordered input schema"
        )

    if dataset.label_schema is None:
        if data.labels is not None:
            raise MLPartitionDataError("unlabeled dataset cannot contain materialized labels")
    else:
        expected_labels = np.asarray([item.class_id for item in observations], dtype=np.int64)
        labels = np.asarray(data.labels)
        if labels.shape != (n_rows,) or not np.array_equal(labels, expected_labels):
            raise MLPartitionDataError(
                f"materialized {role!r} labels do not match the dataset manifest"
            )

    masks = _declared_masks(data, schema)
    validate_mask_arrays(schema, masks, batch_size=n_rows)
    validate_mask_relationships(schema, masks)
    observed_name = next(
        (mask.name for mask in schema.masks if mask.kind == "observed"),
        None,
    )
    if observed_name is not None:
        observed = np.asarray(masks[observed_name], dtype=bool)
        if np.any(observed & ~np.isfinite(values)):
            raise MLPartitionDataError("observed values must be finite")


class MaterializedDataset:
    """Expose validated prepared splits through the canonical dataset protocol."""

    def __init__(
        self,
        dataset: DatasetSnapshotManifest,
        split: SplitManifest,
        data: Mapping[str, MLMaterializedPartitionData],
        *,
        effective_batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.plan = MaterializedDatasetPlan(
            dataset=dataset,
            split=split,
            effective_batch_size=effective_batch_size,
        )
        materialized = {role: _freeze_role_data(role_data) for role, role_data in data.items()}
        represented = {member.split for member in split.members}
        if set(materialized) != represented:
            raise MLPartitionDataError(
                "materialized split roles must exactly match the split manifest; "
                f"expected={sorted(represented)}, received={sorted(materialized)}"
            )
        for role, role_data in materialized.items():
            _validate_role_data(
                role_data,
                role=role,
                dataset=dataset,
                split=split,
            )
        self._data = MappingProxyType(materialized)

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        """Return one prevalidated split without filesystem access or copying."""
        try:
            return self._data[split]
        except KeyError as exc:
            raise MLPartitionDataError(
                f"split role {split!r} is absent; represented roles are {sorted(self._data)}"
            ) from exc


__all__ = [
    "MLDatasetPlanProtocol",
    "MLDatasetProtocol",
    "MaterializedDataset",
    "MaterializedDatasetPlan",
]
