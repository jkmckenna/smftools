"""Deterministic training-only background sampling for explanation methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..artifacts import ExplanationBaseline
from ..contracts import InputSchema, validate_mask_arrays, validate_mask_relationships
from ..data.partition_dataset import MLMaterializedPartitionData
from .contracts import (
    InterpretabilityContractError,
    _array_digest,
    _digest,
    _integer,
    _sha256,
    _string,
)


def _readonly(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BackgroundReference:
    """A bounded, checksummed background selected only from training rows."""

    background_hash: str
    dataset_snapshot_id: str
    split_id: str
    input_schema_hash: str
    split_role: str
    cohort: str
    sampling_rule: str
    random_seed: int
    molecule_uids: tuple[str, ...]
    experiment_uids: tuple[str, ...]
    modalities: tuple[str, ...]
    coordinates: np.ndarray
    channel_names: tuple[str, ...]
    values: np.ndarray
    masks: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        _digest(self.background_hash, "background.background_hash")
        _digest(self.dataset_snapshot_id, "background.dataset_snapshot_id")
        _digest(self.split_id, "background.split_id")
        _digest(self.input_schema_hash, "background.input_schema_hash")
        if self.split_role != "train":
            raise InterpretabilityContractError(
                "data-derived explanation backgrounds must use the train split"
            )
        object.__setattr__(self, "cohort", _string(self.cohort, "background.cohort"))
        object.__setattr__(
            self,
            "sampling_rule",
            _string(self.sampling_rule, "background.sampling_rule"),
        )
        _integer(self.random_seed, "background.random_seed")
        molecule_uids = tuple(self.molecule_uids)
        if not molecule_uids or len(molecule_uids) != len(set(molecule_uids)):
            raise InterpretabilityContractError(
                "background molecule_uids must be non-empty and unique"
            )
        for value in molecule_uids:
            _string(value, "background.molecule_uids[]")
        object.__setattr__(self, "molecule_uids", molecule_uids)
        experiment_uids = tuple(self.experiment_uids)
        modalities = tuple(self.modalities)
        for name, items in (
            ("experiment_uids", experiment_uids),
            ("modalities", modalities),
        ):
            if len(items) != len(molecule_uids):
                raise InterpretabilityContractError(
                    f"background {name} must align with molecule_uids"
                )
            for value in items:
                _string(value, f"background.{name}[]")
        object.__setattr__(self, "experiment_uids", experiment_uids)
        object.__setattr__(self, "modalities", modalities)
        coordinates = _readonly(self.coordinates, dtype=np.int64)
        if coordinates.ndim != 1 or coordinates.size == 0:
            raise InterpretabilityContractError("background coordinates must be a non-empty vector")
        object.__setattr__(self, "coordinates", coordinates)
        channel_names = tuple(self.channel_names)
        if not channel_names or len(channel_names) != len(set(channel_names)):
            raise InterpretabilityContractError(
                "background channel_names must be non-empty and unique"
            )
        for value in channel_names:
            _string(value, "background.channel_names[]")
        object.__setattr__(self, "channel_names", channel_names)
        values = _readonly(self.values, dtype=np.float32)
        expected = (len(molecule_uids), len(coordinates), len(channel_names))
        if values.shape != expected:
            raise InterpretabilityContractError(
                f"background values have shape {values.shape}; expected {expected}"
            )
        object.__setattr__(self, "values", values)
        masks: dict[str, np.ndarray] = {}
        for name, array in sorted(self.masks.items()):
            masks[_string(name, "background.masks key")] = _readonly(array, dtype=bool)
        object.__setattr__(self, "masks", MappingProxyType(masks))
        if self.background_hash != _sha256(self._identity_dict()):
            raise InterpretabilityContractError(
                "background.background_hash does not match background content"
            )

    @classmethod
    def create(
        cls,
        *,
        dataset_snapshot_id: str,
        split_id: str,
        input_schema_hash: str,
        cohort: str,
        sampling_rule: str,
        random_seed: int,
        molecule_uids: tuple[str, ...],
        experiment_uids: tuple[str, ...],
        modalities: tuple[str, ...],
        coordinates: Any,
        channel_names: tuple[str, ...],
        values: Any,
        masks: Mapping[str, Any],
    ) -> BackgroundReference:
        """Create a content-addressed training background."""
        fields = {
            "dataset_snapshot_id": dataset_snapshot_id,
            "split_id": split_id,
            "input_schema_hash": input_schema_hash,
            "split_role": "train",
            "cohort": cohort,
            "sampling_rule": sampling_rule,
            "random_seed": random_seed,
            "molecule_uids": molecule_uids,
            "experiment_uids": experiment_uids,
            "modalities": modalities,
            "coordinates": coordinates,
            "channel_names": channel_names,
            "values": values,
            "masks": masks,
        }
        identity = {
            "dataset_snapshot_id": dataset_snapshot_id,
            "split_id": split_id,
            "input_schema_hash": input_schema_hash,
            "split_role": "train",
            "cohort": cohort,
            "sampling_rule": sampling_rule,
            "random_seed": random_seed,
            "molecule_uids": list(molecule_uids),
            "experiment_uids": list(experiment_uids),
            "modalities": list(modalities),
            "coordinates": _array_digest(np.asarray(coordinates, dtype=np.int64)),
            "channel_names": list(channel_names),
            "values": _array_digest(np.asarray(values, dtype=np.float32)),
            "masks": {
                name: _array_digest(np.asarray(array, dtype=bool))
                for name, array in sorted(masks.items())
            },
        }
        return cls(background_hash=_sha256(identity), **fields)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "split_id": self.split_id,
            "input_schema_hash": self.input_schema_hash,
            "split_role": self.split_role,
            "cohort": self.cohort,
            "sampling_rule": self.sampling_rule,
            "random_seed": self.random_seed,
            "molecule_uids": list(self.molecule_uids),
            "experiment_uids": list(self.experiment_uids),
            "modalities": list(self.modalities),
            "coordinates": _array_digest(self.coordinates),
            "channel_names": list(self.channel_names),
            "values": _array_digest(self.values),
            "masks": {name: _array_digest(array) for name, array in sorted(self.masks.items())},
        }

    def to_baseline(self) -> ExplanationBaseline:
        """Return artifact-compatible baseline provenance for this background."""
        return ExplanationBaseline(
            kind="sampled_training_background",
            description=(
                f"{self.sampling_rule}; n={len(self.molecule_uids)}; seed={self.random_seed}; "
                f"modalities={','.join(sorted(set(self.modalities)))}"
            ),
            baseline_hash=self.background_hash,
            dataset_snapshot_id=self.dataset_snapshot_id,
            cohort="train",
        )

    def validate_against(self, input_schema: InputSchema) -> None:
        """Validate background axes, masks, and observed-value integrity."""
        if self.input_schema_hash != input_schema.schema_hash:
            raise InterpretabilityContractError(
                "background input_schema_hash differs from input schema"
            )
        if len(self.coordinates) != input_schema.n_positions:
            raise InterpretabilityContractError(
                "background position width differs from input schema"
            )
        if self.channel_names != tuple(channel.name for channel in input_schema.channels):
            raise InterpretabilityContractError(
                "background channel order differs from input schema"
            )
        validate_mask_arrays(
            input_schema,
            self.masks,
            batch_size=len(self.molecule_uids),
            require_all=False,
        )
        validate_mask_relationships(input_schema, self.masks)
        observed_spec = next(
            (mask for mask in input_schema.masks if mask.kind == "observed"),
            None,
        )
        if observed_spec is not None and observed_spec.name in self.masks:
            observed = np.asarray(self.masks[observed_spec.name], dtype=bool)
            if np.any(observed & ~np.isfinite(self.values)):
                raise InterpretabilityContractError(
                    "background contains non-finite values marked as observed"
                )


def _data_masks(
    data: MLMaterializedPartitionData,
    input_schema: InputSchema,
) -> dict[str, np.ndarray]:
    by_kind = {
        "observed": data.observed_mask,
        "availability": data.availability_mask,
        "design": data.design_mask,
        "padding": data.padding_mask,
    }
    return {mask.name: by_kind[mask.kind] for mask in input_schema.masks if mask.kind in by_kind}


def sample_training_background(
    data: MLMaterializedPartitionData,
    input_schema: InputSchema,
    *,
    dataset_snapshot_id: str,
    split_id: str,
    max_observations: int,
    random_seed: int = 0,
    cohort: str = "train",
) -> BackgroundReference:
    """Uniformly sample a bounded background while preserving training row order."""
    if data.split != "train":
        raise InterpretabilityContractError(
            "explanation background sampling only accepts materialized train data"
        )
    _integer(max_observations, "max_observations", minimum=1)
    _integer(random_seed, "random_seed")
    if tuple(data.channel_names) != tuple(channel.name for channel in input_schema.channels):
        raise InterpretabilityContractError(
            "background data channel order differs from input schema"
        )
    if len(data.coordinates) != input_schema.n_positions:
        raise InterpretabilityContractError(
            "background data position width differs from input schema"
        )
    n_rows = len(data.molecule_uids)
    if n_rows == 0:
        raise InterpretabilityContractError("background source contains no training rows")
    if n_rows <= max_observations:
        selected = np.arange(n_rows, dtype=np.int64)
        sampling_rule = "all_training_rows"
    else:
        selected = np.sort(
            np.random.default_rng(random_seed).choice(
                n_rows,
                size=max_observations,
                replace=False,
            )
        )
        sampling_rule = "uniform_without_replacement"
    masks = _data_masks(data, input_schema)
    selected_masks: dict[str, np.ndarray] = {}
    by_name = {mask.name: mask for mask in input_schema.masks}
    for name, array in masks.items():
        mask = by_name[name]
        selected_masks[name] = array[selected] if "observation" in mask.axes else array
    validate_mask_arrays(
        input_schema,
        selected_masks,
        batch_size=len(selected),
        require_all=False,
    )
    validate_mask_relationships(input_schema, selected_masks)
    result = BackgroundReference.create(
        dataset_snapshot_id=dataset_snapshot_id,
        split_id=split_id,
        input_schema_hash=input_schema.schema_hash,
        cohort=cohort,
        sampling_rule=sampling_rule,
        random_seed=random_seed,
        molecule_uids=tuple(data.molecule_uids[index] for index in selected),
        experiment_uids=tuple(data.experiment_uids[index] for index in selected),
        modalities=tuple(data.modalities[index] for index in selected),
        coordinates=data.coordinates,
        channel_names=tuple(data.channel_names),
        values=data.values[selected],
        masks=selected_masks,
    )
    result.validate_against(input_schema)
    return result
