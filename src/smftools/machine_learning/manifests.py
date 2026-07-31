"""Immutable, path-neutral dataset and split manifests for machine learning.

The records in this module describe resolved membership and provenance. They
do not read matrices, choose split strategies, balance classes, or organize
runtime artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from smftools.informatics.molecule_identity import molecule_uid, validate_experiment_uid

from .contracts import InputSchema, LabelSchema

ML_DATASET_MANIFEST_VERSION = 1
ML_SPLIT_MANIFEST_VERSION = 1
SPLIT_ROLES = frozenset({"train", "validation", "test"})


class MLManifestError(ValueError):
    """Raised when a dataset or split manifest is invalid."""


class StaleDatasetSourceError(MLManifestError):
    """Raised when current source identities no longer match a snapshot."""


def _fail(path: str, message: str) -> None:
    raise MLManifestError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(path, "keys must be strings")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "must be a sequence")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        _fail(path, f"contains unknown fields: {unknown}")
    missing = sorted(required.difference(value))
    if missing:
        _fail(path, f"is missing required fields: {missing}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _strings(value: Any, path: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    )
    if len(result) != len(set(result)):
        _fail(path, "cannot contain duplicates")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer greater than or equal to {minimum}")
    return value


def _version(value: Any, expected: int, path: str) -> int:
    result = _integer(value, path, minimum=1)
    if result != expected:
        _fail(path, f"unsupported version {result}; supported version is {expected}")
    return result


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MLManifestError(f"value is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: Any, path: str) -> str:
    result = _string(value, path).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(path, "must be a lowercase SHA-256 digest")
    return result


def _freeze_json(value: Any, path: str) -> Any:
    """Return an immutable copy of a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            _fail(path, "must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            _fail(path, "mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    _fail(path, f"contains unsupported value type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _string_mapping(value: Any, path: str) -> Mapping[str, str]:
    raw = _mapping(value, path)
    result = {
        _string(key, f"{path}.key"): _string(item, f"{path}.{key}") for key, item in raw.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class SourceArtifactReference:
    """Portable reference to one immutable source artifact."""

    artifact_id: str
    kind: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _string(self.artifact_id, "source_artifact.artifact_id")
        _string(self.kind, "source_artifact.kind")
        relative_path = _string(self.relative_path, "source_artifact.relative_path")
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative_path
            or "://" in relative_path
        ):
            _fail(
                "source_artifact.relative_path",
                "must be a portable POSIX path relative to its owning experiment or project",
            )
        if relative_path in {".", ""}:
            _fail("source_artifact.relative_path", "must identify an artifact")
        _digest(self.sha256, "source_artifact.sha256")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable artifact reference."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceArtifactReference:
        """Validate and restore an artifact reference."""
        value = _mapping(raw, "source_artifact")
        fields = {"artifact_id", "kind", "relative_path", "sha256"}
        _keys(value, path="source_artifact", allowed=fields, required=fields)
        return cls(
            artifact_id=_string(value["artifact_id"], "source_artifact.artifact_id"),
            kind=_string(value["kind"], "source_artifact.kind"),
            relative_path=_string(value["relative_path"], "source_artifact.relative_path"),
            sha256=_digest(value["sha256"], "source_artifact.sha256"),
        )


@dataclass(frozen=True)
class ExperimentSource:
    """Immutable identity of one experiment stage used by a dataset."""

    experiment_id: str
    experiment_uid: str
    modality: str
    stage: str
    stage_generation_id: str
    membership_fingerprint: str
    feature_fingerprint: str
    artifacts: tuple[SourceArtifactReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_uid", validate_experiment_uid(self.experiment_uid))
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=lambda item: (item.kind, item.artifact_id))),
        )
        _string(self.experiment_id, "source.experiment_id")
        _string(self.modality, "source.modality")
        _string(self.stage, "source.stage")
        _string(self.stage_generation_id, "source.stage_generation_id")
        _string(self.membership_fingerprint, "source.membership_fingerprint")
        _string(self.feature_fingerprint, "source.feature_fingerprint")
        if not self.artifacts:
            _fail("source.artifacts", "must contain at least one portable reference")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            _fail("source.artifacts", "artifact IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable experiment source."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_uid": self.experiment_uid,
            "modality": self.modality,
            "stage": self.stage,
            "stage_generation_id": self.stage_generation_id,
            "membership_fingerprint": self.membership_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentSource:
        """Validate and restore an experiment source."""
        path = "source"
        value = _mapping(raw, path)
        fields = {
            "experiment_id",
            "experiment_uid",
            "modality",
            "stage",
            "stage_generation_id",
            "membership_fingerprint",
            "feature_fingerprint",
            "artifacts",
        }
        _keys(value, path=path, allowed=fields, required=fields)
        return cls(
            experiment_id=_string(value["experiment_id"], f"{path}.experiment_id"),
            experiment_uid=_string(value["experiment_uid"], f"{path}.experiment_uid"),
            modality=_string(value["modality"], f"{path}.modality"),
            stage=_string(value["stage"], f"{path}.stage"),
            stage_generation_id=_string(
                value["stage_generation_id"], f"{path}.stage_generation_id"
            ),
            membership_fingerprint=_string(
                value["membership_fingerprint"], f"{path}.membership_fingerprint"
            ),
            feature_fingerprint=_string(
                value["feature_fingerprint"], f"{path}.feature_fingerprint"
            ),
            artifacts=tuple(
                SourceArtifactReference.from_dict(item)
                for item in _sequence(value["artifacts"], f"{path}.artifacts")
            ),
        )


@dataclass(frozen=True)
class GenomicInterval:
    """Resolved zero-based half-open interval selected for materialization."""

    reference: str
    start: int
    end: int

    def __post_init__(self) -> None:
        _string(self.reference, "interval.reference")
        _integer(self.start, "interval.start")
        _integer(self.end, "interval.end", minimum=1)
        if self.end <= self.start:
            _fail("interval.end", "must be greater than start")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable interval."""
        return {"reference": self.reference, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GenomicInterval:
        """Validate and restore an interval."""
        value = _mapping(raw, "interval")
        fields = {"reference", "start", "end"}
        _keys(value, path="interval", allowed=fields, required=fields)
        return cls(
            reference=_string(value["reference"], "interval.reference"),
            start=_integer(value["start"], "interval.start"),
            end=_integer(value["end"], "interval.end", minimum=1),
        )


@dataclass(frozen=True)
class DatasetSelection:
    """Resolved user selection whose identity is independent of local paths."""

    scope_kind: str
    scope_id: str | None
    set_name: str | None
    dataset_name: str
    plan_hash: str
    samples: tuple[str, ...]
    references: tuple[str, ...]
    intervals: tuple[GenomicInterval, ...]
    filters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(sorted(self.samples)))
        object.__setattr__(self, "references", tuple(sorted(self.references)))
        object.__setattr__(
            self,
            "intervals",
            tuple(sorted(self.intervals, key=lambda item: (item.reference, item.start, item.end))),
        )
        object.__setattr__(self, "filters", _freeze_json(self.filters, "selection.filters"))
        if self.scope_kind not in {"experiment", "project", "set"}:
            _fail("selection.scope_kind", "must be 'experiment', 'project', or 'set'")
        _optional_string(self.scope_id, "selection.scope_id")
        _optional_string(self.set_name, "selection.set_name")
        if self.scope_kind == "set" and self.set_name is None:
            _fail("selection.set_name", "is required for set scope")
        _string(self.dataset_name, "selection.dataset_name")
        _digest(self.plan_hash, "selection.plan_hash")
        if len(self.samples) != len(set(self.samples)):
            _fail("selection.samples", "cannot contain duplicates")
        if len(self.references) != len(set(self.references)):
            _fail("selection.references", "cannot contain duplicates")
        interval_keys = [(item.reference, item.start, item.end) for item in self.intervals]
        if len(interval_keys) != len(set(interval_keys)):
            _fail("selection.intervals", "cannot contain duplicates")
        outside = sorted({item.reference for item in self.intervals}.difference(self.references))
        if outside:
            _fail("selection.intervals", f"references unselected contigs: {outside}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable selection."""
        return {
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "set_name": self.set_name,
            "dataset_name": self.dataset_name,
            "plan_hash": self.plan_hash,
            "samples": list(self.samples),
            "references": list(self.references),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "filters": _thaw_json(self.filters),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DatasetSelection:
        """Validate and restore a resolved selection."""
        path = "selection"
        value = _mapping(raw, path)
        fields = {
            "scope_kind",
            "scope_id",
            "set_name",
            "dataset_name",
            "plan_hash",
            "samples",
            "references",
            "intervals",
            "filters",
        }
        _keys(value, path=path, allowed=fields, required=fields)
        return cls(
            scope_kind=_string(value["scope_kind"], f"{path}.scope_kind"),
            scope_id=_optional_string(value["scope_id"], f"{path}.scope_id"),
            set_name=_optional_string(value["set_name"], f"{path}.set_name"),
            dataset_name=_string(value["dataset_name"], f"{path}.dataset_name"),
            plan_hash=_digest(value["plan_hash"], f"{path}.plan_hash"),
            samples=_strings(value["samples"], f"{path}.samples"),
            references=_strings(value["references"], f"{path}.references"),
            intervals=tuple(
                GenomicInterval.from_dict(item)
                for item in _sequence(value["intervals"], f"{path}.intervals")
            ),
            filters=_mapping(value["filters"], f"{path}.filters"),
        )


@dataclass(frozen=True)
class DatasetObservation:
    """Stable row identity and audit metadata for one selected molecule."""

    molecule_uid: str
    experiment_uid: str
    read_id: str
    sample_id: str
    reference: str
    modality: str
    class_id: int | None
    group_values: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized_experiment = validate_experiment_uid(self.experiment_uid)
        object.__setattr__(self, "experiment_uid", normalized_experiment)
        object.__setattr__(
            self,
            "group_values",
            _string_mapping(self.group_values, "observation.group_values"),
        )
        _string(self.read_id, "observation.read_id")
        _string(self.sample_id, "observation.sample_id")
        _string(self.reference, "observation.reference")
        _string(self.modality, "observation.modality")
        expected = molecule_uid(normalized_experiment, self.read_id)
        if self.molecule_uid != expected:
            _fail(
                "observation.molecule_uid",
                "does not match the stable experiment_uid/read_id identity",
            )
        if self.class_id is not None:
            _integer(self.class_id, "observation.class_id")

    def value_for_group(self, field: str) -> str:
        """Return one declared grouping value for split auditing."""
        core = {
            "molecule_uid": self.molecule_uid,
            "experiment_uid": self.experiment_uid,
            "read_id": self.read_id,
            "sample_id": self.sample_id,
            "reference": self.reference,
            "modality": self.modality,
        }
        if field in core:
            return core[field]
        if field not in self.group_values:
            _fail(
                "split.group_by",
                f"field {field!r} is absent from observation {self.molecule_uid!r}",
            )
        return self.group_values[field]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable observation record."""
        return {
            "molecule_uid": self.molecule_uid,
            "experiment_uid": self.experiment_uid,
            "read_id": self.read_id,
            "sample_id": self.sample_id,
            "reference": self.reference,
            "modality": self.modality,
            "class_id": self.class_id,
            "group_values": dict(self.group_values),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DatasetObservation:
        """Validate and restore an observation record."""
        path = "observation"
        value = _mapping(raw, path)
        fields = {
            "molecule_uid",
            "experiment_uid",
            "read_id",
            "sample_id",
            "reference",
            "modality",
            "class_id",
            "group_values",
        }
        _keys(value, path=path, allowed=fields, required=fields)
        class_id = value["class_id"]
        if class_id is not None:
            class_id = _integer(class_id, f"{path}.class_id")
        return cls(
            molecule_uid=_string(value["molecule_uid"], f"{path}.molecule_uid"),
            experiment_uid=_string(value["experiment_uid"], f"{path}.experiment_uid"),
            read_id=_string(value["read_id"], f"{path}.read_id"),
            sample_id=_string(value["sample_id"], f"{path}.sample_id"),
            reference=_string(value["reference"], f"{path}.reference"),
            modality=_string(value["modality"], f"{path}.modality"),
            class_id=class_id,
            group_values=_string_mapping(value["group_values"], f"{path}.group_values"),
        )


@dataclass(frozen=True)
class CountRecord:
    """One deterministic category count stored in a manifest summary."""

    value: str
    count: int

    def __post_init__(self) -> None:
        _string(self.value, "count.value")
        _integer(self.count, "count.count", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable count."""
        return {"value": self.value, "count": self.count}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CountRecord:
        """Validate and restore a count."""
        value = _mapping(raw, "count")
        fields = {"value", "count"}
        _keys(value, path="count", allowed=fields, required=fields)
        return cls(
            value=_string(value["value"], "count.value"),
            count=_integer(value["count"], "count.count", minimum=1),
        )


def _counts(values: Sequence[str]) -> tuple[CountRecord, ...]:
    return tuple(CountRecord(value, count) for value, count in sorted(Counter(values).items()))


@dataclass(frozen=True)
class DatasetSummary:
    """Inspectable counts for an immutable dataset snapshot."""

    n_observations: int
    n_experiments: int
    counts_by_sample: tuple[CountRecord, ...]
    counts_by_modality: tuple[CountRecord, ...]
    counts_by_class: tuple[CountRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts_by_sample", tuple(self.counts_by_sample))
        object.__setattr__(self, "counts_by_modality", tuple(self.counts_by_modality))
        object.__setattr__(self, "counts_by_class", tuple(self.counts_by_class))
        _integer(self.n_observations, "dataset_summary.n_observations", minimum=1)
        _integer(self.n_experiments, "dataset_summary.n_experiments", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dataset summary."""
        return {
            "n_observations": self.n_observations,
            "n_experiments": self.n_experiments,
            "counts_by_sample": [item.to_dict() for item in self.counts_by_sample],
            "counts_by_modality": [item.to_dict() for item in self.counts_by_modality],
            "counts_by_class": [item.to_dict() for item in self.counts_by_class],
        }


def _dataset_summary(
    observations: tuple[DatasetObservation, ...],
    label_schema: LabelSchema | None,
) -> DatasetSummary:
    class_names: list[str] = []
    if label_schema is not None:
        class_names = [
            label_schema.class_order[observation.class_id]
            for observation in observations
            if observation.class_id is not None
        ]
    return DatasetSummary(
        n_observations=len(observations),
        n_experiments=len({item.experiment_uid for item in observations}),
        counts_by_sample=_counts([item.sample_id for item in observations]),
        counts_by_modality=_counts([item.modality for item in observations]),
        counts_by_class=_counts(class_names),
    )


def _source_digest(sources: tuple[ExperimentSource, ...]) -> str:
    return _sha256({"sources": [source.to_dict() for source in sources]})


def _membership_digest(observations: tuple[DatasetObservation, ...]) -> str:
    return _sha256({"observations": [item.to_dict() for item in observations]})


@dataclass(frozen=True)
class DatasetSnapshotManifest:
    """Immutable identity of resolved ML inputs without loading their matrices."""

    schema_version: int
    snapshot_id: str
    source_digest: str
    membership_digest: str
    selection: DatasetSelection
    input_schema: InputSchema
    label_schema: LabelSchema | None
    sources: tuple[ExperimentSource, ...]
    observations: tuple[DatasetObservation, ...]
    summary: DatasetSummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(self.sources, key=lambda item: item.experiment_uid)),
        )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(self.observations, key=lambda item: item.molecule_uid)),
        )
        _validate_dataset_snapshot(self)

    @classmethod
    def create(
        cls,
        *,
        selection: DatasetSelection,
        input_schema: InputSchema,
        label_schema: LabelSchema | None,
        sources: Sequence[ExperimentSource],
        observations: Sequence[DatasetObservation],
    ) -> DatasetSnapshotManifest:
        """Create a canonical snapshot from already-resolved source records."""
        canonical_sources = tuple(sorted(sources, key=lambda item: item.experiment_uid))
        canonical_observations = tuple(sorted(observations, key=lambda item: item.molecule_uid))
        source_digest = _source_digest(canonical_sources)
        membership_digest = _membership_digest(canonical_observations)
        identity = {
            "schema_version": ML_DATASET_MANIFEST_VERSION,
            "selection": selection.to_dict(),
            "input_schema_hash": input_schema.schema_hash,
            "label_schema_hash": label_schema.schema_hash if label_schema is not None else None,
            "source_digest": source_digest,
            "membership_digest": membership_digest,
        }
        return cls(
            schema_version=ML_DATASET_MANIFEST_VERSION,
            snapshot_id=_sha256(identity),
            source_digest=source_digest,
            membership_digest=membership_digest,
            selection=selection,
            input_schema=input_schema,
            label_schema=label_schema,
            sources=canonical_sources,
            observations=canonical_observations,
            summary=_dataset_summary(canonical_observations, label_schema),
        )

    def assert_sources_current(self, current_sources: Sequence[ExperimentSource]) -> None:
        """Reject source records that changed after this snapshot was created."""
        canonical = tuple(sorted(current_sources, key=lambda item: item.experiment_uid))
        current_digest = _source_digest(canonical)
        if current_digest != self.source_digest:
            raise StaleDatasetSourceError(
                "dataset sources are stale: current source digest "
                f"{current_digest} does not match snapshot digest {self.source_digest}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable snapshot."""
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "source_digest": self.source_digest,
            "membership_digest": self.membership_digest,
            "selection": self.selection.to_dict(),
            "input_schema": self.input_schema.to_dict(),
            "label_schema": self.label_schema.to_dict() if self.label_schema is not None else None,
            "sources": [source.to_dict() for source in self.sources],
            "observations": [item.to_dict() for item in self.observations],
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DatasetSnapshotManifest:
        """Validate, restore, and verify all serialized snapshot identities."""
        path = "dataset_snapshot"
        value = _mapping(raw, path)
        fields = {
            "schema_version",
            "snapshot_id",
            "source_digest",
            "membership_digest",
            "selection",
            "input_schema",
            "label_schema",
            "sources",
            "observations",
            "summary",
        }
        _keys(value, path=path, allowed=fields, required=fields)
        _version(value["schema_version"], ML_DATASET_MANIFEST_VERSION, f"{path}.schema_version")
        label_raw = value["label_schema"]
        restored = cls.create(
            selection=DatasetSelection.from_dict(_mapping(value["selection"], f"{path}.selection")),
            input_schema=InputSchema.from_dict(
                _mapping(value["input_schema"], f"{path}.input_schema")
            ),
            label_schema=(
                None
                if label_raw is None
                else LabelSchema.from_dict(_mapping(label_raw, f"{path}.label_schema"))
            ),
            sources=tuple(
                ExperimentSource.from_dict(item)
                for item in _sequence(value["sources"], f"{path}.sources")
            ),
            observations=tuple(
                DatasetObservation.from_dict(item)
                for item in _sequence(value["observations"], f"{path}.observations")
            ),
        )
        expected = restored.to_dict()
        for field in ("source_digest", "membership_digest", "summary", "snapshot_id"):
            if value[field] != expected[field]:
                _fail(f"{path}.{field}", "does not match the resolved manifest content")
        return restored

    def canonical_json(self) -> str:
        """Return stable canonical JSON for persistence by a later artifact layer."""
        return _canonical_json(self.to_dict())


def _validate_dataset_snapshot(snapshot: DatasetSnapshotManifest) -> None:
    _version(
        snapshot.schema_version,
        ML_DATASET_MANIFEST_VERSION,
        "dataset_snapshot.schema_version",
    )
    _digest(snapshot.snapshot_id, "dataset_snapshot.snapshot_id")
    _digest(snapshot.source_digest, "dataset_snapshot.source_digest")
    _digest(snapshot.membership_digest, "dataset_snapshot.membership_digest")
    if not snapshot.sources:
        _fail("dataset_snapshot.sources", "must contain at least one experiment")
    if not snapshot.observations:
        _fail("dataset_snapshot.observations", "must contain at least one molecule")
    source_uids = [source.experiment_uid for source in snapshot.sources]
    if len(source_uids) != len(set(source_uids)):
        _fail("dataset_snapshot.sources", "experiment UIDs must be unique")
    molecule_uids = [item.molecule_uid for item in snapshot.observations]
    if len(molecule_uids) != len(set(molecule_uids)):
        _fail("dataset_snapshot.observations", "molecule UIDs must be unique")
    sources = {source.experiment_uid: source for source in snapshot.sources}
    selected_modalities = set(snapshot.input_schema.modalities)
    if {source.modality for source in snapshot.sources} != selected_modalities:
        _fail(
            "dataset_snapshot.sources",
            "source modalities must exactly match the input schema modalities",
        )
    valid_classes = (
        set(snapshot.label_schema.value_to_class.values())
        if snapshot.label_schema is not None
        else set()
    )
    for observation in snapshot.observations:
        source = sources.get(observation.experiment_uid)
        if source is None:
            _fail(
                "dataset_snapshot.observations",
                f"molecule {observation.molecule_uid!r} references an unknown experiment",
            )
        if observation.modality != source.modality:
            _fail(
                "dataset_snapshot.observations",
                f"molecule {observation.molecule_uid!r} modality differs from its experiment",
            )
        if snapshot.selection.samples and observation.sample_id not in snapshot.selection.samples:
            _fail(
                "dataset_snapshot.observations",
                f"molecule {observation.molecule_uid!r} has an unselected sample",
            )
        if (
            snapshot.selection.references
            and observation.reference not in snapshot.selection.references
        ):
            _fail(
                "dataset_snapshot.observations",
                f"molecule {observation.molecule_uid!r} has an unselected reference",
            )
        if snapshot.label_schema is None and observation.class_id is not None:
            _fail("dataset_snapshot.observations", "unlabeled datasets cannot store class IDs")
        if snapshot.label_schema is not None and observation.class_id not in valid_classes:
            _fail(
                "dataset_snapshot.observations",
                f"molecule {observation.molecule_uid!r} has an invalid or missing class ID",
            )
    expected_sources = _source_digest(snapshot.sources)
    if snapshot.source_digest != expected_sources:
        _fail("dataset_snapshot.source_digest", "does not match sources")
    expected_membership = _membership_digest(snapshot.observations)
    if snapshot.membership_digest != expected_membership:
        _fail("dataset_snapshot.membership_digest", "does not match observations")
    expected_summary = _dataset_summary(snapshot.observations, snapshot.label_schema)
    if snapshot.summary != expected_summary:
        _fail("dataset_snapshot.summary", "does not match observations")
    expected_id = _sha256(
        {
            "schema_version": snapshot.schema_version,
            "selection": snapshot.selection.to_dict(),
            "input_schema_hash": snapshot.input_schema.schema_hash,
            "label_schema_hash": (
                snapshot.label_schema.schema_hash if snapshot.label_schema is not None else None
            ),
            "source_digest": snapshot.source_digest,
            "membership_digest": snapshot.membership_digest,
        }
    )
    if snapshot.snapshot_id != expected_id:
        _fail("dataset_snapshot.snapshot_id", "does not match manifest identity")


@dataclass(frozen=True)
class SplitMember:
    """Resolved split and biological-group membership for one molecule."""

    molecule_uid: str
    split: str
    group_id: str
    group_values: Mapping[str, str]

    def __post_init__(self) -> None:
        _string(self.molecule_uid, "split_member.molecule_uid")
        if self.split not in SPLIT_ROLES:
            _fail("split_member.split", f"must be one of {sorted(SPLIT_ROLES)}")
        _digest(self.group_id, "split_member.group_id")
        object.__setattr__(
            self,
            "group_values",
            _string_mapping(self.group_values, "split_member.group_values"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable split member."""
        return {
            "molecule_uid": self.molecule_uid,
            "split": self.split,
            "group_id": self.group_id,
            "group_values": dict(self.group_values),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SplitMember:
        """Validate and restore a split member."""
        value = _mapping(raw, "split_member")
        fields = {"molecule_uid", "split", "group_id", "group_values"}
        _keys(value, path="split_member", allowed=fields, required=fields)
        return cls(
            molecule_uid=_string(value["molecule_uid"], "split_member.molecule_uid"),
            split=_string(value["split"], "split_member.split"),
            group_id=_digest(value["group_id"], "split_member.group_id"),
            group_values=_string_mapping(value["group_values"], "split_member.group_values"),
        )


@dataclass(frozen=True)
class SplitSummary:
    """Counts for one train, validation, or test partition."""

    split: str
    n_observations: int
    n_groups: int
    counts_by_modality: tuple[CountRecord, ...]
    counts_by_class: tuple[CountRecord, ...]

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ROLES:
            _fail("split_summary.split", f"must be one of {sorted(SPLIT_ROLES)}")
        _integer(self.n_observations, "split_summary.n_observations", minimum=1)
        _integer(self.n_groups, "split_summary.n_groups", minimum=1)
        object.__setattr__(self, "counts_by_modality", tuple(self.counts_by_modality))
        object.__setattr__(self, "counts_by_class", tuple(self.counts_by_class))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable split summary."""
        return {
            "split": self.split,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "counts_by_modality": [item.to_dict() for item in self.counts_by_modality],
            "counts_by_class": [item.to_dict() for item in self.counts_by_class],
        }


def _split_summaries(
    dataset: DatasetSnapshotManifest,
    members: tuple[SplitMember, ...],
) -> tuple[SplitSummary, ...]:
    observations = {item.molecule_uid: item for item in dataset.observations}
    result: list[SplitSummary] = []
    for split in sorted({item.split for item in members}):
        split_members = [item for item in members if item.split == split]
        split_observations = [observations[item.molecule_uid] for item in split_members]
        class_names: list[str] = []
        if dataset.label_schema is not None:
            class_names = [
                dataset.label_schema.class_order[item.class_id]
                for item in split_observations
                if item.class_id is not None
            ]
        result.append(
            SplitSummary(
                split=split,
                n_observations=len(split_members),
                n_groups=len({item.group_id for item in split_members}),
                counts_by_modality=_counts([item.modality for item in split_observations]),
                counts_by_class=_counts(class_names),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class SplitManifest:
    """Immutable resolved train/validation/test membership for one snapshot."""

    schema_version: int
    split_id: str
    dataset_snapshot_id: str
    membership_digest: str
    group_by: tuple[str, ...]
    members: tuple[SplitMember, ...]
    summaries: tuple[SplitSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_by", tuple(self.group_by))
        object.__setattr__(
            self,
            "members",
            tuple(sorted(self.members, key=lambda item: item.molecule_uid)),
        )
        object.__setattr__(
            self,
            "summaries",
            tuple(sorted(self.summaries, key=lambda item: item.split)),
        )
        _version(self.schema_version, ML_SPLIT_MANIFEST_VERSION, "split.schema_version")
        _digest(self.split_id, "split.split_id")
        _digest(self.dataset_snapshot_id, "split.dataset_snapshot_id")
        _digest(self.membership_digest, "split.membership_digest")
        if not self.group_by:
            _fail("split.group_by", "must contain at least one field")
        if len(self.group_by) != len(set(self.group_by)):
            _fail("split.group_by", "cannot contain duplicates")
        if not self.members:
            _fail("split.members", "must contain at least one molecule")
        molecule_uids = [member.molecule_uid for member in self.members]
        if len(molecule_uids) != len(set(molecule_uids)):
            _fail("split.members", "molecule UIDs must be unique")
        group_splits: dict[str, str] = {}
        for member in self.members:
            if set(member.group_values) != set(self.group_by):
                _fail(
                    "split.members",
                    "each member's group values must exactly match group_by",
                )
            expected_group_id = _sha256(
                {
                    "group_by": list(self.group_by),
                    "values": {field: member.group_values[field] for field in self.group_by},
                }
            )
            if member.group_id != expected_group_id:
                _fail("split.members", "group ID does not match group_by and group values")
            previous = group_splits.setdefault(member.group_id, member.split)
            if previous != member.split:
                _fail("split.members", "one biological group occurs in multiple splits")
        expected_membership_digest = _sha256(
            {"members": [member.to_dict() for member in self.members]}
        )
        if self.membership_digest != expected_membership_digest:
            _fail("split.membership_digest", "does not match members")
        expected_split_id = _sha256(
            {
                "schema_version": self.schema_version,
                "dataset_snapshot_id": self.dataset_snapshot_id,
                "group_by": list(self.group_by),
                "membership_digest": self.membership_digest,
            }
        )
        if self.split_id != expected_split_id:
            _fail("split.split_id", "does not match split identity")
        summary_roles = [summary.split for summary in self.summaries]
        if len(summary_roles) != len(set(summary_roles)):
            _fail("split.summaries", "split roles must be unique")
        member_roles = {member.split for member in self.members}
        if set(summary_roles) != member_roles:
            _fail("split.summaries", "must contain one summary for every represented split")
        for summary in self.summaries:
            members = [member for member in self.members if member.split == summary.split]
            if summary.n_observations != len(members):
                _fail("split.summaries", "observation count does not match members")
            if summary.n_groups != len({member.group_id for member in members}):
                _fail("split.summaries", "group count does not match members")

    @classmethod
    def create(
        cls,
        *,
        dataset: DatasetSnapshotManifest,
        group_by: Sequence[str],
        assignments: Mapping[str, str],
    ) -> SplitManifest:
        """Resolve explicit row assignments and verify biological-group isolation."""
        fields = tuple(_string(field, "split.group_by") for field in group_by)
        if not fields:
            _fail("split.group_by", "must contain at least one field")
        if len(fields) != len(set(fields)):
            _fail("split.group_by", "cannot contain duplicates")
        expected_uids = {item.molecule_uid for item in dataset.observations}
        assigned_uids = set(assignments)
        missing = sorted(expected_uids.difference(assigned_uids))
        unknown = sorted(assigned_uids.difference(expected_uids))
        if missing or unknown:
            _fail(
                "split.assignments",
                f"must cover the dataset exactly; missing={missing}, unknown={unknown}",
            )
        members: list[SplitMember] = []
        group_splits: dict[str, str] = {}
        for observation in dataset.observations:
            split = _string(assignments[observation.molecule_uid], "split.assignments")
            if split not in SPLIT_ROLES:
                _fail("split.assignments", f"split {split!r} is not supported")
            values = MappingProxyType(
                {field: observation.value_for_group(field) for field in fields}
            )
            group_id = _sha256({"group_by": list(fields), "values": dict(values)})
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                _fail(
                    "split.assignments",
                    f"group {dict(values)!r} occurs in both {previous!r} and {split!r}",
                )
            members.append(
                SplitMember(
                    molecule_uid=observation.molecule_uid,
                    split=split,
                    group_id=group_id,
                    group_values=values,
                )
            )
        canonical_members = tuple(sorted(members, key=lambda item: item.molecule_uid))
        membership_digest = _sha256({"members": [member.to_dict() for member in canonical_members]})
        identity = {
            "schema_version": ML_SPLIT_MANIFEST_VERSION,
            "dataset_snapshot_id": dataset.snapshot_id,
            "group_by": list(fields),
            "membership_digest": membership_digest,
        }
        return cls(
            schema_version=ML_SPLIT_MANIFEST_VERSION,
            split_id=_sha256(identity),
            dataset_snapshot_id=dataset.snapshot_id,
            membership_digest=membership_digest,
            group_by=fields,
            members=canonical_members,
            summaries=_split_summaries(dataset, canonical_members),
        )

    def validate_against(self, dataset: DatasetSnapshotManifest) -> None:
        """Verify identities, summaries, and group membership against a dataset."""
        if self.dataset_snapshot_id != dataset.snapshot_id:
            _fail("split.dataset_snapshot_id", "does not match the supplied dataset")
        assignments = {member.molecule_uid: member.split for member in self.members}
        rebuilt = SplitManifest.create(
            dataset=dataset,
            group_by=self.group_by,
            assignments=assignments,
        )
        if rebuilt.to_dict() != self.to_dict():
            _fail("split", "does not match its dataset, grouping fields, or summaries")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable split manifest."""
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "membership_digest": self.membership_digest,
            "group_by": list(self.group_by),
            "members": [member.to_dict() for member in self.members],
            "summaries": [summary.to_dict() for summary in self.summaries],
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        dataset: DatasetSnapshotManifest,
    ) -> SplitManifest:
        """Validate and restore a split manifest against its dataset snapshot."""
        path = "split"
        value = _mapping(raw, path)
        fields = {
            "schema_version",
            "split_id",
            "dataset_snapshot_id",
            "membership_digest",
            "group_by",
            "members",
            "summaries",
        }
        _keys(value, path=path, allowed=fields, required=fields)
        _version(value["schema_version"], ML_SPLIT_MANIFEST_VERSION, f"{path}.schema_version")
        members = tuple(
            SplitMember.from_dict(item) for item in _sequence(value["members"], f"{path}.members")
        )
        assignments = {member.molecule_uid: member.split for member in members}
        if len(assignments) != len(members):
            _fail(f"{path}.members", "molecule UIDs must be unique")
        restored = cls.create(
            dataset=dataset,
            group_by=_strings(value["group_by"], f"{path}.group_by"),
            assignments=assignments,
        )
        if restored.to_dict() != dict(value):
            _fail(path, "serialized fields do not match resolved split content")
        return restored

    def canonical_json(self) -> str:
        """Return stable canonical JSON for persistence by a later artifact layer."""
        return _canonical_json(self.to_dict())
