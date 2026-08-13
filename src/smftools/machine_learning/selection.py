"""Metadata-only planning of eligible machine-learning observations and channels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from smftools.constants import (
    CHIMERIC_DIR,
    HMM_DIR,
    LATENT_DIR,
    PREPROCESS_DIR,
    RAW_DIR,
    SPATIAL_DIR,
    VARIANT_DIR,
)
from smftools.informatics.experiment_manifest import read_experiment_manifest
from smftools.informatics.molecule_identity import (
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
    molecule_uid,
    validate_experiment_uid,
)
from smftools.project.reference_registry import (
    REFERENCE_REGISTRY_FILENAME,
    ReferenceRegistry,
)
from smftools.project.registry import list_experiments, resolve_set_membership

from .plan import DatasetSpec, MLPlan, PhysicalChannelSource

ML_SELECTION_PLAN_VERSION = 1
_STAGE_DIRS = {
    "raw": RAW_DIR,
    "preprocess": PREPROCESS_DIR,
    "spatial": SPATIAL_DIR,
    "hmm": HMM_DIR,
    "latent": LATENT_DIR,
    "variant": VARIANT_DIR,
    "chimeric": CHIMERIC_DIR,
}
_CORE_IDENTITY_COLUMNS = (
    MOLECULE_UID_COLUMN,
    EXPERIMENT_UID_COLUMN,
    "read_id",
    "experiment_id",
    "sample_id",
    "reference",
    "physical_reference",
    "modality",
    "class_id",
)
_FILTER_OPERATORS = ("not_in", "min", "max", "in", "eq")
_ESTIMATED_BYTES_PER_CHANNEL_POSITION = 6


class MLSelectionError(ValueError):
    """Raised when metadata cannot produce one unambiguous ML selection."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _artifact_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise MLSelectionError(f"required metadata artifact does not exist: {path}")
    result = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise MLSelectionError(f"metadata artifact directory is empty: {path}")
    for child in files:
        result.update(child.relative_to(path).as_posix().encode("utf-8"))
        result.update(b"\0")
        result.update(_file_sha256(child).encode("ascii"))
    return result.hexdigest()


def _as_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise MLSelectionError("filter membership values must be a string or sequence")


@dataclass(frozen=True)
class ResolvedChannelSource:
    """One biological channel's physical source for one selected modality."""

    channel_name: str
    biological_role: str
    modality: str
    stage: str
    layer: str
    site_context: str
    catalog_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable resolved channel source."""
        return {
            "channel_name": self.channel_name,
            "biological_role": self.biological_role,
            "modality": self.modality,
            "stage": self.stage,
            "layer": self.layer,
            "site_context": self.site_context,
            "catalog_sha256": self.catalog_sha256,
        }


@dataclass(frozen=True)
class SelectedExperimentSource:
    """Resolved metadata provenance for one selected experiment."""

    experiment_id: str
    experiment_uid: str
    modality: str
    physical_references: tuple[str, ...]
    canonical_references: tuple[str, ...]
    channels: tuple[ResolvedChannelSource, ...]
    membership_artifact: Path
    membership_artifact_sha256: str
    membership_fingerprint: str
    feature_fingerprint: str

    def to_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        """Return path-neutral provenance, optionally including diagnostic paths."""
        result: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_uid": self.experiment_uid,
            "modality": self.modality,
            "physical_references": list(self.physical_references),
            "canonical_references": list(self.canonical_references),
            "channels": [channel.to_dict() for channel in self.channels],
            "membership_artifact_sha256": self.membership_artifact_sha256,
            "membership_fingerprint": self.membership_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
        }
        if include_paths:
            result["membership_artifact"] = self.membership_artifact.as_posix()
        return result


@dataclass(frozen=True)
class MLDataSelectionPlan:
    """Resolved metadata selection and conservative materialization estimate."""

    schema_version: int
    selection_id: str
    dataset_name: str
    plan_hash: str
    scope_kind: str
    scope_id: str
    set_name: str | None
    channel_policy: str
    channel_names: tuple[str, ...]
    group_by: tuple[str, ...]
    sources: tuple[SelectedExperimentSource, ...]
    identity_table: pd.DataFrame
    membership_fingerprint: str
    feature_fingerprint: str
    n_observations: int
    n_features: int
    estimated_materialization_bytes: int
    class_counts: Mapping[str, int]
    modality_counts: Mapping[str, int]
    sample_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        table = self.identity_table.copy(deep=True).reset_index(drop=True)
        missing = sorted(set(_CORE_IDENTITY_COLUMNS).difference(table.columns))
        if missing:
            raise MLSelectionError(f"identity table is missing required columns: {missing}")
        if table[MOLECULE_UID_COLUMN].duplicated().any():
            raise MLSelectionError("identity table contains duplicate molecule_uid values")
        object.__setattr__(self, "identity_table", table)

    def to_dry_run_dict(self) -> dict[str, Any]:
        """Return an explainable selection report without observation-level rows."""
        return {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "dataset_name": self.dataset_name,
            "plan_hash": self.plan_hash,
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "set_name": self.set_name,
            "channel_policy": self.channel_policy,
            "channel_names": list(self.channel_names),
            "group_by": list(self.group_by),
            "sources": [source.to_dict(include_paths=True) for source in self.sources],
            "n_observations": self.n_observations,
            "n_features": self.n_features,
            "estimated_materialization_bytes": self.estimated_materialization_bytes,
            "class_counts": dict(self.class_counts),
            "modality_counts": dict(self.modality_counts),
            "sample_counts": dict(self.sample_counts),
        }


@dataclass(frozen=True)
class _ExperimentMetadata:
    experiment_id: str
    experiment_uid: str
    modality: str
    run_root: Path
    spines: Mapping[str, Path]
    catalogs: Mapping[str, Path]
    references: Mapping[str, str]
    canonical_references: Mapping[str, str]


def _completed_stages(run_root: Path) -> set[str]:
    manifest = read_experiment_manifest(run_root)
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        return set()
    return {
        str(stage)
        for stage, record in stages.items()
        if isinstance(record, Mapping)
        and (record.get("state") == "complete" or "completed_at" in record)
    }


def _experiment_metadata(run_root: Path, experiment_id: str | None) -> _ExperimentMetadata:
    manifest = read_experiment_manifest(run_root)
    if not manifest:
        raise MLSelectionError(f"no experiment manifest at {run_root / 'experiment_manifest.json'}")
    modality = str(manifest.get("modality", "")).lower()
    uid = manifest.get(EXPERIMENT_UID_COLUMN)
    if not modality or modality == "unknown":
        raise MLSelectionError("experiment manifest has no known modality")
    if uid is None:
        raise MLSelectionError("experiment manifest has no experiment_uid")
    uid = validate_experiment_uid(uid)
    resolved_id = str(experiment_id or manifest.get("experiment") or run_root.name)
    completed = _completed_stages(run_root)
    spines = {
        stage: run_root / directory / "spine.h5ad"
        for stage, directory in _STAGE_DIRS.items()
        if stage in completed and (run_root / directory / "spine.h5ad").is_file()
    }
    raw_dir = run_root / RAW_DIR
    catalogs: dict[str, Path] = {}
    for name, path in {
        "interval_catalog": raw_dir / "interval_catalog.parquet",
        "molecule_index": run_root / "molecule_index",
        "raw_obs": raw_dir / "obs.parquet",
    }.items():
        if path.exists():
            catalogs[name] = path
    for stage, spine in spines.items():
        stage_dir = spine.parent
        if stage == "preprocess":
            pointer_path = stage_dir / "current.json"
            if pointer_path.is_file():
                try:
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise MLSelectionError(
                        f"preprocess current pointer is unreadable: {pointer_path}"
                    ) from exc
                relative = Path(str(pointer.get("generation_path", "")))
                generation = (stage_dir / relative).resolve()
                if (
                    not str(relative)
                    or relative.is_absolute()
                    or not generation.is_relative_to(stage_dir.resolve())
                ):
                    raise MLSelectionError(
                        f"preprocess current pointer is not portable: {pointer_path}"
                    )
                stage_dir = generation
        for suffix, candidate in {
            "read_index": stage_dir / "read_index",
            "task_catalog": stage_dir / "task_catalog.parquet",
        }.items():
            if candidate.exists():
                catalogs[f"{stage}_{suffix}"] = candidate
    references = {
        str(name): str(uid_value)
        for name, uid_value in dict(manifest.get("reference_uids", {})).items()
    }
    return _ExperimentMetadata(
        experiment_id=resolved_id,
        experiment_uid=uid,
        modality=modality,
        run_root=run_root,
        spines=spines,
        catalogs=catalogs,
        references=references,
        canonical_references={name: name for name in references},
    )


def _project_metadata(
    project_dir: Path,
    dataset: DatasetSpec,
    *,
    set_name: str | None,
) -> list[_ExperimentMetadata]:
    entries = list_experiments(project_dir)
    by_id = {str(entry["id"]): entry for entry in entries}
    selected_ids = set(by_id)
    if set_name is not None:
        # Shared with the project catalog and `project show-set`, so an ML
        # selection narrows to exactly the membership the CLI reports.
        selected_ids &= set(resolve_set_membership(project_dir, set_name).resolved)
    if dataset.experiments.include:
        selected_ids &= set(dataset.experiments.include)
    selected_ids -= set(dataset.experiments.exclude)
    registry = ReferenceRegistry.load(project_dir / REFERENCE_REGISTRY_FILENAME)
    result = []
    for experiment_id in sorted(selected_ids):
        entry = by_id[experiment_id]
        modality = str(entry.get("modality", "")).lower()
        if not modality or modality == "unknown":
            raise MLSelectionError(f"project experiment {experiment_id!r} has no known modality")
        if modality not in dataset.modalities:
            continue
        references = {
            str(name): str(uid) for name, uid in dict(entry.get("references", {})).items()
        }
        result.append(
            _ExperimentMetadata(
                experiment_id=experiment_id,
                experiment_uid=validate_experiment_uid(entry["experiment_uid"]),
                modality=modality,
                run_root=Path(entry["path"]),
                spines={stage: Path(path) for stage, path in entry.get("spines", {}).items()},
                catalogs={name: Path(path) for name, path in entry.get("catalogs", {}).items()},
                references=references,
                canonical_references={
                    name: registry.canonical_reference(uid) for name, uid in references.items()
                },
            )
        )
    return result


def _source_for_modality(
    dataset: DatasetSpec,
    *,
    modality: str,
) -> list[tuple[str, str, PhysicalChannelSource]]:
    result = []
    for channel in dataset.channels:
        matches = [source for source in channel.sources if source.modality == modality]
        if len(matches) > 1:
            raise MLSelectionError(
                f"channel {channel.name!r} has ambiguous physical sources for modality {modality!r}"
            )
        if not matches:
            if dataset.channel_policy != "union":
                raise MLSelectionError(
                    f"channel {channel.name!r} has no physical source for modality {modality!r}"
                )
            continue
        source = matches[0]
        _validate_channel_semantics(
            modality=modality,
            site_context=source.site_context,
            biological_role=channel.biological_role,
        )
        result.append((channel.name, channel.biological_role, source))
    if not result:
        raise MLSelectionError(f"modality {modality!r} has no available input channels")
    return result


def _validate_channel_semantics(
    *,
    modality: str,
    site_context: str,
    biological_role: str,
) -> None:
    context = site_context.lower()
    role = biological_role.lower()
    if modality == "deaminase" and (context != "c" or role != "accessibility"):
        raise MLSelectionError(
            "deaminase C-site input must be explicitly declared as accessibility"
        )
    if modality == "conversion" and context == "gpc" and role != "accessibility":
        raise MLSelectionError("conversion GpC input must be declared as accessibility")
    if context == "cpg" and role not in {"accessibility", "endogenous_methylation"}:
        raise MLSelectionError(
            "CpG input has ambiguous biological meaning; declare accessibility or "
            "endogenous_methylation"
        )


def _stage_read_index(metadata: _ExperimentMetadata, stage: str) -> Path | None:
    if stage == "raw":
        return metadata.catalogs.get("molecule_index")
    registered = metadata.catalogs.get(f"{stage}_read_index")
    if registered is not None:
        return registered
    spine = metadata.spines.get(stage)
    if spine is not None and (spine.parent / "read_index").exists():
        return spine.parent / "read_index"
    return None


def _stage_task_catalog(metadata: _ExperimentMetadata, stage: str) -> Path | None:
    registered = metadata.catalogs.get(f"{stage}_task_catalog")
    if registered is not None:
        return registered
    read_index = _stage_read_index(metadata, stage)
    if read_index is not None and (read_index.parent / "task_catalog.parquet").is_file():
        return read_index.parent / "task_catalog.parquet"
    spine = metadata.spines.get(stage)
    if spine is not None and (spine.parent / "task_catalog.parquet").is_file():
        return spine.parent / "task_catalog.parquet"
    return None


def _layer_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {value}
        return _layer_values(decoded)
    if isinstance(value, Sequence):
        return {str(item) for item in value}
    if hasattr(value, "tolist"):
        return _layer_values(value.tolist())
    return set()


def _resolve_channels(
    metadata: _ExperimentMetadata,
    dataset: DatasetSpec,
    physical_references: tuple[str, ...],
) -> tuple[ResolvedChannelSource, ...]:
    channels = []
    for channel_name, role, source in _source_for_modality(dataset, modality=metadata.modality):
        if source.stage not in metadata.spines and source.stage != "raw":
            raise MLSelectionError(
                f"experiment {metadata.experiment_id!r} has no complete stage "
                f"{source.stage!r} for channel {channel_name!r}"
            )
        catalog = _stage_task_catalog(metadata, source.stage)
        if catalog is None or not catalog.is_file():
            raise MLSelectionError(
                f"experiment {metadata.experiment_id!r} cannot verify layer {source.layer!r}: "
                f"stage {source.stage!r} has no task catalog"
            )
        frame = pd.read_parquet(catalog)
        if "layers" not in frame:
            raise MLSelectionError(f"stage catalog {catalog} does not declare written layers")
        if "reference" in frame:
            frame = frame.loc[frame["reference"].astype(str).isin(physical_references)]
        if frame.empty:
            raise MLSelectionError(
                f"stage {source.stage!r} has no tasks for selected references in "
                f"experiment {metadata.experiment_id!r}"
            )
        missing = [
            index
            for index, value in frame["layers"].items()
            if source.layer not in _layer_values(value)
        ]
        if missing:
            raise MLSelectionError(
                f"layer {source.layer!r} is unavailable in {len(missing)} selected "
                f"{source.stage!r} task(s) for experiment {metadata.experiment_id!r}"
            )
        channels.append(
            ResolvedChannelSource(
                channel_name=channel_name,
                biological_role=role,
                modality=metadata.modality,
                stage=source.stage,
                layer=source.layer,
                site_context=source.site_context,
                catalog_sha256=_artifact_sha256(catalog),
            )
        )
    return tuple(channels)


def _filter_definition(key: str) -> tuple[str, str]:
    for operator in _FILTER_OPERATORS:
        suffix = f"_{operator}"
        if key.endswith(suffix):
            return key[: -len(suffix)], operator
    return key, "eq"


def _required_metadata_columns(
    dataset: DatasetSpec,
    group_by: tuple[str, ...],
) -> set[str]:
    result = set(group_by)
    if dataset.labels is not None:
        result.add(dataset.labels.column)
    for key in dataset.filters:
        if key not in {"start", "end"}:
            result.add(_filter_definition(key)[0])
    return result


def _read_identity_metadata(
    metadata: _ExperimentMetadata,
    *,
    required_columns: set[str],
) -> tuple[pd.DataFrame, Path]:
    index_path = metadata.catalogs.get("molecule_index")
    if index_path is None:
        index_path = metadata.run_root / "molecule_index"
    if not index_path.exists():
        raise MLSelectionError(
            f"experiment {metadata.experiment_id!r} has no molecule identity index"
        )
    dataset = ds.dataset(index_path, format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    base_columns = {
        MOLECULE_UID_COLUMN,
        EXPERIMENT_UID_COLUMN,
        "read_id",
        "Reference_strand",
        "Sample",
        "Barcode",
    }
    selected = sorted((base_columns | required_columns).intersection(available))
    frame = dataset.to_table(columns=selected).to_pandas()
    missing = required_columns.difference(frame.columns)
    if missing:
        raw_spine = metadata.spines.get("raw")
        obs_path = metadata.catalogs.get("raw_obs") or (
            raw_spine.parent / "obs.parquet" if raw_spine is not None else None
        )
        if obs_path is not None and obs_path.is_file():
            obs_dataset = ds.dataset(obs_path, format="parquet")
            obs_available = set(obs_dataset.schema.names)
            join_columns = sorted((missing | {"read_id"}).intersection(obs_available))
            if "read_id" in join_columns:
                obs = obs_dataset.to_table(columns=join_columns).to_pandas()
                if obs["read_id"].astype(str).duplicated().any():
                    raise MLSelectionError(f"raw obs sidecar has duplicate read IDs: {obs_path}")
                frame = frame.merge(obs, on="read_id", how="left", validate="one_to_one")
    still_missing = sorted(required_columns.difference(frame.columns))
    if still_missing:
        raise MLSelectionError(
            f"experiment {metadata.experiment_id!r} metadata lacks required columns: "
            f"{still_missing}"
        )
    return frame, index_path


def _sample_mask(
    values: pd.Series,
    *,
    experiment_id: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> pd.Series:
    samples = values.astype(str)

    def matches(token: str) -> pd.Series:
        return samples == (
            token.split("/", 1)[1] if token.startswith(f"{experiment_id}/") else token
        )

    mask = pd.Series(True, index=values.index)
    if include:
        mask &= pd.concat([matches(token) for token in include], axis=1).any(axis=1)
    if exclude:
        mask &= ~pd.concat([matches(token) for token in exclude], axis=1).any(axis=1)
    return mask


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, expected in filters.items():
        if key in {"start", "end"}:
            continue
        column, operator = _filter_definition(str(key))
        values = frame[column]
        if operator == "min":
            mask &= values >= expected
        elif operator == "max":
            mask &= values <= expected
        elif operator == "in":
            mask &= values.astype(str).isin(_as_strings(expected))
        elif operator == "not_in":
            mask &= ~values.astype(str).isin(_as_strings(expected))
        else:
            mask &= values == expected
    return frame.loc[mask]


def _canonical_reference_map(
    metadata: _ExperimentMetadata,
    requested: tuple[str, ...],
) -> dict[str, str]:
    result = {}
    for physical, canonical in metadata.canonical_references.items():
        uid = metadata.references.get(physical)
        if not requested or physical in requested or canonical in requested or uid in requested:
            result[physical] = canonical
    return result


def _stage_membership(
    metadata: _ExperimentMetadata,
    stages: set[str],
) -> set[str] | None:
    membership: set[str] | None = None
    for stage in sorted(stages.difference({"raw"})):
        index = _stage_read_index(metadata, stage)
        if index is None or not index.exists():
            raise MLSelectionError(
                f"experiment {metadata.experiment_id!r} has no read index for stage {stage!r}"
            )
        stage_dataset = ds.dataset(index, format="parquet", partitioning="hive")
        if MOLECULE_UID_COLUMN not in stage_dataset.schema.names:
            raise MLSelectionError(f"stage read index lacks {MOLECULE_UID_COLUMN!r}: {index}")
        values = set(
            stage_dataset.to_table(columns=[MOLECULE_UID_COLUMN])
            .column(MOLECULE_UID_COLUMN)
            .to_pylist()
        )
        membership = values if membership is None else membership.intersection(values)
    return membership


def _identity_for_experiment(
    metadata: _ExperimentMetadata,
    dataset: DatasetSpec,
    group_by: tuple[str, ...],
    reference_map: Mapping[str, str],
    channels: tuple[ResolvedChannelSource, ...],
) -> tuple[pd.DataFrame, Path]:
    required = _required_metadata_columns(dataset, group_by)
    core_groups = {
        "experiment_uid",
        "experiment_id",
        "modality",
        "sample_id",
        "reference",
        "physical_reference",
        "Sample",
        "Barcode",
    }
    frame, artifact = _read_identity_metadata(
        metadata,
        required_columns=required.difference(core_groups),
    )
    if "Reference_strand" not in frame:
        raise MLSelectionError("molecule index lacks 'Reference_strand'")
    frame = frame.loc[frame["Reference_strand"].astype(str).isin(reference_map)]
    sample_column = "Sample" if "Sample" in frame else "Barcode" if "Barcode" in frame else None
    if sample_column is None:
        raise MLSelectionError("molecule index lacks Sample and Barcode identity")
    frame = frame.loc[
        _sample_mask(
            frame[sample_column],
            experiment_id=metadata.experiment_id,
            include=dataset.samples.include,
            exclude=dataset.samples.exclude,
        )
    ]
    stages = {channel.stage for channel in channels}
    stage_members = _stage_membership(metadata, stages)
    if stage_members is not None:
        frame = frame.loc[frame[MOLECULE_UID_COLUMN].astype(str).isin(stage_members)]
    frame = _apply_filters(frame, dataset.filters)
    if frame.empty:
        return pd.DataFrame(columns=_CORE_IDENTITY_COLUMNS), artifact
    frame[EXPERIMENT_UID_COLUMN] = frame[EXPERIMENT_UID_COLUMN].astype(str)
    if set(frame[EXPERIMENT_UID_COLUMN]) != {metadata.experiment_uid}:
        raise MLSelectionError(
            f"experiment UID mismatch in molecule index for {metadata.experiment_id!r}"
        )
    expected_uids = [
        molecule_uid(metadata.experiment_uid, read_id) for read_id in frame["read_id"].astype(str)
    ]
    if frame[MOLECULE_UID_COLUMN].astype(str).tolist() != expected_uids:
        raise MLSelectionError(
            f"inconsistent stable molecule identities for experiment {metadata.experiment_id!r}"
        )
    selected = pd.DataFrame(
        {
            MOLECULE_UID_COLUMN: frame[MOLECULE_UID_COLUMN].astype(str),
            EXPERIMENT_UID_COLUMN: metadata.experiment_uid,
            "read_id": frame["read_id"].astype(str),
            "experiment_id": metadata.experiment_id,
            "sample_id": frame[sample_column].astype(str),
            "reference": frame["Reference_strand"].astype(str).map(reference_map),
            "physical_reference": frame["Reference_strand"].astype(str),
            "modality": metadata.modality,
        }
    )
    if dataset.labels is None:
        selected["class_id"] = None
    else:
        label = frame[dataset.labels.column]
        missing = label.isna()
        unknown = sorted(set(label.loc[~missing].astype(str)).difference(dataset.labels.classes))
        if unknown:
            raise MLSelectionError(
                f"label column {dataset.labels.column!r} contains undeclared classes: {unknown}"
            )
        if missing.any() and dataset.labels.missing == "error":
            raise MLSelectionError(
                f"label column {dataset.labels.column!r} contains missing values"
            )
        selected["class_id"] = label.astype(str).map(dataset.labels.classes)
        if dataset.labels.missing == "drop":
            selected = selected.loc[~missing]
            frame = frame.loc[~missing]
    for field in group_by:
        if field in selected:
            continue
        if field == "Sample":
            selected[field] = selected["sample_id"]
        elif field == "Barcode" and "Barcode" in frame:
            selected[field] = frame["Barcode"].astype(str)
        elif field in frame:
            selected[field] = frame[field].astype(str)
        else:
            raise MLSelectionError(f"group field {field!r} is absent from selection metadata")
    return selected.reset_index(drop=True), artifact


def _feature_count(
    metadata: Sequence[_ExperimentMetadata],
    reference_maps: Mapping[str, Mapping[str, str]],
    filters: Mapping[str, Any],
) -> int:
    start = filters.get("start")
    end = filters.get("end")
    if (start is None) != (end is None):
        raise MLSelectionError("filters.start and filters.end must be provided together")
    if start is not None:
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise MLSelectionError("filters.start/end must define a valid half-open interval")
        return end - start
    lengths: dict[str, int] = {}
    for item in metadata:
        catalog = _interval_catalog(item)
        frame = pd.read_parquet(catalog)
        if not {"reference", "max_end"}.issubset(frame):
            raise MLSelectionError(f"interval catalog lacks reference/max_end columns: {catalog}")
        selected = frame.loc[
            frame["reference"].astype(str).isin(reference_maps[item.experiment_id])
        ]
        for physical, maximum in selected.groupby("reference")["max_end"].max().items():
            canonical = reference_maps[item.experiment_id][str(physical)]
            lengths[canonical] = max(lengths.get(canonical, 0), int(maximum))
    if not lengths:
        raise MLSelectionError("selected references have no feature coordinates")
    return sum(lengths.values())


def _interval_catalog(metadata: _ExperimentMetadata) -> Path:
    catalog = metadata.catalogs.get("interval_catalog")
    if catalog is None:
        catalog = metadata.catalogs.get("interval_catalog.parquet")
    if catalog is None:
        raw_spine = metadata.spines.get("raw")
        catalog = raw_spine.parent / "interval_catalog.parquet" if raw_spine else None
    if catalog is None or not catalog.is_file():
        raise MLSelectionError(f"experiment {metadata.experiment_id!r} has no raw interval catalog")
    return catalog


def _automatic_group_fields(plan: MLPlan, dataset_name: str) -> tuple[str, ...]:
    fields = set()
    for job in plan.jobs.values():
        if job.dataset == dataset_name and job.split is not None:
            fields.update(plan.splits[job.split].group_by)
    return tuple(sorted(fields))


def plan_ml_dataset(
    plan: MLPlan,
    dataset_name: str,
    *,
    experiment_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    experiment_id: str | None = None,
    group_by: Sequence[str] | None = None,
) -> MLDataSelectionPlan:
    """Resolve one dataset from metadata without opening feature matrices.

    Exactly one scope directory is required and must agree with ``plan.scope``.
    The returned identity table contains scalar observation metadata only.
    """
    if dataset_name not in plan.datasets:
        raise MLSelectionError(f"unknown dataset {dataset_name!r}")
    if (experiment_dir is None) == (project_dir is None):
        raise MLSelectionError("provide exactly one of experiment_dir or project_dir")
    dataset = plan.datasets[dataset_name]
    resolved_groups = tuple(
        dict.fromkeys(
            str(field)
            for field in (
                _automatic_group_fields(plan, dataset_name) if group_by is None else group_by
            )
        )
    )
    if plan.scope.kind == "experiment":
        if experiment_dir is None:
            raise MLSelectionError("experiment-scoped plan requires experiment_dir")
        metadata = [_experiment_metadata(Path(experiment_dir).resolve(), experiment_id)]
        item = metadata[0]
        if (
            dataset.experiments.include and item.experiment_id not in dataset.experiments.include
        ) or item.experiment_id in dataset.experiments.exclude:
            raise MLSelectionError("dataset selection excludes the scoped experiment")
        if item.modality not in dataset.modalities:
            raise MLSelectionError(f"scoped experiment modality {item.modality!r} is not selected")
        scope_id = metadata[0].experiment_id
    else:
        if project_dir is None:
            raise MLSelectionError("project-scoped plan requires project_dir")
        project_path = Path(project_dir).resolve()
        metadata = _project_metadata(
            project_path,
            dataset,
            set_name=plan.scope.set_name,
        )
        scope_id = project_path.name
    if not metadata:
        raise MLSelectionError("dataset selection matched no active experiments")
    unknown = sorted({item.modality for item in metadata}.difference(dataset.modalities))
    if unknown:
        raise MLSelectionError(f"selected experiments have unsupported modalities: {unknown}")

    source_records = []
    tables = []
    selected_metadata = []
    reference_maps: dict[str, dict[str, str]] = {}
    for item in metadata:
        reference_map = _canonical_reference_map(item, dataset.references)
        if not reference_map:
            continue
        reference_maps[item.experiment_id] = reference_map
        channels = _resolve_channels(item, dataset, tuple(sorted(reference_map)))
        table, membership_artifact = _identity_for_experiment(
            item,
            dataset,
            resolved_groups,
            reference_map,
            channels,
        )
        if table.empty:
            continue
        membership_fingerprint = _sha256(sorted(table[MOLECULE_UID_COLUMN].astype(str)))
        feature_fingerprint = _sha256(
            {
                "references": dict(sorted(reference_map.items())),
                "channels": [channel.to_dict() for channel in channels],
                "interval_catalog_sha256": _artifact_sha256(_interval_catalog(item)),
            }
        )
        source_records.append(
            SelectedExperimentSource(
                experiment_id=item.experiment_id,
                experiment_uid=item.experiment_uid,
                modality=item.modality,
                physical_references=tuple(sorted(reference_map)),
                canonical_references=tuple(sorted(set(reference_map.values()))),
                channels=channels,
                membership_artifact=membership_artifact.resolve(),
                membership_artifact_sha256=_artifact_sha256(membership_artifact),
                membership_fingerprint=membership_fingerprint,
                feature_fingerprint=feature_fingerprint,
            )
        )
        tables.append(table)
        selected_metadata.append(item)
    if not tables:
        raise MLSelectionError("dataset selection matched no eligible observations")
    identity = pd.concat(tables, ignore_index=True).sort_values(MOLECULE_UID_COLUMN, kind="stable")
    if identity[MOLECULE_UID_COLUMN].duplicated().any():
        raise MLSelectionError("selected experiments contain duplicate molecule identities")
    n_features = _feature_count(selected_metadata, reference_maps, dataset.filters)
    membership_fingerprint = _sha256(identity[MOLECULE_UID_COLUMN].astype(str).tolist())
    feature_fingerprint = _sha256(
        [
            source.feature_fingerprint
            for source in sorted(source_records, key=lambda x: x.experiment_id)
        ]
    )
    identity_payload = {
        "dataset_name": dataset_name,
        "plan_hash": plan.plan_hash,
        "scope_kind": plan.scope.kind,
        "scope_id": scope_id,
        "set_name": plan.scope.set_name,
        "membership_fingerprint": membership_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "source_artifacts": [
            {
                "experiment_id": source.experiment_id,
                "membership_artifact_sha256": source.membership_artifact_sha256,
            }
            for source in sorted(source_records, key=lambda item: item.experiment_id)
        ],
    }
    class_values = identity["class_id"].dropna().map(lambda value: str(int(value)))
    estimated_bytes = (
        len(identity)
        * n_features
        * max(1, len(dataset.channels))
        * _ESTIMATED_BYTES_PER_CHANNEL_POSITION
    )
    return MLDataSelectionPlan(
        schema_version=ML_SELECTION_PLAN_VERSION,
        selection_id=_sha256(identity_payload),
        dataset_name=dataset_name,
        plan_hash=plan.plan_hash,
        scope_kind=plan.scope.kind,
        scope_id=scope_id,
        set_name=plan.scope.set_name,
        channel_policy=dataset.channel_policy,
        channel_names=tuple(channel.name for channel in dataset.channels),
        group_by=resolved_groups,
        sources=tuple(sorted(source_records, key=lambda item: item.experiment_id)),
        identity_table=identity,
        membership_fingerprint=membership_fingerprint,
        feature_fingerprint=feature_fingerprint,
        n_observations=len(identity),
        n_features=n_features,
        estimated_materialization_bytes=estimated_bytes,
        class_counts=dict(sorted(Counter(class_values).items())),
        modality_counts=dict(sorted(Counter(identity["modality"].astype(str)).items())),
        sample_counts=dict(sorted(Counter(identity["sample_id"].astype(str)).items())),
    )
