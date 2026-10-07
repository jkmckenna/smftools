"""Metadata-only planning of eligible machine-learning observations and channels."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from smftools.informatics.barcode_sidecar import barcode_number_key
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

from .plan import (
    ALL_POSITIONS,
    SITE_CALL_STAGES,
    CoordinateFrame,
    DatasetSpec,
    LabelSpec,
    MLPlan,
    PhysicalChannelSource,
)

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
    # Execution-time bindings (`MLX-06`), never part of identity: where the
    # experiment lives, and each read stage's spine and generation.
    run_root: Path | None = None
    stage_spines: Mapping[str, Path] = field(default_factory=dict)
    stage_generations: Mapping[str, str] = field(default_factory=dict)
    stage_read_indexes: Mapping[str, Path] = field(default_factory=dict)

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
    label_table_sha256: str | None = None
    # `MLX-03`: per mapped canonical reference, frame position by source
    # position (-1: no counterpart), and each map's sha256.
    coordinate_maps: Mapping[str, Any] = field(default_factory=dict)
    coordinate_map_sha256: Mapping[str, str] = field(default_factory=dict)

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
            **(
                {"label_table_sha256": self.label_table_sha256}
                if self.label_table_sha256 is not None
                else {}
            ),
            **(
                {"coordinate_map_sha256": dict(self.coordinate_map_sha256)}
                if self.coordinate_map_sha256
                else {}
            ),
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
            stage=source.stage,
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
    stage: str = "",
) -> None:
    context = site_context.lower()
    role = biological_role.lower()
    if context == ALL_POSITIONS:
        # Every position (`RPG-01`): for derived layers defined between sites
        # (e.g. HMM features), never for site calls read off their sites.
        if stage.lower() in SITE_CALL_STAGES:
            raise MLSelectionError(
                f"site_context {ALL_POSITIONS!r} reads every position; {stage!r} holds site "
                "calls -- use the site context they were called at"
            )
        if modality == "deaminase" and role != "accessibility":
            raise MLSelectionError("deaminase input must be declared as accessibility")
        return
    # GpC is a subset of a deaminase's C sites -- still accessibility.
    if modality == "deaminase" and (context not in {"c", "gpc"} or role != "accessibility"):
        raise MLSelectionError("deaminase input must be C or GpC sites, declared as accessibility")
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
    if spine is not None:
        # Generation layout: the read index lives in the current generation,
        # not beside the stage's top-level spine (`MLX-10`).
        generation = _current_generation_dir(spine.parent)
        if generation is not None and (generation / "read_index").exists():
            return generation / "read_index"
    return None


def _current_generation_dir(stage_dir: Path) -> Path | None:
    from smftools.informatics.generation import GenerationError, resolve_current_generation

    try:
        current = resolve_current_generation(stage_dir)
    except (GenerationError, OSError, ValueError):
        return None
    return None if current is None else Path(current[0])


def _stage_task_catalog(metadata: _ExperimentMetadata, stage: str) -> Path | None:
    """The catalog that says which layers a stage *wrote*.

    A generation holds two: ``catalog.parquet`` (the written store: ``layers``,
    ``has_x``) and ``task_catalog.parquet`` (the planner's tasks, which list no
    layers). The written one is preferred wherever both exist (`F66`).
    """
    registered = metadata.catalogs.get(f"{stage}_task_catalog")
    if registered is not None:
        return registered
    directories = []
    read_index = _stage_read_index(metadata, stage)
    if read_index is not None:
        directories.append(read_index.parent)
    spine = metadata.spines.get(stage)
    if spine is not None:
        directories.append(spine.parent)
    for directory in directories:
        for name in ("catalog.parquet", "task_catalog.parquet"):
            if (directory / name).is_file():
                return directory / name
    return None


def _stage_generation_id(metadata: _ExperimentMetadata, stage: str) -> str:
    """The generation a stage's reads come from, or ``current`` when unversioned."""
    read_index = _stage_read_index(metadata, stage)
    if read_index is not None and read_index.parent.parent.name == "generations":
        return read_index.parent.name
    return "current"


def _stage_obs_sidecar(metadata: _ExperimentMetadata, stage: str) -> Path | None:
    """A stage's per-read obs (QC and dedup flags live only here)."""
    read_index = _stage_read_index(metadata, stage)
    for directory in (
        read_index.parent if read_index is not None else None,
        metadata.spines[stage].parent if stage in metadata.spines else None,
    ):
        if directory is not None and (directory / "stage_obs.parquet").is_file():
            return directory / "stage_obs.parquet"
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
        has_x = frame["has_x"] if "has_x" in frame else pd.Series(False, index=frame.index)
        available = {
            index: _layer_values(value) | ({"X"} if bool(has_x.loc[index]) else set())
            for index, value in frame["layers"].items()
        }
        missing = [index for index, layers in available.items() if source.layer not in layers]
        if missing:
            written = sorted(set().union(*available.values()))
            raise MLSelectionError(
                f"layer {source.layer!r} is unavailable in {len(missing)} selected "
                f"{source.stage!r} task(s) for experiment {metadata.experiment_id!r}; "
                f"the stage wrote {written}. Stores from partitioned preprocess hold "
                "site calls in 'X': declare layer 'X' with the channel's site_context."
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


@dataclass(frozen=True)
class _LabelTable:
    """A project label table (`MLX-01`), keyed on normalized identity fields."""

    keys: tuple[str, ...]
    frame: pd.DataFrame  # indexed by the normalized key tuple
    columns: frozenset[str]  # every non-key column
    sha256: str


def _barcode_token(value: Any) -> Any:
    """Barcode spellings compare by number: ``4``, ``4.0``, ``NB04``, ``barcode04``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return barcode_number_key(str(value))


def _normalized_key(name: str, values: pd.Series) -> pd.Series:
    if name == "barcode":
        return values.map(_barcode_token)
    return values.astype(str)


def _load_label_table(project_path: Path, labels: LabelSpec) -> _LabelTable:
    assert labels.table is not None
    path = project_path / labels.table
    if not path.is_file():
        raise MLSelectionError(f"label table not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    absent = [column for column in (*labels.keys, labels.column) if column not in frame]
    if absent:
        raise MLSelectionError(f"label table {labels.table!r} lacks columns {absent}")
    keys = list(labels.keys)
    for key in keys:
        frame[key] = _normalized_key(key, frame[key])
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].drop_duplicates().head(3).to_dict("records")
        raise MLSelectionError(
            f"label table {labels.table!r} has {int(duplicated.sum())} rows sharing a key, "
            f"e.g. {examples}"
        )
    indexed = frame.set_index(pd.MultiIndex.from_frame(frame[keys])).drop(columns=keys)
    return _LabelTable(
        keys=tuple(keys),
        frame=indexed,
        columns=frozenset(indexed.columns),
        sha256=_file_sha256(path),
    )


def _join_label_table(
    frame: pd.DataFrame,
    table: _LabelTable,
    *,
    metadata: _ExperimentMetadata,
    reference_map: Mapping[str, str],
    sample_column: str,
    stored_columns: set[str],
) -> pd.DataFrame:
    """Add the label table's columns to each row whose identity it lists."""
    # Against everything stored, not just the columns loaded for this plan: a
    # table column must never stand in for a stored one of the same name.
    collisions = sorted(table.columns.intersection(stored_columns | set(frame.columns)))
    if collisions:
        raise MLSelectionError(
            f"label table columns {collisions} collide with molecule metadata of "
            f"experiment {metadata.experiment_id!r}; rename them in the table"
        )
    if "barcode" in table.keys and "Barcode" not in frame:
        raise MLSelectionError(
            f"label table is keyed on barcode but experiment {metadata.experiment_id!r} "
            "has no Barcode identity"
        )
    physical = frame["Reference_strand"].astype(str)
    sources = {
        "experiment_id": pd.Series(metadata.experiment_id, index=frame.index),
        "experiment_uid": frame[EXPERIMENT_UID_COLUMN],
        "barcode": frame["Barcode"] if "Barcode" in frame else None,
        "sample": frame[sample_column],
        "reference": physical.map(reference_map),
        "physical_reference": physical,
    }
    index = pd.MultiIndex.from_arrays(
        [_normalized_key(key, sources[key]) for key in table.keys], names=list(table.keys)
    )
    joined = table.frame.reindex(index)
    frame = frame.copy()
    for column in sorted(table.columns):
        frame[column] = joined[column].to_numpy()
    return frame


@dataclass(frozen=True)
class _CoordinateMap:
    frame_of: np.ndarray  # frame position by source position; -1 = none
    sha256: str


def _load_coordinate_maps(project_path: Path, frame: CoordinateFrame) -> dict[str, _CoordinateMap]:
    """Validated source -> frame position maps (`MLX-03`)."""
    maps = {}
    for source_name, relative in frame.maps.items():
        path = project_path / relative
        if not path.is_file():
            raise MLSelectionError(f"coordinate map not found: {path}")
        table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if not {"source_position", "frame_position"}.issubset(table):
            raise MLSelectionError(
                f"coordinate map {relative!r} needs source_position and frame_position columns"
            )
        pairs = table[["source_position", "frame_position"]]
        if pairs.isna().any().any() or not all(
            pd.api.types.is_integer_dtype(pairs[column]) for column in pairs
        ):
            raise MLSelectionError(f"coordinate map {relative!r} positions must be integers")
        pairs = pairs.sort_values("source_position")
        source = pairs["source_position"].to_numpy(dtype=np.int64)
        target = pairs["frame_position"].to_numpy(dtype=np.int64)
        if len(source) and (source.min() < 0 or target.min() < 0):
            raise MLSelectionError(f"coordinate map {relative!r} has negative positions")
        if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
            raise MLSelectionError(f"coordinate map {relative!r} must be one-to-one")
        if np.any(np.diff(target) <= 0):
            raise MLSelectionError(
                f"coordinate map {relative!r} must keep order: frame positions must increase "
                "with source positions"
            )
        frame_of = np.full(int(source.max()) + 1 if len(source) else 0, -1, dtype=np.int64)
        frame_of[source] = target
        maps[str(source_name)] = _CoordinateMap(frame_of=frame_of, sha256=_file_sha256(path))
    return maps


def _runs(positions: np.ndarray) -> list[tuple[int, int]]:
    if positions.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = np.concatenate([[positions[0]], positions[breaks + 1]])
    ends = np.concatenate([positions[breaks], [positions[-1]]]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _frame_feature_count(
    metadata: Sequence[_ExperimentMetadata],
    reference_maps: Mapping[str, Mapping[str, str]],
    frame: CoordinateFrame,
    maps: Mapping[str, _CoordinateMap],
    windows: Sequence[tuple[int, int]],
) -> int:
    """Feature count in the frame, refusing positions some reference lacks.

    Which frame positions a molecule *has* would identify its reference -- in
    an intact-vs-deletion task, its class. So every selected frame position
    must exist on every selected reference; there is no override (`MLX-03`).
    """
    lengths = _reference_lengths(metadata, reference_maps)
    if frame.reference not in lengths:
        raise MLSelectionError(
            f"coordinate_frame reference {frame.reference!r} is not among the selected references"
        )
    kept_windows = list(windows) or [(0, lengths[frame.reference])]
    if kept_windows[-1][1] > lengths[frame.reference]:
        raise MLSelectionError(
            f"positions window ends at {kept_windows[-1][1]}, beyond the frame reference length "
            f"{lengths[frame.reference]}"
        )
    kept = np.concatenate([np.arange(start, end) for start, end in kept_windows])
    for reference in sorted(set(lengths).difference({frame.reference})):
        mapped = maps[reference].frame_of
        missing = np.setdiff1d(kept, mapped[mapped >= 0])
        if missing.size:
            raise MLSelectionError(
                f"frame positions {_runs(missing)} have no counterpart on {reference!r}: "
                "a model could tell the references apart by which positions a molecule has. "
                "Select only positions every reference carries (positions.exclude)."
            )
    return int(kept.size)


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
    stages: Sequence[str] = (),
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
    # Then the obs of each stage the dataset reads, and preprocess's in any
    # case: QC and dedup flags (`passes_qc`, `passes_dedup`, ...) exist only
    # there (`F66`), also for a dataset that reads only derived stages (`F73`).
    obs_stages = sorted(set(stages).difference({"raw"}))
    if "preprocess" not in obs_stages and "preprocess" in metadata.spines:
        obs_stages.append("preprocess")
    for stage in obs_stages:
        missing = required_columns.difference(frame.columns)
        if not missing:
            break
        sidecar = _stage_obs_sidecar(metadata, stage)
        if sidecar is None:
            continue
        stage_dataset = ds.dataset(sidecar, format="parquet")
        join_columns = sorted(missing.intersection(stage_dataset.schema.names))
        if not join_columns or "read_id" not in stage_dataset.schema.names:
            continue
        stage_obs = stage_dataset.to_table(columns=["read_id", *join_columns]).to_pandas()
        stage_obs["read_id"] = stage_obs["read_id"].astype(str)
        if stage_obs["read_id"].duplicated().any():
            raise MLSelectionError(f"{stage} stage obs has duplicate read IDs: {sidecar}")
        frame = frame.merge(stage_obs, on="read_id", how="left", validate="one_to_one")
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
    label_table: _LabelTable | None = None,
) -> tuple[pd.DataFrame, Path]:
    required = _required_metadata_columns(dataset, group_by)
    if label_table is not None:
        required -= label_table.columns
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
        stages=sorted({channel.stage for channel in channels}),
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
    if label_table is not None:
        # Before `filters`, so filters and group_by may name table columns.
        frame = _join_label_table(
            frame,
            label_table,
            metadata=metadata,
            reference_map=reference_map,
            sample_column=sample_column,
            stored_columns=set(
                ds.dataset(artifact, format="parquet", partitioning="hive").schema.names
            ),
        )
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
    windows: Sequence[tuple[int, int]] = (),
) -> int:
    if windows:
        # `MLX-02`: only kept positions are features; they must exist.
        length = _reference_lengths(metadata, reference_maps)
        shortest = min(length.values())
        if windows[-1][1] > shortest:
            raise MLSelectionError(
                f"positions window ends at {windows[-1][1]}, beyond the selected "
                f"reference length {shortest}"
            )
        return sum(end - start for start, end in windows)
    start = filters.get("start")
    end = filters.get("end")
    if (start is None) != (end is None):
        raise MLSelectionError("filters.start and filters.end must be provided together")
    if start is not None:
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise MLSelectionError("filters.start/end must define a valid half-open interval")
        return end - start
    return sum(_reference_lengths(metadata, reference_maps).values())


def _reference_lengths(
    metadata: Sequence[_ExperimentMetadata],
    reference_maps: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    """Canonical reference -> its covered length, from the raw interval catalogs."""
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
    return lengths


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
    label_table = None
    coordinate_maps: dict[str, _CoordinateMap] = {}
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
        if dataset.labels is not None and dataset.labels.source == "table":
            label_table = _load_label_table(project_path, dataset.labels)
        if dataset.coordinate_frame is not None:
            coordinate_maps = _load_coordinate_maps(project_path, dataset.coordinate_frame)
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
        if dataset.coordinate_frame is not None:
            allowed = {dataset.coordinate_frame.reference, *dataset.coordinate_frame.maps}
            unmapped = sorted(set(reference_map.values()).difference(allowed))
            if unmapped:
                raise MLSelectionError(
                    f"experiment {item.experiment_id!r} selects references {unmapped} that "
                    "coordinate_frame neither uses as frame nor maps"
                )
        reference_maps[item.experiment_id] = reference_map
        channels = _resolve_channels(item, dataset, tuple(sorted(reference_map)))
        table, membership_artifact = _identity_for_experiment(
            item,
            dataset,
            resolved_groups,
            reference_map,
            channels,
            label_table,
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
                run_root=item.run_root,
                stage_spines={
                    channel.stage: item.spines[channel.stage]
                    for channel in channels
                    if channel.stage in item.spines
                },
                stage_generations={
                    channel.stage: _stage_generation_id(item, channel.stage) for channel in channels
                },
                stage_read_indexes={
                    channel.stage: index
                    for channel in channels
                    if (index := _stage_read_index(item, channel.stage)) is not None
                },
            )
        )
        tables.append(table)
        selected_metadata.append(item)
    if not tables:
        raise MLSelectionError("dataset selection matched no eligible observations")
    identity = pd.concat(tables, ignore_index=True).sort_values(MOLECULE_UID_COLUMN, kind="stable")
    if identity[MOLECULE_UID_COLUMN].duplicated().any():
        raise MLSelectionError("selected experiments contain duplicate molecule identities")
    windows = dataset.positions.windows() if dataset.positions is not None else ()
    if dataset.coordinate_frame is not None:
        n_features = _frame_feature_count(
            selected_metadata, reference_maps, dataset.coordinate_frame, coordinate_maps, windows
        )
    else:
        n_features = _feature_count(selected_metadata, reference_maps, dataset.filters, windows)
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
    if coordinate_maps:
        identity_payload["coordinate_maps"] = {
            reference: item.sha256 for reference, item in sorted(coordinate_maps.items())
        }
    if label_table is not None:
        # Labels changed in the table are a different dataset.
        identity_payload["label_table"] = {
            "sha256": label_table.sha256,
            "keys": list(label_table.keys),
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
        label_table_sha256=label_table.sha256 if label_table is not None else None,
        coordinate_maps={reference: item.frame_of for reference, item in coordinate_maps.items()},
        coordinate_map_sha256={
            reference: item.sha256 for reference, item in coordinate_maps.items()
        },
    )
