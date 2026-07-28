"""Transactional project-wide embeddings with explicit provenance.

An embedding definition identifies the semantic query and algorithm, but never
its currently resolved membership. Each fit or extension is published as an
immutable generation beneath that definition. A small atomic ``current.json``
selects the only generation readers may consume.

PCA and UMAP estimators are Python pickle files. They are project-local trusted
artifacts, not a stable interchange format. Coordinate-only cache reads do not
unpickle them; extending an existing embedding requires the caller to cross the
trust boundary explicitly with ``trust_local_models=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..readwrite import atomic_write_json
from .catalog import project_adata
from .set_store import resolve_set_members, set_label, sets_root

EMBEDDINGS_DIRNAME = "embeddings"
GENERATIONS_DIRNAME = "generations"
STAGING_DIRNAME = ".staging"
CURRENT_FILENAME = "current.json"
GENERATION_MANIFEST_FILENAME = "generation_manifest.json"
PCA_MODEL_FILENAME = "pca_model.pkl"
UMAP_MODEL_FILENAME = "umap_model.pkl"
PCA_SPACE_FILENAME = "pca_space.npy"
COORDS_FILENAME = "coords.npy"
CLUSTERS_FILENAME = "clusters.npy"
OBS_NAMES_FILENAME = "obs_names.json"
ROW_DIGESTS_FILENAME = "feature_row_digests.json"

IDENTITY_SCHEMA_VERSION = 2
SOURCE_SCHEMA_VERSION = 2
GENERATION_SCHEMA_VERSION = 1
EMBEDDING_IMPLEMENTATION_VERSION = 1
_ARTIFACT_FILENAMES = (
    PCA_MODEL_FILENAME,
    UMAP_MODEL_FILENAME,
    PCA_SPACE_FILENAME,
    COORDS_FILENAME,
    CLUSTERS_FILENAME,
    OBS_NAMES_FILENAME,
    ROW_DIGESTS_FILENAME,
)
_MODEL_DEPENDENCIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "pynndescent",
    "numba",
)


class EmbeddingCompositionError(ValueError):
    """Raised when membership shrank or existing feature values changed."""


class EmbeddingCompatibilityError(RuntimeError):
    """Raised when a stored generation is incompatible or invalid."""


class EmbeddingTrustError(RuntimeError):
    """Raised when an extension would unpickle models without explicit trust."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _values(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(sorted({str(item) for item in value}))


def _stable_hash(payload: object, *, length: int | None = None) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest if length is None else digest[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_definition(
    *,
    canonical_reference,
    set_name,
    modality,
    experiments,
    stage,
    layer,
    start,
    end,
    feature_kind,
    leiden_resolution,
    n_neighbors,
    min_reads,
    random_state,
) -> dict[str, object]:
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "implementation_version": EMBEDDING_IMPLEMENTATION_VERSION,
        "canonical_reference": str(canonical_reference),
        "selection": {
            "set_name": None if set_name is None else str(set_name),
            "experiments": list(_values(experiments)),
            "mode": (
                "named_set_intersection"
                if set_name is not None and experiments is not None
                else "named_set"
                if set_name is not None
                else "explicit_experiments"
                if experiments is not None
                else "all_active"
            ),
        },
        "modality": list(_values(modality)),
        "stage": None if stage is None else str(stage),
        "feature": {
            "kind": str(feature_kind),
            "layer": None if layer is None else str(layer),
            "start": start,
            "end": end,
            "coverage_filter_version": 1,
            "acf_parameters": {"rolling_window": 5, "max_lag": 1000},
        },
        "pipeline": {
            "reduction": "pca",
            "pca_max_components": 50,
            "umap_components": 2,
            "umap_min_dist": 0.3,
            "leiden_resolution": float(leiden_resolution),
            "n_neighbors": int(n_neighbors),
            "min_reads": int(min_reads),
            "random_state": int(random_state),
        },
        "molecule_identity_schema_version": 1,
    }


def _definition_hash(definition: dict[str, object]) -> str:
    return _stable_hash(definition, length=16)


def embedding_dir(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
) -> Path:
    """Return the stable definition root without reading or creating artifacts."""
    definition = _embedding_definition(
        canonical_reference=canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    label = set_label(set_name, canonical_reference)
    return sets_root(project_dir) / label / EMBEDDINGS_DIRNAME / _definition_hash(definition)


def _make_features(adata, *, feature_kind: str, layer, start, end):
    from ..analysis.compute.dimensionality_reduction import (
        coverage_filter,
        make_features_acf,
        make_features_raw,
    )

    positions = np.asarray(adata.var_names, dtype=np.int64)
    if start is not None or end is not None:
        window = np.ones(positions.shape[0], dtype=bool)
        if start is not None:
            window &= positions >= int(start)
        if end is not None:
            window &= positions < int(end)
        adata = adata[:, window]
        positions = positions[window]
    matrix_source = adata.layers[layer] if layer is not None else adata.X
    matrix = np.asarray(matrix_source, dtype=np.float64)
    obs_names = list(map(str, adata.obs_names))
    matrix, positions, obs_names, _ = coverage_filter(matrix, positions, obs_names)
    if feature_kind == "acf":
        features, valid = make_features_acf(matrix, positions)
        obs_names = [name for name, keep in zip(obs_names, valid, strict=True) if keep]
    elif feature_kind == "raw":
        features = make_features_raw(matrix)
    else:
        raise ValueError(f"feature_kind must be 'raw' or 'acf', got {feature_kind!r}")
    if len(obs_names) != len(set(obs_names)):
        raise RuntimeError("project embedding selection contains duplicate molecule identities")
    return np.ascontiguousarray(features), obs_names


def _feature_digests(features: np.ndarray, obs_names: list[str]) -> tuple[dict[str, str], str]:
    rows = {}
    for name, row in zip(obs_names, features, strict=True):
        contiguous = np.ascontiguousarray(row)
        digest = hashlib.sha256()
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
        rows[name] = digest.hexdigest()
    return rows, _stable_hash([[name, rows[name]] for name in obs_names])


def _membership_digest(obs_names: list[str]) -> str:
    return _stable_hash(obs_names)


def _portable_path(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _source_members(project_dir: Path, members: list[dict]) -> list[dict[str, object]]:
    from ..informatics.experiment_manifest import read_experiment_manifest
    from ..pipeline.project_graph import project_source_member_record
    from .catalog import ProjectCatalog

    entries = {entry["id"]: entry for entry in ProjectCatalog.open(project_dir).experiments()}
    records = []
    for member in members:
        entry = entries[member["experiment"]]
        stage_record = (
            read_experiment_manifest(entry["path"]).get("stages", {}).get(member["stage"], {})
        )
        if not isinstance(stage_record, dict):
            stage_record = {}
        stage_identity = {
            key: stage_record.get(key)
            for key in ("generation_id", "config_hash", "schema_versions", "input_artifact_ids")
            if stage_record.get(key) is not None
        }
        spine_path = Path(member["spine_path"])
        semantic_source = project_source_member_record(member)
        records.append(
            {
                "experiment": member["experiment"],
                "experiment_uid": member["experiment_uid"],
                "stage": member["stage"],
                "reference_strands": list(member["reference_strands"]),
                "spine_path": _portable_path(spine_path, project_dir),
                "spine_sha256": _file_sha256(spine_path),
                "stage_generation_id": stage_record.get("generation_id"),
                "stage_config_hash": stage_record.get("config_hash"),
                "stage_fingerprint": _stable_hash(stage_identity),
                "source_channels": {
                    "membership": semantic_source["membership_fingerprint"],
                    "features": semantic_source["feature_fingerprint"],
                    "variant_reporting": semantic_source["variant_reporting_fingerprint"],
                },
            }
        )
    return records


def _source_member_changed(previous: dict, current: dict | None) -> bool:
    """Compare only channels consumed by PL-21 when channel metadata is available."""
    if current is None:
        return True
    previous_channels = previous.get("source_channels")
    current_channels = current.get("source_channels")
    if not isinstance(previous_channels, dict) or not isinstance(current_channels, dict):
        # Schema-v1 generations lacked channel separation. Preserve their strict
        # source comparison instead of guessing that an old source is compatible.
        return current != previous
    return any(
        previous_channels.get(channel) != current_channels.get(channel)
        for channel in ("membership", "features")
    )


def _dependencies() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in ("smftools", *_MODEL_DEPENDENCIES):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "unknown"
    return values


def _source_snapshot(
    project_dir: Path,
    members: list[dict],
    features: np.ndarray,
    obs_names: list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    row_digests, feature_digest = _feature_digests(features, obs_names)
    source = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "members": _source_members(project_dir, members),
        "ordered_molecule_membership_digest": _membership_digest(obs_names),
        "feature_input_digest": feature_digest,
        "n_molecules": len(obs_names),
        "n_features": int(features.shape[1]),
    }
    return source, row_digests


def _fit_from_scratch(
    features, obs_names, *, leiden_resolution, min_reads, n_neighbors, random_state
) -> dict:
    from ..analysis.compute.dimensionality_reduction import run_pipeline

    result = run_pipeline(
        features,
        leiden_resolution=leiden_resolution,
        min_reads=min_reads,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    if result is None:
        raise ValueError(f"fewer than min_reads={min_reads} reads survived feature preparation")
    X_pca, X_umap, clusters, variance, pca_model, umap_model = result
    return {
        "X_pca": X_pca,
        "X_umap": X_umap,
        "clusters": clusters,
        "pca_model": pca_model,
        "umap_model": umap_model,
        "obs_names": obs_names,
        "explained_variance_ratio": variance.tolist(),
    }


def _assign_nearest_cluster(new_points, reference_points, reference_clusters):
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=1).fit(reference_points)
    _, indices = model.kneighbors(new_points)
    return reference_clusters[indices[:, 0]]


def _write_generation_artifacts(directory: Path, result: dict, row_digests: dict[str, str]) -> None:
    np.save(directory / PCA_SPACE_FILENAME, result["X_pca"], allow_pickle=False)
    np.save(directory / COORDS_FILENAME, result["X_umap"], allow_pickle=False)
    np.save(directory / CLUSTERS_FILENAME, result["clusters"], allow_pickle=False)
    atomic_write_json(directory / OBS_NAMES_FILENAME, result["obs_names"])
    atomic_write_json(directory / ROW_DIGESTS_FILENAME, row_digests)
    with (directory / PCA_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(result["pca_model"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (directory / UMAP_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(result["umap_model"], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _artifact_checksums(directory: Path) -> dict[str, str]:
    return {name: _file_sha256(directory / name) for name in _ARTIFACT_FILENAMES}


def _validate_generation(
    directory: Path,
    *,
    definition: dict[str, object],
    expected_generation_id: str | None = None,
) -> dict[str, object]:
    manifest_path = directory / GENERATION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise EmbeddingCompatibilityError(f"embedding generation manifest is missing: {directory}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != GENERATION_SCHEMA_VERSION:
        raise EmbeddingCompatibilityError(
            "embedding generation schema is incompatible; pass force_recompute=True"
        )
    if manifest.get("status") != "complete":
        raise EmbeddingCompatibilityError("embedding generation is not complete")
    if manifest.get("definition") != definition:
        raise EmbeddingCompatibilityError("embedding generation definition does not match its path")
    if (
        expected_generation_id is not None
        and manifest.get("generation_id") != expected_generation_id
    ):
        raise EmbeddingCompatibilityError("embedding generation ID does not match current pointer")
    checksums = manifest.get("artifacts", {})
    for name in _ARTIFACT_FILENAMES:
        path = directory / name
        if not path.is_file() or checksums.get(name) != _file_sha256(path):
            raise EmbeddingCompatibilityError(f"embedding artifact is missing or corrupt: {name}")
    obs_names = json.loads((directory / OBS_NAMES_FILENAME).read_text())
    row_digests = json.loads((directory / ROW_DIGESTS_FILENAME).read_text())
    X_pca = np.load(directory / PCA_SPACE_FILENAME, allow_pickle=False)
    X_umap = np.load(directory / COORDS_FILENAME, allow_pickle=False)
    clusters = np.load(directory / CLUSTERS_FILENAME, allow_pickle=False)
    n_obs = len(obs_names)
    if (
        len(set(obs_names)) != n_obs
        or set(row_digests) != set(obs_names)
        or X_pca.ndim != 2
        or X_umap.shape != (n_obs, 2)
        or X_pca.shape[0] != n_obs
        or clusters.shape != (n_obs,)
        or int(manifest.get("source", {}).get("n_molecules", -1)) != n_obs
    ):
        raise EmbeddingCompatibilityError("embedding generation arrays or identities are invalid")
    return manifest


def _resolve_current(root: Path, definition: dict[str, object]) -> tuple[Path, dict] | None:
    pointer_path = root / CURRENT_FILENAME
    if not pointer_path.exists():
        legacy = root / "meta.json"
        if legacy.exists():
            raise EmbeddingCompatibilityError(
                "legacy in-place project embedding requires force_recompute=True migration"
            )
        return None
    with pointer_path.open(encoding="utf-8") as handle:
        pointer = json.load(handle)
    if int(pointer.get("schema_version", -1)) != 1:
        raise EmbeddingCompatibilityError("embedding current-pointer schema is incompatible")
    relative = Path(str(pointer.get("generation_path", "")))
    generation = (root / relative).resolve()
    if relative.is_absolute() or not generation.is_relative_to(root.resolve()):
        raise EmbeddingCompatibilityError("embedding current pointer is not portable")
    manifest_path = generation / GENERATION_MANIFEST_FILENAME
    if (
        pointer.get("manifest_sha256") != _file_sha256(manifest_path)
        if manifest_path.is_file()
        else True
    ):
        raise EmbeddingCompatibilityError("embedding current manifest checksum does not match")
    manifest = _validate_generation(
        generation,
        definition=definition,
        expected_generation_id=str(pointer.get("generation_id")),
    )
    return generation, manifest


def _check_dependencies(manifest: dict[str, object]) -> None:
    stored = manifest.get("dependencies", {})
    current = _dependencies()
    incompatible = {
        name: (stored.get(name), current.get(name))
        for name in _MODEL_DEPENDENCIES
        if stored.get(name) != current.get(name)
    }
    if incompatible:
        raise EmbeddingCompatibilityError(
            f"embedding model dependencies changed: {incompatible}; pass force_recompute=True"
        )


def _read_generation(
    directory: Path,
    manifest: dict,
    *,
    load_models: bool,
) -> dict:
    result = {
        "X_pca": np.load(directory / PCA_SPACE_FILENAME, allow_pickle=False),
        "X_umap": np.load(directory / COORDS_FILENAME, allow_pickle=False),
        "clusters": np.load(directory / CLUSTERS_FILENAME, allow_pickle=False),
        "obs_names": json.loads((directory / OBS_NAMES_FILENAME).read_text()),
        "feature_row_digests": json.loads((directory / ROW_DIGESTS_FILENAME).read_text()),
        "pca_model": None,
        "umap_model": None,
        "explained_variance_ratio": manifest.get("explained_variance_ratio"),
        "meta": manifest,
    }
    if load_models:
        _check_dependencies(manifest)
        with (directory / PCA_MODEL_FILENAME).open("rb") as handle:
            result["pca_model"] = pickle.load(handle)
        with (directory / UMAP_MODEL_FILENAME).open("rb") as handle:
            result["umap_model"] = pickle.load(handle)
    return result


def read_embedding(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
    trust_local_models: bool = False,
) -> dict:
    """Read the validated current generation without resolving source experiments.

    This is relocation-safe because the current pointer and all generation
    artifacts are definition-root-relative. Models remain unloaded unless the
    caller explicitly trusts the local project tree.
    """
    definition = _embedding_definition(
        canonical_reference=canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    root = embedding_dir(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    current = _resolve_current(root, definition)
    if current is None:
        raise FileNotFoundError(f"no current project embedding for definition: {root}")
    return _read_generation(*current, load_models=trust_local_models)


def _publish_generation(
    root: Path,
    *,
    definition: dict[str, object],
    source: dict[str, object],
    row_digests: dict[str, str],
    result: dict,
    fit_kind: str,
    prior_generation_id: str | None,
    prior_fit_at: str | None,
) -> tuple[Path, dict]:
    generation_id = uuid4().hex
    staging = root / STAGING_DIRNAME / generation_id
    final = root / GENERATIONS_DIRNAME / generation_id
    staging.mkdir(parents=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    moved_to_final = False
    try:
        _write_generation_artifacts(staging, result, row_digests)
        timestamp = _now()
        manifest = {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "status": "complete",
            "generation_id": generation_id,
            "definition_hash": _definition_hash(definition),
            "definition": definition,
            "source": source,
            "dependencies": _dependencies(),
            "fit_kind": fit_kind,
            "fit_at": prior_fit_at or timestamp,
            "extended_at": timestamp if fit_kind == "extended" else None,
            "refit_at": timestamp if fit_kind == "full" and prior_generation_id else None,
            "prior_generation_id": prior_generation_id,
            "n_reads": len(result["obs_names"]),
            "n_new_reads": (
                len(result["obs_names"]) - int(result.get("prior_n_reads", 0))
                if fit_kind == "extended"
                else 0
            ),
            "explained_variance_ratio": result.get("explained_variance_ratio"),
            "artifacts": _artifact_checksums(staging),
        }
        atomic_write_json(staging / GENERATION_MANIFEST_FILENAME, manifest)
        _validate_generation(staging, definition=definition, expected_generation_id=generation_id)
        os.replace(staging, final)
        moved_to_final = True
        manifest_path = final / GENERATION_MANIFEST_FILENAME
        atomic_write_json(
            root / CURRENT_FILENAME,
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "generation_path": final.relative_to(root).as_posix(),
                "manifest_sha256": _file_sha256(manifest_path),
            },
        )
        return final, manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if moved_to_final:
            shutil.rmtree(final, ignore_errors=True)
        raise


def fit_or_extend_embedding(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
    force_recompute: bool = False,
    trust_local_models: bool = False,
) -> dict:
    """Fit, safely read, extend, or explicitly refit a project embedding.

    Exact cache hits read validated arrays without unpickling models. Pure
    membership growth preserves old coordinates and transforms only new
    molecules, but requires ``trust_local_models=True`` because persisted
    sklearn/UMAP estimators are pickle files. Removal or changed feature values
    for an existing molecule requires ``force_recompute=True``.
    """
    project_dir = Path(project_dir)
    definition = _embedding_definition(
        canonical_reference=canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    root = embedding_dir(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    members = resolve_set_members(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    adata = project_adata(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layers=[layer] if layer is not None else [],
        start=start,
        end=end,
        allow_large=True,
    )
    features, obs_names = _make_features(
        adata, feature_kind=feature_kind, layer=layer, start=start, end=end
    )
    source, row_digests = _source_snapshot(project_dir, members, features, obs_names)

    current = None if force_recompute else _resolve_current(root, definition)
    if current is not None:
        generation, manifest = current
        existing = _read_generation(generation, manifest, load_models=False)
        old_names = existing["obs_names"]
        removed = set(old_names).difference(obs_names)
        if removed:
            raise EmbeddingCompositionError(
                f"{len(removed)} previously embedded molecule(s) are absent; "
                "pass force_recompute=True"
            )
        changed = [
            name
            for name in old_names
            if existing["feature_row_digests"].get(name) != row_digests.get(name)
        ]
        if changed:
            raise EmbeddingCompositionError(
                f"{len(changed)} existing molecule(s) have changed feature values; "
                "pass force_recompute=True"
            )
        previous_members = {
            (item["experiment"], item["experiment_uid"]): item
            for item in manifest["source"]["members"]
        }
        current_members = {
            (item["experiment"], item["experiment_uid"]): item for item in source["members"]
        }
        changed_sources = [
            owner
            for owner, record in previous_members.items()
            if _source_member_changed(record, current_members.get(owner))
        ]
        if changed_sources:
            raise EmbeddingCompositionError(
                f"{len(changed_sources)} existing experiment source(s) changed; "
                "pass force_recompute=True"
            )
        if set(old_names) == set(obs_names) and old_names != obs_names:
            raise EmbeddingCompositionError(
                "existing molecule order changed; pass force_recompute=True"
            )
        if (
            old_names == obs_names
            and manifest["source"]["feature_input_digest"] == source["feature_input_digest"]
        ):
            return existing
        if not trust_local_models:
            raise EmbeddingTrustError(
                "embedding growth requires trusted local pickle models; rerun with "
                "trust_local_models=True only for a trusted project tree"
            )
        existing = _read_generation(generation, manifest, load_models=True)
        obs_index = {name: index for index, name in enumerate(obs_names)}
        new_names = [name for name in obs_names if name not in set(old_names)]
        new_features = features[[obs_index[name] for name in new_names]]
        new_X_pca = existing["pca_model"].transform(new_features)
        new_X_umap = existing["umap_model"].transform(new_X_pca)
        new_clusters = _assign_nearest_cluster(new_X_pca, existing["X_pca"], existing["clusters"])
        result = {
            "X_pca": np.vstack((existing["X_pca"], new_X_pca)),
            "X_umap": np.vstack((existing["X_umap"], new_X_umap)),
            "clusters": np.concatenate((existing["clusters"], new_clusters)),
            "obs_names": old_names + new_names,
            "pca_model": existing["pca_model"],
            "umap_model": existing["umap_model"],
            "explained_variance_ratio": manifest.get("explained_variance_ratio"),
            "prior_n_reads": len(old_names),
        }
        ordered_digests = {name: row_digests[name] for name in result["obs_names"]}
        final, published = _publish_generation(
            root,
            definition=definition,
            source={
                **source,
                "ordered_molecule_membership_digest": _membership_digest(result["obs_names"]),
                "feature_input_digest": _stable_hash(
                    [[name, ordered_digests[name]] for name in result["obs_names"]]
                ),
            },
            row_digests=ordered_digests,
            result=result,
            fit_kind="extended",
            prior_generation_id=manifest["generation_id"],
            prior_fit_at=manifest["fit_at"],
        )
        return _read_generation(final, published, load_models=True)

    result = _fit_from_scratch(
        features,
        obs_names,
        leiden_resolution=leiden_resolution,
        min_reads=min_reads,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    prior = None
    if force_recompute and (root / CURRENT_FILENAME).exists():
        try:
            prior = _resolve_current(root, definition)
        except EmbeddingCompatibilityError:
            # A forced refit may repair a corrupt/incompatible pointer. Immutable
            # generation directories are retained even when they cannot be trusted
            # enough to name as the new generation's validated predecessor.
            prior = None
    prior_manifest = prior[1] if prior is not None else None
    final, published = _publish_generation(
        root,
        definition=definition,
        source=source,
        row_digests=row_digests,
        result=result,
        fit_kind="full",
        prior_generation_id=(
            str(prior_manifest["generation_id"]) if prior_manifest is not None else None
        ),
        prior_fit_at=None,
    )
    return _read_generation(final, published, load_models=True)
