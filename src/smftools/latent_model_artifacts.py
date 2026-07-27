"""Immutable model artifacts for partitioned latent coordinate owners."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .informatics.molecule_identity import EXPERIMENT_UID_COLUMN, MOLECULE_UID_COLUMN
from .readwrite import atomic_write_json

LATENT_MODEL_SCHEMA_VERSION = 1
LATENT_MODEL_IMPLEMENTATION_VERSION = 1
LATENT_MODEL_MANIFEST = "model_manifest.json"
LATENT_MODEL_STATE = "model_state.pkl"
LATENT_MODEL_ARRAYS = "model_arrays.npz"
_MODEL_DEPENDENCIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "pynndescent",
    "numba",
)


class LatentModelArtifactError(RuntimeError):
    """Raised when a latent model artifact is invalid or incompatible."""


@dataclass(frozen=True)
class LatentModelArtifact:
    """Validated immutable model artifact metadata."""

    model_id: str
    model_checksum: str
    path: Path
    manifest: dict[str, object]


def _stable_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one model artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_fit_membership(
    obs: pd.DataFrame,
    read_ids: list[str],
    *,
    limit: int,
    random_state: int,
    coordinate_owner: str,
) -> tuple[list[str], list[str]]:
    """Select fit molecules by stable identity hash, independent of input order."""
    if limit < 1:
        raise ValueError("latent fit limit must be positive")
    selected = obs.loc[list(map(str, read_ids))]
    missing = [
        column for column in (EXPERIMENT_UID_COLUMN, MOLECULE_UID_COLUMN) if column not in selected
    ]
    if missing:
        raise ValueError(f"latent fit membership requires identity columns: {missing}")
    if selected.index.astype(str).duplicated().any():
        raise ValueError("latent fit membership contains duplicate read IDs")
    records = []
    for read_id, row in selected.iterrows():
        experiment_uid = str(row[EXPERIMENT_UID_COLUMN])
        molecule_uid = str(row[MOLECULE_UID_COLUMN])
        score = hashlib.sha256(
            "\0".join(
                (
                    experiment_uid,
                    molecule_uid,
                    str(int(random_state)),
                    str(coordinate_owner),
                )
            ).encode()
        ).hexdigest()
        records.append((score, molecule_uid, str(read_id)))
    chosen = sorted(records)[: min(len(records), int(limit))]
    return [record[2] for record in chosen], [record[1] for record in chosen]


def fit_membership_digest(molecule_uids: list[str]) -> str:
    """Return the ordered semantic digest for deterministic fit membership."""
    return _stable_hash(list(map(str, molecule_uids)))


def mask_identity(mask: np.ndarray) -> str:
    """Return a stable identity for one selected feature mask."""
    values = np.asarray(mask, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(np.packbits(values).tobytes())
    return digest.hexdigest()


def latent_model_key(
    *,
    source_identity: dict[str, object],
    analysis_core_id: str,
    representation_specs: list[dict[str, object]],
    algorithm_parameters: dict[str, object],
    fit_molecule_uids: list[str],
    forced_fit_revision: str | None,
) -> dict[str, object]:
    """Build the immutable semantic key for one coordinate-owner model bundle."""
    return {
        "schema_version": LATENT_MODEL_SCHEMA_VERSION,
        "implementation_version": LATENT_MODEL_IMPLEMENTATION_VERSION,
        "source": source_identity,
        "analysis_core_id": str(analysis_core_id),
        "representations": representation_specs,
        "algorithm_parameters": algorithm_parameters,
        "fit_membership_digest": fit_membership_digest(fit_molecule_uids),
        "forced_fit_revision": forced_fit_revision,
    }


def latent_model_id(key: dict[str, object]) -> str:
    """Return the stable semantic model ID for a model key."""
    return _stable_hash(key)[:32]


def dependency_versions() -> dict[str, str]:
    """Return versions needed to safely unpickle fitted transform state."""
    dependencies = {"python": platform.python_version()}
    for package in ("smftools", *_MODEL_DEPENDENCIES):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unknown"
    return dependencies


def _artifact_dir(models_root: Path, model_id: str) -> Path:
    return models_root / model_id[-2:] / model_id


def write_latent_model_artifact(
    models_root: str | Path,
    *,
    key: dict[str, object],
    state: dict[str, object],
    fit_molecule_uids: list[str],
    cp_provenance: list[dict[str, object]],
) -> LatentModelArtifact:
    """Atomically write one immutable checksummed trusted-local model bundle."""
    models_root = Path(models_root)
    model_id = latent_model_id(key)
    final = _artifact_dir(models_root, model_id)
    staging_root = models_root / ".staging" / uuid4().hex
    staging = staging_root / model_id
    staging.mkdir(parents=True)
    try:
        state_payload = dict(state)
        cp_factors = {
            str(key): np.asarray(value)
            for key, value in dict(state_payload.pop("cp_factors", {})).items()
        }
        state_path = staging / LATENT_MODEL_STATE
        with state_path.open("wb") as handle:
            pickle.dump(state_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        arrays_path = staging / LATENT_MODEL_ARRAYS
        np.savez_compressed(arrays_path, **cp_factors)
        state_checksum = file_sha256(state_path)
        arrays_checksum = file_sha256(arrays_path)
        checksum = _stable_hash({"state_sha256": state_checksum, "arrays_sha256": arrays_checksum})
        manifest = {
            "schema_version": LATENT_MODEL_SCHEMA_VERSION,
            "model_id": model_id,
            "model_checksum": checksum,
            "model_key": key,
            "fit_molecule_uids": list(map(str, fit_molecule_uids)),
            "fit_membership_digest": fit_membership_digest(fit_molecule_uids),
            "cp_provenance": cp_provenance,
            "state_file": LATENT_MODEL_STATE,
            "state_format": "python-pickle",
            "state_sha256": state_checksum,
            "arrays_file": LATENT_MODEL_ARRAYS,
            "arrays_format": "numpy-npz",
            "arrays_sha256": arrays_checksum,
            "trust_boundary": "trusted-local-smftools-generation",
            "dependencies": dependency_versions(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / LATENT_MODEL_MANIFEST, manifest)
        validate_latent_model_artifact(staging, expected_model_id=model_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            existing = validate_latent_model_artifact(final, expected_model_id=model_id)
            if existing.model_checksum != checksum:
                raise LatentModelArtifactError(
                    f"model ID {model_id} is already bound to different fitted state"
                )
            shutil.rmtree(staging_root)
            return existing
        os.replace(staging, final)
        shutil.rmtree(staging_root, ignore_errors=True)
        return LatentModelArtifact(model_id, checksum, final, manifest)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def validate_latent_model_artifact(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
) -> LatentModelArtifact:
    """Validate model identity, schema, state checksum, and portable manifest."""
    path = Path(path)
    manifest_path = path / LATENT_MODEL_MANIFEST
    state_path = path / LATENT_MODEL_STATE
    arrays_path = path / LATENT_MODEL_ARRAYS
    if not manifest_path.is_file() or not state_path.is_file() or not arrays_path.is_file():
        raise LatentModelArtifactError(f"latent model artifact is incomplete: {path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != LATENT_MODEL_SCHEMA_VERSION:
        raise LatentModelArtifactError("latent model artifact schema is incompatible")
    if (
        manifest.get("state_format") != "python-pickle"
        or manifest.get("arrays_format") != "numpy-npz"
        or manifest.get("trust_boundary") != "trusted-local-smftools-generation"
    ):
        raise LatentModelArtifactError("latent model artifact format or trust boundary is invalid")
    model_id = str(manifest.get("model_id", ""))
    key = manifest.get("model_key")
    if not isinstance(key, dict) or latent_model_id(key) != model_id:
        raise LatentModelArtifactError("latent model semantic ID does not match its key")
    if expected_model_id is not None and model_id != str(expected_model_id):
        raise LatentModelArtifactError("latent model ID does not match task provenance")
    if path.name != model_id:
        raise LatentModelArtifactError("latent model path does not match its semantic ID")
    state_checksum = file_sha256(state_path)
    arrays_checksum = file_sha256(arrays_path)
    if state_checksum != str(manifest.get("state_sha256", "")):
        raise LatentModelArtifactError("latent model estimator-state checksum mismatch")
    if arrays_checksum != str(manifest.get("arrays_sha256", "")):
        raise LatentModelArtifactError("latent model portable-array checksum mismatch")
    checksum = _stable_hash({"state_sha256": state_checksum, "arrays_sha256": arrays_checksum})
    if checksum != str(manifest.get("model_checksum", "")):
        raise LatentModelArtifactError("latent model state checksum mismatch")
    if manifest.get("fit_membership_digest") != fit_membership_digest(
        list(map(str, manifest.get("fit_molecule_uids", ())))
    ):
        raise LatentModelArtifactError("latent model fit-membership digest mismatch")
    return LatentModelArtifact(model_id, checksum, path, manifest)


def load_latent_model_state(
    path: str | Path,
    *,
    expected_model_id: str,
    expected_model_checksum: str,
    trusted_local: bool,
) -> tuple[dict[str, object], LatentModelArtifact]:
    """Load validated pickle state after an explicit trusted-local decision."""
    artifact = validate_latent_model_artifact(path, expected_model_id=expected_model_id)
    if artifact.model_checksum != str(expected_model_checksum):
        raise LatentModelArtifactError("latent task/model checksum mismatch")
    if not trusted_local:
        raise LatentModelArtifactError(
            "latent model state is trusted-local pickle data; explicit trust is required"
        )
    stored = artifact.manifest.get("dependencies", {})
    if not isinstance(stored, dict):
        raise LatentModelArtifactError("latent model dependency provenance is invalid")
    current = dependency_versions()
    incompatible = {
        name: (stored.get(name), current.get(name))
        for name in _MODEL_DEPENDENCIES
        if stored.get(name) != current.get(name)
    }
    if incompatible:
        raise LatentModelArtifactError(
            f"latent model dependency versions are incompatible: {incompatible}"
        )
    with (artifact.path / LATENT_MODEL_STATE).open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise LatentModelArtifactError("latent model state payload is invalid")
    with np.load(artifact.path / LATENT_MODEL_ARRAYS, allow_pickle=False) as arrays:
        state["cp_factors"] = {key: np.asarray(arrays[key]) for key in arrays.files}
    return state, artifact
