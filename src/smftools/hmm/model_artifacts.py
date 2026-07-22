"""Immutable, portable artifacts for fitted HMM checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from smftools.optional_imports import require
from smftools.readwrite import atomic_write_json

MODEL_ARTIFACT_SCHEMA_VERSION = 1
MODEL_ID_PREFIX = "hmm"

HMM_FIT_CONFIG_FIELDS = (
    "smf_modality",
    "output_binary_layer_name",
    "hmm_n_states",
    "hmm_init_emission_probs",
    "hmm_init_transition_probs",
    "hmm_init_start_probs",
    "hmm_eps",
    "hmm_dtype",
    "hmm_distance_aware",
    "hmm_distance_bins",
    "hmm_init_transitions_by_bin",
    "hmm_max_iter",
    "hmm_tol",
    "hmm_fit_scope",
    "hmm_fit_strategy",
    "hmm_groupby",
    "hmm_shared_scope",
    "hmm_adapt_iters",
    "hmm_adapt_emissions",
    "hmm_adapt_startprobs",
    "hmm_emission_adapt_iters",
    "hmm_emission_adapt_tol",
    "hmm_max_fit_reads",
    "hmm_fit_selection_seed",
    "hmm_methbases",
)

HMM_FIT_CONFIG_DEFAULTS = {
    "hmm_n_states": 2,
    "hmm_init_emission_probs": [[0.8, 0.2], [0.2, 0.8]],
    "hmm_init_transition_probs": [[0.9, 0.1], [0.1, 0.9]],
    "hmm_init_start_probs": [0.5, 0.5],
    "hmm_eps": 1e-8,
    "hmm_dtype": "float64",
    "hmm_distance_aware": False,
    "hmm_distance_bins": [1, 5, 10, 25, 50, 100],
    "hmm_max_iter": 50,
    "hmm_tol": 1e-5,
    "hmm_fit_scope": "per_sample",
    "hmm_fit_strategy": "per_group",
    "hmm_groupby": ["sample", "reference", "methbase"],
    "hmm_shared_scope": ["reference", "methbase"],
    "hmm_adapt_iters": 10,
    "hmm_adapt_emissions": True,
    "hmm_adapt_startprobs": True,
    "hmm_emission_adapt_iters": 5,
    "hmm_emission_adapt_tol": 1e-4,
    "hmm_max_fit_reads": 1000,
    "hmm_fit_selection_seed": 0,
}


class HMMArtifactError(RuntimeError):
    """Base error for immutable HMM artifact operations."""


class HMMArtifactConflictError(HMMArtifactError):
    """Raised when different content targets an existing immutable model ID."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    """Hash nested checkpoint content independently of torch serialization bytes."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "cpu"):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            digest.update(b"array:")
            digest.update(str(contiguous.dtype).encode("utf-8"))
            digest.update(_canonical_json(contiguous.shape).encode("utf-8"))
            digest.update(contiguous.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence:")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(b":")
            digest.update(_canonical_json(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _config_values(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        return dict(cfg.to_dict())
    return dict(vars(cfg))


def hmm_fit_config(cfg: Any) -> dict[str, Any]:
    """Return only configuration values capable of changing a fitted model."""
    values = _config_values(cfg)
    return {
        name: values[name] if name in values else HMM_FIT_CONFIG_DEFAULTS[name]
        for name in HMM_FIT_CONFIG_FIELDS
        if name in values or name in HMM_FIT_CONFIG_DEFAULTS
    }


def hmm_fit_config_hash(cfg: Any) -> str:
    """Return a stable digest of the model-fitting configuration."""
    return _sha256_bytes(_canonical_json(hmm_fit_config(cfg)).encode("utf-8"))[:16]


def training_selection_metadata(
    read_ids: Any,
    *,
    n_reads: int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Describe a training selection without persisting molecule identifiers."""
    identifiers = sorted(str(value) for value in (read_ids or []))
    result = {
        "selection_sha256": _sha256_bytes(_canonical_json(identifiers).encode("utf-8")),
        "n_reads": int(len(identifiers) if n_reads is None else n_reads),
    }
    result.update(metadata)
    return result


@dataclass(frozen=True)
class HMMModelKey:
    """Canonical identity of one fitted HMM model."""

    fit_kind: str
    reference: str
    barcode: str
    label: str
    architecture: str
    fit_config_hash: str
    core_start: int | None = None
    core_end: int | None = None
    revision: str | None = None
    schema_version: int = MODEL_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical model key."""
        return asdict(self)

    @property
    def model_id(self) -> str:
        """Return a collision-resistant, path-safe identifier."""
        digest = _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))
        return f"{MODEL_ID_PREFIX}-{digest[:32]}"


def checkpoint_path(models_dir: str | Path, key: HMMModelKey) -> Path:
    """Return a short, OS-portable checkpoint path for a model key."""
    model_id = key.model_id
    return Path(models_dir) / model_id[-2:] / f"{model_id}.pt"


def metadata_path(checkpoint: str | Path) -> Path:
    """Return the commit metadata path paired with a checkpoint."""
    return Path(checkpoint).with_suffix(".json")


def _artifact_record(
    key: HMMModelKey,
    checkpoint: Path,
    checksum: str,
    content_checksum: str,
    training_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "model_id": key.model_id,
        "model_key": key.to_dict(),
        "fit_config_hash": key.fit_config_hash,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checksum,
        "checkpoint_content_sha256": content_checksum,
        "model_checksum": content_checksum,
        "training_selection": dict(training_selection or {}),
    }


def load_artifact_record(checkpoint: str | Path) -> dict[str, Any]:
    """Load and validate the committed metadata for an HMM checkpoint."""
    checkpoint = Path(checkpoint)
    sidecar = metadata_path(checkpoint)
    if not checkpoint.is_file() or not sidecar.is_file():
        raise HMMArtifactError(f"Incomplete HMM artifact: {checkpoint}")
    with sidecar.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise HMMArtifactError(f"Unsupported HMM artifact schema for {checkpoint}")
    try:
        recorded_key = HMMModelKey(**record["model_key"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HMMArtifactError(f"Invalid HMM model key metadata for {checkpoint}") from exc
    expected = str(record.get("checkpoint_sha256", ""))
    actual = file_sha256(checkpoint)
    if not expected or actual != expected:
        raise HMMArtifactError(
            f"HMM checkpoint checksum mismatch for {checkpoint}: expected {expected}, got {actual}"
        )
    if (
        checkpoint.stem != str(record.get("model_id", ""))
        or recorded_key.model_id != checkpoint.stem
    ):
        raise HMMArtifactError(f"HMM checkpoint model ID does not match its path: {checkpoint}")
    record["checkpoint_path"] = checkpoint
    record["metadata_path"] = sidecar
    return record


@contextmanager
def model_fit_lock(
    checkpoint: str | Path,
    *,
    timeout_seconds: float = 86400.0,
    poll_seconds: float = 0.1,
) -> Iterator[None]:
    """Serialize fit/publication for one model ID using an exclusive lock file."""
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_suffix(".lock")
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_unix": time.time(),
                    },
                    handle,
                )
                handle.write("\n")
            break
        except FileExistsError:
            try:
                with lock_path.open("r", encoding="utf-8") as handle:
                    owner = json.load(handle)
                owner_pid = int(owner.get("pid", -1))
                if owner.get("hostname") == socket.gethostname() and owner_pid > 0:
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        lock_path.unlink(missing_ok=True)
                        continue
                    except PermissionError:
                        pass
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise HMMArtifactError(f"Timed out waiting for HMM model lock: {lock_path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Serialize a torch payload and atomically replace the destination."""
    torch = require("torch", extra="torch", purpose="HMM model artifacts")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def publish_checkpoint(
    payload: dict[str, Any],
    checkpoint: str | Path,
    key: HMMModelKey,
    *,
    training_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish one immutable checkpoint while holding its model lock."""
    checkpoint = Path(checkpoint)
    if checkpoint.stem != key.model_id:
        raise HMMArtifactError(
            f"Checkpoint path {checkpoint} does not match immutable model ID {key.model_id}"
        )
    if checkpoint.exists():
        existing = load_artifact_record(checkpoint)
        if existing.get("model_key") != key.to_dict():
            raise HMMArtifactConflictError(
                f"Model ID {key.model_id} is already bound to a different model key"
            )
        candidate_checksum = _content_sha256(payload)
        if existing.get("checkpoint_content_sha256") != candidate_checksum:
            raise HMMArtifactConflictError(
                f"Different checkpoint content cannot be published as immutable model "
                f"{key.model_id}"
            )
        return existing

    artifact_payload = dict(payload)
    artifact_payload.update(
        {
            "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_id": key.model_id,
            "model_key": key.to_dict(),
            "fit_config_hash": key.fit_config_hash,
            "training_selection": dict(training_selection or {}),
        }
    )
    atomic_torch_save(artifact_payload, checkpoint)
    checksum = file_sha256(checkpoint)
    content_checksum = _content_sha256(payload)
    record = _artifact_record(key, checkpoint, checksum, content_checksum, training_selection)
    try:
        atomic_write_json(metadata_path(checkpoint), record)
    except BaseException:
        checkpoint.unlink(missing_ok=True)
        raise
    record["checkpoint_path"] = checkpoint
    record["metadata_path"] = metadata_path(checkpoint)
    return record


def portable_artifact_record(record: Mapping[str, Any], models_dir: str | Path) -> dict[str, Any]:
    """Remove local paths and express artifact references relative to the model store."""
    models_dir = Path(models_dir)
    result = {key: value for key, value in record.items() if not key.endswith("_path")}
    checkpoint = Path(record["checkpoint_path"])
    sidecar = Path(record["metadata_path"])
    result["checkpoint"] = checkpoint.relative_to(models_dir).as_posix()
    result["metadata"] = sidecar.relative_to(models_dir).as_posix()
    return result
