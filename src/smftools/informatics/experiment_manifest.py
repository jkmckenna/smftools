"""Consolidated experiment-level provenance manifest.

One JSON file per experiment (at the run root, alongside ``molecules.parquet``),
appended to -- never replaced -- as each pipeline stage runs. Closes two gaps
``spine.uns`` never covered: config parameters and the raw instrument input path were
previously nowhere in the store at all (only a ``config_hash`` for staleness
detection, not the values); "which stages have run" was previously inferred only from
directory presence, with no single readable index.

Phase 2 of ``dev/experiment_storage_schema.md`` (not tracked in git). Currently wired
into the raw stage only (``cli/load_adata.py``) -- preprocess/spatial/hmm can adopt
``record_stage_completion`` the same way once their executors are touched for the
schema's later phases; nothing here is raw-specific.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "experiment_manifest.json"


def experiment_manifest_path(run_root: str | Path) -> Path:
    return Path(run_root) / MANIFEST_FILENAME


def config_hash(config: dict[str, Any]) -> str:
    """Stable, short hash of a resolved config dict (e.g. ``ExperimentConfig.to_dict()``)."""
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _save(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)


def update_experiment_manifest(run_root: str | Path, **fields: Any) -> Path:
    """Merge top-level fields into the manifest (``None`` values are skipped, so
    callers can pass everything they have without clobbering an already-recorded
    field with an absent one). Never touches the ``"stages"`` sub-dict -- see
    :func:`record_stage_completion` for that.
    """
    path = experiment_manifest_path(run_root)
    manifest = _load(path)
    manifest.update({key: value for key, value in fields.items() if value is not None})
    _save(path, manifest)
    return path


def record_stage_completion(
    run_root: str | Path,
    stage: str,
    *,
    config_hash: str | None = None,
    n_molecules: int | None = None,
    **extra: Any,
) -> Path:
    """Record (or overwrite, on a re-run) one stage's entry in the manifest's
    ``"stages"`` index -- a readable completion/provenance log, in addition to (not
    instead of) the existing structural signal of whether ``<stage>_outputs/`` exists.
    """
    path = experiment_manifest_path(run_root)
    manifest = _load(path)
    stages = manifest.setdefault("stages", {})
    entry: dict[str, Any] = {"completed_at": _now()}
    if config_hash is not None:
        entry["config_hash"] = config_hash
    if n_molecules is not None:
        entry["n_molecules"] = int(n_molecules)
    entry.update(extra)
    stages[stage] = entry
    _save(path, manifest)
    return path


def read_experiment_manifest(run_root: str | Path) -> dict[str, Any]:
    return _load(experiment_manifest_path(run_root))
