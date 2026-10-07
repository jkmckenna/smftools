"""Cache keys for analyses over a plan dataset (`RPG-04`, `F72`).

An analysis that saves its results (``context-bias`` counts, ``periodicity``
periodograms) reuses them only when its key matches. The plan hash alone is
not enough: a plan names its label table and coordinate maps by path, so
editing those files in place leaves the hash unchanged (`F72`). The key adds
the content hash of every file the dataset references.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_files(plan, dataset_name: str, base_dir: str | Path | None) -> dict[str, str]:
    """``{declared path: sha256}`` for the dataset's label table and coordinate maps.

    Paths resolve against ``base_dir`` (the project or experiment directory),
    as selection resolves them. A missing file hashes as ``"missing"``:
    selection reports it; the key only needs to differ.
    """
    dataset = plan.datasets[dataset_name]
    declared = []
    if dataset.labels is not None and dataset.labels.table:
        declared.append(dataset.labels.table)
    if dataset.coordinate_frame is not None:
        declared.extend(dataset.coordinate_frame.maps.values())
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    hashes = {}
    for relative in sorted(set(map(str, declared))):
        path = base / relative
        hashes[relative] = _sha256(path) if path.is_file() else "missing"
    return hashes


def cache_key(
    plan,
    dataset_name: str,
    *,
    base_dir: str | Path | None,
    parameters: Mapping[str, Any],
) -> dict:
    """Everything a saved result depends on: plan, dataset, referenced files, parameters."""
    return json.loads(
        json.dumps(
            {
                "plan_hash": plan.plan_hash,
                "dataset": dataset_name,
                "files": referenced_files(plan, dataset_name, base_dir),
                "parameters": dict(parameters),
            },
            sort_keys=True,
            default=str,
        )
    )
