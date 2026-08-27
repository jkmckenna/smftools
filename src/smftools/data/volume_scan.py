"""`data scan`: index runs on attached volumes into the replica catalog (`PSR-11`).

Walks a stamped volume for published input manifests
(`smftools.informatics.input_manifest`), each of which already carries the
relocation-invariant dataset digest `PSR-10`'s catalog is keyed by, and
registers one replica per run root found: `(volume_id, run root's path
relative to the volume's own mount point)`.

The walk is pruned to stay cheap on a real archive drive: it never descends
into a directory named `generations` (content-addressed, can hold millions of
files across every pipeline stage) and, once inside a `raw_outputs` directory,
only descends into its `input_manifest` child -- the ragged raw store and BAM
artifacts that also live under `raw_outputs` are exactly what this scan does
not need to touch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from ..constants import RAW_DIR
from ..informatics.input_manifest import (
    INPUT_MANIFEST_DIRNAME,
    RESOLVED_INPUT_MANIFEST_JSON,
    InputManifestError,
    read_resolved_input_manifest,
)
from ..logging_utils import get_logger
from .replica_catalog import Replica, add_replica
from .volume_stamp import read_volume_stamp

logger = get_logger(__name__)

#: Never descend into these -- content-addressed, can hold millions of files.
_PRUNED_DIR_NAMES = frozenset({"generations"})


@dataclass(frozen=True)
class ScannedRun:
    """One run root found under a scanned volume."""

    run_root: Path
    #: Relative to the scanned mount's own root -- what gets stored in the
    #: catalog, never the absolute path (`PSR-08`/`PSR-09`).
    relative_path: str
    digest: str
    #: Set instead of `digest` when the manifest exists but could not be read
    #: -- the run root is still reported, since a broken manifest at a known
    #: location is worth surfacing, but nothing is added to the catalog for it.
    warning: Optional[str] = None


def _iter_resolved_manifests(mount: Path) -> Iterator[Path]:
    """Every `resolved_input_manifest.json` under `mount`, walk pruned for cost."""
    for current, dirnames, filenames in os.walk(mount):
        current_path = Path(current)
        dirnames[:] = [name for name in sorted(dirnames) if not name.startswith(".")]
        if current_path.name == RAW_DIR:
            dirnames[:] = [name for name in dirnames if name == INPUT_MANIFEST_DIRNAME]
        else:
            dirnames[:] = [name for name in dirnames if name not in _PRUNED_DIR_NAMES]
        if (
            current_path.name == INPUT_MANIFEST_DIRNAME
            and RESOLVED_INPUT_MANIFEST_JSON in filenames
        ):
            yield current_path / RESOLVED_INPUT_MANIFEST_JSON
            dirnames[:] = []  # nothing more to find beneath a manifest directory


def scan_volume(mount: str | Path) -> list[ScannedRun]:
    """Every run root found under `mount`, with the dataset digest each names.

    Does not touch the catalog -- see `scan_and_catalog` for that. Splitting
    discovery from persistence keeps this half trivially testable against a
    plain directory tree, with no catalog file or volume stamp involved.
    """
    mount_path = Path(mount).resolve()
    found: list[ScannedRun] = []
    for manifest_path in _iter_resolved_manifests(mount_path):
        # <run_root>/<RAW_DIR>/<INPUT_MANIFEST_DIRNAME>/<RESOLVED_INPUT_MANIFEST_JSON>
        run_root = manifest_path.parents[2]
        relative = run_root.relative_to(mount_path).as_posix()
        try:
            manifest = read_resolved_input_manifest(manifest_path)
        except InputManifestError as exc:
            logger.warning("skipping unreadable input manifest at %s: %s", manifest_path, exc)
            found.append(
                ScannedRun(run_root=run_root, relative_path=relative, digest="", warning=str(exc))
            )
            continue
        found.append(ScannedRun(run_root=run_root, relative_path=relative, digest=manifest.digest))
    return sorted(found, key=lambda run: run.relative_path)


def scan_and_catalog(
    mount: str | Path, catalog: Mapping[str, Sequence[Replica]]
) -> tuple[dict[str, list[Replica]], list[ScannedRun]]:
    """Scan `mount` and return an updated catalog plus every run found.

    Args:
        mount: A stamped volume's mount point.
        catalog: The catalog to update. Not mutated; a new dict is returned.

    Returns:
        The updated catalog, and every `ScannedRun` found (including ones
        with an unreadable manifest, which contribute nothing to the catalog).

    Raises:
        ValueError: `mount` has never been stamped -- run `data init-volume`
            first, since a replica cannot be recorded against a volume with
            no identity.
    """
    mount_path = Path(mount).resolve()
    stamp = read_volume_stamp(mount_path)
    if stamp is None:
        raise ValueError(f"{mount_path} has not been stamped; run 'data init-volume' first.")

    runs = scan_volume(mount_path)
    updated: dict[str, list[Replica]] = {key: list(value) for key, value in catalog.items()}
    for run in runs:
        if not run.digest:
            continue
        updated = add_replica(
            updated, run.digest, volume_id=stamp.volume_id, path=run.relative_path
        )
    return updated, runs
