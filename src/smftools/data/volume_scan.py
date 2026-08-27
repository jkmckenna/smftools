"""`data scan`: index runs on attached volumes (`PSR-11`, extended by `PSR-19`).

Two independent things get registered per volume scanned:

- **Raw dataset replicas** (`PSR-11`, original): walks for published input
  manifests (`smftools.informatics.input_manifest`), each of which already
  carries the relocation-invariant dataset digest `PSR-10`'s replica catalog
  is keyed by.
- **Analysis locations** (`PSR-19`): walks for `experiment_manifest.json`,
  present at the root of every modern run regardless of which stages have
  completed, and registers `(volume_id, run root's relative path)` into
  `smftools.data.analysis_catalog`, keyed by the run's durable
  `experiment_uid` (`PSR-17`). Unlike a raw replica, an analysis location is
  never treated as interchangeable with another -- this only records where
  copies are, never which one leads.

Both walks are pruned to stay cheap on a real archive drive: neither
descends into a directory named `generations` (content-addressed, can hold
millions of files across every pipeline stage), and the raw-manifest walk,
once inside a `raw_outputs` directory, only descends into its
`input_manifest` child -- the ragged raw store and BAM artifacts that also
live under `raw_outputs` are exactly what that walk does not need to touch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from ..constants import RAW_DIR
from ..informatics.experiment_manifest import MANIFEST_FILENAME, read_experiment_manifest
from ..informatics.input_manifest import (
    INPUT_MANIFEST_DIRNAME,
    RESOLVED_INPUT_MANIFEST_JSON,
    InputManifestError,
    read_resolved_input_manifest,
)
from ..informatics.molecule_identity import EXPERIMENT_UID_COLUMN, validate_experiment_uid
from ..logging_utils import get_logger
from .analysis_catalog import AnalysisLocation, add_location
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


@dataclass(frozen=True)
class ScannedAnalysisLocation:
    """One run root found under a scanned volume, for analysis-locality tracking."""

    run_root: Path
    #: Relative to the scanned mount's own root.
    relative_path: str
    experiment_uid: str
    #: Set instead of `experiment_uid` when the manifest exists but names no
    #: (or an invalid) identity -- e.g. a legacy run predating `experiment_uid`.
    #: The run root is still reported; nothing is added to the catalog for it.
    warning: Optional[str] = None


def _iter_run_roots(mount: Path) -> Iterator[Path]:
    """Every directory with its own `experiment_manifest.json` under `mount`.

    Present at the root of every modern run regardless of which stages have
    completed, so this is the authoritative run-root marker -- unlike
    `_iter_resolved_manifests`, nothing more of interest lives beneath one
    for this purpose (no experiment nests inside another), so the walk stops
    descending the moment it finds one.
    """
    for current, dirnames, filenames in os.walk(mount):
        current_path = Path(current)
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if not name.startswith(".") and name not in _PRUNED_DIR_NAMES
        ]
        if MANIFEST_FILENAME in filenames:
            yield current_path
            dirnames[:] = []  # no experiment nests inside another


def scan_volume_for_analysis_locations(mount: str | Path) -> list[ScannedAnalysisLocation]:
    """Every run root found under `mount`, with the `experiment_uid` each names.

    Does not touch the catalog -- see `scan_and_catalog_analysis_locations`.
    """
    mount_path = Path(mount).resolve()
    found: list[ScannedAnalysisLocation] = []
    for run_root in _iter_run_roots(mount_path):
        relative = run_root.relative_to(mount_path).as_posix()
        manifest = read_experiment_manifest(run_root)
        raw_uid = manifest.get(EXPERIMENT_UID_COLUMN)
        if raw_uid is None:
            found.append(
                ScannedAnalysisLocation(
                    run_root=run_root,
                    relative_path=relative,
                    experiment_uid="",
                    warning="no experiment_uid recorded (predates the identity system, or raw "
                    "has not completed)",
                )
            )
            continue
        try:
            experiment_uid = validate_experiment_uid(raw_uid)
        except ValueError as exc:
            logger.warning("skipping invalid experiment_uid at %s: %s", run_root, exc)
            found.append(
                ScannedAnalysisLocation(
                    run_root=run_root, relative_path=relative, experiment_uid="", warning=str(exc)
                )
            )
            continue
        found.append(
            ScannedAnalysisLocation(
                run_root=run_root, relative_path=relative, experiment_uid=experiment_uid
            )
        )
    return sorted(found, key=lambda run: run.relative_path)


def scan_and_catalog_analysis_locations(
    mount: str | Path, catalog: Mapping[str, Sequence[AnalysisLocation]]
) -> tuple[dict[str, list[AnalysisLocation]], list[ScannedAnalysisLocation]]:
    """Scan `mount` and return an updated analysis-location catalog plus every run found.

    Args:
        mount: A stamped volume's mount point.
        catalog: The catalog to update. Not mutated; a new dict is returned.

    Raises:
        ValueError: `mount` has never been stamped -- run `data init-volume`
            first.
    """
    mount_path = Path(mount).resolve()
    stamp = read_volume_stamp(mount_path)
    if stamp is None:
        raise ValueError(f"{mount_path} has not been stamped; run 'data init-volume' first.")

    runs = scan_volume_for_analysis_locations(mount_path)
    updated: dict[str, list[AnalysisLocation]] = {
        key: list(value) for key, value in catalog.items()
    }
    for run in runs:
        if not run.experiment_uid:
            continue
        updated = add_location(
            updated, run.experiment_uid, volume_id=stamp.volume_id, path=run.relative_path
        )
    return updated, runs
