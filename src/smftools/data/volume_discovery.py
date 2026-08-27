"""Discover attached volumes stamped by `data init-volume` (`PSR-09`).

Scans the platform's mount roots plus any configured extra search paths for
stamp files, producing a live `volume_id -> mount path` map. This is what lets
`PSR-12` later replace Phase 1's structural approximation of offline vs.
missing (`smftools.config.input_availability`) with an exact answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..config.input_availability import MOUNT_ROOTS
from ..config.roots import extra_volume_search_paths
from ..logging_utils import get_logger
from .volume_stamp import VolumeStamp, read_volume_stamp

logger = get_logger(__name__)


@dataclass(frozen=True)
class DiscoveredVolume:
    """One stamped volume found attached, and where."""

    stamp: VolumeStamp
    mount_path: Path


def _subdirectories(directory: Path) -> list[Path]:
    try:
        return sorted((entry for entry in directory.iterdir() if entry.is_dir()), key=str)
    except OSError:
        return []


def platform_mount_root_candidates() -> list[Path]:
    """Directories whose immediate subdirectories may be mounted volumes.

    Mirrors the mount conventions `smftools.config.input_availability` already
    uses to recognize a detached volume structurally: macOS mounts at
    `/Volumes/<label>`; Linux at `/mnt/<label>`, `/media/<user>/<label>`, or
    `/run/media/<user>/<label>`. A root at depth 2 (the `<user>` layer) is
    expanded to its existing subdirectories first, since the actual username
    is not known in advance.
    """
    candidates: list[Path] = []
    for root_parts, depth in MOUNT_ROOTS:
        root = Path(*root_parts)
        if not root.is_dir():
            continue
        if depth == 1:
            candidates.append(root)
        else:
            candidates.extend(_subdirectories(root))
    return candidates


def discover_volumes(
    *,
    mount_roots: Optional[Iterable[Path]] = None,
    extra_paths: Optional[Iterable[Path]] = None,
    config_dir: Optional[Path] = None,
) -> list[DiscoveredVolume]:
    """Every stamped, currently-reachable volume.

    Args:
        mount_roots: Directories to enumerate one level deep for candidate
            volumes -- each subdirectory is checked for a stamp, matching how
            a platform mount root works (`/Volumes/<label>`). Defaults to
            `platform_mount_root_candidates()`; pass an explicit list in
            tests rather than depending on the real machine's mount points.
        extra_paths: Directories that are themselves candidate volumes,
            checked directly rather than enumerated -- for network mounts
            that do not live under a platform mount root. Defaults to
            `smftools.config.roots.extra_volume_search_paths(config_dir=...)`.
        config_dir: Passed through to `extra_volume_search_paths` when
            `extra_paths` is not given explicitly. Ignored otherwise.

    Returns:
        list[DiscoveredVolume]: One entry per distinct `volume_id` found,
        sorted by mount path. A candidate directory with no stamp is skipped
        silently; one with a corrupt stamp is skipped with a warning, never
        raised, since one bad mount must not fail discovery of every other
        attached volume. A second mount reporting a `volume_id` already found
        is skipped with a warning too -- only one attached copy of a given
        volume can be real at a time.
    """
    roots = list(mount_roots) if mount_roots is not None else platform_mount_root_candidates()
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(_subdirectories(Path(root)))
    if extra_paths is not None:
        candidates.extend(Path(path) for path in extra_paths)
    else:
        candidates.extend(extra_volume_search_paths(config_dir=config_dir))

    discovered: dict[str, DiscoveredVolume] = {}
    for candidate in candidates:
        try:
            stamp = read_volume_stamp(candidate)
        except ValueError as exc:
            logger.warning("skipping unreadable volume stamp at %s: %s", candidate, exc)
            continue
        if stamp is None:
            continue
        if stamp.volume_id in discovered:
            logger.warning(
                "volume %s is attached at both %s and %s; keeping the first",
                stamp.volume_id,
                discovered[stamp.volume_id].mount_path,
                candidate,
            )
            continue
        discovered[stamp.volume_id] = DiscoveredVolume(stamp=stamp, mount_path=candidate)
    return sorted(discovered.values(), key=lambda found: str(found.mount_path))
