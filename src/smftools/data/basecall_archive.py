"""Write a basecall generation back to its POD5 archive (`BCS-08`, `BCS-09`).

Basecalls are derived and regenerable, so archiving them is an optimisation,
not a duty -- but a large one: it is what makes the POD5s optional for
everything downstream. Write-back is always an explicit command, never
automatic, and idempotent: re-running it after an interrupted copy verifies
and re-copies rather than leaving a duplicate or a corrupt file behind.

Layout, as a sibling of the signal rather than mixed into it::

    <archive_root>/
    └── basecalls/
        └── <model>@<dorado_version>/
            ├── basecall_manifest.json
            └── *.bam

Keyed by model (and Dorado version, since a build change can shift model
behaviour even when the model name itself does not) so several models can
coexist and selection can answer "is there a derivative for this model?"
from the directory listing alone, before opening anything.

`BCS-09`'s I/O policy -- basecalls always land in the analysis tree first,
write-back is a separate phase run after a batch, and overlap between
reading signal and writing an archive copy is safe only across devices --
is enforced today by there being no code path that runs basecalling and
archiving in the same process at all (`experiment batch` has no `archive`
task); this module's contribution is reporting whether a source and
destination share a volume (`PSR-08`), via `same_volume` in its result, so a
caller sequencing multiple runs by hand can see it rather than guess.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from ..constants import BASECALL_DIR
from ..informatics.basecall_generation import resolve_current_basecall_generation
from ..informatics.raw_intermediate_manifest import sha256_file
from ..readwrite import atomic_write_json
from .volume_stamp import read_volume_stamp

BASECALLS_ARCHIVE_DIRNAME = "basecalls"
ARCHIVE_MANIFEST_FILENAME = "basecall_manifest.json"
ARCHIVE_MANIFEST_SCHEMA_VERSION = 1


class BasecallArchiveError(RuntimeError):
    """Raised when a basecall generation cannot be safely written back to an archive."""


def enclosing_volume_id(path: str | Path) -> Optional[str]:
    """Return the `volume_id` of the nearest stamped ancestor of `path`, if any.

    Walks upward from `path` itself (whether or not it currently exists) to
    the filesystem root, reading `.smftools-volume.json` at each level
    (`PSR-08`). A corrupt stamp on some unrelated ancestor is skipped rather
    than raised -- a write-back caller's job is comparing volumes, not
    auditing them; `data verify`/`data status` are where a corrupt stamp is
    a reportable problem in its own right.
    """
    resolved = Path(path).resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        try:
            stamp = read_volume_stamp(candidate)
        except ValueError:
            continue
        if stamp is not None:
            return stamp.volume_id
    return None


def archive_basecall_generation(
    run_root: str | Path,
    *,
    archive_root: str | Path,
) -> dict[str, Any]:
    """Copy `run_root`'s current basecall generation into `archive_root`, idempotently.

    Args:
        run_root: An experiment's output directory (or a bare `--output`
            directory from `smftools basecall`'s config-free form), whose
            `basecall_outputs/current.json` generation is what gets archived.
        archive_root: The run's archive directory -- conventionally the
            sibling of wherever its POD5s live, e.g. `<archive>/<run>/`.
            `basecalls/<model>@<dorado_version>/` is created under it.

    Returns:
        A dict with `status` (`"archived"` or `"already_archived"`), `path`
        (the destination model directory), `generation_id`, and
        `same_volume` (whether the source BAM and the archive destination
        currently resolve to the same stamped volume, `PSR-08`; `None` when
        either side is unstamped and cannot be compared).

    Raises:
        BasecallArchiveError: `run_root` has no current basecall generation,
            its manifest is missing the fields write-back needs, or the
            copied BAM's checksum does not match the source -- refused
            rather than publishing a corrupt archive copy.
    """
    run_root = Path(run_root)
    archive_root = Path(archive_root)

    resolved = resolve_current_basecall_generation(run_root / BASECALL_DIR)
    if resolved is None:
        raise BasecallArchiveError(f"{run_root} has no current basecall generation to archive.")
    generation_dir, manifest = resolved

    model = str(manifest.get("model", "")).strip()
    if not model:
        raise BasecallArchiveError(f"basecall generation at {generation_dir} records no model.")
    dorado_version = str(manifest.get("dorado_version") or "unknown")

    bam_record = manifest.get("artifacts", {}).get("bam", {})
    bam_relative = str(bam_record.get("path", ""))
    source_sha256 = str(bam_record.get("sha256", ""))
    if not bam_relative or not source_sha256:
        raise BasecallArchiveError(
            f"basecall generation at {generation_dir} has no recorded BAM artifact."
        )
    source_bam = generation_dir / bam_relative
    if not source_bam.is_file():
        raise BasecallArchiveError(f"basecall generation BAM is missing: {source_bam}")

    same_volume: Optional[bool] = None
    source_volume_id = enclosing_volume_id(source_bam)
    destination_volume_id = enclosing_volume_id(archive_root)
    if source_volume_id is not None and destination_volume_id is not None:
        same_volume = source_volume_id == destination_volume_id

    dest_dir = archive_root / BASECALLS_ARCHIVE_DIRNAME / f"{model}@{dorado_version}"
    dest_manifest_path = dest_dir / ARCHIVE_MANIFEST_FILENAME
    dest_bam_path = dest_dir / Path(bam_relative).name

    if dest_manifest_path.is_file():
        try:
            existing = json.loads(dest_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if (
            existing.get("generation_id") == manifest.get("generation_id")
            and dest_bam_path.is_file()
            and sha256_file(dest_bam_path) == source_sha256
        ):
            return {
                "status": "already_archived",
                "path": dest_dir,
                "generation_id": manifest.get("generation_id"),
                "same_volume": same_volume,
            }

    dest_dir.mkdir(parents=True, exist_ok=True)
    staging_bam = dest_dir / f"{dest_bam_path.name}.partial"
    shutil.copy2(source_bam, staging_bam)
    copied_sha256 = sha256_file(staging_bam)
    if copied_sha256 != source_sha256:
        staging_bam.unlink(missing_ok=True)
        raise BasecallArchiveError(
            f"write-back checksum mismatch copying {source_bam} to {staging_bam}; "
            "not publishing a corrupt archive copy."
        )
    os.replace(staging_bam, dest_bam_path)

    atomic_write_json(
        dest_manifest_path,
        {
            "schema_version": ARCHIVE_MANIFEST_SCHEMA_VERSION,
            "generation_id": manifest.get("generation_id"),
            "model": model,
            "dorado_version": manifest.get("dorado_version"),
            "modality": manifest.get("modality"),
            "config_hash": manifest.get("config_hash"),
            "input_artifact_ids": manifest.get("input_artifact_ids"),
            "max_basecall_reads": manifest.get("max_basecall_reads"),
            "subsample_seed": manifest.get("subsample_seed"),
            "bam": {"path": dest_bam_path.name, "sha256": source_sha256},
            "archived_from": str(run_root),
        },
    )

    return {
        "status": "archived",
        "path": dest_dir,
        "generation_id": manifest.get("generation_id"),
        "same_volume": same_volume,
    }
