"""Classify whether an experiment's raw input is present, offline, or missing.

Raw sequencing input is routinely archived to removable storage once an
experiment's pipeline outputs exist. Every stage used to fail the moment that
volume was detached, because config loading discovered input files eagerly and
`discover_input_files` raises on an absent path -- so `hmm`, `spatial`, and
`export-bundle`, none of which read a byte of raw input, failed alongside `raw`.

Three states replace that single failure:

``present``
    The path resolves and can be discovered, exactly as before.
``offline``
    The path lies on a volume that is not currently attached. Expected, and not
    an error: only the stages that consume raw input object.
``missing``
    The volume is attached (or the path was never on removable storage) and the
    path is simply absent. A real error, reported as early as it ever was.

The volume test here is structural -- a path under a platform mount root whose
mount directory is absent. This is deliberately conservative, because
misreading a deleted path as "offline" would silently skip ingestion.

`PSR-12` adds an exact answer on top of it, when available: if the run has a
published input manifest (identifying its dataset) and the volume catalog
(`smftools.data.replica_catalog`, populated by `data scan`) knows a replica of
that dataset, classification is decided from volume identity instead of a
guess -- no attached replica is a confident ``offline`` even when the path
does not match a recognized mount convention at all (a network share, say),
and an attached replica that still resolves the specific path is ``present``
even though it moved to a different mount point or name. Either input to this
-- a published manifest, or a populated catalog -- being absent falls back to
the structural guess exactly as before; a user who has never run `data scan`
sees no change in behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

INPUT_PRESENT = "present"
INPUT_OFFLINE = "offline"
INPUT_MISSING = "missing"

#: Mount roots by platform, mapped to the depth at which the *volume* directory
#: sits beneath them. macOS mounts at ``/Volumes/<label>``; Linux uses
#: ``/mnt/<label>``, ``/media/<user>/<label>``, or ``/run/media/<user>/<label>``.
#: Public (not underscore-prefixed) because ``smftools.data.volume_discovery``
#: (`PSR-09`) reuses it as the single source of truth for platform mount
#: conventions rather than duplicating this table.
MOUNT_ROOTS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("/", "Volumes"), 1),
    (("/", "mnt"), 1),
    (("/", "media"), 2),
    (("/", "run", "media"), 2),
)


@dataclass(frozen=True)
class InputAvailability:
    """The resolved state of one configured input path."""

    state: str
    path: Optional[Path] = None
    volume: Optional[Path] = None
    detail: str = ""
    #: The dataset's volume, by `volume_id`, when classification came from the
    #: replica catalog (`PSR-12`) rather than the structural guess -- set only
    #: when exactly one candidate volume is known, since `volume` above is a
    #: mount *path* and an unattached volume has none to report.
    volume_id: Optional[str] = None

    @property
    def is_present(self) -> bool:
        """Whether the input can be read right now."""
        return self.state == INPUT_PRESENT


def detached_volume_for(path: Path) -> Optional[Path]:
    """Return the mount directory ``path`` sits on when that directory is absent.

    Args:
        path: An absolute path that does not currently resolve.

    Returns:
        Path or None: The volume directory (e.g. ``/Volumes/<label>``) when
        ``path`` lies under a known mount root and that volume directory does
        not exist, otherwise ``None``.
    """
    parts = Path(path).parts
    for root, depth in MOUNT_ROOTS:
        if parts[: len(root)] != root:
            continue
        volume_parts = parts[: len(root) + depth]
        if len(volume_parts) < len(root) + depth:
            # The path names the mount root itself, not a volume beneath it.
            continue
        volume = Path(*volume_parts)
        if not volume.exists():
            return volume
    return None


def _exact_availability(
    path: Path, output_directory: str | Path | None
) -> Optional[InputAvailability]:
    """Classify `path` from volume identity when the inputs for it exist (`PSR-12`).

    Returns ``None`` -- deferring to the structural approximation -- whenever
    any prerequisite is missing: no ``output_directory``, no published input
    manifest for this run, no populated replica catalog, or nothing
    catalogued for this manifest's dataset. That is the ordinary state for
    anyone who has not run `data scan`, and must not become an error or a
    behavior change for them.
    """
    if output_directory is None:
        return None

    from ..constants import RAW_DIR
    from ..data.replica_catalog import load_catalog, replicas_for, resolve_replica
    from ..data.volume_discovery import discover_volumes
    from ..informatics.input_manifest import (
        INPUT_MANIFEST_DIRNAME,
        RESOLVED_INPUT_MANIFEST_JSON,
        InputManifestError,
        read_resolved_input_manifest,
    )

    manifest_path = (
        Path(output_directory) / RAW_DIR / INPUT_MANIFEST_DIRNAME / RESOLVED_INPUT_MANIFEST_JSON
    )
    if not manifest_path.is_file():
        return None
    try:
        manifest = read_resolved_input_manifest(manifest_path)
    except InputManifestError:
        return None

    catalog = load_catalog()
    replicas = replicas_for(catalog, manifest.digest)
    if not replicas:
        return None

    resolved_replica = resolve_replica(catalog, manifest.digest, attached=discover_volumes())
    if resolved_replica is None:
        known_ids = sorted({replica.volume_id for replica in replicas})
        return InputAvailability(
            state=INPUT_OFFLINE,
            path=path,
            volume_id=known_ids[0] if len(known_ids) == 1 else None,
            detail=(
                f"{path} belongs to a dataset with {len(replicas)} known replica(s) "
                f"(volume(s): {', '.join(known_ids)}), none of which is currently attached."
            ),
        )

    try:
        relative = path.resolve(strict=False).relative_to(
            Path(manifest.base_directory).resolve(strict=False)
        )
    except ValueError:
        return None
    relocated = resolved_replica.resolved_path / relative
    if not relocated.exists():
        return None
    return InputAvailability(state=INPUT_PRESENT, path=relocated)


def resolve_input_availability(
    path: str | Path | None, *, output_directory: str | Path | None = None
) -> InputAvailability:
    """Classify one configured input path as present, offline, or missing.

    Args:
        path: The configured ``input_data_path`` or manifest path. ``None``
            resolves to ``present``, since an unset path is not a failure here;
            the caller decides whether one was required.
        output_directory: The run's own output directory, used only to reach
            its published input manifest for the exact classification path
            (`PSR-12`). ``None`` (the default) skips that path entirely and
            preserves the structural approximation's behavior unchanged.

    Returns:
        InputAvailability: The state, plus the detached volume and a
        human-readable sentence when the input cannot be read.
    """
    if path is None:
        return InputAvailability(state=INPUT_PRESENT)
    resolved = Path(path).expanduser()
    if resolved.exists():
        return InputAvailability(state=INPUT_PRESENT, path=resolved)

    exact = _exact_availability(resolved, output_directory)
    if exact is not None:
        return exact

    volume = detached_volume_for(resolved)
    if volume is not None:
        return InputAvailability(
            state=INPUT_OFFLINE,
            path=resolved,
            volume=volume,
            detail=(
                f"input path {resolved} is on volume {volume}, which is not attached. "
                "Stages that do not read raw input continue normally; attach the volume "
                "to run ingestion."
            ),
        )
    return InputAvailability(
        state=INPUT_MISSING,
        path=resolved,
        detail=f"input path does not exist: {resolved}",
    )


def require_input_available(availability: InputAvailability, *, stage: str) -> None:
    """Raise unless the input can be read, naming what to attach.

    Args:
        availability: The state resolved at config load.
        stage: The stage requiring the input, named in the error.

    Raises:
        FileNotFoundError: If the input is offline or missing.
    """
    if availability.is_present:
        return
    if availability.state == INPUT_OFFLINE:
        known_as = availability.volume or availability.volume_id or "an unattached volume"
        raise FileNotFoundError(
            f"stage {stage!r} reads raw input, which is currently offline: "
            f"{availability.path} is on {known_as}, which is not "
            "attached. Attach the volume and re-run. Stages that do not read raw "
            "input (preprocess, spatial, hmm, latent, export-bundle) run without it."
        )
    raise FileNotFoundError(
        f"stage {stage!r} reads raw input, which is missing: {availability.path}. "
        "The path's volume is attached, so this is an absent or mistyped path rather "
        "than archived data."
    )


def restore_offline_input_identity(
    output_directory: str | Path | None, *, bam_suffix: str
) -> Optional[tuple[str, list[Path]]]:
    """Recover the input type and file list a completed run already recorded.

    A stage decides whether it is already complete by comparing config hashes, and
    ``input_type``/``input_files`` feed that hash. Discovery cannot produce them
    while the input volume is detached, so without this every stage would see a
    changed hash, judge a finished raw generation incompatible, and try to
    re-ingest data it cannot reach (`PSR-01`).

    The values come from the run's resolved input manifest -- the sources raw
    actually ingested -- and **not** from the manifest's config snapshot, which
    records ``input_type`` after the FASTQ-to-BAM rewrite has already replaced it
    and so would not reproduce the hash.

    Args:
        output_directory: The run root, or ``None`` when unset.
        bam_suffix: The configured BAM suffix, for classification.

    Returns:
        tuple or None: ``(input_type, input_files)`` when a resolved input
        manifest is readable, otherwise ``None``.
    """
    if output_directory is None:
        return None
    manifest = (
        Path(output_directory) / "raw_outputs" / "input_manifest" / "resolved_input_manifest.json"
    )
    if not manifest.is_file():
        return None
    try:
        with manifest.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    base = payload.get("base_directory")
    sources = payload.get("sources") or []
    paths: list[Path] = []
    for source in sources:
        raw_path = source.get("path")
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute() and base:
            candidate = Path(base) / candidate
        paths.append(candidate)
    if not paths:
        return None

    from .discover_input_files import input_kind_for_path

    kinds = {input_kind_for_path(path, bam_suffix=bam_suffix) for path in paths}
    if len(kinds) != 1:
        # Mixed recorded types would have been refused at ingestion, so this means
        # the record is not one this restoration understands. Fall back rather
        # than assert an identity that could differ from the hashed one.
        return None
    return kinds.pop(), paths
