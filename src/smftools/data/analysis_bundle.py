"""Bundle a run's published generations into few, large files for transfer (`TAB-01`).

An experiment's analysis tree is written as many small, independent partition
stores -- fine-grained on purpose, to bound per-task memory and make each task
independently resumable (see `dev/plans/audits/pipeline_scaling_audit.md`).
That is the right tradeoff for *writing*, but it makes the tree brutally slow
to copy to another drive: a real run's `analyses/` tree measured over a
million files for a few hundred gigabytes, and `rsync` between two local
drives on that tree slows from ~100MB/s to under 1MB/s once it reaches the
zarr-chunk-dense parts -- per-file negotiation overhead, not byte count.

This module does not touch the write path at all. It bundles what is already
on disk, one archive per *published generation* -- a generation directory
(`<stage>_outputs/generations/<generation_id>/`) is the codebase's own
immutability boundary (`staged_generation` publishes it atomically and it
never changes again), so a generation needs bundling exactly once, ever. A
later run only adds bundles for new generations; it never re-touches old
ones. See `dev/plans/in-progress/transfer_time_analysis_bundling_plan.md` for
the full design and the alternatives (zarr v3 sharding, coarser source-side
partitioning) it deliberately does not revisit.

Each bundle is a plain, uncompressed `tar` -- zarr chunk data is normally
already compressed by its own codec, so a second compression pass mostly
spends CPU for negligible size benefit -- and is self-contained: the
generation's own `generation_manifest.json` travels inside it, so unbundling
(`TAB-02`) never needs to reach back to the source to validate what it
extracted.

`unbundle_analysis_generations` (`TAB-02`) is the inverse: extract each
bundle into `run_root`'s matching `<stage>_outputs/generations/<id>/` path,
stage-then-atomic-rename so an interrupted extraction never leaves a partial
generation directory that looks real, and verify twice before trusting the
result -- the bundle's own recorded checksum before extracting (proves the
*transfer* did not corrupt it), then, for the stages that record one
(`basecall`/`raw`/`preprocess`), every artifact's own recorded checksum
after extracting (proves the *tar round-trip* preserved content the original
pipeline vouched for, not merely that the tar file itself is intact). Other
stages (`spatial`/`hmm`/`latent`, ...) do not record per-artifact checksums
in their manifests today, so unbundling one can only confirm the manifest
parses and its `generation_id` matches -- reported honestly as such, not
silently claimed as full verification. `current.json` stays untouched here
too; `data sync` already reconciles it.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Optional

from ..informatics.experiment_manifest import artifact_record
from ..informatics.generation_listing import (
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STAGE_GENERATION_DIRS,
    STATE_OK,
    GenerationRecord,
    list_experiment_generations,
)
from ..informatics.raw_intermediate_manifest import sha256_file
from ..readwrite import atomic_write_json

BUNDLE_SUFFIX = ".tar"
BUNDLE_SIDECAR_SUFFIX = ".tar.json"
BUNDLE_MANIFEST_SCHEMA_VERSION = 1
UNBUNDLE_STAGING_SUBDIR = ".bundle-staging"


class AnalysisBundleError(RuntimeError):
    """Raised when a generation cannot be safely bundled."""


def _bundle_paths(bundle_root: Path, kind: str, generation_id: str) -> tuple[Path, Path]:
    stage_dir = bundle_root / kind
    return (
        stage_dir / f"{generation_id}{BUNDLE_SUFFIX}",
        stage_dir / f"{generation_id}{BUNDLE_SIDECAR_SUFFIX}",
    )


def _already_bundled(bundle_path: Path, sidecar_path: Path, generation_id: str) -> bool:
    if not (bundle_path.is_file() and sidecar_path.is_file()):
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if sidecar.get("generation_id") != generation_id:
        return False
    recorded_sha256 = sidecar.get("sha256")
    if not recorded_sha256:
        return False
    try:
        return sha256_file(bundle_path) == recorded_sha256
    except OSError:
        return False


def _bundle_one(
    run_root: Path,
    record: GenerationRecord,
    bundle_root: Path,
) -> dict[str, Any]:
    bundle_path, sidecar_path = _bundle_paths(bundle_root, record.kind, record.generation_id)
    if _already_bundled(bundle_path, sidecar_path, record.generation_id):
        return {
            "kind": record.kind,
            "generation_id": record.generation_id,
            "status": "already_bundled",
            "path": bundle_path,
        }

    source_dir = run_root / record.path
    if not source_dir.is_dir():
        raise AnalysisBundleError(
            f"generation {record.generation_id!r} ({record.kind}) has no directory at "
            f"{source_dir}, despite a complete manifest"
        )

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = bundle_path.with_name(bundle_path.name + ".partial")
    with tarfile.open(staging_path, "w") as tar:
        tar.add(source_dir, arcname=record.generation_id)

    digest = sha256_file(staging_path)
    os.replace(staging_path, bundle_path)
    atomic_write_json(
        sidecar_path,
        {
            "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            "generation_id": record.generation_id,
            "kind": record.kind,
            "source_path": record.path,
            "sha256": digest,
        },
    )
    return {
        "kind": record.kind,
        "generation_id": record.generation_id,
        "status": "bundled",
        "path": bundle_path,
    }


def bundle_analysis_generations(
    run_root: str | Path,
    *,
    bundle_root: str | Path,
    stage: Optional[str] = None,
    generation_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Tar every complete, not-yet-bundled generation under `run_root`.

    Args:
        run_root: An experiment's output directory (the one holding
            `raw_outputs/`, `preprocess_adata_outputs/`, ...).
        bundle_root: Directory to write bundles into --
            `bundle_root/<kind>/<generation_id>.tar` plus a `.tar.json`
            sidecar recording the bundle's own checksum for idempotency.
        stage: Restrict to one stage's generations (`raw`, `preprocess`,
            `spatial`, `hmm`, `latent`, `basecall`, ...). All stages if `None`.
        generation_id: Restrict to one generation id. All if `None`.

    Returns:
        One dict per generation considered: `kind`, `generation_id`,
        `status` (`"bundled"`, `"already_bundled"`, or `"skipped"`), `path`
        (the bundle path, absent for `"skipped"`), and for `"skipped"` a
        `reason` naming why (not `state == STATE_OK`, not `status ==
        "complete"`, or filtered out by `stage`/`generation_id`).

    Raises:
        AnalysisBundleError: A generation's manifest reports it complete but
            its directory is missing -- refused rather than silently
            producing an empty or partial bundle.
    """
    run_root = Path(run_root)
    bundle_root = Path(bundle_root)
    results: list[dict[str, Any]] = []
    for record in list_experiment_generations(run_root):
        if stage is not None and record.kind != stage:
            continue
        if generation_id is not None and record.generation_id != generation_id:
            continue
        if record.state != STATE_OK:
            results.append(
                {
                    "kind": record.kind,
                    "generation_id": record.generation_id,
                    "status": "skipped",
                    "reason": f"generation state is {record.state!r}, not readable",
                }
            )
            continue
        if record.status != "complete":
            results.append(
                {
                    "kind": record.kind,
                    "generation_id": record.generation_id,
                    "status": "skipped",
                    "reason": f"generation status is {record.status!r}, not 'complete'",
                }
            )
            continue
        results.append(_bundle_one(run_root, record, bundle_root))
    return results


def _verify_extracted_generation(
    generation_dir: Path, expected_generation_id: str
) -> tuple[bool, list[str]]:
    """Re-verify a freshly-extracted generation against its own manifest.

    Returns ``(any_checksum_verified, problems)``. ``problems`` is non-empty
    only for a real integrity failure (missing/unreadable manifest, a
    `generation_id` mismatch, a missing or corrupt artifact) -- never for a
    stage simply not recording per-artifact checksums yet, which is reported
    through ``any_checksum_verified`` being `False` with an empty
    ``problems`` list instead.
    """
    manifest_path = generation_dir / GENERATION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"generation manifest is missing or unreadable: {exc}"]

    declared_id = str(manifest.get("generation_id", ""))
    if declared_id != expected_generation_id:
        return False, [
            f"manifest generation_id {declared_id!r} does not match bundle "
            f"{expected_generation_id!r}"
        ]

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False, []

    checked_any = False
    problems: list[str] = []
    for key, record in artifacts.items():
        if not (isinstance(record, dict) and "path" in record and "sha256" in record):
            continue
        checked_any = True
        artifact_path = generation_dir / str(record["path"])
        if not artifact_path.exists():
            problems.append(f"artifact {key!r} is missing: {artifact_path}")
            continue
        try:
            # `artifact_record`'s own checksum -- not `sha256_file` -- since an
            # artifact may be a directory (e.g. preprocess's bulk `store/`),
            # hashed by name+content across every file in it, not one file's
            # bytes; reusing the same function that *wrote* the recorded
            # checksum is what makes this a real re-verification rather than
            # a mismatched algorithm reported as corruption.
            actual = str(
                artifact_record(artifact_path, artifact_path.parent, checksum=True)["sha256"]
            )
        except OSError as exc:
            problems.append(f"artifact {key!r} could not be checksummed: {exc}")
            continue
        if actual != str(record["sha256"]):
            problems.append(f"artifact {key!r} checksum mismatch after unbundling")
    return checked_any, problems


def _unbundle_one(
    bundle_path: Path,
    sidecar_path: Path,
    kind: str,
    generation_id: str,
    run_root: Path,
) -> dict[str, Any]:
    stage_dirname = STAGE_GENERATION_DIRS.get(kind)
    if stage_dirname is None:
        raise AnalysisBundleError(
            f"unknown stage kind {kind!r} in bundle {bundle_path}; no generation directory mapping"
        )
    destination = run_root / stage_dirname / GENERATIONS_SUBDIR / generation_id

    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisBundleError(
            f"bundle sidecar is missing or unreadable: {sidecar_path}"
        ) from exc
    recorded_sha256 = str(sidecar.get("sha256", ""))
    if not recorded_sha256 or sha256_file(bundle_path) != recorded_sha256:
        raise AnalysisBundleError(
            f"bundle {bundle_path} does not match its recorded checksum -- refusing to "
            "extract a possibly-corrupt transfer"
        )

    if destination.is_dir():
        checked, problems = _verify_extracted_generation(destination, generation_id)
        if not problems:
            return {
                "kind": kind,
                "generation_id": generation_id,
                "status": "already_unbundled",
                "path": destination,
                "checksums_verified": checked,
            }
        # A destination exists but fails verification: fall through and
        # re-extract rather than trusting or silently leaving it in place.

    staging_parent = run_root / stage_dirname / GENERATIONS_SUBDIR / UNBUNDLE_STAGING_SUBDIR
    staging_parent.mkdir(parents=True, exist_ok=True)
    staged_generation_dir = staging_parent / generation_id
    if staged_generation_dir.exists():
        shutil.rmtree(staged_generation_dir)
    with tarfile.open(bundle_path, "r") as tar:
        tar.extractall(staging_parent, filter="data")

    checked, problems = _verify_extracted_generation(staged_generation_dir, generation_id)
    if problems:
        shutil.rmtree(staged_generation_dir, ignore_errors=True)
        raise AnalysisBundleError(
            f"unbundled generation {generation_id!r} ({kind}) failed verification: "
            + "; ".join(problems)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        shutil.rmtree(destination)
    os.replace(staged_generation_dir, destination)
    try:
        staging_parent.rmdir()
    except OSError:
        pass  # not empty -- another generation for this stage is mid-extraction

    return {
        "kind": kind,
        "generation_id": generation_id,
        "status": "unbundled",
        "path": destination,
        "checksums_verified": checked,
    }


def unbundle_analysis_generations(
    bundle_root: str | Path,
    *,
    run_root: str | Path,
    stage: Optional[str] = None,
    generation_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Extract bundles from `bundle_root` into `run_root`'s matching generation paths.

    Args:
        bundle_root: Directory `bundle_analysis_generations` wrote bundles
            into -- `bundle_root/<kind>/<generation_id>.tar` plus its
            `.tar.json` sidecar.
        run_root: The destination experiment output directory. Need not
            exist yet; `<stage>_outputs/generations/` is created as needed.
        stage: Restrict to one stage's bundles. All stages if `None`.
        generation_id: Restrict to one generation id. All if `None`.

    Returns:
        One dict per bundle considered: `kind`, `generation_id`, `status`
        (`"unbundled"`, `"already_unbundled"`), `path` (the destination
        generation directory), and `checksums_verified` (whether the stage's
        manifest recorded per-artifact checksums to actually verify against
        -- `False` does not mean a problem was found, only that this stage
        does not yet record enough to fully prove content integrity).

    Raises:
        AnalysisBundleError: A bundle's own checksum does not match its
            sidecar (refuses to extract a possibly-corrupt transfer), or the
            freshly-extracted generation fails verification against its own
            manifest (refuses to leave a corrupt generation at the
            destination) -- either way nothing partial is left behind.
    """
    bundle_root = Path(bundle_root)
    run_root = Path(run_root)
    results: list[dict[str, Any]] = []
    if not bundle_root.is_dir():
        return results
    for kind_dir in sorted(p for p in bundle_root.iterdir() if p.is_dir()):
        kind = kind_dir.name
        if stage is not None and kind != stage:
            continue
        for bundle_path in sorted(kind_dir.glob(f"*{BUNDLE_SUFFIX}")):
            found_generation_id = bundle_path.stem
            if generation_id is not None and found_generation_id != generation_id:
                continue
            sidecar_path = kind_dir / f"{found_generation_id}{BUNDLE_SIDECAR_SUFFIX}"
            results.append(
                _unbundle_one(bundle_path, sidecar_path, kind, found_generation_id, run_root)
            )
    return results
