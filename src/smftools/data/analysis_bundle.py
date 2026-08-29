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
ones. See `dev/plans/proposed/transfer_time_analysis_bundling_plan.md` for
the full design and the alternatives (zarr v3 sharding, coarser source-side
partitioning) it deliberately does not revisit.

Each bundle is a plain, uncompressed `tar` -- zarr chunk data is normally
already compressed by its own codec, so a second compression pass mostly
spends CPU for negligible size benefit -- and is self-contained: the
generation's own `generation_manifest.json` travels inside it, so unbundling
(`TAB-02`) never needs to reach back to the source to validate what it
extracted.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any, Optional

from ..informatics.generation_listing import (
    STATE_OK,
    GenerationRecord,
    list_experiment_generations,
)
from ..informatics.raw_intermediate_manifest import sha256_file
from ..readwrite import atomic_write_json

BUNDLE_SUFFIX = ".tar"
BUNDLE_SIDECAR_SUFFIX = ".tar.json"
BUNDLE_MANIFEST_SCHEMA_VERSION = 1


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
