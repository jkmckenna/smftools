"""Immutable generation publication for the basecall stage (`BCS-05`).

Promotes basecalling out of `raw`'s inline step and gives it the same
generation lifecycle every other stage got in 2.21.0: `basecall_outputs/`
publishing immutable, checksummed generations selected by `current.json`,
inventoried by `smftools experiment generations` for free through
`informatics.generation_listing.STAGE_GENERATION_DIRS`.

This module owns publication and validation only. Running dorado itself is
unchanged -- see `informatics.basecall_execution` -- and reading the model a
set of reads already carries is `informatics.basecall_provenance` (`BCS-02`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

from ..constants import BASECALL_DIR
from ..readwrite import atomic_write_json
from .experiment_manifest import artifact_record
from .generation import (
    GENERATION_MANIFEST,
    GenerationError,
    resolve_current_generation,
    staged_generation,
)

BASECALL_GENERATION_SCHEMA_VERSION = 1
BASECALL_ARTIFACT_NAME = "bam"


class BasecallGenerationError(RuntimeError):
    """Raised when a basecall generation cannot be published or validated safely."""


def _checksum(path: Path) -> str:
    return str(artifact_record(path, path.parent, checksum=True)["sha256"])


def validate_basecall_generation(
    generation_dir: str | Path,
    *,
    expected_generation_id: str | None = None,
) -> dict[str, Any]:
    """Validate one complete basecall generation without mutating it."""
    generation_dir = Path(generation_dir)
    manifest_path = generation_dir / GENERATION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BasecallGenerationError(
            "basecall generation manifest is missing or unreadable"
        ) from exc
    if int(manifest.get("schema_version", -1)) != BASECALL_GENERATION_SCHEMA_VERSION:
        raise BasecallGenerationError("basecall generation schema is incompatible")
    if manifest.get("status") != "complete":
        raise BasecallGenerationError("basecall generation is not complete")
    generation_id = str(manifest.get("generation_id", ""))
    if not generation_id or (
        expected_generation_id is not None and generation_id != expected_generation_id
    ):
        raise BasecallGenerationError("basecall generation ID does not match")
    if not str(manifest.get("model", "")).strip():
        raise BasecallGenerationError("basecall generation records no model")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or BASECALL_ARTIFACT_NAME not in artifacts:
        raise BasecallGenerationError("basecall generation artifact manifest is missing")
    record = artifacts[BASECALL_ARTIFACT_NAME]
    if not isinstance(record, dict):
        raise BasecallGenerationError("basecall generation BAM artifact record is malformed")
    bam_path = generation_dir / str(record.get("path", ""))
    if not bam_path.is_file() or str(record.get("sha256", "")) != _checksum(bam_path):
        raise BasecallGenerationError("basecall generation BAM is missing or corrupt")

    input_ids = manifest.get("input_artifact_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise BasecallGenerationError("basecall generation records no input identity")
    return manifest


def resolve_current_basecall_generation(
    basecall_output_dir: str | Path,
) -> Optional[tuple[Path, dict[str, Any]]]:
    """Resolve and validate the generation selected by basecall's `current.json`."""
    basecall_output_dir = Path(basecall_output_dir)
    try:
        selected = resolve_current_generation(
            basecall_output_dir,
            manifest_checksum=_checksum,
            require_generation_id=True,
        )
    except GenerationError as exc:
        raise BasecallGenerationError(str(exc)) from exc
    if selected is None:
        return None
    generation_dir, pointer_manifest = selected
    manifest = validate_basecall_generation(
        generation_dir, expected_generation_id=str(pointer_manifest.get("generation_id", ""))
    )
    return generation_dir, manifest


def publish_basecall_generation(
    run_root: str | Path,
    *,
    bam_path: str | Path,
    model: str,
    modality: str,
    config_hash: str,
    input_artifact_ids: list[str],
    dorado_version: str | None = None,
    bam_suffix: str = ".bam",
    generation_id: str | None = None,
    extra_manifest_fields: Mapping[str, Any] | None = None,
) -> dict[str, Path | str]:
    """Snapshot, validate, and atomically publish one immutable basecall generation.

    Args:
        run_root: The experiment's output directory root.
        bam_path: The BAM dorado just wrote. Copied into the generation, not
            moved -- the caller's own workspace (an `IntermediateSpec` commit,
            typically) stays intact for its own reuse bookkeeping.
        model: The resolved basecall model name, exactly as dorado reports it.
        modality: The experiment's SMF modality, recorded for reference.
        config_hash: The stage config hash, for the generation manifest's
            record of what produced it -- not (yet) a reuse key on its own;
            see `BCS-07` for identity that survives an offline POD5 archive.
        input_artifact_ids: Per-source content identities of the POD5/FAST5
            input consumed -- `cli.helpers.basecall_input_artifact_ids`'s
            `input-manifest:<digest>` plus one `source:<source_id>:<sha256>`
            per file, not one aggregate checksum, so a later comparison can
            tell a superset/subset/disjoint change in the source set apart
            from "identical" (`BCS-07`; classification itself is `BCS-11`).
        dorado_version: The installed Dorado version, recorded and never
            gating -- a release that leaves the model identity unchanged must
            not force a re-basecall of anything.
        bam_suffix: Extension for the published BAM artifact.
        generation_id: Override the generated id (tests; republishing a known id).
        extra_manifest_fields: Additional top-level manifest fields, merged in
            without overwriting the fields this function itself controls.

    Returns:
        A dict of artifact paths, plus ``generation``/``generation_manifest``/
        ``current``/``generation_id``, matching the shape other stage
        publishers return.
    """
    run_root = Path(run_root)
    basecall_output_dir = run_root / BASECALL_DIR
    bam_path = Path(bam_path)
    if not bam_path.is_file():
        raise BasecallGenerationError(f"basecall publication source BAM is missing: {bam_path}")
    if not str(model).strip():
        raise BasecallGenerationError("basecall publication requires a model name")
    if not input_artifact_ids:
        raise BasecallGenerationError("basecall publication requires input_artifact_ids")

    artifact_relative = f"basecalls{bam_suffix}"

    def validate(staging: Path, _final: Path, _root: Path) -> None:
        validate_basecall_generation(staging, expected_generation_id=staged.generation_id)

    try:
        with staged_generation(
            basecall_output_dir,
            run_root=run_root,
            validate=validate,
            generation_id=generation_id,
            manifest_checksum=_checksum,
            write_json=atomic_write_json,
        ) as staged:
            generation_id = staged.generation_id
            staging_dir = staged.staging_dir
            destination = staged.artifact(artifact_relative)
            shutil.copy2(bam_path, destination)
            manifest = {
                **dict(extra_manifest_fields or {}),
                "schema_version": BASECALL_GENERATION_SCHEMA_VERSION,
                "status": "complete",
                "generation_id": generation_id,
                "config_hash": str(config_hash),
                "model": str(model),
                "dorado_version": str(dorado_version) if dorado_version else None,
                "modality": str(modality),
                "input_artifact_ids": list(input_artifact_ids),
                "artifacts": {
                    BASECALL_ARTIFACT_NAME: artifact_record(destination, staging_dir, checksum=True)
                },
            }
            staged.record_manifest(manifest)
    except GenerationError as exc:
        raise BasecallGenerationError(str(exc)) from exc

    final_dir = staged.final_dir
    return {
        BASECALL_ARTIFACT_NAME: final_dir / artifact_relative,
        "generation": final_dir,
        "generation_manifest": final_dir / GENERATION_MANIFEST,
        "current": basecall_output_dir / "current.json",
        "generation_id": generation_id,
    }
