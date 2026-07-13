"""Project registry: register experiments by pointer (append-only) + named sets.

The registry (``registry.json``) is the project's source of truth for *which*
experiments belong to it and where they live. Experiments are never copied; each
entry is a pointer to a run's output_directory, plus every pipeline stage spine
discovered under it (raw, preprocess, spatial, hmm, ...) -- registering an
experiment doesn't pin it to one stage, since a later stage's spine already
carries forward everything an earlier stage produced (see
``informatics.partition_read.materialize``'s layer/read-metric overlays), and a
project consumer may want any of them. Registration reads the raw spine's
``uns`` for modality + per-reference ``reference_uids`` so the project can
harmonize references without opening matrices.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smftools.constants import (
    CHIMERIC_DIR,
    HMM_DIR,
    LATENT_DIR,
    PREPROCESS_DIR,
    RAW_DIR,
    SPATIAL_DIR,
    VARIANT_DIR,
)
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

REGISTRY_FILENAME = "registry.json"
SETS_SUBDIR = "sets"
SCHEMA_VERSION = 2
SPINE_FILENAME = "spine.h5ad"
_CATALOG_NAMES = ("interval_catalog.parquet", "catalog.parquet")

# Stage name -> output subdirectory, for discovering every stage spine under one
# run's output_directory (see cli.helpers.AdataPaths -- the same names).
STAGE_DIRS = {
    "raw": RAW_DIR,
    "preprocess": PREPROCESS_DIR,
    "spatial": SPATIAL_DIR,
    "hmm": HMM_DIR,
    "latent": LATENT_DIR,
    "variant": VARIANT_DIR,
    "chimeric": CHIMERIC_DIR,
}

# Auto-fallback order when a caller doesn't request a specific stage: the
# sequential main pipeline chain, most-derived (most information available)
# first. latent/variant/chimeric are side branches off preprocess, not deeper
# in this chain, so they're only used when explicitly requested by stage name.
STAGE_PRIORITY = ("hmm", "spatial", "preprocess", "raw")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_registry_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / REGISTRY_FILENAME


def init_project(project_dir: str | Path) -> Path:
    """Create a project directory + empty registry (idempotent)."""
    project_dir = Path(project_dir)
    (project_dir / SETS_SUBDIR).mkdir(parents=True, exist_ok=True)
    path = project_registry_path(project_dir)
    if not path.exists():
        save_registry(
            project_dir,
            {"schema_version": SCHEMA_VERSION, "created_at": _now(), "experiments": {}, "sets": {}},
        )
        logger.info("Initialized project at %s", project_dir)
    return path


def load_registry(project_dir: str | Path) -> dict:
    path = project_registry_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"no project registry at {path}; run 'project init' first")
    with path.open() as handle:
        return json.load(handle)


def save_registry(project_dir: str | Path, registry: dict) -> Path:
    path = project_registry_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry["updated_at"] = _now()
    with path.open("w") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    return path


def discover_stage_spines(experiment_dir: Path) -> tuple[Path, dict[str, Path]]:
    """Find the run root and every stage spine under one experiment's run directory.

    ``experiment_dir`` may point at the run's top-level ``output_directory``
    (containing sibling ``raw_outputs/``, ``preprocess_adata_outputs/``, ...), or
    directly at one stage directory (e.g. ``raw_outputs/``, the original
    ``project add`` convention, or any other directory holding a ``spine.h5ad``)
    -- both resolve to the same run root and the same discovered stage spines, so
    registering an experiment doesn't depend on which one a caller happened to
    point at.

    Returns:
        ``(run_root, {stage_name: spine_path})`` for whichever stages exist.

    Raises:
        FileNotFoundError: If no stage spine is found under either interpretation.
    """
    experiment_dir = Path(experiment_dir)
    direct_spine = experiment_dir / SPINE_FILENAME
    if direct_spine.exists():
        # experiment_dir is one stage dir; its parent is the run root, and any
        # sibling stage dirs there are discovered too. If experiment_dir's name
        # doesn't match a known stage dir (e.g. it's not literally "raw_outputs"),
        # its own spine is still real and usable -- register it as "raw" by
        # default, matching the original single-spine registration behavior.
        run_root = experiment_dir.parent
        stage_name = next(
            (stage for stage, stage_dir in STAGE_DIRS.items() if stage_dir == experiment_dir.name),
            "raw",
        )
        found = {stage_name: direct_spine}
    else:
        # No spine directly inside -- treat experiment_dir itself as the run's
        # top-level output_directory and look for stage dirs inside it.
        run_root = experiment_dir
        found = {}

    found.update(
        {
            stage: candidate
            for stage, stage_dir in STAGE_DIRS.items()
            if stage not in found and (candidate := run_root / stage_dir / SPINE_FILENAME).exists()
        }
    )
    if not found:
        raise FileNotFoundError(
            f"no stage spine found under {experiment_dir} (looked for "
            f"{SPINE_FILENAME} directly inside it, and under "
            f"{sorted(STAGE_DIRS.values())})"
        )
    return run_root, found


def _relative_registry_path(path: Path, anchor: Path) -> str:
    """Return ``path`` as a POSIX-style string relative to ``anchor``, for ``registry.json``.

    Kept relative (not resolved to an absolute string) so the registry -- and the
    project directory it lives in -- survives being copied to a different machine
    or mount point, the same way the project's ``runs/`` symlinks already do.
    Pair with :func:`_resolve_registry_path` on the read side.
    """
    import os

    return Path(os.path.relpath(path.resolve(), start=anchor.resolve())).as_posix()


def _resolve_registry_path(value: str, anchor: Path) -> Path:
    """Resolve a ``registry.json`` path value written by :func:`_relative_registry_path`.

    Accepts both the new relative encoding (resolved against ``anchor``, the
    project dir) and the historical absolute-string encoding (used as-is, for
    registries written before this fix), so existing registries keep working
    without needing every experiment re-added.
    """
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (anchor / candidate).resolve()


def _discover_catalogs(spines: dict[str, Path], project_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    raw_dir = spines["raw"].parent if "raw" in spines else None
    if raw_dir is not None:
        for name in _CATALOG_NAMES:
            path = raw_dir / name
            if path.exists():
                found[name] = _relative_registry_path(path, project_dir)
    return found


# Legacy monolithic filename suffix -> stage, matching cli.helpers.get_adata_paths'
# naming (e.g. "<experiment>_preprocessed_duplicates_removed.h5ad.gz"). Checked
# longest/most-specific first. A bare "<experiment>.h5ad.gz" (no suffix) has no
# match and falls back to "raw" in infer_legacy_stage.
_LEGACY_STAGE_SUFFIXES = (
    ("_preprocessed_duplicates_removed", "preprocess"),
    ("_preprocessed", "preprocess"),
    ("_spatial", "spatial"),
    ("_hmm", "hmm"),
    ("_latent", "latent"),
    ("_variant", "variant"),
    ("_chimeric", "chimeric"),
)


def infer_legacy_stage(spine_path: Path) -> str:
    """Best-effort pipeline stage for a legacy monolithic h5ad, from its filename.

    Pass an explicit ``stage=`` to :func:`add_experiment` instead when a file
    doesn't follow the standard naming (or the guess is wrong).
    """
    stem = spine_path.name
    for suffix in (".h5ad.gz", ".h5ad"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for suffix, stage in _LEGACY_STAGE_SUFFIXES:
        if stem.endswith(suffix):
            return stage
    return "raw"


def _legacy_reference_uids(spine, ref_column: str = "Reference_strand") -> dict[str, str]:
    """Best-effort ``reference_uids`` for a spine, without mutating its source file.

    Modern spines already carry ``uns["reference_uids"]`` (a sequence-hash
    identity computed once at raw-ingestion time, see ``cli.raw_adata``) and are
    returned as-is. A legacy monolithic AnnData (pre-partitioned-store pipeline)
    predates that field, but still carries ``uns["References"][f"{chromosome}_
    FASTA_sequence"]`` -- the same reference-sequence storage the modern
    computation itself reads from -- so the identity hash can be computed here
    on the fly at registration time instead of requiring the source file to be
    rewritten. Not cached back into it: input data stays untouched, so this
    re-runs (cheaply -- it's a handful of sha256 hashes, not a matrix op) every
    time the experiment is (re-)registered.

    Caveat: without the exact reference length the modern computation trims to
    (see ``informatics.reference_identity.reference_uid``), this hashes the
    full stored sequence instead. That's internally consistent across legacy
    experiments, and matches modern uids in the common case where
    ``uns["References"]`` already stores the untrimmed canonical sequence, but
    isn't guaranteed to byte-for-byte match a modern uid for the same locus if
    trimming/padding conventions ever differed.
    """
    from ..informatics.reference_identity import reference_uid

    existing = dict(spine.uns.get("reference_uids", {}) or {})
    if existing:
        return {str(k): str(v) for k, v in existing.items()}

    sequences = dict(spine.uns.get("References", {}) or {})
    if not sequences or ref_column not in spine.obs:
        return {}

    uids: dict[str, str] = {}
    for reference_strand in spine.obs[ref_column].astype(str).unique():
        chromosome = str(reference_strand).rsplit("_", 1)[0]
        sequence = sequences.get(f"{chromosome}_FASTA_sequence")
        if sequence:
            uids[str(reference_strand)] = reference_uid(str(sequence))
    return uids


def _read_spine_metadata(spine_path: Path) -> dict:
    from ..readwrite import safe_read_h5ad

    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    uns = spine.uns
    return {
        "modality": str(uns.get("modality", "unknown")),
        "reference_uids": _legacy_reference_uids(spine),
        "experiment": uns.get("experiment"),
        "schema_version": int(uns.get("raw_schema_version", 0) or 0),
        "n_reads": int(spine.n_obs),
    }


def add_experiment(
    project_dir: str | Path,
    experiment_dir: str | Path,
    *,
    experiment_id: str | None = None,
    name: str | None = None,
    stage: str | None = None,
) -> tuple[str, dict]:
    """Register (or refresh) one experiment by pointer. Append-only; O(1).

    ``experiment_dir`` may be a directory or a file:

    - **Directory** (the common case): the run's top-level output directory, or
      one stage directory inside it (e.g. ``raw_outputs/``) -- see
      :func:`discover_stage_spines`. Every partitioned stage spine found under
      it is recorded.
    - **File**: one legacy monolithic ``.h5ad``/``.h5ad.gz``, from before the
      partitioned-store pipeline. ``stage`` names which pipeline stage it
      represents; omit it to best-effort infer from the filename (see
      :func:`infer_legacy_stage`). Register each legacy stage file for the same
      experiment with repeated calls (same resulting ``experiment_id``) --
      spines accumulate rather than replace, so nothing registered earlier is
      lost. The source file is only ever read here, never modified: reference
      identity for harmonization is computed on the fly instead of being
      cached back into it (see :func:`_legacy_reference_uids`).

    Either way, a later `project` query treats the result the same --
    ``materialize()`` detects a legacy spine (no ``uns["is_spine"]``) and reads
    it directly instead of through the partition machinery, so callers never
    need to know which kind of spine they got.

    Calling again with the same directory/experiment_id refreshes: newly
    discovered or newly registered stages merge into the existing entry rather
    than replacing it.
    """
    project_dir = Path(project_dir)
    experiment_path = Path(experiment_dir).resolve()
    registry = load_registry(project_dir)

    is_legacy_call = experiment_path.is_file()
    if is_legacy_call:
        stage_name = stage or infer_legacy_stage(experiment_path)
        spines = {stage_name: experiment_path}
        run_root = experiment_path.parent
    else:
        run_root, spines = discover_stage_spines(experiment_path)

    metadata_stage = (
        "raw" if "raw" in spines else next((s for s in STAGE_PRIORITY if s in spines), next(iter(spines)))
    )
    meta = _read_spine_metadata(spines[metadata_stage])
    exp_id = str(experiment_id or meta.get("experiment") or run_root.name)

    existing = registry["experiments"].get(exp_id)
    if existing and "spines" in existing and not is_legacy_call:
        # Directory-discovered entries are anchored at run_root, so a mismatch
        # means this id is already used by a different run -- refuse to merge.
        # Legacy (file) registrations are explicit and per-file, with no shared
        # root across repeated calls for the same experiment, so this check
        # doesn't apply there -- accumulating legacy stage files under one id
        # is the whole point.
        existing_root = _resolve_registry_path(existing["path"], project_dir)
        if existing_root != run_root:
            raise ValueError(
                f"experiment id '{exp_id}' already registered at {existing_root}; "
                "pass a distinct --id"
            )

    merged_spines = dict(existing.get("spines", {})) if existing else {}
    merged_spines.update(
        {stg: _relative_registry_path(path, project_dir) for stg, path in spines.items()}
    )
    merged_references = dict(existing.get("references", {})) if existing else {}
    merged_references.update(meta["reference_uids"])
    merged_catalogs = dict(existing.get("catalogs", {})) if existing else {}
    merged_catalogs.update(_discover_catalogs(spines, project_dir))

    entry = {
        "path": _relative_registry_path(run_root, project_dir),
        "name": name or (existing["name"] if existing else exp_id),
        "modality": meta["modality"],
        "schema_version": meta["schema_version"],
        "spines": merged_spines,
        "references": merged_references,
        "n_reads": meta["n_reads"],
        "date_added": existing["date_added"] if existing else _now(),
        "status": "active",
        "catalogs": merged_catalogs,
    }
    registry["experiments"][exp_id] = entry
    save_registry(project_dir, registry)
    logger.info(
        "Registered experiment '%s' (%d reads, %d references, stages: %s)",
        exp_id,
        entry["n_reads"],
        len(entry["references"]),
        ", ".join(sorted(merged_spines)),
    )
    return exp_id, entry


def remove_experiment(project_dir: str | Path, experiment_id: str) -> None:
    """Mark an experiment inactive (queries filter to active experiments)."""
    registry = load_registry(project_dir)
    if experiment_id in registry["experiments"]:
        registry["experiments"][experiment_id]["status"] = "inactive"
        save_registry(project_dir, registry)


def list_experiments(project_dir: str | Path, *, active_only: bool = True) -> list[dict]:
    """Return registered experiments with ``path``/``spines``/``catalogs`` resolved.

    ``registry.json`` stores these relative to ``project_dir`` (see
    :func:`_relative_registry_path`); this is the single point where callers get
    back concrete, directly-usable paths, so the relative on-disk encoding stays
    an implementation detail. Every entry always has a ``spines`` dict
    (``{stage_name: absolute_spine_path}``) regardless of registry schema version
    -- pre-schema-2 entries (single ``path`` + ``spine``, always the raw stage)
    are synthesized into the same shape here, so callers never need to branch on
    schema version.
    """
    project_dir = Path(project_dir)
    registry = load_registry(project_dir)
    results = []
    for exp_id, entry in registry["experiments"].items():
        if active_only and entry.get("status") != "active":
            continue
        resolved = dict(entry)
        resolved["path"] = str(_resolve_registry_path(entry["path"], project_dir))
        if "spines" in entry:
            resolved["spines"] = {
                stage: str(_resolve_registry_path(value, project_dir))
                for stage, value in entry["spines"].items()
            }
        else:
            resolved["spines"] = {
                "raw": str(Path(resolved["path"]) / entry.get("spine", SPINE_FILENAME))
            }
        resolved["catalogs"] = {
            name: str(_resolve_registry_path(value, project_dir))
            for name, value in entry.get("catalogs", {}).items()
        }
        results.append({"id": exp_id, **resolved})
    return results


def resolve_experiment_spine(entry: dict, stage: str | None = None) -> tuple[str, Path] | None:
    """Pick one stage's spine from a :func:`list_experiments` entry.

    Args:
        entry: One entry from :func:`list_experiments` (already has absolute
            paths in ``spines``).
        stage: Explicit stage name to require, or ``None`` to fall back through
            :data:`STAGE_PRIORITY` (most-derived available stage first).

    Returns:
        ``(stage_name, path)`` for the selected spine, or ``None`` if the
        requested stage (or, in fallback mode, no stage at all) is available.
    """
    spines = entry.get("spines", {})
    if stage is not None:
        path = spines.get(stage)
        return (stage, Path(path)) if path else None
    for candidate in STAGE_PRIORITY:
        path = spines.get(candidate)
        if path:
            return candidate, Path(path)
    return None


def add_set(
    project_dir: str | Path,
    name: str,
    *,
    experiments: list[str] | None = None,
    query: str | None = None,
) -> None:
    """Define a named set as an explicit experiment list OR a saved query."""
    if (experiments is None) == (query is None):
        raise ValueError("provide exactly one of experiments= or query=")
    registry = load_registry(project_dir)
    registry.setdefault("sets", {})[name] = (
        {"kind": "list", "experiments": [str(e) for e in experiments]}
        if experiments is not None
        else {"kind": "query", "sql": str(query)}
    )
    save_registry(project_dir, registry)


def resolve_set(project_dir: str | Path, name: str) -> dict:
    registry = load_registry(project_dir)
    the_set = registry.get("sets", {}).get(name)
    if the_set is None:
        raise KeyError(f"no set '{name}' in project")
    return the_set
