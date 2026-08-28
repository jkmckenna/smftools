"""CLI logic for `smftools data`: machine- and volume-scoped storage operations."""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def data_init_volume(mount: str | Path, *, label: str, kind: str) -> tuple[dict, bool, list[str]]:
    """Stamp `mount` with a permanent volume identity.

    Returns `(stamp_dict, created, warnings)`. `created` is False when the
    volume already carried a stamp, in which case `warnings` names any
    requested `label`/`kind` that differs from what is already stamped --
    the stamp is never rewritten (`PSR-08`), so the request is honored by
    reporting the mismatch rather than by changing the file.
    """
    from ..data.volume_stamp import init_volume

    stamp, created = init_volume(mount, label=label, kind=kind)
    warnings: list[str] = []
    if not created:
        if stamp.label != label:
            warnings.append(
                f"requested label {label!r} ignored; volume is already labeled {stamp.label!r}"
            )
        if stamp.kind != kind:
            warnings.append(
                f"requested kind {kind!r} ignored; volume is already kind {stamp.kind!r}"
            )
    return stamp.to_dict(), created, warnings


def data_list_volumes(*, config_dir: str | Path | None = None) -> list[dict]:
    """Every stamped volume currently attached to this machine, as plain dicts.

    Scans platform mount roots plus any configured `extra_search_paths`
    (`PSR-09`). Reports only what is attached right now -- a volume that is
    stamped but not currently reachable is invisible here; nothing in `PSR`
    tracks "every volume ever stamped" as its own registry. `data locate`
    (`PSR-11`) can still name a *detached* volume by `volume_id` when the
    replica catalog references it.
    """
    from ..data.volume_discovery import discover_volumes

    found = discover_volumes(config_dir=Path(config_dir) if config_dir is not None else None)
    return [{**item.stamp.to_dict(), "mount_path": str(item.mount_path)} for item in found]


def data_roots_list(*, config_dir: str | Path | None = None) -> list[dict]:
    """Every named root bound on this machine, as plain dicts.

    Resolution order: `SMFTOOLS_ROOT_<NAME>` env vars, the user roots file,
    then any `roots.toml` walked up from `config_dir` -- nearest layer
    winning per name, so a root bound at more than one layer is only
    reported once, from whichever layer actually supplied it (`source`).
    """
    from ..config.roots import known_roots

    bindings = known_roots(config_dir=Path(config_dir) if config_dir is not None else None)
    return [
        {
            "name": name,
            "path": str(binding.path),
            "source": binding.source,
            "all_paths": [str(path) for path in binding.all_paths],
        }
        for name, binding in sorted(bindings.items())
    ]


def _resolve_dataset_digest(target: str) -> str:
    """A dataset digest from `target`: a literal digest, a run root, or a manifest path.

    Accepts, in order: a path to a `resolved_input_manifest.json`; a run root
    directory that has one at its conventional location; or a bare 64-character
    hex sha256 digest.
    """
    from ..data.volume_verify import manifest_path_for
    from ..informatics.input_manifest import (
        RESOLVED_INPUT_MANIFEST_JSON,
        read_resolved_input_manifest,
    )

    candidate = Path(target)
    if candidate.is_file() and candidate.name == RESOLVED_INPUT_MANIFEST_JSON:
        return read_resolved_input_manifest(candidate).digest
    if candidate.is_dir():
        manifest_path = manifest_path_for(candidate)
        if manifest_path.is_file():
            return read_resolved_input_manifest(manifest_path).digest
        raise ValueError(f"no resolved input manifest found under {candidate}")
    text = str(target).strip().lower()
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
        return text
    raise ValueError(
        f"{target!r} is neither a run root directory, a resolved_input_manifest.json path, "
        "nor a 64-character sha256 digest"
    )


def data_scan(
    mounts: list[str] | None,
    *,
    config_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict:
    """Scan one or more stamped volumes and merge what's found into both catalogs.

    `mounts` defaults to every volume `data volumes` currently finds attached.
    Updates the replica catalog (raw dataset replicas, `PSR-10`/`PSR-11`) at
    `catalog_path` and the analysis-location catalog (`PSR-19`) at its own
    default location -- the two are unrelated files with unrelated keys
    (dataset digest vs. `experiment_uid`), so one override does not reach
    both; isolate both together in tests via `SMFTOOLS_CONFIG_DIR`.
    """
    from ..data.analysis_catalog import load_catalog as load_analysis_catalog
    from ..data.analysis_catalog import save_catalog as save_analysis_catalog
    from ..data.replica_catalog import load_catalog, save_catalog
    from ..data.volume_discovery import discover_volumes
    from ..data.volume_scan import scan_and_catalog, scan_and_catalog_analysis_locations

    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    if mounts:
        targets = [Path(mount) for mount in mounts]
    else:
        targets = [found.mount_path for found in discover_volumes(config_dir=resolved_config_dir)]

    catalog = load_catalog(catalog_path)
    analysis_catalog = load_analysis_catalog()
    scanned = []
    for mount in targets:
        catalog, runs = scan_and_catalog(mount, catalog)
        analysis_catalog, analysis_runs = scan_and_catalog_analysis_locations(
            mount, analysis_catalog
        )
        scanned.append(
            {
                "mount": str(mount),
                "runs": [
                    {"path": run.relative_path, "digest": run.digest, "warning": run.warning}
                    for run in runs
                ],
                "analysis_locations": [
                    {
                        "path": run.relative_path,
                        "experiment_uid": run.experiment_uid,
                        "warning": run.warning,
                    }
                    for run in analysis_runs
                ],
            }
        )
    save_catalog(catalog, path=catalog_path)
    save_analysis_catalog(analysis_catalog)
    return {"scanned": scanned}


def data_locate(
    target: str,
    *,
    config_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict:
    """Every catalogued replica of `target`'s dataset, and which are attached."""
    from ..data.replica_catalog import load_catalog, replicas_for
    from ..data.volume_discovery import discover_volumes

    digest = _resolve_dataset_digest(target)
    catalog = load_catalog(catalog_path)
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    attached_by_id = {
        found.stamp.volume_id: found for found in discover_volumes(config_dir=resolved_config_dir)
    }

    rows = []
    for replica in replicas_for(catalog, digest):
        found = attached_by_id.get(replica.volume_id)
        rows.append(
            {
                "volume_id": replica.volume_id,
                "path": replica.path,
                "digest": replica.digest,
                "verified_at": replica.verified_at,
                "attached": found is not None,
                "label": found.stamp.label if found is not None else None,
                "resolved_path": str(found.mount_path / replica.path)
                if found is not None
                else None,
            }
        )
    return {"dataset_digest": digest, "replicas": rows}


def data_verify(
    target: str,
    *,
    volume_id: str | None = None,
    config_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict:
    """Re-checksum every currently-attached replica of `target`'s dataset.

    Restricted to one replica with `volume_id`. A replica whose volume is not
    attached is reported as such rather than silently skipped.
    """
    from ..data.replica_catalog import ResolvedReplica, load_catalog, replicas_for
    from ..data.volume_discovery import discover_volumes
    from ..data.volume_verify import verify_replica
    from ..informatics.input_manifest import InputManifestError

    digest = _resolve_dataset_digest(target)
    catalog = load_catalog(catalog_path)
    candidates = replicas_for(catalog, digest)
    if volume_id is not None:
        candidates = [replica for replica in candidates if replica.volume_id == volume_id]
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    attached_by_id = {
        found.stamp.volume_id: found for found in discover_volumes(config_dir=resolved_config_dir)
    }

    results = []
    for replica in candidates:
        found = attached_by_id.get(replica.volume_id)
        if found is None:
            results.append({"volume_id": replica.volume_id, "status": "not_attached"})
            continue
        resolved = ResolvedReplica(replica=replica, mount_path=found.mount_path)
        try:
            outcome = verify_replica(resolved)
        except InputManifestError as exc:
            results.append(
                {
                    "volume_id": replica.volume_id,
                    "status": "manifest_unreadable",
                    "detail": str(exc),
                }
            )
            continue
        results.append(
            {
                "volume_id": replica.volume_id,
                "status": "ok" if outcome.ok else "mismatch",
                "checked": len(outcome.rows),
                "mismatches": outcome.mismatch_count,
                "unreachable": outcome.unreachable_count,
                "rows": [
                    {
                        "path": row.path,
                        "status": row.status,
                        "expected_sha256": row.expected_sha256,
                        "actual_sha256": row.actual_sha256,
                    }
                    for row in outcome.rows
                    if row.status != "ok"
                ],
            }
        )
    return {"dataset_digest": digest, "results": results}


def data_localize(
    config_path: str | Path, *, apply: bool, out_config_path: str | Path | None = None
) -> dict:
    """Preview, or apply, localizing `config_path`'s small referenced inputs.

    Always computes the plan; `apply=False` (the default at the CLI) stops
    there. `apply=True` copies the files and writes a new config -- the
    original is never modified.
    """
    from ..data.localize import apply_localize_plan, build_localize_plan

    plan = build_localize_plan(config_path)
    result = {
        "config_path": str(plan.config_path),
        "output_directory": str(plan.output_directory),
        "items": [
            {
                "field": item.field,
                "source": str(item.source),
                "size_bytes": item.size_bytes,
                "destination": str(item.destination),
            }
            for item in plan.items
        ],
        "total_bytes": plan.total_bytes,
        "applied": False,
    }
    if apply:
        new_config_path, copied = apply_localize_plan(plan, out_config_path=out_config_path)
        result["applied"] = True
        result["new_config_path"] = str(new_config_path)
        result["copied_fields"] = [item.field for item in copied]
    return result


def data_init(
    lab_root: str | Path, *, stamp_volume: bool, label: str | None, kind: str
) -> tuple[list[str], tuple[dict, bool] | None]:
    """Scaffold `lab_root` (`data/` + `analyses/{runs,projects}/`).

    Returns `(created_paths, stamp_result)`. `stamp_result` is `None` when
    `stamp_volume` is False; otherwise `(stamp_dict, created)`, matching
    `data_init_volume`'s return shape.
    """
    from ..data.lab_init import scaffold_lab_root

    created = [str(path) for path in scaffold_lab_root(lab_root)]

    stamp_result: tuple[dict, bool] | None = None
    if stamp_volume:
        from ..data.volume_stamp import init_volume

        stamp, was_created = init_volume(
            lab_root, label=label or Path(lab_root).resolve().name, kind=kind
        )
        stamp_result = (stamp.to_dict(), was_created)
    return created, stamp_result


def _resolve_experiment_uid(target: str) -> str:
    """An experiment_uid from `target`: a run root directory, or a literal uid."""
    import uuid

    from ..data.run_locality import run_identity

    candidate = Path(target)
    if candidate.is_dir():
        uid = run_identity(candidate)
        if uid is not None:
            return uid
        raise ValueError(f"no experiment_uid recorded under {candidate}")
    try:
        return str(uuid.UUID(str(target).strip()))
    except ValueError:
        raise ValueError(
            f"{target!r} is neither a run root directory nor a valid experiment_uid"
        ) from None


def _one_run_status(
    experiment_uid: str,
    *,
    attached_by_id: dict,
    analysis_catalog,
    replica_catalog,
) -> dict:
    from ..data.analysis_catalog import locations_for as analysis_locations_for
    from ..data.replica_catalog import replicas_for
    from ..data.run_locality import compare_run_locations
    from ..data.volume_verify import manifest_path_for
    from ..informatics.input_manifest import InputManifestError, read_resolved_input_manifest

    locations = analysis_locations_for(analysis_catalog, experiment_uid)
    location_rows = []
    attached_roots = []  # (AnalysisLocation, resolved run root)
    for location in locations:
        found = attached_by_id.get(location.volume_id)
        row = {
            "volume_id": location.volume_id,
            "path": location.path,
            "scanned_at": location.scanned_at,
            "attached": found is not None,
        }
        if found is not None:
            run_root = found.mount_path / location.path
            row["resolved_path"] = str(run_root)
            attached_roots.append((location, run_root))
        location_rows.append(row)

    comparisons = []
    if len(attached_roots) >= 2:
        primary_location, primary_root = attached_roots[0]
        for location, run_root in attached_roots[1:]:
            comparison = compare_run_locations(primary_root, run_root)
            comparisons.append(
                {
                    "a": primary_location.volume_id,
                    "b": location.volume_id,
                    "stages": [
                        {
                            "kind": stage.kind,
                            "state": stage.state,
                            "a_only": list(stage.a_only),
                            "b_only": list(stage.b_only),
                        }
                        for stage in comparison.stages
                    ],
                }
            )

    raw = None
    if attached_roots:
        _, primary_root = attached_roots[0]
        manifest_path = manifest_path_for(primary_root)
        if manifest_path.is_file():
            try:
                manifest = read_resolved_input_manifest(manifest_path)
            except InputManifestError:
                manifest = None
            if manifest is not None:
                replicas = replicas_for(replica_catalog, manifest.digest)
                raw = {
                    "digest": manifest.digest,
                    "replicas": [
                        {
                            "volume_id": replica.volume_id,
                            "path": replica.path,
                            "attached": replica.volume_id in attached_by_id,
                        }
                        for replica in replicas
                    ],
                }

    return {
        "experiment_uid": experiment_uid,
        "locations": location_rows,
        "comparisons": comparisons,
        "raw": raw,
    }


def data_status(
    targets: list[str] | None,
    *,
    config_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict:
    """Where every known run's data and analyses are, attached or not.

    `targets` (run roots or bare `experiment_uid`s) restricts the report;
    omitted, every run in the analysis-location catalog is reported.
    """
    from ..data.analysis_catalog import load_catalog as load_analysis_catalog
    from ..data.replica_catalog import load_catalog as load_replica_catalog
    from ..data.volume_discovery import discover_volumes

    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    attached_by_id = {
        found.stamp.volume_id: found for found in discover_volumes(config_dir=resolved_config_dir)
    }
    analysis_catalog = load_analysis_catalog()
    replica_catalog = load_replica_catalog(catalog_path)

    if targets:
        experiment_uids = [_resolve_experiment_uid(target) for target in targets]
    else:
        experiment_uids = sorted(analysis_catalog)

    runs = [
        _one_run_status(
            uid,
            attached_by_id=attached_by_id,
            analysis_catalog=analysis_catalog,
            replica_catalog=replica_catalog,
        )
        for uid in experiment_uids
    ]
    return {"runs": runs}


def data_sync(
    target: str,
    *,
    from_volume: str | None = None,
    to_volume: str | None = None,
    dry_run: bool = False,
    config_dir: str | Path | None = None,
) -> dict:
    """Additively sync generations between two attached locations of a run.

    `from_volume`/`to_volume` name which two catalogued locations to sync,
    by `volume_id`; both or neither. With neither, exactly two of the run's
    catalogued locations must currently be attached, or the request is
    refused rather than guessing which pair was meant.
    """
    from ..data.analysis_catalog import load_catalog as load_analysis_catalog
    from ..data.analysis_catalog import locations_for
    from ..data.run_sync import sync_run_locations
    from ..data.volume_discovery import discover_volumes

    if (from_volume is None) != (to_volume is None):
        raise ValueError("--from and --to must be given together, or neither")

    experiment_uid = _resolve_experiment_uid(target)
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    attached_by_id = {
        found.stamp.volume_id: found for found in discover_volumes(config_dir=resolved_config_dir)
    }
    analysis_catalog = load_analysis_catalog()
    locations = locations_for(analysis_catalog, experiment_uid)
    attached = [
        (location, attached_by_id[location.volume_id])
        for location in locations
        if location.volume_id in attached_by_id
    ]

    if from_volume is not None:
        by_id = {location.volume_id: (location, found) for location, found in attached}
        for volume_id in (from_volume, to_volume):
            if volume_id not in by_id:
                raise ValueError(
                    f"volume {volume_id!r} is not an attached, catalogued analysis location "
                    "of this run"
                )
        location_a, found_a = by_id[from_volume]
        location_b, found_b = by_id[to_volume]
    else:
        if len(attached) != 2:
            raise ValueError(
                f"expected exactly 2 attached analysis locations to sync, found {len(attached)}; "
                "disambiguate with --from/--to"
            )
        (location_a, found_a), (location_b, found_b) = attached

    root_a = found_a.mount_path / location_a.path
    root_b = found_b.mount_path / location_b.path
    result = sync_run_locations(root_a, root_b, dry_run=dry_run)

    return {
        "experiment_uid": experiment_uid,
        "location_a": {"volume_id": location_a.volume_id, "path": str(root_a)},
        "location_b": {"volume_id": location_b.volume_id, "path": str(root_b)},
        "dry_run": dry_run,
        "stages": [
            {
                "kind": stage.kind,
                "state": stage.state,
                "copied_a_to_b": list(stage.copied_a_to_b),
                "copied_b_to_a": list(stage.copied_b_to_a),
                "skipped_reason": stage.skipped_reason,
            }
            for stage in result.stages
        ],
    }


def data_archive_basecall(run_root: str | Path, *, archive_root: str | Path) -> dict:
    """Write `run_root`'s current basecall generation back to `archive_root` (`BCS-08`)."""
    from ..data.basecall_archive import archive_basecall_generation

    result = archive_basecall_generation(run_root, archive_root=archive_root)
    return {**result, "path": str(result["path"])}
