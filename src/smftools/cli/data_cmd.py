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
    """Scan one or more stamped volumes and merge what's found into the catalog.

    `mounts` defaults to every volume `data volumes` currently finds attached.
    Persists the updated catalog before returning.
    """
    from ..data.replica_catalog import load_catalog, save_catalog
    from ..data.volume_discovery import discover_volumes
    from ..data.volume_scan import scan_and_catalog

    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    if mounts:
        targets = [Path(mount) for mount in mounts]
    else:
        targets = [found.mount_path for found in discover_volumes(config_dir=resolved_config_dir)]

    catalog = load_catalog(catalog_path)
    scanned = []
    for mount in targets:
        catalog, runs = scan_and_catalog(mount, catalog)
        scanned.append(
            {
                "mount": str(mount),
                "runs": [
                    {"path": run.relative_path, "digest": run.digest, "warning": run.warning}
                    for run in runs
                ],
            }
        )
    save_catalog(catalog, path=catalog_path)
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
