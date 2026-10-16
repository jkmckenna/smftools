"""Apply a generation pruning plan: archive or delete (`SLC-08`).

Only `prune_candidate` generations of a fresh plan are touched -- not current,
not pinned, not read (transitively) by a live generation, not named by a
``--protect-from`` file. Each is re-checked against a fresh reference graph
immediately before it moves. While applying, the experiment's workflow lock is
held, and nothing starts while a workflow holds it or a stage has staging
content (a stage may be running).

``archive`` copies the generation to ``<archive_dir>/<experiment root
name>/<stage dir>/generations/<id>``, verifies file count and bytes, then
removes the original. ``delete`` removes it; the ids must be confirmed. Either
way a stub stays in place -- the generation's manifest plus ``pruned.json``
(action, when, why, size, archive path) -- so inventories show what happened
and a later read fails with that reason. :func:`restore_generation` reverses an
archive.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from smftools.readwrite import atomic_write_json

from .generation_listing import (
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    PRUNED_RECORD,
    STAGE_GENERATION_DIRS,
    STAGING_SUBDIR,
    list_experiment_generations,
)
from .generation_pruning import (
    PRUNE_CANDIDATE,
    GenerationPruneError,
    plan_experiment_generation_prune,
)
from .generation_references import build_reference_graph, external_references, protected_generations

PRUNED_SCHEMA_VERSION = 1
WORKFLOW_RUNTIME_DIRECTORY = ".smftools-workflow"
WORKFLOW_LOCK_FILENAME = "run.lock"
_ASIDE = ".pruning"  # per-container holding area while a generation is removed


class PruneApplyError(GenerationPruneError):
    """A prune cannot be applied safely."""


@dataclass(frozen=True)
class PruneAction:
    kind: str
    generation_id: str
    action: str
    size_bytes: int
    archive_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _container(output_root: Path, kind: str) -> Path:
    return output_root / STAGE_GENERATION_DIRS[kind]


def _busy(output_root: Path) -> list[str]:
    """Signs that a workflow or stage is running in this experiment."""
    reasons = []
    lock = output_root / WORKFLOW_RUNTIME_DIRECTORY / WORKFLOW_LOCK_FILENAME
    if lock.exists():
        reasons.append(f"workflow lock {lock} exists")
    for directory in STAGE_GENERATION_DIRS.values():
        for staging in (
            output_root / directory / STAGING_SUBDIR,
            output_root / directory / GENERATIONS_SUBDIR / STAGING_SUBDIR,
        ):
            if staging.is_dir() and any(staging.iterdir()):
                reasons.append(f"{staging} has staging content (a stage may be running)")
    return reasons


@contextmanager
def _experiment_lock(output_root: Path) -> Iterator[None]:
    busy = _busy(output_root)
    if busy:
        raise PruneApplyError("refusing to prune: " + "; ".join(busy))
    runtime = output_root / WORKFLOW_RUNTIME_DIRECTORY
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / WORKFLOW_LOCK_FILENAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PruneApplyError(f"refusing to prune: {lock} appeared") from exc
    try:
        os.write(descriptor, f"{os.getpid()} generations prune\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _inventory(path: Path) -> tuple[int, int]:
    """(regular files, bytes) under ``path``; symlinks counted, not followed."""
    files = size = 0
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            files += 1
            if not os.path.islink(full):
                size += os.lstat(full).st_size
    return files, size


def _still_candidate(
    output_root: Path, kind: str, generation_id: str, protect_from: Sequence[Path]
) -> str | None:
    """``None`` if nothing live reads the generation now, else why it must stay."""
    graph = build_reference_graph(output_root)
    if graph.unreadable:
        return "some generation's references cannot be read"
    records = list_experiment_generations(output_root, include_size=False)
    roots = {
        (r.kind, r.generation_id): ("current" if r.is_current else "pinned")
        for r in records
        if r.is_current or r.pinned
    }
    roots.update(external_references(protect_from, graph))
    key = (kind, generation_id)
    if key in roots:
        return roots[key]
    reached = protected_generations(graph, roots)
    return reached.get(key)


def _stub(generation: Path, record: dict) -> None:
    """Replace a removed generation's directory with its manifest and record."""
    generation.mkdir()
    atomic_write_json(generation / PRUNED_RECORD, record)


def apply_prune(
    output_root: str | Path,
    *,
    keep_last: int | None = None,
    older_than: str | None = None,
    stages: Iterable[str] | None = None,
    protect_from: Iterable[str | Path] = (),
    generation_ids: Sequence[str] = (),
    archive_dir: str | Path | None = None,
    delete: bool = False,
    confirm: Sequence[str] = (),
    allow_same_volume: bool = False,
) -> list[PruneAction]:
    """Archive or delete the plan's prune candidates (optionally only
    ``generation_ids``). ``delete`` needs every target id in ``confirm``."""
    output_root = Path(output_root)
    protect = [Path(p) for p in protect_from]
    if (archive_dir is None) == (not delete):
        raise PruneApplyError("choose exactly one of archive or delete")
    with _experiment_lock(output_root):
        plan = plan_experiment_generation_prune(
            output_root,
            keep_last=keep_last,
            older_than=older_than,
            stages=stages,
            protect_from=protect,
        )
        candidates = {
            d.generation_id: d for d in plan.decisions if d.disposition == PRUNE_CANDIDATE
        }
        wanted = list(generation_ids) or sorted(candidates)
        not_candidates = sorted(set(wanted) - set(candidates))
        if not_candidates:
            raise PruneApplyError(f"not prune candidates in a fresh plan: {not_candidates}")
        if delete:
            unconfirmed = sorted(set(wanted) - set(confirm))
            if unconfirmed:
                raise PruneApplyError(
                    f"deleting needs every id in --confirm; unconfirmed: {unconfirmed}"
                )
        targets = [candidates[gid] for gid in wanted]
        total = sum(d.size_bytes or 0 for d in targets)
        archive_root = None
        if archive_dir is not None:
            archive_root = Path(archive_dir).resolve() / output_root.resolve().name
            archive_root.mkdir(parents=True, exist_ok=True)
            if (
                not allow_same_volume
                and os.stat(archive_root).st_dev == os.stat(output_root).st_dev
            ):
                raise PruneApplyError(
                    f"archive {archive_root} is on the same volume as the experiment: moving "
                    "there frees no space (allow_same_volume to do it anyway)"
                )
            free = shutil.disk_usage(archive_root).free
            if free < total:
                raise PruneApplyError(
                    f"archive volume has {free} bytes free; the generations need {total}"
                )
        actions = []
        for decision in targets:
            why = _still_candidate(output_root, decision.kind, decision.generation_id, protect)
            if why is not None:
                raise PruneApplyError(
                    f"{decision.kind} {decision.generation_id} is no longer a candidate ({why}); "
                    f"{len(actions)} generation(s) were pruned before it"
                )
            actions.append(_prune_one(output_root, decision, archive_root))
        return actions


def _prune_one(output_root: Path, decision, archive_root: Path | None) -> PruneAction:
    container = _container(output_root, decision.kind)
    generation = container / GENERATIONS_SUBDIR / decision.generation_id
    manifest = json.loads((generation / GENERATION_MANIFEST).read_text(encoding="utf-8"))
    files, size = _inventory(generation)
    destination = None
    if archive_root is not None:
        destination = (
            archive_root
            / STAGE_GENERATION_DIRS[decision.kind]
            / GENERATIONS_SUBDIR
            / decision.generation_id
        )
        if destination.exists():
            raise PruneApplyError(f"archive destination already exists: {destination}")
        partial = destination.with_name(destination.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(generation, partial, symlinks=True)
        if _inventory(partial) != (files, size):
            shutil.rmtree(partial, ignore_errors=True)
            raise PruneApplyError(f"archive copy of {generation} does not match; nothing removed")
        partial.rename(destination)
    record = {
        "schema_version": PRUNED_SCHEMA_VERSION,
        "action": "archived" if destination is not None else "deleted",
        "kind": decision.kind,
        "generation_id": decision.generation_id,
        "pruned_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size,
        "files": files,
        "reasons": list(decision.reasons),
        "archive_path": destination.as_posix() if destination is not None else None,
    }
    if destination is not None:
        atomic_write_json(
            destination / PRUNED_RECORD.replace(".json", "_source.json"),
            {
                **record,
                "source_path": generation.resolve().as_posix(),
            },
        )
    aside = container / _ASIDE / decision.generation_id
    aside.parent.mkdir(exist_ok=True)
    generation.rename(aside)  # same volume: atomic; the stub replaces it next
    _stub(generation, record)
    atomic_write_json(generation / GENERATION_MANIFEST, manifest)
    shutil.rmtree(aside)
    if not any(aside.parent.iterdir()):
        aside.parent.rmdir()
    return PruneAction(
        kind=decision.kind,
        generation_id=decision.generation_id,
        action=record["action"],
        size_bytes=size,
        archive_path=record["archive_path"],
    )


def restore_generation(output_root: str | Path, kind: str, generation_id: str) -> Path:
    """Bring an archived generation back into place (verified copy, then the
    archive copy is removed)."""
    output_root = Path(output_root)
    if kind not in STAGE_GENERATION_DIRS:
        raise PruneApplyError(f"unknown stage {kind!r}")
    container = _container(output_root, kind)
    generation = container / GENERATIONS_SUBDIR / generation_id
    record_path = generation / PRUNED_RECORD
    if not record_path.is_file():
        raise PruneApplyError(f"{kind} {generation_id} is not a pruned stub")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("action") != "archived" or not record.get("archive_path"):
        raise PruneApplyError(f"{kind} {generation_id} was deleted, not archived")
    archive = Path(record["archive_path"])
    if not archive.is_dir():
        raise PruneApplyError(f"archive copy not found: {archive}")
    with _experiment_lock(output_root):
        incoming = container / _ASIDE / f"{generation_id}.restore"
        incoming.parent.mkdir(exist_ok=True)
        shutil.rmtree(incoming, ignore_errors=True)
        shutil.copytree(archive, incoming, symlinks=True)
        (incoming / PRUNED_RECORD.replace(".json", "_source.json")).unlink(missing_ok=True)
        expected = (record["files"], record["size_bytes"])
        if _inventory(incoming) != expected:
            shutil.rmtree(incoming, ignore_errors=True)
            raise PruneApplyError(f"restored copy of {archive} does not match its record")
        shutil.rmtree(generation)
        incoming.rename(generation)
        if not any(incoming.parent.iterdir()):
            incoming.parent.rmdir()
        shutil.rmtree(archive)
    return generation
