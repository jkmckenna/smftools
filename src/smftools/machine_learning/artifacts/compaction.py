"""Compact stored model payloads losslessly (`SLC-02`).

skops payloads are zip archives that were written uncompressed; their
members (a fitted forest's node arrays above all) deflate several-fold.
Compaction rewrites a payload with every member deflated, after checking that
each member's bytes are identical to the original's.

A model's identity and manifest stay as published: the manifest's payload
reference (sha256, size) is the original encoding. The workspace records each
compacted payload in ``index/compacted_models.json`` (original and stored
digest and size, per-member content digests, encoding), and bundle
validation accepts a payload matching either the manifest or that record.

Order per model, so every crash point leaves a valid bundle: write the
compacted copy outside the bundle and verify it member by member; add its
record; replace the payload atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from smftools.readwrite import atomic_write_json

from ..workspace import MLWorkspace
from .indexing import _published_directories
from .publication import MODEL_MANIFEST_FILENAME, _publication_lock, file_sha256

COMPACTED_MODELS_FILENAME = "compacted_models.json"
COMPACTION_SCHEMA_VERSION = 1
ZIP_MEDIA_SUFFIX = "+zip"
_STAGING = ".compaction"
_ENCODING = "zip-deflate-6"


class ModelCompactionError(RuntimeError):
    """A payload cannot be compacted safely."""


@dataclass(frozen=True)
class CompactionCandidate:
    model_id: str
    payload: Path
    original_sha256: str
    original_size: int


@dataclass
class CompactionPlan:
    workspace_root: str
    candidates: list[CompactionCandidate] = field(default_factory=list)
    already_compacted: int = 0
    not_zip: int = 0
    unreadable: list[str] = field(default_factory=list)

    @property
    def candidate_bytes(self) -> int:
        return sum(c.original_size for c in self.candidates)

    def to_dict(self) -> dict:
        return {
            "schema_version": COMPACTION_SCHEMA_VERSION,
            "workspace_root": self.workspace_root,
            "candidates": len(self.candidates),
            "candidate_bytes": self.candidate_bytes,
            "already_compacted": self.already_compacted,
            "not_zip": self.not_zip,
            "unreadable": list(self.unreadable),
        }


def _index_path(workspace: MLWorkspace) -> Path:
    return workspace.root / "index" / COMPACTED_MODELS_FILENAME


def compacted_models(workspace: MLWorkspace) -> dict[str, dict]:
    """model_id -> compaction record (empty when nothing was compacted)."""
    path = _index_path(workspace)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {record["model_id"]: record for record in payload.get("records", [])}


def _write_index(workspace: MLWorkspace, records: dict[str, dict]) -> None:
    path = _index_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "schema_version": COMPACTION_SCHEMA_VERSION,
            "records": [records[key] for key in sorted(records)],
        },
    )


def _uncompressed(path: Path) -> bool | None:
    try:
        with zipfile.ZipFile(path) as archive:
            return all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return None


def plan_compaction(workspace: MLWorkspace) -> CompactionPlan:
    """Model payloads still stored uncompressed. Writes nothing."""
    plan = CompactionPlan(workspace_root=str(workspace.root))
    for bundle in _published_directories(workspace.models_root):
        manifest_path = bundle / MODEL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = manifest["artifact"]
            model_id = str(manifest["model_id"])
        except (OSError, ValueError, KeyError):
            plan.unreadable.append(bundle.name)
            continue
        # Judged by the payload's own encoding, not the index: a compaction
        # interrupted after its record but before the swap is offered again.
        if not str(artifact.get("media_type", "")).endswith(ZIP_MEDIA_SUFFIX):
            plan.not_zip += 1
            continue
        payload = bundle / str(artifact["relative_path"])
        state = _uncompressed(payload)
        if state is None:
            plan.unreadable.append(model_id)
        elif state:
            plan.candidates.append(
                CompactionCandidate(
                    model_id=model_id,
                    payload=payload,
                    original_sha256=str(artifact["sha256"]),
                    original_size=int(artifact["size_bytes"]),
                )
            )
        else:
            plan.already_compacted += 1  # written compressed (SLC-01)
    return plan


def _member_digests(path: Path) -> dict[str, str]:
    digests = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            digests[item.filename] = hashlib.sha256(archive.read(item)).hexdigest()
    return digests


def prepare_compacted(workspace: MLWorkspace, candidate: CompactionCandidate) -> dict:
    """Write and verify the compacted copy outside the bundle (safe to run in
    parallel); returns its record. Nothing in the bundle or index changes."""
    staging = workspace.root / _STAGING
    staging.mkdir(parents=True, exist_ok=True)
    partial = staging / f"{candidate.model_id}.skops"
    if file_sha256(candidate.payload) != candidate.original_sha256:
        raise ModelCompactionError(
            f"{candidate.model_id}: payload does not match its manifest; not compacted"
        )
    original_members = _member_digests(candidate.payload)
    with (
        zipfile.ZipFile(candidate.payload) as source,
        zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as target,
    ):
        for item in source.infolist():
            info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = item.external_attr
            target.writestr(info, source.read(item))
    if _member_digests(partial) != original_members:
        partial.unlink(missing_ok=True)
        raise ModelCompactionError(f"{candidate.model_id}: compacted members differ")
    return {
        "model_id": candidate.model_id,
        "encoding": _ENCODING,
        "original_sha256": candidate.original_sha256,
        "original_size": candidate.original_size,
        "stored_sha256": file_sha256(partial),
        "stored_size": partial.stat().st_size,
        "member_sha256": original_members,
        "compacted_at": datetime.now(timezone.utc).isoformat(),
        "partial": partial.as_posix(),
    }


def commit_compacted(workspace: MLWorkspace, candidate: CompactionCandidate, record: dict) -> dict:
    """Record a prepared copy, then swap it in -- under the model's lock, after
    re-checking that the payload is still the original."""
    partial = Path(record.pop("partial"))
    with _publication_lock(workspace.models_root, candidate.model_id, 600.0):
        if file_sha256(candidate.payload) != candidate.original_sha256:
            partial.unlink(missing_ok=True)
            raise ModelCompactionError(f"{candidate.model_id}: payload changed; not compacted")
        records = compacted_models(workspace)
        records[candidate.model_id] = record
        _write_index(workspace, records)  # before the swap: both encodings validate
        _replace_payload(partial, candidate.payload)
    return record


def compact_payload(workspace: MLWorkspace, candidate: CompactionCandidate) -> dict:
    """Compact one payload; returns its compaction record."""
    return commit_compacted(workspace, candidate, prepare_compacted(workspace, candidate))


def _replace_payload(partial: Path, payload: Path) -> None:
    """Swap in the compacted payload (atomic: same filesystem)."""
    os.replace(partial, payload)


def apply_compaction(
    workspace: MLWorkspace,
    plan: CompactionPlan,
    *,
    limit: int | None = None,
    workers: int = 1,
    progress=None,
) -> list[dict]:
    """Compact the plan's candidates (the first ``limit``): copies are prepared
    by ``workers`` processes, committed one at a time by this one."""
    candidates = plan.candidates[: limit if limit is not None else None]
    done: list[dict] = []
    if workers <= 1:
        for candidate in candidates:
            done.append(compact_payload(workspace, candidate))
            if progress is not None:
                progress(len(done), len(candidates))
        return done
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(prepare_compacted, workspace, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            done.append(commit_compacted(workspace, futures[future], future.result()))
            if progress is not None:
                progress(len(done), len(candidates))
    return done


def accepted_digests(workspace: MLWorkspace, model_id: str) -> Iterable[tuple[str, int]]:
    """Extra (sha256, size) a model payload may have: its compacted encoding."""
    record = compacted_models(workspace).get(model_id)
    if record is None:
        return ()
    return ((record["stored_sha256"], int(record["stored_size"])),)
