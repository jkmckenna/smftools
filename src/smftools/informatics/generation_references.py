"""Which generations a generation reads (`SLC-06`).

A published generation is not always self-contained: an annotation refresh
keeps its parent's rows, a raw refresh keeps an older raw generation's
molecules, and a downstream stage's spine points at the upstream stores it
was built from -- which need not be the upstream stage's *current*
generation. Pruning by "not current" alone would delete data that live
generations read. This module records those edges, read-only:

- **pointers**: any ``<stage dir>/generations/<id>`` path in a generation's
  spine ``uns`` or in its small JSON files (manifest, sidecars, spine
  metadata);
- **lineage**: a manifest key ending in ``generation_id`` that names another
  generation of the experiment (e.g. ``annotation_refresh.parent_generation_id``),
  except bookkeeping keys that record what was current at the time
  (``current_generation_id``) rather than what is read.

:func:`protected_generations` walks the edges from the generations that must
stay (current, pinned, named by an external file such as a project registry)
and returns every generation they reach, with the first path that reaches it.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .generation_listing import (
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STAGE_GENERATION_DIRS,
)

REFERENCE_SCHEMA_VERSION = 1
# Ids are matched against the experiment's own generations, so any directory
# name the listing accepts is recognised.
_POINTER = re.compile(r"([A-Za-z0-9_.-]+)/" + GENERATIONS_SUBDIR + r"/([A-Za-z0-9_.-]+)")
_NON_DEPENDENCY_KEYS = frozenset({"generation_id", "current_generation_id"})
# Small metadata files only: stores (parquet, zarr chunks) are never opened.
_JSON_MAX_BYTES = 4 * 1024 * 1024
_JSON_MAX_DEPTH = 2
_SPINE_FILENAME = "spine.h5ad"
_DIR_KINDS = {directory: kind for kind, directory in STAGE_GENERATION_DIRS.items()}


class GenerationReferenceError(RuntimeError):
    """A generation's references could not be read."""


@dataclass(frozen=True)
class GenerationReference:
    """``source`` reads ``target`` (both ``(kind, generation_id)``)."""

    source_kind: str
    source_id: str
    target_kind: str
    target_id: str
    via: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "via": self.via,
        }


@dataclass
class ReferenceGraph:
    """Every generation of one experiment and the generations each reads."""

    generations: dict[tuple[str, str], Path]
    references: list[GenerationReference]
    unreadable: dict[tuple[str, str], str]

    def targets(self, kind: str, generation_id: str) -> list[GenerationReference]:
        return [
            ref
            for ref in self.references
            if (ref.source_kind, ref.source_id) == (kind, generation_id)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "references": [ref.to_dict() for ref in self.references],
            "unreadable": [
                {"kind": kind, "generation_id": gid, "error": error}
                for (kind, gid), error in sorted(self.unreadable.items())
            ],
        }


def _experiment_generations(output_root: Path) -> dict[tuple[str, str], Path]:
    found = {}
    for kind, directory in STAGE_GENERATION_DIRS.items():
        container = output_root / directory / GENERATIONS_SUBDIR
        if not container.is_dir():
            continue
        for path in sorted(container.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                found[(kind, path.name)] = path
    return found


def _json_files(generation: Path) -> Iterator[Path]:
    base = len(generation.parts)
    for root, dirs, files in os.walk(generation):
        depth = len(Path(root).parts) - base
        if depth >= _JSON_MAX_DEPTH:
            dirs[:] = []
        for name in files:
            if name.endswith(".json"):
                path = Path(root) / name
                if path.stat().st_size <= _JSON_MAX_BYTES:
                    yield path


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """``(key path, string)`` for every string in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _spine_strings(spine: Path) -> Iterator[tuple[str, str]]:
    """``(uns key, string)`` for every string in a spine's ``uns``."""
    import h5py
    import numpy as np

    def values(obj) -> Iterator[str]:
        data = obj[()]
        for item in np.atleast_1d(data).ravel():
            if isinstance(item, bytes):
                yield item.decode("utf-8", "replace")
            elif isinstance(item, str):
                yield item

    found: list[tuple[str, str]] = []
    with h5py.File(spine, "r") as handle:
        if "uns" not in handle:
            return iter(())

        def visit(name, obj):
            for attr, value in obj.attrs.items():
                if isinstance(value, (str, bytes)):
                    text = value.decode() if isinstance(value, bytes) else value
                    found.append((f"uns/{name}@{attr}", text))
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "OSU":
                for text in values(obj):
                    found.append((f"uns/{name}", text))

        handle["uns"].visititems(visit)
    return iter(found)


def _pointer_targets(text: str, known: dict[tuple[str, str], Path]) -> Iterator[tuple[str, str]]:
    for directory, generation_id in _POINTER.findall(text):
        kind = _DIR_KINDS.get(directory)
        if kind is not None and (kind, generation_id) in known:
            yield kind, generation_id


def _lineage_targets(
    manifest: dict, known: dict[tuple[str, str], Path], by_id: dict[str, str]
) -> Iterator[tuple[str, str, str]]:
    for key_path, value in _strings(manifest):
        key = key_path.rsplit(".", 1)[-1]
        if key.endswith("generation_id") and key not in _NON_DEPENDENCY_KEYS and value in by_id:
            yield by_id[value], value, key_path


def build_reference_graph(output_root: str | Path) -> ReferenceGraph:
    """Read every generation's references. Opens metadata only, writes nothing."""
    output_root = Path(output_root)
    generations = _experiment_generations(output_root)
    by_id = {gid: kind for kind, gid in generations}
    references: list[GenerationReference] = []
    unreadable: dict[tuple[str, str], str] = {}
    for (kind, gid), path in generations.items():
        seen: set[tuple[str, str]] = set()

        def add(target_kind: str, target_id: str, via: str) -> None:
            if (target_kind, target_id) == (kind, gid) or (target_kind, target_id) in seen:
                return
            seen.add((target_kind, target_id))
            references.append(GenerationReference(kind, gid, target_kind, target_id, via))

        try:
            for json_path in _json_files(path):
                text = json_path.read_text(encoding="utf-8", errors="replace")
                relative = json_path.relative_to(path).as_posix()
                for target in _pointer_targets(text, generations):
                    add(*target, via=relative)
                if json_path.name == GENERATION_MANIFEST and json_path.parent == path:
                    for target_kind, target_id, key in _lineage_targets(
                        json.loads(text), generations, by_id
                    ):
                        add(target_kind, target_id, via=f"{relative}:{key}")
            spine = path / _SPINE_FILENAME
            if spine.is_file():
                for key, text in _spine_strings(spine):
                    for target in _pointer_targets(text, generations):
                        add(*target, via=f"{_SPINE_FILENAME}:{key}")
        except Exception as exc:  # noqa: BLE001 -- any failure means "references unknown"
            unreadable[(kind, gid)] = f"{type(exc).__name__}: {exc}"
    return ReferenceGraph(generations=generations, references=references, unreadable=unreadable)


def external_references(
    paths: Iterable[str | Path], graph: ReferenceGraph
) -> dict[tuple[str, str], str]:
    """Generations of this experiment named by outside files (e.g. a project
    registry). A directory stands for its ``registry.json``."""
    named: dict[tuple[str, str], str] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "registry.json"
        if not path.is_file():
            raise GenerationReferenceError(f"no file to read references from: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in _pointer_targets(text, graph.generations):
            named.setdefault(target, f"named by {path.as_posix()}")
    return named


def protected_generations(
    graph: ReferenceGraph, roots: dict[tuple[str, str], str]
) -> dict[tuple[str, str], str]:
    """Every generation reached from ``roots`` (``(kind, id) -> why it is a
    root``), mapped to the first chain of reads that reaches it."""
    reached: dict[tuple[str, str], str] = {}
    queue = deque(roots)
    while queue:
        source = queue.popleft()
        for ref in graph.targets(*source):
            target = (ref.target_kind, ref.target_id)
            if target in roots or target in reached:
                continue
            reached[target] = f"read by {ref.source_kind} {ref.source_id[:8]} ({ref.via})"
            queue.append(target)
    return reached
