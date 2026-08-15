"""Shared publication vocabulary for immutable generations.

Four subsystems already publish immutable generations, each with its own
publish/resolve pair: raw (:mod:`.raw_generation`), preprocess
(:mod:`smftools.preprocessing.preprocess_generation`), latent
(:mod:`smftools.tools.partitioned_latent`), and project embeddings
(:mod:`smftools.project.embedding_store`). They converged independently on one
on-disk shape::

    <output_dir>/
      current.json                       # atomic pointer to the readable generation
      generations/<generation_id>/
        generation_manifest.json
        ...                              # kind-specific artifacts
      .staging/<generation_id>/          # build area, never read by consumers

This module is that shape, factored out once, for kinds that do not have it yet.
It deliberately does **not** rewrite the four existing implementations: they are
working and tested, their on-disk layouts are load-bearing for published data,
and consolidating them changes no layout, so it can follow separately.

The transaction is the point. A generation becomes visible only when it is
complete: artifacts are built under ``.staging/``, validated, moved into place
with a single :func:`os.replace`, and only then does ``current.json`` advance. A
failure at any step leaves the previously current generation untouched.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from ..logging_utils import get_logger
from ..readwrite import atomic_write_json

logger = get_logger(__name__)

GENERATIONS_SUBDIR = "generations"
STAGING_SUBDIR = ".staging"
CURRENT_FILENAME = "current.json"
GENERATION_MANIFEST = "generation_manifest.json"
CURRENT_SCHEMA_VERSION = 1


class GenerationError(RuntimeError):
    """Raised when a generation cannot be published or selected safely."""


@dataclass
class StagedGeneration:
    """One in-flight generation. Build into :attr:`staging_dir`, then record a manifest."""

    generation_id: str
    staging_dir: Path
    final_dir: Path
    output_dir: Path
    run_root: Path
    _manifest: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def record_manifest(self, payload: dict[str, Any]) -> None:
        """Declare this generation's manifest. Required before a clean exit.

        ``generation_id`` is stamped here rather than trusted from the caller, so
        the manifest and the directory name cannot disagree.
        """
        if not isinstance(payload, dict):
            raise GenerationError("generation manifest payload must be a dict")
        self._manifest = {**payload, "generation_id": self.generation_id}

    def artifact(self, *parts: str) -> Path:
        """Return a path inside the staging directory, creating parent directories."""
        target = self.staging_dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


def _current_path(output_dir: Path) -> Path:
    return output_dir / CURRENT_FILENAME


@contextmanager
def staged_generation(
    output_dir: str | Path,
    *,
    run_root: str | Path | None = None,
    validate: Callable[[Path, Path, Path], None] | None = None,
    generation_id: str | None = None,
) -> Iterator[StagedGeneration]:
    """Build and atomically publish one immutable generation.

    Args:
        output_dir: The kind's directory (e.g. ``<run>/hmm_adata_outputs``).
        run_root: Anchor for relative artifact pointers. Defaults to
            ``output_dir.parent``.
        validate: Optional ``(staging_dir, final_dir, run_root)`` callable run
            after the manifest is written and before the move. Raise from it to
            abort; the staging tree is removed and ``current.json`` never moves.
        generation_id: Override the generated id. For tests and for republishing
            a known id; normally leave unset.

    Yields:
        A :class:`StagedGeneration`. The caller must call
        :meth:`StagedGeneration.record_manifest` before the block exits cleanly.

    Raises:
        GenerationError: If no manifest was recorded, or the generation id is
            already published.
    """
    output_dir = Path(output_dir)
    run_root = Path(run_root) if run_root is not None else output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_id = generation_id or uuid4().hex
    staging_dir = output_dir / STAGING_SUBDIR / generation_id
    final_dir = output_dir / GENERATIONS_SUBDIR / generation_id
    if final_dir.exists():
        raise GenerationError(f"generation {generation_id!r} is already published")

    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    staged = StagedGeneration(
        generation_id=generation_id,
        staging_dir=staging_dir,
        final_dir=final_dir,
        output_dir=output_dir,
        run_root=run_root,
    )

    moved = False
    try:
        yield staged
        if staged._manifest is None:
            raise GenerationError(
                f"generation {generation_id!r} recorded no manifest; "
                "call record_manifest() before leaving the staged_generation block"
            )
        atomic_write_json(staging_dir / GENERATION_MANIFEST, staged._manifest)
        if validate is not None:
            validate(staging_dir, final_dir, run_root)
        os.replace(staging_dir, final_dir)
        moved = True
        atomic_write_json(
            _current_path(output_dir),
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "generation_id": generation_id,
                "generation_path": final_dir.relative_to(output_dir).as_posix(),
            },
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if moved:
            # The move succeeded but advancing `current` did not. Remove the
            # orphan so it cannot be mistaken for a published generation; the
            # previously current one is still selected and untouched.
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    logger.info("Published %s generation %s", output_dir.name, generation_id)


def resolve_current_generation(
    output_dir: str | Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Return ``(generation_dir, manifest)`` for the selected generation.

    Returns ``None`` when nothing is published yet, which is how a legacy
    in-place output directory presents. Raises rather than guessing when a
    pointer exists but does not resolve safely.
    """
    output_dir = Path(output_dir)
    pointer_path = _current_path(output_dir)
    if not pointer_path.is_file():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"{output_dir.name} current pointer is unreadable") from exc
    if int(pointer.get("schema_version", -1)) != CURRENT_SCHEMA_VERSION:
        raise GenerationError(f"{output_dir.name} current-pointer schema is incompatible")

    # Check the raw string before Path(): Path("") is Path("."), which would
    # resolve to output_dir itself and pass every containment test below.
    raw_path = str(pointer.get("generation_path", "")).strip()
    relative = Path(raw_path)
    generation = (output_dir / relative).resolve()
    if (
        not raw_path
        or relative.is_absolute()
        or ".." in relative.parts
        or generation == output_dir.resolve()
        or not generation.is_relative_to(output_dir.resolve())
    ):
        raise GenerationError(f"{output_dir.name} current pointer is not portable")

    manifest_path = generation / GENERATION_MANIFEST
    if not manifest_path.is_file():
        raise GenerationError(f"{output_dir.name} current generation has no manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"{output_dir.name} current manifest is unreadable") from exc

    expected = str(pointer.get("generation_id", ""))
    if expected and str(manifest.get("generation_id", "")) != expected:
        raise GenerationError(
            f"{output_dir.name} current pointer names {expected!r} but the manifest does not"
        )
    return generation, manifest


def has_published_generations(output_dir: str | Path) -> bool:
    """True when at least one generation directory exists under ``output_dir``."""
    root = Path(output_dir) / GENERATIONS_SUBDIR
    return root.is_dir() and any(child.is_dir() for child in root.iterdir())
