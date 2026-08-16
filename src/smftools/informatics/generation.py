"""Shared publication vocabulary for immutable generations.

Experiment stages and project embeddings publish immutable generations using
one on-disk shape::

    <output_dir>/
      current.json                       # atomic pointer to the readable generation
      generations/<generation_id>/
        generation_manifest.json
        ...                              # kind-specific artifacts
      .staging/<generation_id>/          # build area, never read by consumers

This module owns that shared publication and selection mechanism. Kind-specific
modules remain responsible for building artifacts and validating their schema.
The optional manifest checksum preserves the stronger pointer used by raw,
preprocess, and project embeddings; latent and the post-preprocess experiment
stages retain the checksum-free pointer they historically published.

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


# One generation kind can be a re-basecalled descendant of another. The block is
# optional and its absence is meaningful: an ordinary generation has no lineage
# provenance and a reader must not invent one. Per `D2` in the
# generation-lifecycle plan, ``generation_kind`` is *derived* from the basecall
# generation rather than independently asserted by each descendant stage.
LINEAGE_PROVENANCE_KEYS = frozenset(
    {
        "lineage_id",
        "origin_experiment_uid",
        "parent_raw_generation_id",
        "parent_preprocess_generation_id",
        "selection_id",
        "source_resolution_digest",
        "basecall_id",
        "generation_kind",
        "identity_map",
    }
)
LINEAGE_GENERATION_KINDS = frozenset({"full_source", "parent_universe", "selected_cohort"})
_LINEAGE_REQUIRED_TEXT_KEYS = (
    "lineage_id",
    "origin_experiment_uid",
    "parent_raw_generation_id",
    "selection_id",
    "basecall_id",
    "generation_kind",
)
_LINEAGE_OPTIONAL_TEXT_KEYS = (
    "parent_preprocess_generation_id",
    "source_resolution_digest",
    "identity_map",
)


def validate_lineage_provenance(lineage: Any) -> dict[str, Any] | None:
    """Validate a generation's lineage block, if it carries one.

    Returns ``None`` for an ordinary generation. A malformed block is an error
    rather than a warning: a descendant that cannot state which selection and
    basecall produced it is exactly the artifact this program exists to prevent.
    """
    if lineage is None:
        return None
    if not isinstance(lineage, dict) or set(lineage) != LINEAGE_PROVENANCE_KEYS:
        raise GenerationError("generation lineage provenance is malformed")
    for key in _LINEAGE_REQUIRED_TEXT_KEYS:
        value = lineage.get(key)
        if not isinstance(value, str) or not value.strip():
            raise GenerationError(f"generation lineage provenance lacks {key}")
    if lineage["generation_kind"] not in LINEAGE_GENERATION_KINDS:
        raise GenerationError("generation lineage generation kind is invalid")
    for key in _LINEAGE_OPTIONAL_TEXT_KEYS:
        value = lineage.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise GenerationError(f"generation lineage provenance has an invalid {key}")
    return lineage


@contextmanager
def staged_generation(
    output_dir: str | Path,
    *,
    run_root: str | Path | None = None,
    validate: Callable[[Path, Path, Path], None] | None = None,
    generation_id: str | None = None,
    manifest_checksum: Callable[[Path], str] | None = None,
    write_json: Callable[[str | Path, Any], None] | None = None,
    after_publish: Callable[[Path, Path, Path], None] | None = None,
    after_current: Callable[[Path, Path, Path], None] | None = None,
    select_current: bool = True,
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
        manifest_checksum: Optional checksum function. When provided, the
            resulting digest is recorded as ``manifest_sha256`` in
            ``current.json``.
        write_json: JSON writer used for the manifest, selector, and selector
            rollback. Defaults to :func:`smftools.readwrite.atomic_write_json`.
            Kind-specific callers may pass their imported writer to preserve
            failure-injection seams.
        after_publish: Optional ``(staging_dir, final_dir, run_root)`` callable
            run after the tree moves and *before* the selector advances, whether
            or not it will. Use it for work that describes the generation itself,
            such as validating it at its published location. A failure removes
            the new generation and leaves the selector untouched.
        after_current: Optional ``(staging_dir, final_dir, run_root)`` callable
            run only when the selector actually advances. Use it for work that
            follows *selection* rather than publication -- publishing a canonical
            stage-root spine, for instance, which ordinary readers resolve and
            which a non-selected generation must therefore never overwrite. A
            failure restores the previous selector and removes the new
            generation.
        select_current: Whether publication also advances ``current.json``.
            Set ``False`` to publish without selecting, which is how a
            re-basecalling lineage adds a descendant generation beside the
            parent's: the generation becomes addressable immediately, and only
            explicit promotion changes what ordinary readers resolve.

    Yields:
        A :class:`StagedGeneration`. The caller must call
        :meth:`StagedGeneration.record_manifest` before the block exits cleanly.

    Raises:
        GenerationError: If no manifest was recorded, or the generation id is
            already published.
    """
    output_dir = Path(output_dir)
    run_root = Path(run_root) if run_root is not None else output_dir.parent
    write_json = write_json or atomic_write_json
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

    pointer_path = _current_path(output_dir)
    previous_current: dict[str, Any] | None = None
    if pointer_path.is_file():
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            previous_current = payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            # Publishing a valid generation may repair a corrupt selector.
            previous_current = None

    moved = False
    current_advanced = False
    try:
        yield staged
        if staged._manifest is None:
            raise GenerationError(
                f"generation {generation_id!r} recorded no manifest; "
                "call record_manifest() before leaving the staged_generation block"
            )
        write_json(staging_dir / GENERATION_MANIFEST, staged._manifest)
        if validate is not None:
            validate(staging_dir, final_dir, run_root)
        os.replace(staging_dir, final_dir)
        moved = True
        if after_publish is not None:
            after_publish(staging_dir, final_dir, run_root)
        if select_current:
            pointer = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "generation_id": generation_id,
                "generation_path": final_dir.relative_to(output_dir).as_posix(),
            }
            if manifest_checksum is not None:
                pointer["manifest_sha256"] = manifest_checksum(final_dir / GENERATION_MANIFEST)
            write_json(pointer_path, pointer)
            current_advanced = True
            if after_current is not None:
                after_current(staging_dir, final_dir, run_root)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if moved:
            if current_advanced:
                if previous_current is None:
                    pointer_path.unlink(missing_ok=True)
                else:
                    write_json(pointer_path, previous_current)
            # The move succeeded but the publication transaction did not.
            # Remove the orphan so it cannot be mistaken for a generation.
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    logger.info("Published %s generation %s", output_dir.name, generation_id)


def resolve_current_generation(
    output_dir: str | Path,
    *,
    manifest_checksum: Callable[[Path], str] | None = None,
    require_generation_id: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """Return ``(generation_dir, manifest)`` for the selected generation.

    Returns ``None`` when nothing is published yet, which is how a legacy
    in-place output directory presents. Raises rather than guessing when a
    pointer exists but does not resolve safely. Set ``require_generation_id``
    for formats whose historical validator requires the selector to name the
    generation explicitly.
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
    if manifest_checksum is not None:
        expected_checksum = str(pointer.get("manifest_sha256", ""))
        if not expected_checksum or manifest_checksum(manifest_path) != expected_checksum:
            raise GenerationError(f"{output_dir.name} current manifest checksum does not match")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"{output_dir.name} current manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise GenerationError(f"{output_dir.name} current manifest is not an object")

    expected = str(pointer.get("generation_id", ""))
    if require_generation_id and not expected:
        raise GenerationError(f"{output_dir.name} current pointer names no generation")
    if expected and str(manifest.get("generation_id", "")) != expected:
        raise GenerationError(
            f"{output_dir.name} current pointer names {expected!r} but the manifest does not"
        )
    return generation, manifest


def resolve_stage_generation(
    stage_dir: str | Path,
    lineage: str | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve the selected generation for one experiment stage.

    Args:
        stage_dir: Directory that owns the stage's ``current.json`` and
            ``generations/`` tree.
        lineage: Optional generation ID pinned for this stage by a processing
            lineage. When omitted, resolve the stage's current pointer.

    Returns:
        ``(generation_dir, manifest)`` for the selected generation, or ``None``
        when the stage still uses the legacy in-place layout and no lineage was
        requested.

    Raises:
        GenerationError: If a requested lineage generation is unsafe, missing,
            or inconsistent with its manifest.
    """
    stage_dir = Path(stage_dir)
    if lineage is None:
        return resolve_current_generation(stage_dir)

    generation_id = str(lineage).strip()
    relative = Path(generation_id)
    if (
        not generation_id
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 1
    ):
        raise GenerationError(f"{stage_dir.name} lineage generation id is not portable")

    generation = (stage_dir / GENERATIONS_SUBDIR / relative).resolve()
    generations_root = (stage_dir / GENERATIONS_SUBDIR).resolve()
    if generation.parent != generations_root or not generation.is_dir():
        raise GenerationError(
            f"{stage_dir.name} lineage generation {generation_id!r} does not exist"
        )

    manifest_path = generation / GENERATION_MANIFEST
    if not manifest_path.is_file():
        raise GenerationError(
            f"{stage_dir.name} lineage generation {generation_id!r} has no manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(
            f"{stage_dir.name} lineage generation {generation_id!r} manifest is unreadable"
        ) from exc
    if not isinstance(manifest, dict):
        raise GenerationError(
            f"{stage_dir.name} lineage generation {generation_id!r} manifest is not an object"
        )
    if str(manifest.get("generation_id", "")) != generation_id:
        raise GenerationError(
            f"{stage_dir.name} lineage generation {generation_id!r} disagrees with its manifest"
        )
    return generation, manifest


def remap_staged_paths(outputs: dict[str, Any], staged: StagedGeneration) -> dict[str, Any]:
    """Rebase artifact paths from the staging directory to the published one.

    A stage executor writes into ``staged.staging_dir``; publication moves that
    whole tree to ``staged.final_dir``, so the absolute paths the executor
    returned no longer exist. Values outside the staged tree (a source spine,
    for instance) are passed through untouched.
    """
    staging = staged.staging_dir.resolve()
    remapped: dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, Path):
            resolved = value.resolve()
            if resolved == staging or staging in resolved.parents:
                remapped[key] = staged.final_dir / resolved.relative_to(staging)
                continue
        remapped[key] = value
    return remapped


def publish_canonical_spine(generation_spine: str | Path, canonical_path: str | Path) -> None:
    """Copy a published generation spine to the stage root, atomically.

    Readers that predate generations resolve ``<stage>/spine.h5ad``. That copy
    sits two levels below the run root, so the run-root-relative pointers inside
    it resolve unchanged from either location.
    """
    generation_spine = Path(generation_spine)
    canonical_path = Path(canonical_path)
    temporary = canonical_path.with_name(f".{canonical_path.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(generation_spine, temporary)
        os.replace(temporary, canonical_path)
    finally:
        temporary.unlink(missing_ok=True)


def rebind_staged_spine_pointers(
    spine_path: str | Path,
    *,
    staging_dir: str | Path,
    publication_dir: str | Path,
    run_root: str | Path,
) -> list[str]:
    """Repoint a staged spine's ``uns`` paths at the published generation.

    Stage executors record artifact pointers relative to the run root, so while
    building they faithfully encode ``<stage>/.staging/<id>/...`` -- which
    dangles the moment the tree is moved to ``<stage>/generations/<id>/``. Raw
    solves this by binding its spine to a ``publication_dir``
    (``raw_generation._bind_generation_spine``); this is the same idea, applied
    generically so a stage cannot silently miss a pointer when its key set
    changes.

    Call while the spine is still in staging, before publication. Pointers to
    artifacts outside the staged tree (a source spine, for instance) are left
    alone.

    Returns:
        The ``uns`` keys that were rewritten, for logging and assertions.
    """
    from ..readwrite import normalize_uns_string_lists, safe_read_h5ad, safe_write_h5ad
    from .partition_read import relative_uns_path

    staging_dir = Path(staging_dir)
    publication_dir = Path(publication_dir)
    run_root = Path(run_root)

    old_prefix = relative_uns_path(staging_dir, run_root)
    new_prefix = relative_uns_path(publication_dir, run_root)
    if not old_prefix or old_prefix == new_prefix:
        return []

    spine, _ = safe_read_h5ad(Path(spine_path), verbose=False)
    rewritten: dict[str, str] = {}
    for key, value in spine.uns.items():
        if isinstance(value, str) and value.startswith(f"{old_prefix}/"):
            rewritten[key] = f"{new_prefix}{value[len(old_prefix) :]}"
    if not rewritten:
        return []

    spine.uns.update(rewritten)
    # The spine was just read from disk, so its string-list uns entries are numpy
    # arrays the writer's sanitizer would otherwise store as repr strings.
    normalize_uns_string_lists(spine)
    safe_write_h5ad(spine, Path(spine_path), backup=False, verbose=False)
    return sorted(rewritten)


def has_published_generations(output_dir: str | Path) -> bool:
    """True when at least one generation directory exists under ``output_dir``."""
    root = Path(output_dir) / GENERATIONS_SUBDIR
    return root.is_dir() and any(child.is_dir() for child in root.iterdir())
