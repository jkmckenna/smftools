"""Set store: caches a project-level cross-experiment materialization, keyed by the
*resolved* composition that produced it -- which experiment/stage/spine and reference
strands actually went in, plus every parameter that changes what gets materialized.

Re-running the same query is then a cache read instead of a full
materialize-and-concat; a query whose resolved composition has changed (a new
experiment registered, an existing one re-registered with new data, a different
stage/window/layer selection, ...) is detected automatically by a hash mismatch,
never served stale.

Phase 3 of ``dev/project_sample_and_set_stores.md``. Wraps
``project.catalog.project_adata`` -- same materialize + concat behavior, just cached.
Per-sample-store joins and embedding persistence (Phase 4) are not implemented yet;
there is no project-computed per-sample analysis catalog to join in yet either (see
``project.sample_store``).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SETS_OUTPUT_DIRNAME = "sets"
BASE_CACHE_FILENAME = "base.h5ad"
COMPOSITION_FILENAME = "composition.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return slug or "x"


def _sets_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_outputs" / SETS_OUTPUT_DIRNAME


def _composition_hash(composition: dict) -> str:
    encoded = json.dumps(composition, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _build_composition(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name,
    modality,
    experiments,
    stage,
    start,
    end,
    layers,
    read_metrics,
) -> tuple[dict, list[dict]]:
    from .catalog import ProjectCatalog, resolve_set_members

    catalog = ProjectCatalog.open(project_dir)
    members = resolve_set_members(
        catalog,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    composition = {
        "canonical_reference": canonical_reference,
        "set_name": set_name,
        "modality": modality,
        "experiments": sorted(experiments) if experiments else None,
        "stage": stage,
        "start": start,
        "end": end,
        "layers": sorted(layers) if layers else None,
        "read_metrics": read_metrics,
        "members": [
            {
                "experiment": m["experiment"],
                "stage": m["stage"],
                "spine_path": str(m["spine_path"]),
                "reference_strands": m["reference_strands"],
            }
            for m in members
        ],
    }
    return composition, members


def set_cache_dir(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    read_metrics: bool = False,
) -> Path:
    """Return the cache directory a :func:`materialize_set` call with these exact
    parameters would use, without resolving membership or touching the cache."""
    composition, _ = _build_composition(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
    )
    label = _slug(set_name) if set_name else _slug(canonical_reference)
    return _sets_root(project_dir) / label / _composition_hash(composition)


def materialize_set(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    read_metrics: bool = False,
    force_recompute: bool = False,
):
    """Materialize + concat a canonical reference across a set, caching by composition.

    Same parameters and result as ``project.catalog.project_adata`` -- this only adds
    caching around it. If a cache already exists for the current *resolved* composition
    (which experiments/stages/spines/reference-strands the query resolves to right now,
    not just the parameters passed in), it's returned directly, no materialize. If the
    project has changed since the cache was written (an experiment registered,
    re-registered, or removed such that the resolved composition differs), the
    composition hash no longer matches, so this transparently falls through to a fresh
    materialize -- never a stale result. ``force_recompute=True`` skips the cache read
    outright (still writes a fresh cache afterward, at the same, current composition
    hash).

    Raises the same ``ValueError`` as ``project_adata`` when nothing matches.
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad
    from .catalog import project_adata

    composition, members = _build_composition(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
    )
    if not members:
        raise ValueError(f"no experiment references match canonical_reference={canonical_reference!r}")

    label = _slug(set_name) if set_name else _slug(canonical_reference)
    cache_dir = _sets_root(project_dir) / label / _composition_hash(composition)
    cache_path = cache_dir / BASE_CACHE_FILENAME

    if not force_recompute and cache_path.exists():
        adata, _ = safe_read_h5ad(cache_path, verbose=False)
        return adata

    adata = project_adata(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_write_h5ad(adata, cache_path, backup=False, verbose=False)
    (cache_dir / COMPOSITION_FILENAME).write_text(
        json.dumps(composition, indent=2, sort_keys=True, default=str)
    )
    return adata
