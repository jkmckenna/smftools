"""Consolidated ``experiment_spine.h5ad``: a generated, superset spine per experiment.

Coexists with (does not replace) every stage's own ``spine.h5ad`` -- see
``dev/plans/completed/experiment_storage_schema.md``'s Phase 3 status for why: ~15 existing call
sites (``project/registry.py``, ``sample_store.py``, explicit ``--stage`` pinning,
...) rely on resolving one *specific* stage's self-contained spine, and that stays
exactly as it is. This file is an additional, opt-in-preferred artifact for the
auto-fallback ("give me whatever's most complete") case.

Two things a single per-stage spine can't do today, that this fixes:

- Its ``obs`` is the join of every stage's ``obs.parquet`` that has run
  (``informatics.stage_obs``), rather than whichever one stage's obs snapshot.
- Its ``uns`` is the *union* of every stage spine's catalog pointers found under the
  run root, not one linear ``.copy()`` lineage. Spatial and hmm are sibling branches
  off preprocess, not sequential to each other -- no single existing spine's ``uns``
  carries both ``spatial_task_catalog`` and ``hmm_catalog`` at once, so
  ``materialize()`` against one of them alone can never resolve both stages' derived
  data in the same call. Unioning fixes that.

Lives at ``<run_root>/experiment_spine_outputs/spine.h5ad`` -- its own stage-shaped
sibling directory to ``raw_outputs/``, ``preprocess_adata_outputs/``, etc., **not**
literal top-level, despite earlier design-doc sketches. Reason: ``materialize()``'s
``_resolve_spine``/``_run_root_from_spine_path`` hardcode "a spine lives two
directory-levels under the run root" to resolve ``uns['source_base_dir']`` -- a true
top-level file would resolve that anchor one directory too high. Reusing the same
directory-depth convention as every other stage avoids touching that code at all.
"""

from __future__ import annotations

from pathlib import Path

from smftools.constants import HMM_DIR, LATENT_DIR, PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR

from .generation import resolve_stage_generation
from .partition_read import load_spine, relative_uns_path, resolve_relative_path
from .stage_obs import OBS_FILENAME, read_stage_obs

EXPERIMENT_SPINE_DIR = "experiment_spine_outputs"
EXPERIMENT_SPINE_FILENAME = "spine.h5ad"

# Preprocess's normalized obs artifact isn't named "obs.parquet" -- that name is
# already taken by its own denormalized QC sidecar (PREPROCESS_OBS_SIDECAR in
# preprocessing/partitioned_executor.py). Duplicated here (not imported) to avoid a
# reverse dependency: preprocessing/ already imports from informatics/, so
# informatics/ can't import back from preprocessing/ without a cycle. Keep this in
# sync with PREPROCESS_STAGE_OBS if that ever changes.
_PREPROCESS_STAGE_OBS_FILENAME = "stage_obs.parquet"

# Fixed composition order. Each stage owns only its stage-prefixed keys; shared,
# non-prefixed keys come from the earliest stage that declares them. Descendant
# spines are copies of their source and can therefore carry stale pointers for a
# sibling stage, so a later spine must never replace keys owned by that sibling.
_STAGE_DIRS_IN_UNION_ORDER = (
    ("raw", RAW_DIR),
    ("preprocess", PREPROCESS_DIR),
    ("spatial", SPATIAL_DIR),
    ("hmm", HMM_DIR),
    ("latent", LATENT_DIR),
)
_STAGE_NAMES = tuple(stage for stage, _stage_dir in _STAGE_DIRS_IN_UNION_ORDER)


def _resolved_stage_dir(stage: str, stage_dir: Path) -> Path:
    """Return the current generation directory, or the legacy stage directory."""
    if stage == "raw":
        # Preserve raw's stronger artifact/checksum validation until EGL-01b's
        # consolidation folds that validator into the shared helper.
        from .raw_generation import resolve_current_raw_generation

        resolved = resolve_current_raw_generation(stage_dir)
    else:
        resolved = resolve_stage_generation(stage_dir)
    return resolved[0] if resolved is not None else stage_dir


def _merge_stage_uns(
    target: dict[str, object],
    source: dict[str, object],
    *,
    stage: str,
) -> None:
    """Merge one spine's metadata according to per-stage key ownership."""
    for key, value in source.items():
        owner = next(
            (candidate for candidate in _STAGE_NAMES if key.startswith(f"{candidate}_")),
            None,
        )
        if owner == stage:
            target[key] = value
        elif owner is None and key not in target:
            target[key] = value


def experiment_spine_path(run_root: str | Path) -> Path:
    return Path(run_root) / EXPERIMENT_SPINE_DIR / EXPERIMENT_SPINE_FILENAME


def write_experiment_spine(run_root: str | Path) -> Path | None:
    """(Re)generate the consolidated ``experiment_spine.h5ad`` for one experiment.

    Returns ``None`` (writes nothing) if raw's ``obs.parquet`` doesn't exist yet --
    an old run predating Phase 3, nothing to build the consolidated obs from. No
    forced backfill; the run keeps working through the existing per-stage spines.
    """
    import anndata as ad

    from ..readwrite import safe_write_h5ad

    run_root = Path(run_root)
    stage_dirs = {
        stage: _resolved_stage_dir(stage, run_root / stage_dir)
        for stage, stage_dir in _STAGE_DIRS_IN_UNION_ORDER
    }
    raw_dir = stage_dirs["raw"]
    if not (raw_dir / OBS_FILENAME).exists():
        return None

    obs = read_stage_obs(raw_dir)
    preprocess_dir = stage_dirs["preprocess"]
    preprocess_stage_obs = preprocess_dir / _PREPROCESS_STAGE_OBS_FILENAME
    preprocess_spine_path = preprocess_dir / "spine.h5ad"
    if preprocess_spine_path.exists():
        preprocess_spine = load_spine(preprocess_spine_path, verbose=False)
        generation_stage_obs = resolve_relative_path(
            preprocess_spine.uns.get("preprocess_stage_obs"),
            run_root,
        )
        if generation_stage_obs is not None:
            preprocess_stage_obs = generation_stage_obs
    if preprocess_stage_obs.exists():
        extra = read_stage_obs(
            preprocess_stage_obs.parent,
            filename=preprocess_stage_obs.name,
        )
        overlap = [column for column in extra.columns if column in obs.columns]
        obs = obs.join(extra.drop(columns=overlap), how="inner")
    obs.index.name = None

    uns: dict[str, object] = {}
    for stage, stage_dir in _STAGE_DIRS_IN_UNION_ORDER:
        spine_path = stage_dirs[stage] / "spine.h5ad"
        if not spine_path.exists():
            continue
        stage_spine = load_spine(spine_path, verbose=False)
        _merge_stage_uns(uns, dict(stage_spine.uns), stage=stage)

    uns["source_base_dir"] = relative_uns_path(raw_dir, run_root)
    uns["is_spine"] = True

    experiment_spine = ad.AnnData(obs=obs, uns=uns)
    output_path = experiment_spine_path(run_root)
    safe_write_h5ad(experiment_spine, output_path, backup=False, verbose=False)
    return output_path
