"""HMM layer sets (`HPS-07`): one fitting plan's layers as their own generation.

A layer set holds the decoded layers of one named fitting plan -- ``default``
or an ``hmm_variants`` entry (`HPS-05`) -- for one experiment, published under
``hmm_adata_outputs/layer_sets/<plan>/`` with the HMM stage's own generation
machinery. Fitting and decoding a plan never regenerates the HMM stage and
never touches its current generation.

A layer set is decoded from the experiment's current spatial spine (the HMM
stage's input), so its spine points at the same upstream stores: ``materialize``
reads it as it reads an HMM spine, the generation reference graph protects what
it reads, and pruning handles superseded layer-set generations. Its manifest
records the plan, each model's ``fit_config_hash`` and the spatial generation
it was decoded from; a layer set whose input is no longer current is reported
as stale. Decoding while upstream stages are stale is allowed and recorded --
the HMM's stage-only refit (`SUG-02`).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smftools.constants import HMM_DIR, SPATIAL_DIR
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_PLAN = "default"
LAYER_SET_STAGE = "hmm_layer_set"
LAYER_SET_SCHEMA_VERSION = 1


class LayerSetError(ValueError):
    """A layer-set request cannot be served."""


def plan_names(cfg) -> list[str]:
    """``default`` and every configured ``hmm_variants`` plan."""
    from ..tools.partitioned_hmm import hmm_variants

    return [DEFAULT_PLAN, *hmm_variants(cfg)]


def plan_config(cfg, plan: str):
    """``cfg`` restricted to one fitting plan's models."""
    from ..tools.partitioned_hmm import hmm_variants

    restricted = copy.copy(cfg)
    if plan == DEFAULT_PLAN:
        restricted.hmm_variants = {}
        restricted._hmm_only_variant = None
        return restricted
    variants = hmm_variants(cfg)
    if plan not in variants:
        raise LayerSetError(f"unknown HMM plan {plan!r}; configured plans: {plan_names(cfg)}")
    restricted._hmm_only_variant = plan
    return restricted


def plan_fit_hashes(cfg_plan) -> dict[str, str]:
    """model label -> fit_config_hash for a restricted config's models."""
    from ..hmm.model_artifacts import hmm_fit_config_hash
    from ..tools.partitioned_hmm import _configured_model_specs

    return {
        spec.label: hmm_fit_config_hash(spec.config(cfg_plan))
        for spec in _configured_model_specs(cfg_plan)
    }


def layer_set_container(run_root: str | Path, plan: str) -> Path:
    from ..informatics.generation_listing import HMM_LAYER_SETS_SUBDIR

    return Path(run_root) / HMM_DIR / HMM_LAYER_SETS_SUBDIR / plan


def _current(container: Path) -> tuple[str | None, dict | None]:
    """(current generation id, its manifest) of a stage-like container."""
    pointer = container / "current.json"
    if not pointer.is_file():
        return None, None
    generation_id = json.loads(pointer.read_text(encoding="utf-8")).get("generation_id")
    manifest_path = container / "generations" / str(generation_id) / "generation_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    )
    return generation_id, manifest


def publish_layer_set(
    run_root: str | Path,
    source_spine: str | Path,
    cfg,
    plan: str,
    *,
    source_generation: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit and decode one plan into a new layer-set generation (or reuse the
    current one when its settings and input are unchanged)."""
    from ..informatics.generation import (
        rebind_staged_spine_pointers,
        staged_generation,
    )
    from ..tools.partitioned_hmm import execute_partitioned_hmm

    run_root = Path(run_root)
    cfg_plan = plan_config(cfg, plan)
    hashes = plan_fit_hashes(cfg_plan)
    if not hashes:
        raise LayerSetError(f"HMM plan {plan!r} has no models for the configured signals")
    container = layer_set_container(run_root, plan)
    current_id, manifest = _current(container)
    if (
        not force
        and manifest is not None
        and manifest.get("fit_config_hashes") == hashes
        and (manifest.get("source_generation") or {}).get("generation_id") == source_generation
    ):
        logger.info("HMM plan %s: layer set %s is current; reused", plan, current_id)
        return {
            "plan": plan,
            "generation_id": current_id,
            "reused": True,
            "spine": container / "generations" / str(current_id) / "spine.h5ad",
        }
    container.mkdir(parents=True, exist_ok=True)
    with staged_generation(container, run_root=run_root) as staged:
        outputs = execute_partitioned_hmm(Path(source_spine), cfg_plan, staged.staging_dir)
        rebound = rebind_staged_spine_pointers(
            outputs["spine"],
            staging_dir=staged.staging_dir,
            publication_dir=staged.final_dir,
            run_root=run_root,
        )
        staged.record_manifest(
            {
                "schema_version": LAYER_SET_SCHEMA_VERSION,
                "status": "complete",
                "stage": LAYER_SET_STAGE,
                "plan": plan,
                "fit_config_hashes": hashes,
                "source_generation": {"kind": "spatial", "generation_id": source_generation},
                "rebound_spine_pointers": rebound,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    generation_id = staged.final_dir.name
    logger.info("HMM plan %s: published layer set %s", plan, generation_id)
    return {
        "plan": plan,
        "generation_id": generation_id,
        "reused": False,
        "spine": staged.final_dir / "spine.h5ad",
    }


def list_layer_sets(run_root: str | Path, cfg) -> list[dict[str, Any]]:
    """Each plan's current layer set and whether it is still current."""
    from ..informatics.generation_listing import HMM_LAYER_SETS_SUBDIR

    run_root = Path(run_root)
    spatial_current, _ = _current(run_root / SPATIAL_DIR)
    configured = plan_names(cfg)
    on_disk = []
    root = run_root / HMM_DIR / HMM_LAYER_SETS_SUBDIR
    if root.is_dir():
        on_disk = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    rows = []
    for plan in dict.fromkeys([*configured, *on_disk]):
        generation_id, manifest = _current(layer_set_container(run_root, plan))
        expected = plan_fit_hashes(plan_config(cfg, plan)) if plan in configured else None
        source = ((manifest or {}).get("source_generation") or {}).get("generation_id")
        rows.append(
            {
                "plan": plan,
                "configured": plan in configured,
                "generation_id": generation_id,
                "settings_current": (
                    None
                    if manifest is None or expected is None
                    else manifest.get("fit_config_hashes") == expected
                ),
                "source_generation": source,
                "input_current": None if manifest is None else source == spatial_current,
            }
        )
    return rows


def run_from_config(config_path, plan: str, *, force: bool = False) -> dict[str, Any]:
    """Resolve an experiment config and publish ``plan``'s layer set."""
    from .helpers import get_adata_paths, load_experiment_config

    cfg = load_experiment_config(config_path)
    paths = get_adata_paths(cfg)
    source = getattr(paths, "spatial_spine", None)
    if source is None or not Path(source).exists():
        raise LayerSetError("an HMM layer set needs the experiment's spatial stage (spatial spine)")
    run_root = Path(cfg.output_directory)
    spatial_current, _ = _current(run_root / SPATIAL_DIR)
    return publish_layer_set(
        run_root, source, cfg, plan, source_generation=spatial_current, force=force
    )


def list_from_config(config_path) -> list[dict[str, Any]]:
    from .helpers import load_experiment_config

    cfg = load_experiment_config(config_path)
    return list_layer_sets(Path(cfg.output_directory), cfg)
