"""Refresh a preprocess generation's demux annotations without recomputing it.

`demux_type` (double / single barcode ends) comes from the raw stage. When it
changes after preprocessing -- typically because an already-demultiplexed
input's sequencing summary is applied later (`reassemble-raw`) -- nothing the
preprocess matrices hold depends on it, but two read annotations do:

- the copied demux columns themselves, and
- duplicate keepers: `_select_duplicate_keeper` prefers
  ``duplicate_detection_demux_types_to_use`` when it picks the read a duplicate
  cluster keeps, so with every read ``unclassified`` it chose blind. A keeper
  that turns out single while a double member exists drops the whole cluster
  from a double-only selection.

`refresh_preprocess_demux` publishes a sibling preprocess generation: the
current generation's files hardlinked, with only ``obs.parquet``,
``stage_obs.parquet`` and the spine rewritten -- demux columns from the current
raw generation, duplicate flags recomputed over the *existing* clusters
(``duplicate_cluster_id``) with the same keeper rule. Clusters, QC, matrices,
spatial and HMM outputs are untouched. The manifest keeps the parent's source
and node results (the matrices were computed from that raw generation) and
records an ``annotation_refresh`` block naming the raw generation the demux
calls came from.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

from ..informatics.generation import resolve_current_generation, staged_generation
from ..informatics.partition_read import load_spine
from ..readwrite import atomic_write_json, safe_write_h5ad
from .partitioned_executor import _select_duplicate_keeper
from .preprocess_generation import (
    PREPROCESS_GENERATION_MANIFEST,
    PreprocessGenerationError,
    _atomic_publish_spine,
    _bind_generation_spine,
    _checksum,
    _generation_artifact_record,
    resolve_current_preprocess_generation,
    validate_preprocess_generation,
)

logger = get_logger(__name__)

DEMUX_COLUMNS = ("demux_type", "demux_type_source", "demux_type_confidence")
DEDUP_COLUMNS = ("is_duplicate", "is_duplicate_reason", "passes_dedup")
REWRITTEN = {"obs": "obs.parquet", "stage_obs": "stage_obs.parquet", "spine": "spine.h5ad"}
READ_ID = "read_id"


def recompute_duplicate_keepers(
    obs: pd.DataFrame, *, preferred_demux: set[str], metric: str
) -> pd.DataFrame:
    """Duplicate flags re-chosen over the existing clusters.

    ``obs`` is indexed by read id, in the generation's row order, with
    ``duplicate_cluster_id`` / ``duplicate_cluster_size`` from the original
    deduplication. Returns a frame of ``DEDUP_COLUMNS``; with unchanged demux
    calls it reproduces the original flags.
    """
    duplicate = pd.Series(False, index=obs.index)
    clustered = obs[pd.to_numeric(obs["duplicate_cluster_size"], errors="coerce") > 1]
    for _cluster, members in clustered.groupby("duplicate_cluster_id", sort=False).groups.items():
        members = [str(member) for member in members]
        keeper = _select_duplicate_keeper(
            obs, members, preferred_demux=preferred_demux, metric=metric
        )
        duplicate.loc[[member for member in members if member != keeper]] = True
    flags = pd.DataFrame(index=obs.index)
    flags["is_duplicate"] = duplicate
    flags["is_duplicate_reason"] = np.where(duplicate, "sequence_cluster", "")
    flags["passes_dedup"] = obs["passes_qc"].astype(bool) & ~duplicate
    return flags


def _update(frame: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """``frame`` (with a read-id column) with ``values`` (indexed by read id)
    written into the columns both share; rows keep their order."""
    frame = frame.copy()
    keys = frame[READ_ID].astype(str)
    for column in values.columns:
        if column not in frame.columns:
            continue
        aligned = values[column].reindex(keys)
        present = aligned.notna().to_numpy()
        if not present.any():
            continue
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype(object)
        frame.loc[present, column] = aligned.to_numpy()[present]
    return frame


def refresh_preprocess_demux(
    run_root: str | Path,
    *,
    preferred_demux: set[str],
    metric: str,
    select_current: bool = True,
) -> dict[str, Any]:
    """Publish a sibling preprocess generation with refreshed demux columns
    and duplicate keepers (see the module docstring). Returns counts."""
    run_root = Path(run_root)
    output_dir = run_root / "preprocess_adata_outputs"
    current = resolve_current_preprocess_generation(output_dir)
    if current is None:
        raise PreprocessGenerationError(f"{output_dir} has no current preprocess generation")
    parent_dir, parent_manifest = Path(current[0]), dict(current[1])
    raw = resolve_current_generation(run_root / "raw_outputs")
    if raw is None:
        raise PreprocessGenerationError(f"{run_root} has no current raw generation")
    raw_dir, raw_manifest = Path(raw[0]), raw[1]
    raw_obs = pd.read_parquet(raw_dir / "obs.parquet", columns=[READ_ID, *DEMUX_COLUMNS])
    demux = raw_obs.assign(**{READ_ID: raw_obs[READ_ID].astype(str)}).set_index(READ_ID)

    obs = pd.read_parquet(parent_dir / REWRITTEN["obs"])
    obs = _update(obs, demux)
    indexed = obs.set_index(obs[READ_ID].astype(str), drop=False)
    before = indexed["passes_dedup"].astype(bool).copy()
    flags = recompute_duplicate_keepers(indexed, preferred_demux=preferred_demux, metric=metric)
    obs = _update(obs, flags)
    after = flags["passes_dedup"].astype(bool)
    changes = {
        "keepers_changed": int((before != after).sum() // 2),
        "kept_before": int(before.sum()),
        "kept_after": int(after.sum()),
        "double_kept_before": int((before & (indexed["demux_type"].astype(str) == "double")).sum()),
        "double_kept_after": int((after & (indexed["demux_type"].astype(str) == "double")).sum()),
    }
    refreshed = pd.concat([demux[[c for c in DEMUX_COLUMNS if c in demux]], flags], axis=1)

    def validate(staging: Path, final: Path, root: Path) -> None:
        validate_preprocess_generation(
            staging, expected_generation_id=staged.generation_id, final_dir=final, run_root=root
        )

    def publish_spine(_staging: Path, final: Path, _root: Path) -> None:
        _atomic_publish_spine(final / REWRITTEN["spine"], output_dir / REWRITTEN["spine"])

    with staged_generation(
        output_dir,
        run_root=run_root,
        validate=validate,
        manifest_checksum=_checksum,
        write_json=atomic_write_json,
        after_current=publish_spine,
        select_current=select_current,
    ) as staged:
        staging = staged.staging_dir
        skip = {PREPROCESS_GENERATION_MANIFEST, *REWRITTEN.values()}
        shutil.copytree(
            parent_dir,
            staging,
            dirs_exist_ok=True,
            copy_function=os.link,
            ignore=lambda directory, names: [
                name for name in names if Path(directory) == parent_dir and name in skip
            ],
        )
        obs.to_parquet(staging / REWRITTEN["obs"], index=False)
        stage_obs = _update(pd.read_parquet(parent_dir / REWRITTEN["stage_obs"]), refreshed)
        stage_obs.to_parquet(staging / REWRITTEN["stage_obs"], index=False)
        spine = load_spine(parent_dir / REWRITTEN["spine"], verbose=False)
        spine_obs = spine.obs
        names = (
            spine_obs[READ_ID].astype(str) if READ_ID in spine_obs else spine_obs.index.astype(str)
        )
        for column in refreshed.columns:
            if column in spine_obs.columns:
                values = refreshed[column].reindex(names.to_numpy())
                present = values.notna().to_numpy()
                if isinstance(spine_obs[column].dtype, pd.CategoricalDtype):
                    spine_obs[column] = spine_obs[column].astype(object)
                spine_obs.loc[present, column] = values.to_numpy()[present]
        safe_write_h5ad(spine, staging / REWRITTEN["spine"], backup=False, verbose=False)
        _bind_generation_spine(
            staging / REWRITTEN["spine"],
            generation_id=staged.generation_id,
            publication_dir=staged.final_dir,
            run_root=run_root,
        )
        manifest = dict(parent_manifest)
        manifest["generation_id"] = staged.generation_id
        artifacts = dict(manifest.get("artifacts") or {})
        for key, relative in REWRITTEN.items():
            artifacts[key] = _generation_artifact_record(staging / relative, staging)
        manifest["artifacts"] = artifacts
        checksums = {key: artifacts[key]["sha256"] for key in REWRITTEN}
        node_results = []
        for result in manifest.get("node_results") or []:
            result = dict(result)
            result["artifacts"] = [
                {**artifact, "checksum": checksums[artifact["artifact_id"]]}
                if artifact.get("artifact_id") in checksums
                else artifact
                for artifact in result.get("artifacts") or []
            ]
            node_results.append(result)
        manifest["node_results"] = node_results
        manifest["annotation_refresh"] = {
            "kind": "demux",
            "parent_generation_id": str(parent_manifest.get("generation_id")),
            "raw_generation_id": str(raw_manifest.get("generation_id", raw_dir.name)),
            "columns": [*DEMUX_COLUMNS, *DEDUP_COLUMNS],
            "preferred_demux": sorted(preferred_demux),
            "keep_best_metric": metric,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **changes,
        }
        staged.record_manifest(manifest)
    from ..informatics.experiment_spine import write_experiment_spine

    write_experiment_spine(run_root)
    logger.info(
        "Published preprocess generation %s (demux refresh of %s): %s",
        staged.generation_id,
        parent_manifest.get("generation_id"),
        changes,
    )
    return {"generation_id": staged.generation_id, **changes}
