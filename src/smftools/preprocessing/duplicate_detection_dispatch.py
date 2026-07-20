"""Bounded, parallel, multi-round duplicate-read detection dispatch.

``reduce_duplicate_reads`` (partitioned_executor.py) used to materialize an
entire (reference, sample[, core]) group's read x site matrix in one call and
run a single synchronous, unparallelized ``_process_group`` pass over it --
for locus-mode references the "core window" already spans the whole
reference, so there was no chunking axis at all. On a real production
experiment (~1.3M reads, some groups up to ~46,000 reads) this crashed
(silent OOM) inside ``_process_group``'s uncapped hierarchical clustering
step.

This module adds the missing chunking axis *above* ``_process_group``: reads
in a group are split into bounded chunks, each dispatched as an independent
task through the shared memory-watchdog-covered ``run_tasks_parallel`` pool.
Reads surviving a round (chunk keepers -- reads not flagged as duplicates by
their chunk's own comparison) pool into a smaller "round 2" candidate set,
and so on, until the pool fits in one chunk or stops shrinking. A single
global ``UnionFind`` (owned by the caller, e.g. ``reduce_duplicate_reads``)
receives every round's chunk-local duplicate pairs -- union-find composes
correctly across chunks and rounds regardless of processing order, so no
separate cluster-ID-remapping logic is needed.

See dev/duplicate_detection_scaling.md for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

from ..informatics.partition_read import materialize
from .dispatch_plan import BYTES_PER_WORKING_POSITION
from .flag_duplicate_reads import UnionFind, _process_group

logger = get_logger(__name__)


@dataclass(frozen=True)
class DuplicateDetectionChunkTask:
    """One independently dispatchable duplicate-detection work unit."""

    task_id: str
    reference: str
    sample: str
    core_start: int
    core_end: int
    load_start: int
    load_end: int
    round_index: int
    chunk_index: int
    n_reads: int
    estimated_memory_bytes: int
    read_ids: tuple[str, ...]


def plan_duplicate_detection_chunks(
    obs_slice: pd.DataFrame,
    *,
    reference: str,
    sample: str,
    core_start: int,
    core_end: int,
    load_start: int,
    load_end: int,
    round_index: int,
    max_reads_per_chunk: int,
    target_task_memory_mb: int,
    presort_metric: str,
    round_shuffle_seed: int,
) -> list[DuplicateDetectionChunkTask]:
    """Split a group's (or a prior round's survivor pool's) reads into bounded chunks.

    Round 0 orders reads by ``presort_metric`` (an existing cheap obs scalar) before
    splitting -- true duplicate reads have near-identical aggregate metrics, so this
    front-loads recall into round 0 (fully parallel, no dependency on a prior round).
    Later rounds use a freshly reseeded random shuffle each time, so a pair split
    apart by one round's boundaries gets an independent chance in the next.
    """
    loaded_width = max(1, int(load_end) - int(load_start))
    memory_budget = int(target_task_memory_mb) * 1024**2
    reads_per_chunk = max(1, memory_budget // (loaded_width * BYTES_PER_WORKING_POSITION))
    reads_per_chunk = max(1, min(reads_per_chunk, int(max_reads_per_chunk)))

    if round_index == 0 and presort_metric in obs_slice:
        metric_values = pd.to_numeric(obs_slice[presort_metric], errors="coerce")
        ordered_ids = list(metric_values.sort_values(na_position="last").index)
    else:
        rng = np.random.default_rng(int(round_shuffle_seed) + int(round_index))
        order = rng.permutation(len(obs_slice))
        ordered_ids = [obs_slice.index[i] for i in order]

    tasks: list[DuplicateDetectionChunkTask] = []
    for chunk_index, start in enumerate(range(0, len(ordered_ids), reads_per_chunk)):
        chunk_ids = tuple(map(str, ordered_ids[start : start + reads_per_chunk]))
        tasks.append(
            DuplicateDetectionChunkTask(
                task_id=(
                    f"{reference}|{sample}|{core_start}-{core_end}|"
                    f"round{round_index:03d}|chunk{chunk_index:05d}"
                ),
                reference=reference,
                sample=sample,
                core_start=core_start,
                core_end=core_end,
                load_start=load_start,
                load_end=load_end,
                round_index=round_index,
                chunk_index=chunk_index,
                n_reads=len(chunk_ids),
                estimated_memory_bytes=len(chunk_ids) * loaded_width * BYTES_PER_WORKING_POSITION,
                read_ids=chunk_ids,
            )
        )
    return tasks


def _build_duplicate_detection_context_mask(window, reference: str, cfg) -> np.ndarray:
    """Boolean mask over ``window.var`` selecting the configured comparison site types."""
    context_mask = np.zeros(window.n_vars, dtype=bool)
    for site_type in cfg.duplicate_detection_site_types:
        column = f"{reference}_{site_type}_site"
        if column in window.var:
            context_mask |= window.var[column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    valid_column = f"position_in_{reference}"
    if valid_column in window.var:
        context_mask &= (
            window.var[valid_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
        )
    return context_mask


def execute_duplicate_detection_chunk_task(
    spine_path, task: DuplicateDetectionChunkTask, cfg, obs_slice: pd.DataFrame
) -> dict | None:
    """Materialize one chunk's reads and run duplicate detection on it.

    Module-level and picklable so ``run_tasks_parallel`` can dispatch it to a
    separate process. Returns ``_process_group``'s result dict (or ``None`` if
    the chunk has no valid comparison sites or fewer than 2 reads).
    """
    window = materialize(
        spine_path,
        references=task.reference,
        read_ids=list(task.read_ids),
        start=task.load_start,
        end=task.load_end,
        layers=[],
    )
    context_mask = _build_duplicate_detection_context_mask(window, task.reference, cfg)
    if not context_mask.any():
        return None

    return _process_group(
        {
            "X_sub": np.asarray(window.X)[:, context_mask].astype(np.float32),
            "obs_df": obs_slice,
            "obs_index": list(map(str, window.obs_names)),
            "sample": task.sample,
            "ref": task.reference,
            "distance_threshold": float(cfg.duplicate_detection_distance_threshold),
            "window_size": int(cfg.duplicate_detection_window_size_for_hamming_neighbors),
            "min_overlap_positions": int(cfg.duplicate_detection_min_overlapping_positions),
            "keep_best_metric": cfg.duplicate_detection_keep_best_metric,
            "keep_best_higher": True,
            "do_hierarchical": bool(cfg.duplicate_detection_do_hierarchical),
            "hierarchical_linkage": cfg.duplicate_detection_hierarchical_linkage,
            "hierarchical_metric": "euclidean",
            "hierarchical_window": int(cfg.duplicate_detection_window_size_for_hamming_neighbors),
            "hierarchical_max_representatives": int(
                getattr(cfg, "duplicate_detection_hierarchical_max_representatives", 5000)
            ),
            "do_pca": bool(cfg.duplicate_detection_do_pca),
            "pca_n_components": int(getattr(cfg, "duplicate_detection_pca_n_components", 50)),
            "pca_center": bool(getattr(cfg, "duplicate_detection_pca_center", True)),
            "random_state": 0,
            "demux_col": "demux_type",
            "demux_types": list(cfg.duplicate_detection_demux_types_to_use),
            "n_permutation_passes": int(
                getattr(cfg, "duplicate_detection_n_permutation_passes", 4)
            ),
            "permutation_seed": int(getattr(cfg, "duplicate_detection_permutation_seed", 0)),
        }
    )


def _fold_chunk_result_into_union_find(
    result: dict,
    union_find: UnionFind,
    read_position: dict[str, int],
    hamming_minima: dict[str, np.ndarray],
) -> None:
    """Union one chunk task's local duplicate pairs into the shared global UnionFind.

    Identical in kind to what ``reduce_duplicate_reads`` already did for a single
    whole-group ``_process_group`` call -- reused unchanged per chunk, per round,
    since union-find composition doesn't care which pass/chunk/round discovered a
    given pair.
    """
    local_ids = list(map(str, result["obs_index"]))
    cluster_ids = np.asarray(result["sequence__merged_cluster_id"])
    cluster_sizes = np.asarray(result["sequence__cluster_size"])
    for cluster_id in np.unique(cluster_ids[cluster_sizes > 1]):
        members = np.where(cluster_ids == cluster_id)[0]
        anchor = read_position[local_ids[int(members[0])]]
        for member in members[1:]:
            union_find.union(anchor, read_position[local_ids[int(member)]])
    for column, global_values in hamming_minima.items():
        if column not in result:
            continue
        local_values = np.asarray(result[column], dtype=float)
        for local_index, read_id in enumerate(local_ids):
            value = local_values[local_index]
            global_index = read_position[read_id]
            if not np.isnan(value) and (
                np.isnan(global_values[global_index]) or value < global_values[global_index]
            ):
                global_values[global_index] = value


def _chunk_survivors(result: dict) -> list[str]:
    """Reads not flagged as duplicates within their chunk -- one per chunk-local cluster.

    A non-survivor's eventual duplicate status is already fully determined
    transitively through the union-find once its chunk keeper participates in a
    later round; nothing is lost by not carrying it forward directly.
    """
    is_duplicate = np.asarray(result["sequence__is_duplicate"], dtype=bool)
    obs_index = result["obs_index"]
    return [read_id for read_id, dup in zip(obs_index, is_duplicate) if not dup]


def _dispatch_and_fold(
    tasks: list[DuplicateDetectionChunkTask],
    spine_path,
    cfg,
    group_obs: pd.DataFrame,
    union_find: UnionFind,
    read_position: dict[str, int],
    hamming_minima: dict[str, np.ndarray],
    *,
    pool_label: str | None = None,
) -> list[str]:
    """Dispatch one round's chunk tasks in parallel, fold results, return survivors."""
    if not tasks:
        return []
    from ..memory_guard import run_tasks_parallel

    task_args = [
        (spine_path, task, cfg, group_obs.loc[list(task.read_ids)]) for task in tasks
    ]
    results = run_tasks_parallel(
        execute_duplicate_detection_chunk_task, task_args, cfg=cfg, pool_label=pool_label
    )

    survivors: list[str] = []
    for task, result in zip(tasks, results):
        if result is None:
            # Chunk had <2 reads or no valid comparison sites -- every one of its
            # reads trivially survives (nothing to compare them against within
            # this chunk); they get a fresh chance in the next round/final pass
            # rather than being silently dropped.
            survivors.extend(task.read_ids)
            continue
        _fold_chunk_result_into_union_find(result, union_find, read_position, hamming_minima)
        survivors.extend(_chunk_survivors(result))
    return survivors


def run_duplicate_detection_rounds(
    spine_path,
    group_obs: pd.DataFrame,
    *,
    reference: str,
    sample: str,
    core_start: int,
    core_end: int,
    load_start: int,
    load_end: int,
    cfg,
    union_find: UnionFind,
    read_position: dict[str, int],
    hamming_minima: dict[str, np.ndarray],
) -> None:
    """Own one (reference, sample, core) group's full multi-round duplicate-detection loop.

    Mutates ``union_find``/``hamming_minima`` in place (both owned by the caller and
    shared across every group in the dataset) -- workers never see or mutate them
    directly, only ever returning chunk-local results that get folded in here, in
    the main process, exactly as a single-window call already did before this
    module existed.
    """
    max_reads_per_chunk = int(getattr(cfg, "duplicate_detection_max_reads_per_window", 20_000))
    target_task_memory_mb = int(getattr(cfg, "target_task_memory_mb", 512))
    max_rounds = int(getattr(cfg, "duplicate_detection_max_rounds", 6))
    min_progress_rounds = int(
        getattr(cfg, "duplicate_detection_min_progress_rounds_before_stop", 1)
    )
    presort_metric = str(
        getattr(cfg, "duplicate_detection_chunk_presort_metric", "Fraction_any_C_site_modified")
    )
    round_shuffle_seed = int(getattr(cfg, "duplicate_detection_round_shuffle_seed", 0))

    current_pool = group_obs
    no_progress_rounds = 0
    round_index = 0

    while len(current_pool) >= 2:
        fits_in_one_chunk = len(current_pool) <= max_reads_per_chunk
        if not fits_in_one_chunk and round_index >= max_rounds:
            logger.warning(
                "duplicate detection round cap (%d) reached for reference=%s sample=%s "
                "core=%d-%d with %d unresolved reads remaining; accepting remaining "
                "survivors as distinct without further cross-comparison -- recall may be "
                "incomplete for this group; consider raising duplicate_detection_max_rounds "
                "or duplicate_detection_max_reads_per_window",
                max_rounds,
                reference,
                sample,
                core_start,
                core_end,
                len(current_pool),
            )
            break

        tasks = plan_duplicate_detection_chunks(
            current_pool,
            reference=reference,
            sample=sample,
            core_start=core_start,
            core_end=core_end,
            load_start=load_start,
            load_end=load_end,
            round_index=round_index,
            max_reads_per_chunk=max_reads_per_chunk,
            target_task_memory_mb=target_task_memory_mb,
            presort_metric=presort_metric,
            round_shuffle_seed=round_shuffle_seed,
        )
        survivors = _dispatch_and_fold(
            tasks,
            spine_path,
            cfg,
            group_obs,
            union_find,
            read_position,
            hamming_minima,
            pool_label=(
                f"dedup {reference}/{sample} core={core_start}-{core_end} "
                f"round={round_index} ({len(tasks)} chunk(s), {len(current_pool)} reads)"
            ),
        )

        if fits_in_one_chunk:
            # Guaranteed-exact stop: the whole pool fit in one chunk, so this pass
            # is exactly as good as this algorithm gets for this group.
            if round_index > 0:
                logger.info(
                    "duplicate detection: reference=%s sample=%s core=%d-%d converged after "
                    "%d round(s), final pass on %d reads",
                    reference,
                    sample,
                    core_start,
                    core_end,
                    round_index + 1,
                    len(current_pool),
                )
            break

        next_pool = current_pool.loc[current_pool.index.isin(survivors)]
        if len(next_pool) == len(current_pool):
            no_progress_rounds += 1
            if no_progress_rounds >= min_progress_rounds:
                logger.info(
                    "duplicate detection: reference=%s sample=%s core=%d-%d stopped after "
                    "%d round(s) (%d consecutive round(s) with no new merges), %d reads remaining",
                    reference,
                    sample,
                    core_start,
                    core_end,
                    round_index + 1,
                    no_progress_rounds,
                    len(next_pool),
                )
                break
        else:
            no_progress_rounds = 0
        current_pool = next_pool
        round_index += 1
