"""Unit tests for the duplicate-detection chunking/multi-round dispatch module.

See dev/duplicate_detection_scaling.md for the design this implements.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

import smftools.memory_guard as memory_guard_module
from smftools.preprocessing.duplicate_detection_dispatch import (
    DuplicateDetectionChunkTask,
    _dispatch_and_fold,
    plan_duplicate_detection_chunks,
    run_duplicate_detection_rounds,
)
from smftools.preprocessing.flag_duplicate_reads import UnionFind

# ---------------------------------------------------------------------------
# plan_duplicate_detection_chunks
# ---------------------------------------------------------------------------


def _obs_slice(n_reads: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"Fraction_any_C_site_modified": rng.random(n_reads)},
        index=[f"read{i}" for i in range(n_reads)],
    )


def test_round_zero_chunks_are_contiguous_by_presort_metric():
    obs = _obs_slice(20)
    tasks = plan_duplicate_detection_chunks(
        obs,
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=100,
        load_start=0,
        load_end=100,
        round_index=0,
        max_reads_per_chunk=5,
        target_task_memory_mb=1024,
        presort_metric="Fraction_any_C_site_modified",
        round_shuffle_seed=0,
    )
    ordered_ids = [read_id for task in tasks for read_id in task.read_ids]
    metric = obs["Fraction_any_C_site_modified"]
    assert list(metric.loc[ordered_ids]) == sorted(metric.loc[ordered_ids])


def test_round_one_reshuffles_instead_of_repeating_round_zero():
    obs = _obs_slice(20)
    kwargs = dict(
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=100,
        load_start=0,
        load_end=100,
        max_reads_per_chunk=5,
        target_task_memory_mb=1024,
        presort_metric="Fraction_any_C_site_modified",
        round_shuffle_seed=0,
    )
    round0_tasks = plan_duplicate_detection_chunks(obs, round_index=0, **kwargs)
    round1_tasks = plan_duplicate_detection_chunks(obs, round_index=1, **kwargs)

    round0_order = [read_id for task in round0_tasks for read_id in task.read_ids]
    round1_order = [read_id for task in round1_tasks for read_id in task.read_ids]
    assert round0_order != round1_order
    # Reproducible: same round_index -> identical plan.
    round1_again = plan_duplicate_detection_chunks(obs, round_index=1, **kwargs)
    round1_again_order = [read_id for task in round1_again for read_id in task.read_ids]
    assert round1_order == round1_again_order


def test_chunk_size_respects_max_reads_per_chunk():
    obs = _obs_slice(23)
    tasks = plan_duplicate_detection_chunks(
        obs,
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=100,
        load_start=0,
        load_end=100,
        round_index=0,
        max_reads_per_chunk=5,
        target_task_memory_mb=1024,
        presort_metric="Fraction_any_C_site_modified",
        round_shuffle_seed=0,
    )
    assert all(task.n_reads <= 5 for task in tasks)
    assert sum(task.n_reads for task in tasks) == 23
    all_read_ids = {read_id for task in tasks for read_id in task.read_ids}
    assert all_read_ids == set(obs.index)


def test_chunk_size_bounded_by_target_task_memory_mb_when_smaller():
    obs = _obs_slice(20)
    # 1MB budget / (100,000-wide window * 8 bytes/position) = 1 read/chunk --
    # far below max_reads_per_chunk=1000, so this exercises the memory-derived
    # cap specifically, not the max_reads_per_chunk ceiling.
    tasks = plan_duplicate_detection_chunks(
        obs,
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=100_000,
        load_start=0,
        load_end=100_000,
        round_index=0,
        max_reads_per_chunk=1000,
        target_task_memory_mb=1,
        presort_metric="Fraction_any_C_site_modified",
        round_shuffle_seed=0,
    )
    assert isinstance(tasks[0], DuplicateDetectionChunkTask)
    assert len(tasks) == 20
    assert all(task.n_reads == 1 for task in tasks)


# ---------------------------------------------------------------------------
# _dispatch_and_fold: wiring + fold-in correctness
# ---------------------------------------------------------------------------


def test_dispatch_and_fold_wires_run_tasks_parallel_and_folds_results(monkeypatch):
    hamming_columns = (
        "fwd_hamming_to_next",
        "rev_hamming_to_prev",
        "sequence__hier_hamming_to_pair",
        "sequence__min_hamming_to_pair",
    )
    captured = {}

    def fake_run_tasks_parallel(
        worker, task_args_list, *, cfg, force_sequential=False, pool_label=None
    ):
        captured["worker"] = worker
        captured["task_args_list"] = task_args_list
        captured["cfg"] = cfg
        # Canned result: read0/read1 merge into one cluster, read2 is a singleton.
        return [
            {
                "obs_index": ["read0", "read1", "read2"],
                "sequence__is_duplicate": [False, True, False],
                "sequence__merged_cluster_id": [0, 0, 1],
                "sequence__cluster_size": [2, 2, 1],
                "fwd_hamming_to_next": [0.01, np.nan, np.nan],
                "rev_hamming_to_prev": [np.nan, 0.01, np.nan],
                "sequence__hier_hamming_to_pair": [np.nan, np.nan, np.nan],
                "sequence__min_hamming_to_pair": [0.01, 0.01, np.nan],
            }
        ]

    monkeypatch.setattr(memory_guard_module, "run_tasks_parallel", fake_run_tasks_parallel)

    task = DuplicateDetectionChunkTask(
        task_id="t",
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=10,
        load_start=0,
        load_end=10,
        round_index=0,
        chunk_index=0,
        n_reads=3,
        estimated_memory_bytes=0,
        read_ids=("read0", "read1", "read2"),
    )
    group_obs = pd.DataFrame(index=["read0", "read1", "read2"])
    read_position = {"read0": 0, "read1": 1, "read2": 2}
    union_find = UnionFind(3)
    hamming_minima = {column: np.full(3, np.nan) for column in hamming_columns}
    cfg = SimpleNamespace()

    survivors = _dispatch_and_fold(
        [task], "spine.h5ad", cfg, group_obs, union_find, read_position, hamming_minima
    )

    assert captured["cfg"] is cfg
    assert len(captured["task_args_list"]) == 1
    spine_arg, task_arg, cfg_arg, obs_slice_arg = captured["task_args_list"][0]
    assert spine_arg == "spine.h5ad"
    assert task_arg is task
    assert cfg_arg is cfg
    assert list(obs_slice_arg.index) == ["read0", "read1", "read2"]

    assert sorted(survivors) == ["read0", "read2"]
    assert union_find.find(read_position["read0"]) == union_find.find(read_position["read1"])
    assert union_find.find(read_position["read2"]) != union_find.find(read_position["read0"])
    assert hamming_minima["fwd_hamming_to_next"][read_position["read0"]] == 0.01


def test_dispatch_and_fold_carries_forward_reads_from_none_chunk_results(monkeypatch):
    """A chunk with <2 reads or no valid sites returns None from _process_group --
    those reads must still survive to the next round, not be silently dropped.
    """

    def fake_run_tasks_parallel(
        worker, task_args_list, *, cfg, force_sequential=False, pool_label=None
    ):
        return [None for _ in task_args_list]

    monkeypatch.setattr(memory_guard_module, "run_tasks_parallel", fake_run_tasks_parallel)

    task = DuplicateDetectionChunkTask(
        task_id="t",
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=10,
        load_start=0,
        load_end=10,
        round_index=0,
        chunk_index=0,
        n_reads=1,
        estimated_memory_bytes=0,
        read_ids=("read0",),
    )
    group_obs = pd.DataFrame(index=["read0"])
    read_position = {"read0": 0}
    union_find = UnionFind(1)
    hamming_minima = {"fwd_hamming_to_next": np.full(1, np.nan)}
    cfg = SimpleNamespace()

    survivors = _dispatch_and_fold(
        [task], "spine.h5ad", cfg, group_obs, union_find, read_position, hamming_minima
    )
    assert survivors == ["read0"]


# ---------------------------------------------------------------------------
# run_duplicate_detection_rounds: round-cap graceful degradation
# ---------------------------------------------------------------------------


def test_round_cap_terminates_without_crashing_and_logs_warning(monkeypatch, caplog):
    """Engineer a group where exactly one merge happens per round (in the first
    chunk only) so the survivor pool shrinks by 1 read/round -- far too slowly to
    ever fit under max_reads_per_chunk within the small max_rounds cap used here.
    Must terminate cleanly (no exception), with a warning naming the cap, and
    without raising -- not a silent infinite loop or crash.
    """
    n_reads = 12
    read_ids = [f"read{i}" for i in range(n_reads)]
    group_obs = pd.DataFrame(index=read_ids)
    read_position = {read_id: i for i, read_id in enumerate(read_ids)}
    union_find = UnionFind(n_reads)
    hamming_minima = {"fwd_hamming_to_next": np.full(n_reads, np.nan)}

    def fake_execute_chunk_task(spine_path, task, cfg, obs_slice):
        ids = list(task.read_ids)
        n = len(ids)
        is_duplicate = [False] * n
        cluster_ids = list(range(n))
        cluster_sizes = [1] * n
        if task.chunk_index == 0 and n >= 2:
            # Merge exactly the first pair in the first chunk of this round only.
            is_duplicate[1] = True
            cluster_ids[1] = cluster_ids[0]
            cluster_sizes[0] = 2
            cluster_sizes[1] = 2
        return {
            "obs_index": ids,
            "sequence__is_duplicate": is_duplicate,
            "sequence__merged_cluster_id": cluster_ids,
            "sequence__cluster_size": cluster_sizes,
            "fwd_hamming_to_next": [np.nan] * n,
        }

    def fake_run_tasks_parallel(
        worker, task_args_list, *, cfg, force_sequential=False, pool_label=None
    ):
        return [fake_execute_chunk_task(*args) for args in task_args_list]

    monkeypatch.setattr(memory_guard_module, "run_tasks_parallel", fake_run_tasks_parallel)

    cfg = SimpleNamespace(
        duplicate_detection_max_reads_per_window=5,
        target_task_memory_mb=512,
        duplicate_detection_max_rounds=3,
        duplicate_detection_min_progress_rounds_before_stop=1,
        duplicate_detection_chunk_presort_metric="Fraction_any_C_site_modified",
        duplicate_detection_round_shuffle_seed=0,
    )

    with caplog.at_level(logging.WARNING):
        run_duplicate_detection_rounds(
            "spine.h5ad",
            group_obs,
            reference="ref",
            sample="bc1",
            core_start=0,
            core_end=10,
            load_start=0,
            load_end=10,
            cfg=cfg,
            union_find=union_find,
            read_position=read_position,
            hamming_minima=hamming_minima,
        )

    assert any("round cap" in record.message for record in caplog.records)
    assert any("3" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# run_duplicate_detection_rounds: cross-chunk correctness with real
# _process_group/execute_duplicate_detection_chunk_task logic (materialize
# faked, chunk planning forced) -- proves union-find composition actually
# works end to end, not just with canned results.
# ---------------------------------------------------------------------------


def _fake_materialize_factory(x_by_read: dict, var: pd.DataFrame):
    def fake_materialize(spine_path, *, references, read_ids, start, end, layers):
        import anndata as ad

        read_ids = list(read_ids)
        x = np.vstack([x_by_read[read_id] for read_id in read_ids])
        return ad.AnnData(
            X=x,
            obs=pd.DataFrame(index=read_ids),
            var=var.copy(),
        )

    return fake_materialize


def test_cross_chunk_duplicate_pair_reconciled_after_round_two(monkeypatch):
    import smftools.preprocessing.duplicate_detection_dispatch as dispatch_module

    n_sites = 20
    rng = np.random.default_rng(5)
    read_a = rng.choice([0.0, 1.0], size=n_sites)
    read_b = read_a.copy()
    read_b[0] = 1.0 - read_b[0]  # true near-duplicate: differs at exactly 1/20 sites
    read_c = rng.choice([0.0, 1.0], size=n_sites)  # unrelated filler
    read_d = rng.choice([0.0, 1.0], size=n_sites)  # unrelated filler

    x_by_read = {"readA": read_a, "readB": read_b, "readC": read_c, "readD": read_d}
    var = pd.DataFrame({"ref_GpC_site": [True] * n_sites}, index=[str(i) for i in range(n_sites)])
    monkeypatch.setattr(dispatch_module, "materialize", _fake_materialize_factory(x_by_read, var))

    def fake_run_tasks_parallel(
        worker, task_args_list, *, cfg, force_sequential=False, pool_label=None
    ):
        return [worker(*args) for args in task_args_list]

    monkeypatch.setattr(memory_guard_module, "run_tasks_parallel", fake_run_tasks_parallel)

    def _task(round_index, chunk_index, read_ids):
        return DuplicateDetectionChunkTask(
            task_id=f"round{round_index}chunk{chunk_index}",
            reference="ref",
            sample="bc1",
            core_start=0,
            core_end=n_sites,
            load_start=0,
            load_end=n_sites,
            round_index=round_index,
            chunk_index=chunk_index,
            n_reads=len(read_ids),
            estimated_memory_bytes=0,
            read_ids=tuple(read_ids),
        )

    def fake_plan_chunks(obs_slice, *, round_index, **kwargs):
        remaining = list(obs_slice.index)
        if round_index == 0:
            # Force readA and readB into DIFFERENT chunks (each with an
            # unrelated filler) -- round 0 must NOT be able to merge them.
            return [
                _task(0, 0, [r for r in remaining if r in ("readA", "readC")]),
                _task(0, 1, [r for r in remaining if r in ("readB", "readD")]),
            ]
        # Round 1+: whatever survived pools into a single chunk.
        return [_task(round_index, 0, remaining)]

    monkeypatch.setattr(dispatch_module, "plan_duplicate_detection_chunks", fake_plan_chunks)

    read_ids = ["readA", "readB", "readC", "readD"]
    group_obs = pd.DataFrame(index=read_ids)
    read_position = {read_id: i for i, read_id in enumerate(read_ids)}
    union_find = UnionFind(len(read_ids))
    hamming_columns = (
        "fwd_hamming_to_next",
        "rev_hamming_to_prev",
        "sequence__hier_hamming_to_pair",
        "sequence__min_hamming_to_pair",
    )
    hamming_minima = {column: np.full(len(read_ids), np.nan) for column in hamming_columns}

    cfg = SimpleNamespace(
        duplicate_detection_max_reads_per_window=2,
        target_task_memory_mb=512,
        duplicate_detection_max_rounds=6,
        # Round 0 is deliberately engineered (via fake_plan_chunks) to find zero
        # merges -- readA/readB are forced apart -- so the default
        # min_progress_rounds_before_stop=1 would stop right after round 0 and
        # never reach round 1's reunification. This test isn't exercising that
        # default heuristic (a separate, accepted recall-vs-compute tradeoff);
        # it's proving the underlying chunk/round/union-find machinery correctly
        # reconciles a cross-chunk pair once given the chance, so progress-
        # detection is relaxed to allow one non-progressing round through.
        duplicate_detection_min_progress_rounds_before_stop=2,
        duplicate_detection_chunk_presort_metric="Fraction_any_C_site_modified",
        duplicate_detection_round_shuffle_seed=0,
        duplicate_detection_site_types=["GpC"],
        duplicate_detection_distance_threshold=0.2,
        duplicate_detection_window_size_for_hamming_neighbors=50,
        duplicate_detection_min_overlapping_positions=5,
        duplicate_detection_keep_best_metric=None,
        duplicate_detection_do_hierarchical=False,
        duplicate_detection_hierarchical_linkage="average",
        duplicate_detection_do_pca=False,
        duplicate_detection_pca_n_components=50,
        duplicate_detection_pca_center=True,
        duplicate_detection_demux_types_to_use=[],
        duplicate_detection_hierarchical_max_representatives=5000,
        duplicate_detection_n_permutation_passes=0,
        duplicate_detection_permutation_seed=0,
    )

    run_duplicate_detection_rounds(
        "spine.h5ad",
        group_obs,
        reference="ref",
        sample="bc1",
        core_start=0,
        core_end=n_sites,
        load_start=0,
        load_end=n_sites,
        cfg=cfg,
        union_find=union_find,
        read_position=read_position,
        hamming_minima=hamming_minima,
    )

    assert union_find.find(read_position["readA"]) == union_find.find(read_position["readB"])
    # Fillers must not have been swept into that cluster.
    assert union_find.find(read_position["readC"]) != union_find.find(read_position["readA"])
    assert union_find.find(read_position["readD"]) != union_find.find(read_position["readA"])
