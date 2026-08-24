"""Duplicate detection dispatches groups together, with the same answer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smftools.preprocessing.duplicate_detection_dispatch import merge_duplicate_detection_group
from smftools.preprocessing.flag_duplicate_reads import UnionFind

pytestmark = pytest.mark.unit

COLUMNS = ("fwd_hamming_to_next", "sequence__min_hamming_to_pair")


def _state(read_ids):
    position = {read_id: index for index, read_id in enumerate(read_ids)}
    return (
        UnionFind(len(read_ids)),
        position,
        {c: np.full(len(read_ids), np.nan, dtype=float) for c in COLUMNS},
    )


def _clusters(union_find, read_ids, position):
    groups: dict[int, set[str]] = {}
    for read_id in read_ids:
        groups.setdefault(union_find.find(position[read_id]), set()).add(read_id)
    return sorted(sorted(members) for members in groups.values())


def test_merge_order_does_not_change_the_clusters():
    """Order-independence is what licenses dispatching groups in parallel.

    Union-find composition does not care which pass found a pair, so folding
    contributions in any order must give the sequential answer.
    """
    reads = [f"r{i}" for i in range(6)]
    contributions = [
        ([("r0", "r1")], {}),
        ([("r2", "r3")], {}),
        ([("r1", "r2")], {}),  # bridges the first two groups
    ]

    forward_uf, forward_pos, forward_min = _state(reads)
    for contribution in contributions:
        merge_duplicate_detection_group(contribution, forward_uf, forward_pos, forward_min)

    reverse_uf, reverse_pos, reverse_min = _state(reads)
    for contribution in reversed(contributions):
        merge_duplicate_detection_group(contribution, reverse_uf, reverse_pos, reverse_min)

    assert _clusters(forward_uf, reads, forward_pos) == _clusters(reverse_uf, reads, reverse_pos)
    assert _clusters(forward_uf, reads, forward_pos) == [["r0", "r1", "r2", "r3"], ["r4"], ["r5"]]


def test_a_read_in_two_groups_is_merged_idempotently():
    """Reads spanning a tile boundary appear in two groups by design."""
    reads = ["r0", "r1", "r2"]
    union_find, position, minima = _state(reads)

    for _ in range(3):
        merge_duplicate_detection_group(([("r0", "r1")], {}), union_find, position, minima)

    assert _clusters(union_find, reads, position) == [["r0", "r1"], ["r2"]]


def test_hamming_minima_take_the_smallest_across_groups():
    """A per-read minimum is order-independent, so groups may merge in any order."""
    reads = ["r0", "r1"]
    union_find, position, minima = _state(reads)

    merge_duplicate_detection_group(([], {COLUMNS[0]: {"r0": 5.0}}), union_find, position, minima)
    merge_duplicate_detection_group(([], {COLUMNS[0]: {"r0": 2.0}}), union_find, position, minima)
    merge_duplicate_detection_group(([], {COLUMNS[0]: {"r0": 9.0}}), union_find, position, minima)

    assert minima[COLUMNS[0]][0] == 2.0
    assert np.isnan(minima[COLUMNS[0]][1])


def test_unknown_columns_are_ignored_rather_than_raising():
    """A worker built against a different column set must not abort the merge."""
    reads = ["r0"]
    union_find, position, minima = _state(reads)

    merge_duplicate_detection_group(
        ([], {"not_a_column": {"r0": 1.0}}), union_find, position, minima
    )

    assert all(np.isnan(values).all() for values in minima.values())


def test_the_executor_dispatches_groups_rather_than_walking_them():
    """The sequential walk is what put 314 groups on a five-hour path."""
    import inspect

    from smftools.preprocessing import partitioned_executor

    source = inspect.getsource(partitioned_executor.reduce_duplicate_reads)
    assert "dedup_groups.append" in source
    assert "run_tasks_parallel" in source
    assert "run_duplicate_detection_rounds(" not in source


def test_a_group_worker_does_not_start_a_nested_pool(monkeypatch, tmp_path):
    """Chunk dispatch inside a group worker must run sequentially (`F49`).

    Without this the worker re-reads `cfg.threads` and starts its own pool, so
    12 group workers each spawn their own children: a four-level process tree,
    63.6 GiB aggregate against a 76.8 GiB budget on a real run, and watchdog
    kills measured on process-tree RSS that looked like a bad memory estimate.
    """
    import smftools.preprocessing.duplicate_detection_dispatch as dispatch

    seen: list[bool] = []

    def _capture(worker, task_args, *, cfg, force_sequential=False, **kwargs):
        seen.append(force_sequential)
        return []

    import pandas as pd

    monkeypatch.setattr("smftools.memory_guard.run_tasks_parallel", _capture)
    task = SimpleNamespace(read_ids=["r0"], estimated_memory_bytes=1024)

    dispatch._dispatch_and_fold(
        [task],
        tmp_path / "spine.h5ad",
        object(),
        pd.DataFrame(index=["r0"]),
        UnionFind(1),
        {"r0": 0},
        {},
        force_sequential=True,
    )

    assert seen == [True], "the inner dispatch must be told it is already in a worker"


def test_the_group_worker_requests_sequential_chunk_dispatch():
    """Pin the call site, not just the plumbing."""
    import inspect

    from smftools.preprocessing.duplicate_detection_dispatch import (
        execute_duplicate_detection_group,
    )

    source = inspect.getsource(execute_duplicate_detection_group)
    assert "force_sequential=True" in source
