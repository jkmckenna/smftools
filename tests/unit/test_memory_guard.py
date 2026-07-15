from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from smftools.memory_guard import (
    _cgroup_v2_self_path,
    _live_worker_pids,
    enable_aggregate_memory_cap,
    resolve_max_workers,
    resolve_memory_budget_bytes,
    run_tasks_parallel,
    start_worker_watchdog,
)


def _cfg(**overrides):
    defaults = dict(
        threads=4,
        max_memory_percent=60.0,
        max_memory_gb=None,
        target_task_memory_mb=512,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _square(x: int) -> int:
    """Module-level (picklable) worker for run_tasks_parallel's process-pool
    path -- a spawned worker process can only unpickle a top-level function."""
    return x * x


def test_resolve_memory_budget_bytes_uses_percent_of_total(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 100 * 1024**3)
    budget = resolve_memory_budget_bytes(_cfg(max_memory_percent=60.0, max_memory_gb=None))
    assert budget == int(60 * 1024**3)


def test_resolve_memory_budget_bytes_uses_fixed_gb(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 100 * 1024**3)
    budget = resolve_memory_budget_bytes(_cfg(max_memory_percent=None, max_memory_gb=32.0))
    assert budget == int(32 * 1024**3)


def test_resolve_memory_budget_bytes_takes_the_more_restrictive(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 100 * 1024**3)
    # 60% of 100GiB = 60GiB, fixed cap = 10GiB -- the fixed cap should win.
    tighter = resolve_memory_budget_bytes(_cfg(max_memory_percent=60.0, max_memory_gb=10.0))
    assert tighter == int(10 * 1024**3)
    # 60% of 100GiB = 60GiB, fixed cap = 200GiB -- the percent should win.
    looser = resolve_memory_budget_bytes(_cfg(max_memory_percent=60.0, max_memory_gb=200.0))
    assert looser == int(60 * 1024**3)


def test_resolve_memory_budget_bytes_falls_back_without_config_fields(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 100 * 1024**3)
    budget = resolve_memory_budget_bytes(SimpleNamespace())
    assert budget == int(100 * 1024**3 * 0.9)


def test_resolve_max_workers_caps_by_threads(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 1024 * 1024**3)
    assert resolve_max_workers(_cfg(threads=4), n_items=100, per_item_memory_mb=1.0) == 4


def test_resolve_max_workers_caps_by_item_count(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 1024 * 1024**3)
    assert resolve_max_workers(_cfg(threads=8), n_items=3, per_item_memory_mb=1.0) == 3


def test_resolve_max_workers_caps_by_memory_budget(monkeypatch) -> None:
    # 10 GiB total, 60% budget = 6 GiB; 2048 MB/item -> 3 items fit (6144/2048).
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 10 * 1024**3)
    workers = resolve_max_workers(
        _cfg(threads=8, max_memory_percent=60.0), n_items=100, per_item_memory_mb=2048.0
    )
    assert workers == 3


def test_resolve_max_workers_uses_target_task_memory_mb_by_default(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 10 * 1024**3)
    workers = resolve_max_workers(
        _cfg(threads=8, max_memory_percent=60.0, target_task_memory_mb=2048), n_items=100
    )
    assert workers == 3


def test_resolve_max_workers_always_at_least_one() -> None:
    assert resolve_max_workers(_cfg(threads=4), n_items=0) == 1


def test_live_worker_pids_from_multiprocessing_pool_shape() -> None:
    alive = SimpleNamespace(pid=111, is_alive=lambda: True)
    dead = SimpleNamespace(pid=222, is_alive=lambda: False)
    pool = SimpleNamespace(_pool=[alive, dead])
    assert _live_worker_pids(pool) == [111]


def test_live_worker_pids_from_process_pool_executor_shape() -> None:
    alive = SimpleNamespace(is_alive=lambda: True)
    dead = SimpleNamespace(is_alive=lambda: False)
    pool = SimpleNamespace(_processes={111: alive, 222: dead})
    assert _live_worker_pids(pool) == [111]


def test_live_worker_pids_empty_for_unrecognized_pool_shape() -> None:
    assert _live_worker_pids(SimpleNamespace()) == []


def test_enable_aggregate_memory_cap_budget_override_used_directly(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    captured = {}

    def _fake_try_enable(safety_fraction, *, budget_bytes=None):
        captured["budget_bytes"] = budget_bytes
        return True

    monkeypatch.setattr("smftools.memory_guard._try_enable_linux_cgroup_cap", _fake_try_enable)
    assert enable_aggregate_memory_cap(budget_bytes=12345) is True
    assert captured["budget_bytes"] == 12345


def test_run_tasks_parallel_sequential_path_matches_expected(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 1024 * 1024**3)
    cfg = _cfg(threads=1)
    results = run_tasks_parallel(_square, [(1,), (2,), (3,), (4,)], cfg=cfg)
    assert results == [1, 4, 9, 16]


def test_run_tasks_parallel_process_pool_path_preserves_order(monkeypatch) -> None:
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 1024 * 1024**3)
    cfg = _cfg(threads=4, target_task_memory_mb=1)
    results = run_tasks_parallel(_square, [(1,), (2,), (3,), (4,), (5,)], cfg=cfg)
    assert results == [1, 4, 9, 16, 25]


def test_run_tasks_parallel_empty_task_list() -> None:
    assert run_tasks_parallel(_square, [], cfg=_cfg()) == []


def test_run_tasks_parallel_force_sequential_skips_pool_regardless_of_resources(
    monkeypatch,
) -> None:
    # Plenty of threads/memory to warrant a pool -- force_sequential must
    # still win. Regression test for HMM's GPU-sharing crash: multiple
    # worker *processes* concurrently touching one GPU context isn't safe
    # the way CPU-bound task parallelism is, so callers with that
    # constraint need a way to opt out of pooling entirely regardless of
    # what resolve_max_workers would otherwise decide.
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 1024 * 1024**3)
    cfg = _cfg(threads=8, target_task_memory_mb=1)

    def _fail_if_pool_created(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor must not be created when force_sequential=True")

    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor", _fail_if_pool_created, raising=True
    )
    results = run_tasks_parallel(
        _square, [(1,), (2,), (3,), (4,)], cfg=cfg, force_sequential=True
    )
    assert results == [1, 4, 9, 16]


def test_enable_aggregate_memory_cap_noop_on_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert enable_aggregate_memory_cap() is False


def test_enable_aggregate_memory_cap_noop_without_cgroup_v2(monkeypatch) -> None:
    # Force the platform check too, so this is meaningful even when actually run
    # on a Linux machine that happens to have cgroup v2 mounted.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("smftools.memory_guard._cgroup_v2_self_path", lambda: None)
    assert enable_aggregate_memory_cap() is False


def test_cgroup_v2_self_path_returns_none_when_unmounted() -> None:
    if sys.platform == "linux":
        pytest.skip("cgroup v2 may genuinely be mounted here; covered by the mocked test above")
    # No /sys/fs/cgroup/cgroup.controllers on macOS -- exercises the real,
    # unmocked code path rather than simulating it.
    assert _cgroup_v2_self_path() is None


def test_start_worker_watchdog_noop_on_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    stop = start_worker_watchdog(pool=None, per_worker_budget_bytes=1)
    stop()  # must not raise, and must not have tried to touch `pool`


def test_start_worker_watchdog_noop_with_nonpositive_budget(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    stop = start_worker_watchdog(pool=None, per_worker_budget_bytes=0)
    stop()  # must not raise


def _hog_memory(mb: int, hold_seconds: float) -> None:
    block = bytearray(mb * 1024 * 1024)
    # Touch every page so RSS actually reflects the allocation immediately,
    # rather than relying on lazy fault-in timing.
    for i in range(0, len(block), 4096):
        block[i] = 1
    time.sleep(hold_seconds)


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="watchdog is a no-op on Linux by design (aggregate cgroup cap covers it instead)",
)
def test_watchdog_kills_overbudget_worker() -> None:
    """End-to-end: a worker that blows well past its budget actually gets killed."""
    import multiprocessing as mp

    import psutil

    budget_bytes = 20 * 1024 * 1024  # 20 MiB
    with mp.Pool(processes=1) as pool:
        stop = start_worker_watchdog(pool, budget_bytes, poll_interval=0.2)
        try:
            # Allocate 200 MiB (10x budget) and hold it for 10s -- long enough
            # that "the worker finished normally" can't be mistaken for "the
            # watchdog killed it".
            pool.apply_async(_hog_memory, (200, 10.0))

            pid = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pid is None:
                pids = [p.pid for p in pool._pool if p.pid is not None]
                if pids:
                    pid = pids[0]
                time.sleep(0.05)
            assert pid is not None, "worker process never started"

            alive = True
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    alive = psutil.Process(pid).is_running()
                except psutil.NoSuchProcess:
                    alive = False
                if not alive:
                    break
                time.sleep(0.1)
            assert not alive, f"worker pid={pid} was not killed by the watchdog within 8s"
        finally:
            stop()


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="watchdog is a no-op on Linux by design (aggregate cgroup cap covers it instead)",
)
def test_watchdog_kills_overbudget_worker_via_process_pool_executor() -> None:
    """Same as test_watchdog_kills_overbudget_worker but for ProcessPoolExecutor,
    the pool type the parallel task/extraction loops use -- proves
    _live_worker_pids' generalization actually works end-to-end, not just at
    the unit level."""
    from concurrent.futures import ProcessPoolExecutor

    import psutil

    budget_bytes = 20 * 1024 * 1024  # 20 MiB
    with ProcessPoolExecutor(max_workers=1) as pool:
        stop = start_worker_watchdog(pool, budget_bytes, poll_interval=0.2)
        try:
            future = pool.submit(_hog_memory, 200, 10.0)

            pid = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pid is None:
                pids = _live_worker_pids(pool)
                if pids:
                    pid = pids[0]
                time.sleep(0.05)
            assert pid is not None, "worker process never started"

            alive = True
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    alive = psutil.Process(pid).is_running()
                except psutil.NoSuchProcess:
                    alive = False
                if not alive:
                    break
                time.sleep(0.1)
            assert not alive, f"worker pid={pid} was not killed by the watchdog within 8s"
            with pytest.raises(Exception):
                future.result(timeout=5)
        finally:
            stop()
