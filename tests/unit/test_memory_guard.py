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
    results = run_tasks_parallel(_square, [(1,), (2,), (3,), (4,)], cfg=cfg, force_sequential=True)
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


def test_watchdog_tolerance_and_grace_period(monkeypatch) -> None:
    # Regression test for a real batch failure: the watchdog used to kill on
    # the very first sample over the *bare* per-worker budget, with no
    # tolerance or grace period -- since that budget is an even split of the
    # aggregate (resolve_memory_budget_bytes // max_workers), not a measured
    # per-task estimate, ordinary run-to-run variance regularly puts several
    # workers a few percent over it at once, aborting the whole pool.
    # Confirmed on real data: 8 of 11 batch experiments were aborted this
    # way, each killed worker only 0-4% over budget.
    monkeypatch.setattr(sys, "platform", "darwin")

    # pid 101: sustained ~10% over budget, forever -- within the 20% default
    # tolerance, must never be killed.
    # pid 102: alternates over/under threshold every poll -- never *sustains*
    # over-threshold for grace_polls consecutive polls, must never be killed.
    # pid 103: sustained ~30% over budget, forever -- beyond tolerance, must
    # be killed once it has been over-threshold for grace_polls consecutive
    # polls.
    rss_sequences_mb = {101: [110], 102: [130, 90], 103: [130]}
    call_counts = {pid: 0 for pid in rss_sequences_mb}
    killed: list[int] = []

    class _FakeMemInfo:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class _FakeNoSuchProcess(Exception):
        pass

    class _FakePsutilProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self) -> _FakeMemInfo:
            seq = rss_sequences_mb[self.pid]
            idx = call_counts[self.pid] % len(seq)
            call_counts[self.pid] += 1
            return _FakeMemInfo(seq[idx] * 1024 * 1024)

        def kill(self) -> None:
            killed.append(self.pid)

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_alive(self) -> bool:
            return True

    fake_pool = SimpleNamespace(_processes={pid: _FakeProc(pid) for pid in rss_sequences_mb})
    fake_psutil = SimpleNamespace(Process=_FakePsutilProcess, NoSuchProcess=_FakeNoSuchProcess)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    budget_bytes = 100 * 1024 * 1024
    stop = start_worker_watchdog(
        fake_pool,
        budget_bytes,
        poll_interval=0.02,
        tolerance_fraction=0.2,
        grace_polls=3,
    )
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and 103 not in killed:
            time.sleep(0.02)
    finally:
        stop()

    assert 101 not in killed, "sustained-but-within-tolerance worker must not be killed"
    assert 102 not in killed, "worker with only transient spikes must not be killed"
    assert 103 in killed, "worker sustained beyond tolerance for grace_polls must be killed"


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


def _hog_and_return(mb: int, hold_seconds: float, value: int) -> int:
    block = bytearray(mb * 1024 * 1024)
    for i in range(0, len(block), 4096):
        block[i] = 1
    time.sleep(hold_seconds)
    return value


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="watchdog is a no-op on Linux by design (aggregate cgroup cap covers it instead)",
)
def test_run_tasks_parallel_retries_with_fewer_workers_after_pool_breaks(monkeypatch) -> None:
    # Regression test for a real batch failure: when the watchdog kills a
    # worker, the whole ProcessPoolExecutor becomes unusable and every other
    # pending future raises BrokenProcessPool -- the previous behavior let
    # that propagate straight out of run_tasks_parallel, aborting the entire
    # task list (and, for batch callers, the whole experiment) over a single
    # killed worker. Confirmed on real data: 8 of 11 batch experiments were
    # aborted this way in a single run, because resolve_max_workers's
    # per-item memory estimate badly underestimated a task type's real
    # footprint, packing more concurrent workers than the machine could
    # actually hold. run_tasks_parallel now retries whatever tasks hadn't
    # produced a result yet in a fresh, smaller pool instead of raising.
    import smftools.memory_guard as memory_guard_module

    real_start_watchdog = memory_guard_module.start_worker_watchdog
    monkeypatch.setattr(
        memory_guard_module,
        "start_worker_watchdog",
        lambda pool, budget: real_start_watchdog(pool, budget, poll_interval=0.1),
    )
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 10 * 1024**3)

    # 2 workers initially (threads=2). Aggregate budget ~450 MiB: at 2
    # workers the per-worker budget+tolerance (~270 MiB) is comfortably
    # under each task's real ~300 MiB footprint, so both get killed; halved
    # to 1 worker on retry, the per-worker budget becomes the full ~450 MiB
    # aggregate (+tolerance ~540 MiB), comfortably enough for one 300 MiB
    # task at a time.
    cfg = _cfg(
        threads=2,
        max_memory_percent=None,
        max_memory_gb=450.0 / 1024,
        target_task_memory_mb=1,
    )
    results = run_tasks_parallel(_hog_and_return, [(300, 6.0, 1), (300, 6.0, 2)], cfg=cfg)
    assert results == [1, 2]
