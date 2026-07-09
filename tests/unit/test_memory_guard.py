from __future__ import annotations

import sys
import time

import pytest

from smftools.memory_guard import (
    _cgroup_v2_self_path,
    enable_aggregate_memory_cap,
    start_worker_watchdog,
)


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
