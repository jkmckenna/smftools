"""Platform-aware safety nets to keep smftools pipelines from taking down the host machine.

Two independent mechanisms, chosen because neither one is available/reliable on both
platforms smftools runs on:

- Linux: `enable_aggregate_memory_cap()` places the *entire* CLI process tree (the
  command itself, plus every worker it forks -- multiprocessing children inherit
  their parent's cgroup automatically) into a cgroup v2 with `memory.max` set to a
  safety fraction of total RAM. The kernel enforces this across the whole tree, so
  it protects the machine regardless of how many workers a given command spawns
  internally, and regardless of what else on the machine is also using memory. This
  is called once, at CLI startup, before any worker pool exists.

- macOS (and any platform without a usable process-tree memory cap): there is no
  per-process-tree ceiling available to an unprivileged process, so protection
  instead happens per worker: `start_worker_watchdog()` polls each of a
  multiprocessing.Pool's worker RSS values via psutil and kills any worker that
  exceeds its estimated per-worker budget. This is a narrower guarantee than the
  Linux cgroup (it only sees smftools' own workers, not overall machine pressure),
  which is why it's a fallback rather than the primary mechanism where cgroups are
  available.

Both are best-effort and fail open: if setup fails for any reason (missing
permissions, unsupported kernel, missing psutil, cgroup v2 not mounted), a warning
is logged and the pipeline proceeds without the protection rather than being
blocked by it.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from .logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_AGGREGATE_SAFETY_FRACTION = 0.9
DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 2.0


def enable_aggregate_memory_cap(safety_fraction: float = DEFAULT_AGGREGATE_SAFETY_FRACTION) -> bool:
    """Best-effort: cap this process's (and all its future children's) combined
    memory use via a Linux cgroup v2 `memory.max`.

    No-op (returns False) on any non-Linux platform, or if cgroup v2 isn't mounted,
    isn't delegated to this process, or setup otherwise fails for any reason. Never
    raises -- a failure here must never block the actual pipeline from running.

    Call this once, as early as possible in the CLI entry point, before any
    multiprocessing worker pool is created (children inherit their parent's cgroup
    membership at fork/spawn time, so everything spawned after this call is
    automatically covered).
    """
    if sys.platform != "linux":
        return False
    try:
        return _try_enable_linux_cgroup_cap(safety_fraction)
    except Exception:
        logger.debug("Could not enable aggregate cgroup memory cap", exc_info=True)
        return False


def _cgroup_v2_self_path() -> Path | None:
    """Return this process's own cgroup v2 directory, or None if cgroup v2 isn't
    the active hierarchy (e.g. cgroup v1, or cgroups unavailable)."""
    controllers_file = Path("/sys/fs/cgroup/cgroup.controllers")
    if not controllers_file.exists():
        return None
    try:
        line = Path("/proc/self/cgroup").read_text().strip()
    except OSError:
        return None
    # Unified (v2) hierarchy: exactly one line, "0::/path/to/cgroup".
    for entry in line.splitlines():
        parts = entry.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2].lstrip("/")
            return Path("/sys/fs/cgroup") / rel
    return None


def _try_enable_linux_cgroup_cap(safety_fraction: float) -> bool:
    self_cgroup = _cgroup_v2_self_path()
    if self_cgroup is None or not self_cgroup.is_dir():
        logger.info("cgroup v2 not available; skipping aggregate memory cap")
        return False

    controllers = (self_cgroup / "cgroup.controllers").read_text().split()
    if "memory" not in controllers:
        logger.info(
            "cgroup v2 'memory' controller not available at %s; skipping aggregate memory cap",
            self_cgroup,
        )
        return False

    total_mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    budget_bytes = int(total_mem_bytes * safety_fraction)

    child_cgroup = self_cgroup / f"smftools-{os.getpid()}"
    child_cgroup.mkdir(exist_ok=True)
    try:
        # Delegate the memory controller down to the new child cgroup. May already
        # be enabled, or may not be permitted -- either way, try to proceed; the
        # memory.max write below is the real permission test.
        subtree_control = self_cgroup / "cgroup.subtree_control"
        current = subtree_control.read_text().split()
        if "memory" not in current:
            subtree_control.write_text("+memory\n")
    except OSError:
        pass

    (child_cgroup / "memory.max").write_text(str(budget_bytes))
    (child_cgroup / "cgroup.procs").write_text(str(os.getpid()))

    def _cleanup() -> None:
        try:
            child_cgroup.rmdir()
        except OSError:
            pass  # not empty yet, or already gone -- nothing more we can do at exit

    atexit.register(_cleanup)

    logger.info(
        "Aggregate memory cap enabled via cgroup v2: %.1f GiB budget at %s",
        budget_bytes / (1024**3),
        child_cgroup,
    )
    return True


def start_worker_watchdog(
    pool,
    per_worker_budget_bytes: int,
    poll_interval: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
) -> Callable[[], None]:
    """Best-effort: poll `pool`'s worker processes' RSS and kill any worker that
    exceeds `per_worker_budget_bytes`.

    No-op on Linux (returns a no-op stop callable): the aggregate cgroup cap from
    `enable_aggregate_memory_cap()` already covers the whole process tree there,
    and running both mechanisms at once would just mean two different things can
    kill a worker for the same reason. This is the fallback for platforms (macOS)
    where no process-tree memory cap is available.

    Returns a `stop()` callable that must be called (typically in a `finally`)
    once the pool's work is done, to stop the polling thread.
    """
    if sys.platform == "linux":
        return lambda: None

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed; per-worker memory watchdog disabled")
        return lambda: None

    if per_worker_budget_bytes <= 0:
        logger.debug("No positive per-worker memory budget provided; watchdog disabled")
        return lambda: None

    stop_event = threading.Event()

    def _poll_loop() -> None:
        while not stop_event.is_set():
            for proc in list(getattr(pool, "_pool", [])):
                pid = getattr(proc, "pid", None)
                if pid is None or not proc.is_alive():
                    continue
                try:
                    rss = psutil.Process(pid).memory_info().rss
                except psutil.NoSuchProcess:
                    continue
                if rss > per_worker_budget_bytes:
                    logger.error(
                        "MemoryGuard: worker pid=%d using %.2f GiB > %.2f GiB budget; "
                        "terminating to protect the machine",
                        pid,
                        rss / (1024**3),
                        per_worker_budget_bytes / (1024**3),
                    )
                    try:
                        psutil.Process(pid).kill()
                    except psutil.NoSuchProcess:
                        pass
            stop_event.wait(poll_interval)

    thread = threading.Thread(target=_poll_loop, name="smftools-memory-watchdog", daemon=True)
    thread.start()
    logger.info(
        "Per-worker memory watchdog enabled: %.2f GiB budget/worker, polling every %.1fs",
        per_worker_budget_bytes / (1024**3),
        poll_interval,
    )

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=poll_interval * 2)

    return _stop
