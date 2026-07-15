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


def total_system_memory_bytes() -> int:
    """Total physical RAM, cross-platform (mac/linux), via psutil."""
    import psutil

    return int(psutil.virtual_memory().total)


def resolve_memory_budget_bytes(cfg) -> int:
    """Resolve the aggregate memory budget (bytes) for the whole workflow from
    ``cfg.max_memory_percent``/``cfg.max_memory_gb`` -- the more restrictive of
    the two applies when both are set (a fixed GB ceiling doesn't relax the
    percent-of-RAM default, and vice versa).

    Falls back to ``DEFAULT_AGGREGATE_SAFETY_FRACTION`` of total RAM if neither
    config field is set (e.g. a bare/legacy config, or ``cfg`` not an
    ``ExperimentConfig`` at all) -- matches this module's pre-existing default
    before these fields existed.
    """
    total = total_system_memory_bytes()
    candidates: list[float] = []
    percent = getattr(cfg, "max_memory_percent", None)
    if percent is not None:
        candidates.append(total * (float(percent) / 100.0))
    gb = getattr(cfg, "max_memory_gb", None)
    if gb is not None:
        candidates.append(float(gb) * (1024**3))
    if not candidates:
        candidates.append(total * DEFAULT_AGGREGATE_SAFETY_FRACTION)
    return max(1, int(min(candidates)))


def resolve_max_workers(cfg, n_items: int, *, per_item_memory_mb: float | None = None) -> int:
    """How many workers to run concurrently for a pool of ``n_items`` tasks.

    Caps ``cfg.threads`` by both the item count (no point starting idle
    workers) and the aggregate memory budget (``resolve_memory_budget_bytes``)
    divided by a per-item memory estimate -- defaults to
    ``cfg.target_task_memory_mb``, the same per-task estimate already used for
    preprocess task planning, when ``per_item_memory_mb`` isn't given
    explicitly (e.g. raw-ingestion extraction buckets, which don't have a
    ``target_task_memory_mb``-shaped estimate of their own).
    """
    threads = max(1, int(getattr(cfg, "threads", 1) or 1))
    if n_items <= 0:
        return 1
    if per_item_memory_mb is None:
        per_item_memory_mb = float(getattr(cfg, "target_task_memory_mb", 512) or 512)
    per_item_bytes = max(1.0, per_item_memory_mb) * (1024**2)
    budget_bytes = resolve_memory_budget_bytes(cfg)
    by_memory = max(1, int(budget_bytes // per_item_bytes))
    return max(1, min(threads, n_items, by_memory))


def enable_aggregate_memory_cap(
    safety_fraction: float = DEFAULT_AGGREGATE_SAFETY_FRACTION,
    *,
    budget_bytes: int | None = None,
) -> bool:
    """Best-effort: cap this process's (and all its future children's) combined
    memory use via a Linux cgroup v2 `memory.max`.

    No-op (returns False) on any non-Linux platform, or if cgroup v2 isn't mounted,
    isn't delegated to this process, or setup otherwise fails for any reason. Never
    raises -- a failure here must never block the actual pipeline from running.

    Call this once, as early as possible in the CLI entry point, before any
    multiprocessing worker pool is created (children inherit their parent's cgroup
    membership at fork/spawn time, so everything spawned after this call is
    automatically covered). Safe to call again later in the same process (e.g.
    once an ``ExperimentConfig`` is available and ``budget_bytes`` can be
    resolved from it via ``resolve_memory_budget_bytes``) -- it re-targets the
    same per-PID cgroup directory rather than creating a new one, so a later,
    more precise call simply refines the budget already in effect.

    ``budget_bytes``, when given, is used directly instead of
    ``safety_fraction * total_system_memory_bytes()``.
    """
    if sys.platform != "linux":
        return False
    try:
        return _try_enable_linux_cgroup_cap(safety_fraction, budget_bytes=budget_bytes)
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


def _try_enable_linux_cgroup_cap(safety_fraction: float, *, budget_bytes: int | None = None) -> bool:
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

    if budget_bytes is None:
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


def _live_worker_pids(pool) -> list[int]:
    """Extract live worker PIDs from either a ``multiprocessing.pool.Pool``
    (``._pool``, a list of ``Process`` objects) or a
    ``concurrent.futures.ProcessPoolExecutor`` (``._processes``, a dict of
    ``pid -> Process``) -- the two pool primitives used across this codebase's
    parallel task/extraction loops.
    """
    pids: list[int] = []
    for proc in list(getattr(pool, "_pool", [])):
        pid = getattr(proc, "pid", None)
        if pid is not None and proc.is_alive():
            pids.append(pid)
    processes = getattr(pool, "_processes", None)
    if isinstance(processes, dict):
        for pid, proc in list(processes.items()):
            if proc is not None and proc.is_alive():
                pids.append(pid)
    return pids


def start_worker_watchdog(
    pool,
    per_worker_budget_bytes: int,
    poll_interval: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
) -> Callable[[], None]:
    """Best-effort: poll `pool`'s worker processes' RSS and kill any worker that
    exceeds `per_worker_budget_bytes`.

    `pool` may be a `multiprocessing.pool.Pool` or a
    `concurrent.futures.ProcessPoolExecutor` -- see `_live_worker_pids`.

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
            for pid in _live_worker_pids(pool):
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


def _limit_blas_threads_in_worker() -> None:
    """``ProcessPoolExecutor`` initializer: force single-threaded BLAS/OMP/
    numexpr in this worker process.

    Without this, task-level (process-count) parallelism and BLAS's own
    internal thread pool (numpy matrix ops, autocorrelation, etc.) compound:
    each worker independently spins up threads across every physical core
    for its own numpy calls, so N concurrent task workers can end up
    scheduling N x (BLAS's own thread count) threads total -- severe CPU
    oversubscription that makes wall-clock time *worse* than fewer, purely
    task-parallel workers would give. Confirmed via real-data testing: task
    workers individually using 200%+ CPU each while running concurrently.
    Must run before numpy is imported in the worker to take effect (these
    are read once, at BLAS backend init) -- set as this pool's
    ``initializer``, which runs first thing in each freshly started worker
    process, before any of this codebase's own numpy-importing code does.
    """
    import os

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def run_tasks_parallel(
    worker: Callable, task_args_list: list[tuple], *, cfg, force_sequential: bool = False
) -> list:
    """Run ``worker(*args)`` once per item in ``task_args_list``, in a
    memory-aware ``ProcessPoolExecutor`` pool.

    Shared orchestration for every parallel task-execution loop in this
    codebase (preprocess/spatial/hmm task dispatch, raw-ingestion extraction
    buckets): sizes the pool via ``resolve_max_workers`` (bounded by
    ``cfg.threads``, item count, and the aggregate memory budget divided by a
    per-task estimate) and wires ``start_worker_watchdog`` around it (a no-op
    on Linux, where the aggregate cgroup cap from ``enable_aggregate_memory_
    cap`` already covers the whole process tree).

    Falls back to a plain sequential loop when ``resolve_max_workers`` decides
    only one worker is warranted (small task counts, or a tight memory
    budget) -- no process pool overhead paid for work that wouldn't benefit
    from one anyway. ``worker`` must be a module-level function (picklable),
    not a closure, and every element of ``task_args_list`` and its own
    positional args must be picklable -- the same constraints as any other
    ``ProcessPoolExecutor`` use.

    ``force_sequential``, when true, skips the pool entirely regardless of
    ``resolve_max_workers``'s decision. For callers whose tasks share a
    resource that isn't safe for multiple *processes* to touch concurrently
    -- confirmed via real-data testing: HMM tasks on a GPU device (MPS on
    Apple Silicon) reliably crashed the whole pool (``BrokenProcessPool``)
    when several worker processes each tried to initialize/use the same GPU
    context at once. CPU-bound BLAS oversubscription (the problem
    ``_limit_blas_threads_in_worker`` solves) and this are different
    failure modes -- BLAS threads within one process don't fight over a
    single shared GPU context the way independent *processes* do.

    Returns results in the same order as ``task_args_list`` regardless of
    actual completion order (each future is still submitted concurrently;
    only the order results are *collected* in is preserved), so callers don't
    need to handle reordering the way completion-order APIs (e.g.
    ``as_completed``) would require.
    """
    max_workers = 1 if force_sequential else resolve_max_workers(cfg, len(task_args_list))
    if max_workers <= 1:
        return [worker(*args) for args in task_args_list]

    from concurrent.futures import ProcessPoolExecutor

    per_worker_budget_bytes = resolve_memory_budget_bytes(cfg) // max_workers
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_limit_blas_threads_in_worker
    ) as pool:
        stop_watchdog = start_worker_watchdog(pool, per_worker_budget_bytes)
        try:
            futures = [pool.submit(worker, *args) for args in task_args_list]
            return [future.result() for future in futures]
        finally:
            stop_watchdog()
