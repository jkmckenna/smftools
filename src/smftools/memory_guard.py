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

- macOS, Windows, and Linux when cgroup activation fails: `PoolBudget` snapshots
  combine current system/cgroup headroom with recursive smftools process-tree use
  before pool creation and while bounded work is admitted. A watchdog samples
  worker process trees and terminates sustained over-budget workers. Sequential,
  reducer, plotting, and external-tool entry points use the same live preflight.

OS enforcement remains best-effort, but admission fails clearly when the live
headroom cannot fit even one estimated task.
"""

from __future__ import annotations

import atexit
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_AGGREGATE_SAFETY_FRACTION = 0.9
DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 2.0
POOL_BUDGET_ESTIMATOR_VERSION = "1"
_ACTIVE_CGROUP_PATH: Path | None = None
_UPSTREAM_CGROUP_CPU_COUNT: int | None = None
_UPSTREAM_CGROUP_MEMORY_LIMIT: int | None = None
_UPSTREAM_CGROUP_MEMORY_CURRENT: int | None = None


@dataclass(frozen=True)
class ResourceEnvelope:
    """Immutable requested, detected, and resolved resources for one invocation."""

    platform: str
    logical_cpu_count: int
    affinity_cpu_count: int | None
    cgroup_cpu_count: int | None
    scheduler_cpu_count: int | None
    requested_threads: int
    resolved_threads: int
    total_memory_bytes: int
    available_memory_bytes: int
    cgroup_memory_limit_bytes: int | None
    cgroup_memory_current_bytes: int | None
    scheduler_memory_limit_bytes: int | None
    requested_memory_bytes: int
    memory_reserve_bytes: int
    resolved_memory_bytes: int
    enforcement_capability: str
    enforcement_mode: str
    enforcement_active: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resource provenance record."""
        return asdict(self)


@dataclass(frozen=True)
class PoolBudget:
    """Point-in-time resource budget resolved immediately before pool use."""

    estimator: str
    estimator_version: str
    n_items: int
    per_item_memory_bytes: int
    process_tree_rss_bytes: int
    process_tree_private_bytes: int
    system_available_bytes: int
    cgroup_headroom_bytes: int | None
    run_headroom_bytes: int
    usable_headroom_bytes: int
    max_workers: int
    max_in_flight: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable pool decision record."""
        return asdict(self)


class PoolBudgetError(MemoryError):
    """Raised when even one estimated task cannot fit the live memory headroom."""


def _minimum_positive(values: list[int | None], *, fallback: int) -> int:
    positive = [int(value) for value in values if value is not None and int(value) > 0]
    return min(positive) if positive else max(1, int(fallback))


def affinity_cpu_count() -> int | None:
    """Return CPUs available through process affinity, when the OS exposes it."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            import psutil

            affinity = psutil.Process().cpu_affinity()
            return len(affinity) if affinity else None
        except (AttributeError, OSError, NotImplementedError):
            return None


def _read_cgroup_value(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def cgroup_cpu_quota_count(cgroup_path: Path | None = None) -> int | None:
    """Return the integral CPU ceiling from a cgroup v2 ``cpu.max`` file."""
    if cgroup_path is None and _ACTIVE_CGROUP_PATH is not None:
        return _UPSTREAM_CGROUP_CPU_COUNT
    cgroup_path = cgroup_path or _cgroup_v2_self_path()
    if cgroup_path is None:
        return None
    try:
        raw = (cgroup_path / "cpu.max").read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(raw) < 2 or raw[0] == "max":
        return None
    try:
        quota, period = int(raw[0]), int(raw[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.floor(quota / period))


def cgroup_memory_values(cgroup_path: Path | None = None) -> tuple[int | None, int | None]:
    """Return cgroup v2 memory limit and current use, when available."""
    if cgroup_path is None and _ACTIVE_CGROUP_PATH is not None:
        return _UPSTREAM_CGROUP_MEMORY_LIMIT, _UPSTREAM_CGROUP_MEMORY_CURRENT
    cgroup_path = cgroup_path or _cgroup_v2_self_path()
    if cgroup_path is None:
        return None, None
    return (
        _read_cgroup_value(cgroup_path / "memory.max"),
        _read_cgroup_value(cgroup_path / "memory.current"),
    )


_SCHEDULER_CPU_VARIABLES = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "PBS_NP",
    "NSLOTS",
    "LSB_DJOB_NUMPROC",
)


def scheduler_cpu_count(environ: Mapping[str, str] | None = None) -> int | None:
    """Return the most restrictive recognized scheduler CPU allocation."""
    environ = os.environ if environ is None else environ
    values: list[int] = []
    for key in _SCHEDULER_CPU_VARIABLES:
        raw = str(environ.get(key, "")).strip().split("(", 1)[0]
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return min(values) if values else None


def _parse_memory_size(value: str, *, default_unit_bytes: int = 1) -> int | None:
    raw = str(value).strip().upper()
    if not raw:
        return None
    units = {
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    multiplier = default_unit_bytes
    for suffix in sorted(units, key=len, reverse=True):
        if raw.endswith(suffix):
            multiplier = units[suffix]
            raw = raw[: -len(suffix)]
            break
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return int(parsed * multiplier) if parsed > 0 else None


def scheduler_memory_limit_bytes(
    environ: Mapping[str, str] | None = None,
    *,
    allocated_cpus: int | None = None,
) -> int | None:
    """Return a recognized scheduler job-memory ceiling in bytes."""
    environ = os.environ if environ is None else environ
    candidates: list[int] = []
    per_node = _parse_memory_size(environ.get("SLURM_MEM_PER_NODE", ""), default_unit_bytes=1024**2)
    if per_node is not None:
        candidates.append(per_node)
    per_cpu = _parse_memory_size(environ.get("SLURM_MEM_PER_CPU", ""), default_unit_bytes=1024**2)
    if per_cpu is not None and allocated_cpus is not None:
        candidates.append(per_cpu * allocated_cpus)
    for key in ("PBS_RESC_MEM", "PBS_RESOURCE_LIST_MEM", "SGE_H_VMEM", "LSB_MAX_MEM"):
        parsed = _parse_memory_size(environ.get(key, ""))
        if parsed is not None:
            candidates.append(parsed)
    return min(candidates) if candidates else None


def detected_usable_cpu_count(environ: Mapping[str, str] | None = None) -> int:
    """Return the CPU ceiling imposed by the host, affinity, cgroup, and scheduler."""
    logical = max(1, int(os.cpu_count() or 1))
    return _minimum_positive(
        [
            logical,
            affinity_cpu_count(),
            cgroup_cpu_quota_count(),
            scheduler_cpu_count(environ),
        ],
        fallback=logical,
    )


def total_system_memory_bytes() -> int:
    """Total physical RAM, cross-platform (mac/linux), via psutil."""
    import psutil

    return int(psutil.virtual_memory().total)


def available_system_memory_bytes() -> int:
    """Memory currently available to new work according to the host OS."""
    import psutil

    return int(psutil.virtual_memory().available)


def resolve_resource_envelope(
    cfg,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResourceEnvelope:
    """Resolve one portable run-level CPU and memory ceiling."""
    logical_cpus = max(1, int(os.cpu_count() or 1))
    affinity_cpus = affinity_cpu_count()
    cgroup_cpus = cgroup_cpu_quota_count()
    scheduler_cpus = scheduler_cpu_count(environ)
    requested_threads = int(getattr(cfg, "threads", None) or logical_cpus)
    resolved_threads = _minimum_positive(
        [requested_threads, logical_cpus, affinity_cpus, cgroup_cpus, scheduler_cpus],
        fallback=1,
    )

    total_memory = total_system_memory_bytes()
    available_memory = available_system_memory_bytes()
    cgroup_limit, cgroup_current = cgroup_memory_values()
    scheduler_memory = scheduler_memory_limit_bytes(
        environ,
        allocated_cpus=scheduler_cpus,
    )
    reserve_bytes = int(float(getattr(cfg, "memory_reserve_gb", 1.0) or 0.0) * (1024**3))

    requested_candidates: list[int] = []
    percent = getattr(cfg, "max_memory_percent", None)
    if percent is not None:
        requested_candidates.append(int(total_memory * float(percent) / 100.0))
    fixed_gb = getattr(cfg, "max_memory_gb", None)
    if fixed_gb is not None:
        requested_candidates.append(int(float(fixed_gb) * (1024**3)))
    if not requested_candidates:
        requested_candidates.append(int(total_memory * DEFAULT_AGGREGATE_SAFETY_FRACTION))
    requested_memory = min(requested_candidates)

    detected_candidates = [max(1, available_memory - reserve_bytes)]
    if cgroup_limit is not None:
        current = cgroup_current or 0
        detected_candidates.append(max(1, cgroup_limit - current - reserve_bytes))
    if scheduler_memory is not None:
        detected_candidates.append(max(1, scheduler_memory - reserve_bytes))
    resolved_memory = max(1, min(requested_memory, *detected_candidates))

    if sys.platform == "linux":
        capability = "cgroup_v2" if _cgroup_v2_self_path() is not None else "advisory"
    elif sys.platform in {"darwin", "win32"}:
        capability = "worker_watchdog"
    else:
        capability = "advisory"
    return ResourceEnvelope(
        platform=sys.platform,
        logical_cpu_count=logical_cpus,
        affinity_cpu_count=affinity_cpus,
        cgroup_cpu_count=cgroup_cpus,
        scheduler_cpu_count=scheduler_cpus,
        requested_threads=requested_threads,
        resolved_threads=resolved_threads,
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        cgroup_memory_limit_bytes=cgroup_limit,
        cgroup_memory_current_bytes=cgroup_current,
        scheduler_memory_limit_bytes=scheduler_memory,
        requested_memory_bytes=requested_memory,
        memory_reserve_bytes=reserve_bytes,
        resolved_memory_bytes=resolved_memory,
        enforcement_capability=capability,
        enforcement_mode="not_activated",
        enforcement_active=False,
    )


def activate_resource_envelope(envelope: ResourceEnvelope) -> ResourceEnvelope:
    """Attempt platform enforcement and return the resulting immutable record."""
    active = enable_aggregate_memory_cap(budget_bytes=envelope.resolved_memory_bytes)
    if active:
        mode = "cgroup_v2"
    elif sys.platform in {"darwin", "win32"}:
        mode = "worker_watchdog"
    else:
        mode = "advisory"
    return replace(envelope, enforcement_mode=mode, enforcement_active=active)


def resource_envelope_for_config(cfg) -> ResourceEnvelope:
    """Return the attached run envelope or resolve one for a library caller."""
    envelope = getattr(cfg, "_resource_envelope", None)
    if isinstance(envelope, ResourceEnvelope):
        return envelope
    envelope = resolve_resource_envelope(cfg)
    try:
        cfg._resource_envelope = envelope
    except (AttributeError, TypeError):
        pass
    return envelope


def resolve_memory_budget_bytes(cfg) -> int:
    """Return the immutable run budget after user and detected-memory caps."""
    return resource_envelope_for_config(cfg).resolved_memory_bytes


def process_tree_rss_bytes(pid: int | None = None) -> int:
    """Return RSS for a process and all currently reachable descendants."""
    import psutil

    try:
        parent = psutil.Process(os.getpid() if pid is None else pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    try:
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        # Restricted macOS sandboxes can expose the current process RSS while
        # denying the system-wide PID enumeration psutil uses for descendants.
        processes = [parent]
    total = 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            if process.status() in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}:
                continue
            total += int(process.memory_info().rss)
        except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied):
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        except OSError:
            continue
    return total


def process_tree_private_bytes(pid: int | None = None) -> int:
    """Return unique memory for a process tree without fork-shared double counting."""
    import psutil

    try:
        parent = psutil.Process(os.getpid() if pid is None else pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    try:
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        processes = [parent]
    total = 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            if process.status() in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}:
                continue
            info = process.memory_full_info()
            total += int(getattr(info, "uss", info.rss))
        except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
    return total


def current_cgroup_memory_headroom_bytes() -> int | None:
    """Return live cgroup memory headroom, when a finite limit is visible."""
    if _ACTIVE_CGROUP_PATH is not None:
        limit, current = cgroup_memory_values(_ACTIVE_CGROUP_PATH)
    else:
        limit, current = cgroup_memory_values()
    if limit is None:
        return None
    return max(0, int(limit) - int(current or 0))


def resolve_pool_budget(
    cfg,
    n_items: int,
    *,
    per_item_memory_mb: float | None = None,
    estimator: str = "target_task_memory_mb",
) -> PoolBudget:
    """Resolve workers and admission depth from a fresh live-memory snapshot.

    The run envelope remains immutable and portable. This pool-level decision
    is deliberately ephemeral: it accounts for memory consumed since command
    startup and is recalculated at every pool allocation and admission refill.
    """
    envelope = resource_envelope_for_config(cfg)
    if per_item_memory_mb is None:
        per_item_memory_mb = float(getattr(cfg, "target_task_memory_mb", 512) or 512)
    per_item_bytes = max(1, math.ceil(max(1.0, per_item_memory_mb) * (1024**2)))
    tree_rss = process_tree_rss_bytes()
    tree_private = process_tree_private_bytes()
    system_available = available_system_memory_bytes()
    reserve = max(0, envelope.memory_reserve_bytes)
    # RSS double-counts copy-on-write pages shared by forked workers. Unique
    # set size gives a stable admission signal while RSS remains logged for
    # conventional peak reporting and estimator calibration.
    run_headroom = max(0, envelope.resolved_memory_bytes - tree_private)
    system_headroom = max(0, system_available - reserve)
    cgroup_headroom = current_cgroup_memory_headroom_bytes()
    candidates = [run_headroom, system_headroom]
    if cgroup_headroom is not None:
        # A dedicated smftools cgroup's memory.max is already the resolved
        # post-reserve run ceiling. An upstream/shared cgroup still needs the
        # reserve held back from its currently free allocation.
        cgroup_reserve = 0 if _ACTIVE_CGROUP_PATH is not None else reserve
        candidates.append(max(0, cgroup_headroom - cgroup_reserve))
    usable_headroom = min(candidates)
    by_memory = usable_headroom // per_item_bytes
    max_workers = min(
        envelope.resolved_threads,
        max(0, int(n_items)),
        max(0, int(by_memory)),
    )
    return PoolBudget(
        estimator=str(estimator),
        estimator_version=POOL_BUDGET_ESTIMATOR_VERSION,
        n_items=max(0, int(n_items)),
        per_item_memory_bytes=per_item_bytes,
        process_tree_rss_bytes=tree_rss,
        process_tree_private_bytes=tree_private,
        system_available_bytes=system_available,
        cgroup_headroom_bytes=cgroup_headroom,
        run_headroom_bytes=run_headroom,
        usable_headroom_bytes=usable_headroom,
        max_workers=max_workers,
        max_in_flight=max_workers,
    )


def require_task_admission(budget: PoolBudget, *, pool_label: str | None = None) -> None:
    """Raise a clear error when a live budget cannot admit one task."""
    if budget.n_items <= 0 or budget.max_in_flight > 0:
        return
    label = f" for {pool_label}" if pool_label else ""
    raise PoolBudgetError(
        f"Cannot admit one task{label}: estimated task memory "
        f"({budget.per_item_memory_bytes / (1024**2):.1f} MiB) exceeds live usable "
        f"headroom ({budget.usable_headroom_bytes / (1024**2):.1f} MiB). Close other "
        "memory-intensive processes, reduce task/partition size, or increase the configured "
        "memory ceiling."
    )


def require_memory_headroom(
    cfg,
    *,
    n_items: int = 1,
    estimated_memory_mb: float | None = None,
    operation_label: str | None = None,
    estimator: str = "target_task_memory_mb",
) -> PoolBudget:
    """Preflight one sequential, reducer, plotting, or external-tool operation."""
    budget = resolve_pool_budget(
        cfg,
        n_items,
        per_item_memory_mb=estimated_memory_mb,
        estimator=estimator,
    )
    from .perf_log import get_perf_logger

    perf = get_perf_logger()
    if perf is not None:
        perf.operation_budget(
            operation_label=operation_label,
            pool_budget=budget.as_dict(),
        )
    require_task_admission(budget, pool_label=operation_label)
    return budget


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
    threads = resource_envelope_for_config(cfg).resolved_threads
    if n_items <= 0:
        return 1
    if per_item_memory_mb is None:
        per_item_memory_mb = float(getattr(cfg, "target_task_memory_mb", 512) or 512)
    per_item_bytes = max(1.0, per_item_memory_mb) * (1024**2)
    budget_bytes = resolve_memory_budget_bytes(cfg)
    by_memory = max(1, int(budget_bytes // per_item_bytes))
    return max(1, min(threads, n_items, by_memory))


def bounded_executor_map(
    executor,
    worker: Callable,
    items,
    *,
    cfg,
    max_workers: int,
    pool_label: str | None = None,
    per_item_memory_mb: float | None = None,
    estimator: str = "target_task_memory_mb",
) -> list:
    """Map a lazy iterable with live, bounded admission and ordered results."""
    from concurrent.futures import FIRST_COMPLETED, wait

    iterator = enumerate(items)
    exhausted = object()
    next_item = next(iterator, exhausted)
    futures = {}
    results: dict[int, Any] = {}
    while futures or next_item is not exhausted:
        budget = resolve_pool_budget(
            cfg,
            max_workers,
            per_item_memory_mb=per_item_memory_mb,
            estimator=estimator,
        )
        target_in_flight = min(max_workers, budget.max_in_flight)
        while next_item is not exhausted and len(futures) < target_in_flight:
            index, item = next_item
            futures[executor.submit(worker, item)] = index
            next_item = next(iterator, exhausted)
        if not futures:
            require_task_admission(budget, pool_label=pool_label)
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            results[futures.pop(future)] = future.result()
    return [results[index] for index in sorted(results)]


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
        if _ACTIVE_CGROUP_PATH is not None and _ACTIVE_CGROUP_PATH.is_dir():
            if budget_bytes is not None:
                (_ACTIVE_CGROUP_PATH / "memory.max").write_text(str(budget_bytes))
            return True
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


def _try_enable_linux_cgroup_cap(
    safety_fraction: float, *, budget_bytes: int | None = None
) -> bool:
    global _ACTIVE_CGROUP_PATH
    global _UPSTREAM_CGROUP_CPU_COUNT
    global _UPSTREAM_CGROUP_MEMORY_CURRENT
    global _UPSTREAM_CGROUP_MEMORY_LIMIT

    self_cgroup = _cgroup_v2_self_path()
    if self_cgroup is None or not self_cgroup.is_dir():
        logger.info("cgroup v2 not available; skipping aggregate memory cap")
        return False

    _UPSTREAM_CGROUP_CPU_COUNT = cgroup_cpu_quota_count(self_cgroup)
    _UPSTREAM_CGROUP_MEMORY_LIMIT, _UPSTREAM_CGROUP_MEMORY_CURRENT = cgroup_memory_values(
        self_cgroup
    )

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
    _ACTIVE_CGROUP_PATH = child_cgroup

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


DEFAULT_WATCHDOG_TOLERANCE_FRACTION = 0.2
DEFAULT_WATCHDOG_GRACE_POLLS = 3


def start_worker_watchdog(
    pool,
    per_worker_budget_bytes: int,
    poll_interval: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
    *,
    tolerance_fraction: float = DEFAULT_WATCHDOG_TOLERANCE_FRACTION,
    grace_polls: int = DEFAULT_WATCHDOG_GRACE_POLLS,
    perf_logger=None,
    pool_id=None,
    pool_label: str | None = None,
) -> Callable[[], None]:
    """Best-effort: poll `pool`'s worker processes' RSS and kill any worker that
    *sustains* usage over `per_worker_budget_bytes` (plus `tolerance_fraction`
    headroom) for `grace_polls` consecutive polls.

    `pool` may be a `multiprocessing.pool.Pool` or a
    `concurrent.futures.ProcessPoolExecutor` -- see `_live_worker_pids`.

    Worker termination is disabled only when this process successfully entered
    its dedicated Linux cgroup. If cgroup activation failed, Linux uses the same
    watchdog fallback as macOS and Windows. With a performance logger attached,
    sampling remains active even under cgroup enforcement.

    `per_worker_budget_bytes` is `resolve_memory_budget_bytes(cfg) // max_workers`
    -- an even split of the aggregate budget, not a measured per-task estimate
    (`resolve_max_workers` sizes the pool from `cfg.target_task_memory_mb`,
    which can be off by an order of magnitude from a task's real footprint).
    When real per-worker usage naturally clusters right at that even-split
    boundary, multiple workers can each land a few percent over it from
    ordinary run-to-run variance -- a bare "kill on the first sample over
    budget" check (the previous behavior) then aborts the whole pool over a
    handful of MB, even though the pool's *aggregate* usage may still be safe.
    Confirmed on real batch data: 8 of 11 experiments were aborted this way in
    a single run, each killed worker only 0-4% over budget (see
    dev/pipeline_scaling_audit.md). `tolerance_fraction` absorbs that
    variance, and `grace_polls` absorbs single-sample spikes (a worker must be
    over budget on `grace_polls` *consecutive* polls, not just once) --
    together these still catch genuine runaway growth (which keeps climbing
    across polls) while not aborting a pool for a worker that's merely
    hovering near its fair share.

    Returns a `stop()` callable that must be called (typically in a `finally`)
    once the pool's work is done, to stop the polling thread.
    """
    cgroup_enforced = bool(
        sys.platform == "linux" and _ACTIVE_CGROUP_PATH is not None and _ACTIVE_CGROUP_PATH.is_dir()
    )
    if cgroup_enforced and perf_logger is None:
        return lambda: None

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed; per-worker memory watchdog disabled")
        return lambda: None
    access_denied = getattr(psutil, "AccessDenied", PermissionError)

    if per_worker_budget_bytes <= 0:
        logger.debug("No positive per-worker memory budget provided; watchdog disabled")
        return lambda: None

    kill_threshold_bytes = per_worker_budget_bytes * (1.0 + max(0.0, tolerance_fraction))
    grace_polls = max(1, grace_polls)
    stop_event = threading.Event()
    label_prefix = f"[{pool_label}] " if pool_label else ""

    parent_process = psutil.Process() if perf_logger is not None else None

    def _worker_tree_rss(pid: int) -> int:
        process = psutil.Process(pid)
        rss = int(process.memory_info().rss)
        try:
            descendants = process.children(recursive=True)
        except (AttributeError, access_denied, OSError):
            descendants = []
        for descendant in descendants:
            try:
                rss += int(descendant.memory_info().rss)
            except (psutil.NoSuchProcess, access_denied, OSError):
                continue
        return rss

    def _kill_worker_tree(pid: int) -> None:
        process = psutil.Process(pid)
        try:
            descendants = process.children(recursive=True)
        except (AttributeError, access_denied, OSError):
            descendants = []
        for descendant in reversed(descendants):
            try:
                descendant.kill()
            except (psutil.NoSuchProcess, access_denied, OSError):
                continue
        process.kill()

    def _emit_perf_sample(live_pids, workers_rss_bytes) -> None:
        if perf_logger is None or parent_process is None:
            return
        try:
            parent_rss = parent_process.memory_info().rss
            virtual = psutil.virtual_memory()
            perf_logger.sample(
                pool_id,
                tree_rss_gb=(parent_rss + workers_rss_bytes) / (1024**3),
                parent_rss_gb=round(parent_rss / (1024**3), 3),
                workers_rss_gb=round(workers_rss_bytes / (1024**3), 3),
                predicted_workers_peak_gb=round(
                    len(live_pids) * per_worker_budget_bytes / (1024**3), 3
                ),
                n_live_workers=len(live_pids),
                system_used_gb=round(virtual.used / (1024**3), 3),
                system_available_gb=round(virtual.available / (1024**3), 3),
            )
        except Exception:
            pass

    def _poll_loop() -> None:
        consecutive_over: dict[int, int] = {}
        while not stop_event.is_set():
            live_pids = set(_live_worker_pids(pool))
            for pid in list(consecutive_over):
                if pid not in live_pids:
                    consecutive_over.pop(pid, None)
            workers_rss_bytes = 0
            for pid in live_pids:
                try:
                    rss = _worker_tree_rss(pid)
                except psutil.NoSuchProcess:
                    consecutive_over.pop(pid, None)
                    continue
                workers_rss_bytes += rss
                if cgroup_enforced:
                    continue
                if rss <= kill_threshold_bytes:
                    consecutive_over.pop(pid, None)
                    continue
                consecutive_over[pid] = consecutive_over.get(pid, 0) + 1
                if consecutive_over[pid] < grace_polls:
                    logger.warning(
                        "%sMemoryGuard: worker pid=%d using %.2f GiB > %.2f GiB threshold "
                        "(budget %.2f GiB + %.0f%% tolerance) for %d/%d consecutive poll(s)",
                        label_prefix,
                        pid,
                        rss / (1024**3),
                        kill_threshold_bytes / (1024**3),
                        per_worker_budget_bytes / (1024**3),
                        tolerance_fraction * 100,
                        consecutive_over[pid],
                        grace_polls,
                    )
                    continue
                logger.error(
                    "%sMemoryGuard: worker pid=%d using %.2f GiB > %.2f GiB threshold "
                    "for %d consecutive polls; terminating to protect the machine",
                    label_prefix,
                    pid,
                    rss / (1024**3),
                    kill_threshold_bytes / (1024**3),
                    grace_polls,
                )
                try:
                    _kill_worker_tree(pid)
                except (psutil.NoSuchProcess, access_denied):
                    logger.warning(
                        "%sMemoryGuard could not terminate worker pid=%d", label_prefix, pid
                    )
                consecutive_over.pop(pid, None)
            _emit_perf_sample(live_pids, workers_rss_bytes)
            stop_event.wait(poll_interval)

    thread = threading.Thread(target=_poll_loop, name="smftools-memory-watchdog", daemon=True)
    thread.start()
    logger.info(
        "%sMemory watchdog enabled (%s): %.2f GiB budget/worker (+%.0f%% tolerance, "
        "%d-poll grace period), polling every %.1fs",
        label_prefix,
        "cgroup sampling" if cgroup_enforced else "worker enforcement",
        per_worker_budget_bytes / (1024**3),
        tolerance_fraction * 100,
        grace_polls,
        poll_interval,
    )

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=poll_interval * 2)

    return _stop


def run_tasks_parallel(
    worker: Callable,
    task_args_list: list[tuple],
    *,
    cfg,
    force_sequential: bool = False,
    pool_label: str | None = None,
    per_item_memory_mb: float | None = None,
    estimator: str = "target_task_memory_mb",
) -> list:
    """Run ``worker(*args)`` once per item in ``task_args_list``, in a
    memory-aware ``ProcessPoolExecutor`` pool.

    ``pool_label``: short human-readable description of what this pool is
    doing (e.g. ``"spatial task pool"``, or, for callers invoked many times
    per stage, something identifying enough to distinguish one call from the
    next -- e.g. duplicate detection's ``f"dedup {reference}/{sample} round
    {round_index}"``). Included as a ``[label]`` prefix on this pool's
    watchdog/retry log lines, which are otherwise generic regardless of which
    of this function's several call sites (or, for a repeatedly-invoked
    caller, which specific invocation) produced them -- confirmed as a real
    gap while live-monitoring a production run: long stretches of
    "Per-worker memory watchdog enabled"/"MemoryGuard: worker pid=..." with
    no indication of what task was actually running.

    Shared orchestration for partitioned preprocess/spatial/HMM and duplicate
    detection dispatch: resolves a fresh ``PoolBudget`` before allocation,
    bounds workers and submitted futures, refreshes live headroom after each
    completion, and wires ``start_worker_watchdog`` around the executor.

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
    ``parallel_utils.configure_worker_threads`` solves) and this are different
    failure modes -- BLAS threads within one process don't fight over a
    single shared GPU context the way independent *processes* do.

    Returns results in the same order as ``task_args_list`` regardless of
    completion order. No more than the live ``max_in_flight`` futures are
    submitted, so completed results cannot accumulate in an unbounded queue.

    If the watchdog kills a worker, every other future still pending in that
    same pool also raises ``BrokenProcessPool`` (the whole executor is
    unusable once one worker dies unexpectedly) -- the previous behavior let
    that propagate straight out of this function, aborting the entire task
    list (and, for batch callers, the whole experiment) over a single killed
    worker. This is common when the per-item memory estimate genuinely
    underestimates a task type's real footprint (not just watchdog noise --
    ``resolve_max_workers`` then packs more concurrent workers than the
    machine can actually hold): confirmed on real data, this aborted 8 of 11
    batch experiments in a single run (see dev/pipeline_scaling_audit.md).
    Tasks that hadn't produced a result yet when the pool broke are now
    retried in a fresh pool sized to half as many workers (floor 1), halving
    again on each repeated break, down to a final sequential (single-worker)
    attempt that always succeeds short of one task alone exceeding the
    *aggregate* budget. Already-completed results are kept, not redone.
    """
    from .perf_log import get_perf_logger

    if not task_args_list:
        return []

    perf = get_perf_logger()
    pool_id = perf.next_pool_id() if perf is not None else None
    initial_budget = resolve_pool_budget(
        cfg,
        len(task_args_list),
        per_item_memory_mb=per_item_memory_mb,
        estimator=estimator,
    )
    require_task_admission(initial_budget, pool_label=pool_label)
    max_workers = 1 if force_sequential else initial_budget.max_workers
    predicted_peak_bytes = (
        initial_budget.process_tree_rss_bytes + max_workers * initial_budget.per_item_memory_bytes
    )
    if perf is not None:
        perf.pool_start(
            pool_id,
            n_tasks=len(task_args_list),
            max_workers=max_workers,
            force_sequential=bool(force_sequential),
            predicted_peak_gb=round(predicted_peak_bytes / (1024**3), 3),
            pool_budget=initial_budget.as_dict(),
            **_worker_decision_inputs(
                cfg, len(task_args_list), per_item_memory_mb=per_item_memory_mb
            ),
        )
    if max_workers <= 1:
        # The parallel branch below always logs something (the watchdog's
        # "enabled" line at minimum) -- without an equivalent here, a caller
        # that resolves to a single worker (small task counts, or a tight
        # memory budget) runs with zero indication of what's happening until
        # it returns, which can be a long, log-silent stretch for a
        # repeatedly-invoked caller like duplicate detection's per-group,
        # per-round dispatch.
        if pool_label:
            logger.info("[%s] running %d task(s) sequentially", pool_label, len(task_args_list))
        try:
            results = []
            for args in task_args_list:
                require_memory_headroom(
                    cfg,
                    estimated_memory_mb=per_item_memory_mb,
                    operation_label=pool_label,
                    estimator=estimator,
                )
                results.append(worker(*args))
                if perf is not None:
                    measured_rss = process_tree_rss_bytes()
                    perf.sample(
                        pool_id,
                        tree_rss_gb=measured_rss / (1024**3),
                        predicted_peak_gb=round(predicted_peak_bytes / (1024**3), 3),
                        n_live_workers=0,
                    )
            return results
        finally:
            if perf is not None:
                perf.pool_end(
                    pool_id,
                    final_max_workers=1,
                    n_retries=0,
                    predicted_peak_gb=round(predicted_peak_bytes / (1024**3), 3),
                )

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    from .parallel_utils import configure_worker_threads

    poll_interval = float(getattr(cfg, "perf_log_sample_interval_seconds", 2.0) or 2.0)
    results: list = [None] * len(task_args_list)
    pending_indices = list(range(len(task_args_list)))
    workers_for_attempt = max_workers
    n_retries = 0
    try:
        while pending_indices:
            attempt_budget = resolve_pool_budget(
                cfg,
                len(pending_indices),
                per_item_memory_mb=per_item_memory_mb,
                estimator=estimator,
            )
            require_task_admission(attempt_budget, pool_label=pool_label)
            workers_for_attempt = min(workers_for_attempt, attempt_budget.max_workers)
            per_worker_budget_bytes = resolve_memory_budget_bytes(cfg) // workers_for_attempt
            with ProcessPoolExecutor(
                max_workers=workers_for_attempt,
                initializer=configure_worker_threads,
                initargs=(1,),
            ) as pool:
                stop_watchdog = start_worker_watchdog(
                    pool,
                    per_worker_budget_bytes,
                    poll_interval,
                    perf_logger=perf,
                    pool_id=pool_id,
                    pool_label=pool_label,
                )
                try:
                    from collections import deque

                    queued_indices = deque(pending_indices)
                    future_to_index = {}
                    broke = False
                    still_pending: list[int] = []
                    while queued_indices or future_to_index:
                        refill_budget = resolve_pool_budget(
                            cfg,
                            len(queued_indices) + len(future_to_index),
                            per_item_memory_mb=per_item_memory_mb,
                            estimator=estimator,
                        )
                        target_in_flight = min(
                            workers_for_attempt,
                            refill_budget.max_in_flight,
                        )
                        while queued_indices and len(future_to_index) < target_in_flight:
                            index = queued_indices.popleft()
                            future_to_index[pool.submit(worker, *task_args_list[index])] = index
                        if not future_to_index:
                            require_task_admission(refill_budget, pool_label=pool_label)
                        done, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = future_to_index.pop(future)
                            try:
                                results[index] = future.result()
                            except BrokenProcessPool:
                                broke = True
                                still_pending.append(index)
                        if broke:
                            still_pending.extend(queued_indices)
                            still_pending.extend(future_to_index.values())
                            for future in future_to_index:
                                future.cancel()
                            break
                    if broke:
                        still_pending = list(dict.fromkeys(still_pending))
                    pending_indices = sorted(still_pending)
                finally:
                    stop_watchdog()
            if pending_indices:
                if workers_for_attempt <= 1:
                    raise RuntimeError(
                        f"{len(pending_indices)} task(s) still failing after retrying down to a "
                        "single worker -- at least one task alone exceeds the aggregate memory "
                        "budget; increase cfg.max_memory_gb/max_memory_percent or "
                        "cfg.target_task_memory_mb to shrink task size"
                    )
                workers_for_attempt = max(1, workers_for_attempt // 2)
                n_retries += 1
                if perf is not None:
                    perf.pool_retry(
                        pool_id,
                        reason="broken_pool",
                        n_pending=len(pending_indices),
                        new_max_workers=workers_for_attempt,
                    )
                logger.warning(
                    "%srun_tasks_parallel: pool broke (a worker was likely killed by the memory "
                    "watchdog); retrying %d task(s) with reduced worker count %d",
                    f"[{pool_label}] " if pool_label else "",
                    len(pending_indices),
                    workers_for_attempt,
                )
    finally:
        if perf is not None:
            perf.pool_end(
                pool_id,
                final_max_workers=workers_for_attempt,
                n_retries=n_retries,
                predicted_peak_gb=round(predicted_peak_bytes / (1024**3), 3),
            )
    return results


def _worker_decision_inputs(cfg, n_items: int, *, per_item_memory_mb: float | None = None) -> dict:
    """The inputs ``resolve_max_workers`` used, for the perf log's ``pool_start``.

    Surfacing *why* a worker count was chosen (threads vs. item count vs. the
    aggregate-budget / per-task-estimate cap) is what makes the perf log
    actionable when the estimate is wrong -- the exact failure mode behind the
    batch OOM (see dev/pipeline_scaling_audit.md).
    """
    envelope = resource_envelope_for_config(cfg)
    threads = envelope.resolved_threads
    per_item_mb = (
        float(per_item_memory_mb)
        if per_item_memory_mb is not None
        else float(getattr(cfg, "target_task_memory_mb", 512) or 512)
    )
    budget_bytes = resolve_memory_budget_bytes(cfg)
    by_memory = max(1, int(budget_bytes // (max(1.0, per_item_mb) * (1024**2))))
    return {
        "threads": threads,
        "n_items": int(n_items),
        "aggregate_budget_gb": round(budget_bytes / (1024**3), 3),
        "target_task_memory_mb": per_item_mb,
        "by_memory_workers": by_memory,
        "resource_envelope": envelope.as_dict(),
    }
