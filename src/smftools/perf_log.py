"""Per-command performance logging: worker counts + memory, as JSONL.

A ``PerfLogger`` writes one JSON object per line to ``<stage>/logs/<ts>_perf.jsonl``
(the same timestamp stem as the stage's human log), capturing how many workers a
parallel dispatch used (and the inputs that decided that), periodic memory
samples while workers run, the retry ladder when a pool breaks, and a per-command
summary. It is observability only -- it never changes scheduling or memory
behavior.

The active logger for the current stage is held in a ``contextvars.ContextVar``
set once by ``logging_utils.setup_stage_logging``. ``memory_guard`` reads it at
its pool chokepoints; when it is ``None`` (tests, library use, or
``emit_perf_log=False``) every call is a cheap no-op. Only the parent process
writes -- workers are sampled *by* the parent's watchdog thread -- so there is no
cross-process contention. The ContextVar is not inherited by the watchdog thread,
so callers pass the logger explicitly to ``start_worker_watchdog``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Optional

_CURRENT_PERF_LOGGER: contextvars.ContextVar[Optional["PerfLogger"]] = contextvars.ContextVar(
    "smftools_perf_logger", default=None
)


def get_perf_logger() -> Optional["PerfLogger"]:
    """Return the perf logger for the current stage, or ``None`` if disabled."""
    return _CURRENT_PERF_LOGGER.get()


def set_perf_logger(logger: Optional["PerfLogger"]) -> None:
    """Install ``logger`` as the current stage's perf logger (``None`` disables)."""
    _CURRENT_PERF_LOGGER.set(logger)


class PerfLogger:
    """Append-only JSONL writer for one stage invocation's performance events."""

    def __init__(self, path: str | Path, stage: str, *, sample_interval_seconds: float = 2.0):
        self.path = Path(path)
        self.stage = str(stage)
        self.sample_interval_seconds = max(0.01, float(sample_interval_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        self._start = time.monotonic()
        self._pool_seq = 0
        self._pool_started: dict[int, float] = {}
        self._substep_seq = 0
        self._substeps_completed = 0
        self._substep_seconds = 0.0
        self._n_pools = 0
        self._n_retries = 0
        self._max_workers_used = 0
        self._peak_tree_rss_gb = 0.0
        self._outcome: str | None = None
        self._outcome_fields: dict = {}
        self._tasks_completed = 0
        self._tasks_total = 0
        self._rows_completed = 0
        self._bases_completed = 0
        self._io_baseline = self._process_tree_io_bytes()
        self._bytes_read = 0
        self._bytes_written = 0
        self._sampler_stop = threading.Event()
        self._closed = False
        self._sampler = threading.Thread(
            target=self._sample_resources_until_stopped,
            name=f"smftools-{self.stage}-perf-sampler",
            daemon=True,
        )
        self._sampler.start()

    def _emit(self, event: str, **fields) -> None:
        record = {"ts": round(time.time(), 3), "stage": self.stage, "event": event, **fields}
        line = json.dumps(record, default=str)
        with self._lock:
            if self._closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    def next_pool_id(self) -> int:
        with self._lock:
            self._pool_seq += 1
            return self._pool_seq

    def next_substep_id(self) -> int:
        """Return the next invocation-local substep identifier."""
        with self._lock:
            self._substep_seq += 1
            return self._substep_seq

    def pool_start(self, pool_id: int, **fields) -> None:
        max_workers = int(fields.get("max_workers", 0) or 0)
        n_tasks = int(fields.get("n_tasks", 0) or 0)
        with self._lock:
            self._n_pools += 1
            self._tasks_total += n_tasks
            self._pool_started[pool_id] = time.monotonic()
            self._max_workers_used = max(self._max_workers_used, max_workers)
        self._emit("pool_start", worker_pool_id=pool_id, **fields)

    def resource_envelope(self, **fields) -> None:
        """Record the immutable run-level resource resolution."""
        self._emit("resource_envelope", **fields)

    def operation_budget(self, **fields) -> None:
        """Record a point-in-time preflight for non-pool work."""
        self._emit("operation_budget", **fields)

    def substep_start(self, substep_id: int, name: str, **fields) -> None:
        """Record the start of sequential parent-process work."""
        self._emit("substep_start", substep_id=substep_id, name=name, **fields)

    def substep_end(
        self,
        substep_id: int,
        name: str,
        *,
        duration_seconds: float,
        outcome: str,
        **fields,
    ) -> None:
        """Record a terminal substep outcome and live resource counters."""
        with self._lock:
            self._substeps_completed += 1
            self._substep_seconds += float(duration_seconds)
        resources = self._resource_snapshot()
        self._emit(
            "substep_end",
            substep_id=substep_id,
            name=name,
            duration_seconds=round(float(duration_seconds), 3),
            outcome=outcome,
            **resources,
            **fields,
        )
        logging.getLogger(__name__).info(
            "[%s] substep %s %s in %.2fs; RSS %.3f GiB (peak %.3f GiB)",
            self.stage,
            name,
            outcome,
            duration_seconds,
            resources["current_tree_rss_gb"],
            resources["peak_tree_rss_gb"],
        )

    def sample(self, pool_id: int, *, tree_rss_gb: float, **fields) -> None:
        with self._lock:
            self._peak_tree_rss_gb = max(self._peak_tree_rss_gb, tree_rss_gb)
        self._emit("sample", worker_pool_id=pool_id, tree_rss_gb=round(tree_rss_gb, 3), **fields)

    @staticmethod
    def _process_tree_io_bytes() -> tuple[int, int]:
        """Return cumulative OS I/O counters for the current process tree."""
        try:
            import psutil

            root = psutil.Process()
            processes = [root, *root.children(recursive=True)]
            read_bytes = 0
            write_bytes = 0
            for process in processes:
                try:
                    counters = process.io_counters()
                except (psutil.AccessDenied, psutil.NoSuchProcess, NotImplementedError):
                    continue
                read_bytes += int(getattr(counters, "read_bytes", 0) or 0)
                write_bytes += int(getattr(counters, "write_bytes", 0) or 0)
            return read_bytes, write_bytes
        except Exception:
            return 0, 0

    def _resource_snapshot(self) -> dict[str, float | int]:
        """Capture current RSS plus cumulative read/write bytes best-effort."""
        try:
            from .memory_guard import process_tree_rss_bytes

            rss_bytes = int(process_tree_rss_bytes())
        except Exception:
            rss_bytes = 0
        read_bytes, write_bytes = self._process_tree_io_bytes()
        with self._lock:
            self._peak_tree_rss_gb = max(self._peak_tree_rss_gb, rss_bytes / (1024**3))
            self._bytes_read = max(self._bytes_read, read_bytes - self._io_baseline[0])
            self._bytes_written = max(self._bytes_written, write_bytes - self._io_baseline[1])
            peak = self._peak_tree_rss_gb
            cumulative_read = self._bytes_read
            cumulative_written = self._bytes_written
        return {
            "current_tree_rss_gb": round(rss_bytes / (1024**3), 3),
            "peak_tree_rss_gb": round(peak, 3),
            "bytes_read": max(0, cumulative_read),
            "bytes_written": max(0, cumulative_written),
        }

    def _sample_resources_until_stopped(self) -> None:
        """Sample the whole stage independently of pool enforcement/watchdogs."""
        while not self._sampler_stop.wait(self.sample_interval_seconds):
            self._emit("resource_sample", **self._resource_snapshot())

    def task_complete(
        self,
        pool_id: int,
        *,
        task_index: int,
        completed: int,
        total: int,
        duration_seconds: float,
        retry_count: int = 0,
        rows: int = 0,
        bases: int = 0,
    ) -> None:
        """Emit one completion-order progress record for a bounded task."""
        with self._lock:
            pool_started = self._pool_started.get(pool_id, self._start)
        elapsed = max(time.monotonic() - pool_started, 1e-9)
        throughput = completed / elapsed
        eta = max(0.0, (total - completed) / throughput) if throughput > 0 else None
        with self._lock:
            self._tasks_completed += 1
            self._rows_completed += max(0, int(rows))
            self._bases_completed += max(0, int(bases))
        resources = self._resource_snapshot()
        self._emit(
            "task_progress",
            worker_pool_id=pool_id,
            task_index=int(task_index),
            completed=int(completed),
            total=int(total),
            throughput_tasks_per_second=round(throughput, 3),
            eta_seconds=None if eta is None else round(eta, 2),
            duration_seconds=round(float(duration_seconds), 3),
            retry_count=int(retry_count),
            rows=max(0, int(rows)),
            bases=max(0, int(bases)),
            **resources,
        )
        logging.getLogger(__name__).info(
            "[%s pool %s] completed %d/%d tasks; %.2f tasks/s; ETA %s; "
            "task %.2fs; retry=%d; rows=%d; bases=%d; RSS %.3f GiB (peak %.3f GiB); "
            "I/O read=%d written=%d bytes",
            self.stage,
            pool_id,
            completed,
            total,
            throughput,
            "unknown" if eta is None else f"{eta:.1f}s",
            duration_seconds,
            retry_count,
            rows,
            bases,
            resources["current_tree_rss_gb"],
            resources["peak_tree_rss_gb"],
            resources["bytes_read"],
            resources["bytes_written"],
        )

    def mark_outcome(self, outcome: str, **fields) -> None:
        """Record the terminal semantic outcome used by ``stage_summary``."""
        with self._lock:
            self._outcome = str(outcome)
            self._outcome_fields.update(fields)

    def pool_retry(self, pool_id: int, **fields) -> None:
        with self._lock:
            self._n_retries += 1
        self._emit("pool_retry", worker_pool_id=pool_id, **fields)

    def pool_end(self, pool_id: int, **fields) -> None:
        peak = fields.get("peak_tree_rss_gb")
        if peak is not None:
            with self._lock:
                self._peak_tree_rss_gb = max(self._peak_tree_rss_gb, float(peak))
        self._emit("pool_end", worker_pool_id=pool_id, **fields)

    def close(self, *, outcome: str | None = None, **fields) -> None:
        if self._closed:
            return
        if outcome is not None:
            self.mark_outcome(outcome, **fields)
        self._sampler_stop.set()
        self._sampler.join(timeout=max(1.0, self.sample_interval_seconds + 0.5))
        resources = self._resource_snapshot()
        with self._lock:
            terminal_outcome = self._outcome or "completed"
            outcome_fields = dict(self._outcome_fields)
        self._emit(
            "stage_summary",
            wall_seconds=round(time.monotonic() - self._start, 2),
            n_pools=self._n_pools,
            n_retries_total=self._n_retries,
            max_workers_used=self._max_workers_used,
            tasks_completed=self._tasks_completed,
            tasks_total=self._tasks_total,
            substeps_completed=self._substeps_completed,
            substep_seconds=round(self._substep_seconds, 3),
            rows=self._rows_completed,
            bases=self._bases_completed,
            outcome=terminal_outcome,
            **resources,
            **outcome_fields,
        )
        with self._lock:
            self._closed = True
            try:
                self._fh.close()
            except Exception:
                pass


def summarize_perf_logs(root: str | Path) -> "list[dict]":
    """Roll up every ``*_perf.jsonl`` under ``root`` into one row per file.

    Walks ``root`` recursively (a run's or a batch's output tree), reading each
    perf log's ``pool_start``/``sample``/``pool_end``/``stage_summary`` lines into
    a compact per-file summary: peak tree RSS, worker counts used, retries, and
    the log's path/stage. Returns rows sorted by descending peak RSS -- the
    "how much memory did this command/batch need, at what parallelism" answer,
    without any live monitoring. Pure post-hoc reader; imports nothing heavy.
    """
    root = Path(root)
    summary_rows: list[dict] = []
    for path in sorted(root.rglob("*_perf.jsonl")):
        peak = 0.0
        predicted_peak = 0.0
        max_workers = 0
        n_pools = 0
        n_retries = 0
        wall_seconds = None
        stage = None
        n_samples = 0
        tasks_completed = 0
        tasks_total = 0
        completed_rows = 0
        bases = 0
        bytes_read = 0
        bytes_written = 0
        outcome = None
        substeps_completed = 0
        substep_seconds = 0.0
        try:
            with open(path, encoding="utf-8") as handle:
                for raw in handle:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    stage = record.get("stage", stage)
                    event = record.get("event")
                    if event == "pool_start":
                        n_pools += 1
                        max_workers = max(max_workers, int(record.get("max_workers") or 0))
                        predicted_peak = max(
                            predicted_peak, float(record.get("predicted_peak_gb") or 0.0)
                        )
                    elif event == "sample":
                        n_samples += 1
                        peak = max(peak, float(record.get("tree_rss_gb") or 0.0))
                    elif event == "resource_sample":
                        n_samples += 1
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                        bytes_read = max(bytes_read, int(record.get("bytes_read") or 0))
                        bytes_written = max(bytes_written, int(record.get("bytes_written") or 0))
                    elif event == "task_progress":
                        tasks_completed += 1
                        completed_rows += int(record.get("rows") or 0)
                        bases += int(record.get("bases") or 0)
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                        bytes_read = max(bytes_read, int(record.get("bytes_read") or 0))
                        bytes_written = max(bytes_written, int(record.get("bytes_written") or 0))
                    elif event == "substep_end":
                        substeps_completed += 1
                        substep_seconds += float(record.get("duration_seconds") or 0.0)
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                        bytes_read = max(bytes_read, int(record.get("bytes_read") or 0))
                        bytes_written = max(bytes_written, int(record.get("bytes_written") or 0))
                    elif event == "pool_retry":
                        n_retries += 1
                    elif event == "pool_end":
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                    elif event == "stage_summary":
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                        max_workers = max(max_workers, int(record.get("max_workers_used") or 0))
                        n_retries = max(n_retries, int(record.get("n_retries_total") or 0))
                        wall_seconds = record.get("wall_seconds", wall_seconds)
                        tasks_completed = max(
                            tasks_completed, int(record.get("tasks_completed") or 0)
                        )
                        tasks_total = max(tasks_total, int(record.get("tasks_total") or 0))
                        completed_rows = max(completed_rows, int(record.get("rows") or 0))
                        bases = max(bases, int(record.get("bases") or 0))
                        bytes_read = max(bytes_read, int(record.get("bytes_read") or 0))
                        bytes_written = max(bytes_written, int(record.get("bytes_written") or 0))
                        outcome = record.get("outcome", outcome)
                        substeps_completed = max(
                            substeps_completed, int(record.get("substeps_completed") or 0)
                        )
                        substep_seconds = max(
                            substep_seconds, float(record.get("substep_seconds") or 0.0)
                        )
        except OSError:
            continue
        summary_rows.append(
            {
                "path": str(path),
                "stage": stage,
                "peak_tree_rss_gb": round(peak, 3),
                "max_predicted_peak_gb": round(predicted_peak, 3),
                "peak_to_predicted_ratio": (
                    round(peak / predicted_peak, 3) if predicted_peak > 0 else None
                ),
                "max_workers_used": max_workers,
                "n_pools": n_pools,
                "n_retries_total": n_retries,
                "n_samples": n_samples,
                "wall_seconds": wall_seconds,
                "tasks_completed": tasks_completed,
                "tasks_total": tasks_total,
                "rows": completed_rows,
                "bases": bases,
                "bytes_read": bytes_read,
                "bytes_written": bytes_written,
                "outcome": outcome,
                "substeps_completed": substeps_completed,
                "substep_seconds": round(substep_seconds, 3),
            }
        )
    summary_rows.sort(key=lambda row: row["peak_tree_rss_gb"], reverse=True)
    return summary_rows


@contextmanager
def perf_substep(name: str, **fields):
    """Time and resource-sample one named parent-process stage substep."""
    logger = get_perf_logger()
    if logger is None:
        yield
        return
    substep_id = logger.next_substep_id()
    logger.substep_start(substep_id, str(name), **fields)
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        logger.substep_end(
            substep_id,
            str(name),
            duration_seconds=time.monotonic() - started,
            outcome="failed",
            exception_type=type(exc).__name__,
            exception=str(exc),
            **fields,
        )
        raise
    logger.substep_end(
        substep_id,
        str(name),
        duration_seconds=time.monotonic() - started,
        outcome="completed",
        **fields,
    )


def measured_substep(name: str):
    """Decorate a parent-process function as one named performance substep."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with perf_substep(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate
