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
import threading
import time
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
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        self._start = time.monotonic()
        self._pool_seq = 0
        self._n_pools = 0
        self._n_retries = 0
        self._max_workers_used = 0
        self._peak_tree_rss_gb = 0.0
        self._closed = False

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

    def pool_start(self, pool_id: int, **fields) -> None:
        max_workers = int(fields.get("max_workers", 0) or 0)
        with self._lock:
            self._n_pools += 1
            self._max_workers_used = max(self._max_workers_used, max_workers)
        self._emit("pool_start", worker_pool_id=pool_id, **fields)

    def sample(self, pool_id: int, *, tree_rss_gb: float, **fields) -> None:
        with self._lock:
            self._peak_tree_rss_gb = max(self._peak_tree_rss_gb, tree_rss_gb)
        self._emit("sample", worker_pool_id=pool_id, tree_rss_gb=round(tree_rss_gb, 3), **fields)

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

    def close(self) -> None:
        if self._closed:
            return
        self._emit(
            "stage_summary",
            wall_seconds=round(time.monotonic() - self._start, 2),
            peak_tree_rss_gb=round(self._peak_tree_rss_gb, 3),
            n_pools=self._n_pools,
            n_retries_total=self._n_retries,
            max_workers_used=self._max_workers_used,
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
    rows: list[dict] = []
    for path in sorted(root.rglob("*_perf.jsonl")):
        peak = 0.0
        max_workers = 0
        n_pools = 0
        n_retries = 0
        wall_seconds = None
        stage = None
        n_samples = 0
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
                    elif event == "sample":
                        n_samples += 1
                        peak = max(peak, float(record.get("tree_rss_gb") or 0.0))
                    elif event == "pool_retry":
                        n_retries += 1
                    elif event == "pool_end":
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                    elif event == "stage_summary":
                        peak = max(peak, float(record.get("peak_tree_rss_gb") or 0.0))
                        max_workers = max(max_workers, int(record.get("max_workers_used") or 0))
                        n_retries = max(n_retries, int(record.get("n_retries_total") or 0))
                        wall_seconds = record.get("wall_seconds", wall_seconds)
        except OSError:
            continue
        rows.append(
            {
                "path": str(path),
                "stage": stage,
                "peak_tree_rss_gb": round(peak, 3),
                "max_workers_used": max_workers,
                "n_pools": n_pools,
                "n_retries_total": n_retries,
                "n_samples": n_samples,
                "wall_seconds": wall_seconds,
            }
        )
    rows.sort(key=lambda row: row["peak_tree_rss_gb"], reverse=True)
    return rows
