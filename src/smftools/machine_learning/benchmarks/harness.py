"""Measurement primitives for ML-700: RSS sampling, repeats, and result records.

Records follow the JSONL conventions of :mod:`smftools.perf_log` -- one JSON
object per line -- so existing tooling can read them. The ``PerfLogger``
ContextVar itself is deliberately not reused: it is bound to stage logging and
would tie a benchmark run to a live pipeline stage.
"""

from __future__ import annotations

import gc
import json
import platform
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

# Sampling interval for the RSS watchdog. Fast enough to catch a transient
# concatenation spike, slow enough not to perturb what it measures.
RSS_SAMPLE_SECONDS = 0.02

# One discarded warmup pays import, page-cache, and lazy-init costs.
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3

# A cell whose spread exceeds this fraction of its median is reported unstable
# rather than silently averaged into the published limits.
UNSTABLE_SPREAD_FRACTION = 0.20


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cell cannot be measured meaningfully."""


@dataclass(frozen=True)
class BenchmarkCell:
    """One point in the qualification matrix."""

    sweep: str
    name: str
    parameters: Mapping[str, Any]
    estimated_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep": self.sweep,
            "name": self.name,
            "parameters": dict(self.parameters),
            "estimated_bytes": self.estimated_bytes,
        }


@dataclass(frozen=True)
class EnvironmentRecord:
    """Everything needed to reproduce or discount a measurement."""

    python: str
    platform: str
    machine: str
    cpu_count: int
    total_memory_bytes: int
    numpy: str
    sklearn: str | None
    torch: str | None
    torch_devices: tuple[str, ...]
    git_commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "total_memory_bytes": self.total_memory_bytes,
            "numpy": self.numpy,
            "sklearn": self.sklearn,
            "torch": self.torch,
            "torch_devices": list(self.torch_devices),
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured outcome for one cell across its repeats."""

    cell: BenchmarkCell
    state: str  # supported | slow | refused | error
    seconds: tuple[float, ...]
    peak_rss_deltas: tuple[int, ...]
    rows: int | None = None
    refusal_message: str | None = None
    error: str | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def median_seconds(self) -> float | None:
        return _median(self.seconds)

    @property
    def median_peak_bytes(self) -> int | None:
        value = _median(self.peak_rss_deltas)
        return int(value) if value is not None else None

    @property
    def headroom(self) -> float | None:
        """Measured peak as a fraction of the analytic estimate.

        ``<= 1.0`` means the estimate bounded reality. ``None`` when the cell was
        refused or carries no estimate -- a refused cell never allocates, so its
        peak is not comparable to an estimate it never used.
        """
        estimate = self.cell.estimated_bytes
        peak = self.median_peak_bytes
        if not estimate or peak is None:
            return None
        return peak / estimate

    @property
    def rows_per_second(self) -> float | None:
        median = self.median_seconds
        if self.rows is None or median is None or median <= 0:
            return None
        return self.rows / median

    @property
    def unstable(self) -> bool:
        """True when repeat spread is too wide to publish as a limit."""
        median = self.median_seconds
        if median is None or median <= 0 or len(self.seconds) < 2:
            return False
        return (max(self.seconds) - min(self.seconds)) / median > UNSTABLE_SPREAD_FRACTION

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.cell.to_dict(),
            "state": self.state,
            "seconds": list(self.seconds),
            "median_seconds": self.median_seconds,
            "peak_rss_deltas": list(self.peak_rss_deltas),
            "median_peak_bytes": self.median_peak_bytes,
            "headroom": self.headroom,
            "rows": self.rows,
            "rows_per_second": self.rows_per_second,
            "unstable": self.unstable,
            "refusal_message": self.refusal_message,
            "error": self.error,
            "notes": dict(self.notes),
        }


def _median(values: tuple[float, ...] | tuple[int, ...]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


class _RSSWatchdog:
    """Samples process RSS on a thread and retains the peak above a baseline."""

    def __init__(self, interval: float = RSS_SAMPLE_SECONDS) -> None:
        self._process = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline = 0
        self.peak = 0

    def _rss(self) -> int:
        return int(self._process.memory_info().rss)

    def __enter__(self) -> _RSSWatchdog:
        # Settle allocations from fixture construction so the delta attributes
        # only the measured region.
        gc.collect()
        self.baseline = self._rss()
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._run, name="ml700-rss", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            current = self._rss()
            if current > self.peak:
                self.peak = current

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # One final read: a short region can finish between samples.
        current = self._rss()
        if current > self.peak:
            self.peak = current

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)


def measure(
    cell: BenchmarkCell,
    operation: Callable[[], Any],
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    expect_refusal: type[BaseException] | None = None,
    rows: int | None = None,
    notes: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Run ``operation`` under RSS sampling and return a measured result.

    ``operation`` must be self-contained and side-effect free across repeats.
    When ``expect_refusal`` is supplied the cell is expected to raise that type;
    raising it is recorded as ``refused`` and *not* raising is an error, because
    a refusal boundary that silently stopped refusing is exactly the regression
    this package exists to catch.
    """
    if repeats < 1:
        raise BenchmarkError("repeats must be at least 1")

    for _ in range(max(0, warmup)):
        try:
            operation()
        except Exception:  # noqa: BLE001 - a refused warmup is informative, not fatal
            break

    seconds: list[float] = []
    peaks: list[int] = []
    refusal_message: str | None = None

    for _ in range(repeats):
        with _RSSWatchdog() as watchdog:
            start = time.perf_counter()
            try:
                operation()
            except Exception as exc:  # noqa: BLE001 - classified below
                elapsed = time.perf_counter() - start
                if expect_refusal is not None and isinstance(exc, expect_refusal):
                    seconds.append(elapsed)
                    peaks.append(watchdog.delta)
                    refusal_message = str(exc)
                    continue
                return BenchmarkResult(
                    cell=cell,
                    state="error",
                    seconds=tuple(seconds),
                    peak_rss_deltas=tuple(peaks),
                    rows=rows,
                    error=f"{type(exc).__name__}: {exc}",
                    notes=dict(notes or {}),
                )
            elapsed = time.perf_counter() - start
        if expect_refusal is not None:
            return BenchmarkResult(
                cell=cell,
                state="error",
                seconds=tuple(seconds),
                peak_rss_deltas=tuple(peaks),
                rows=rows,
                error=(
                    f"expected {expect_refusal.__name__} but the operation succeeded; "
                    "a refusal boundary stopped refusing"
                ),
                notes=dict(notes or {}),
            )
        seconds.append(elapsed)
        peaks.append(watchdog.delta)

    return BenchmarkResult(
        cell=cell,
        state="refused" if expect_refusal is not None else "supported",
        seconds=tuple(seconds),
        peak_rss_deltas=tuple(peaks),
        rows=rows,
        refusal_message=refusal_message,
        notes=dict(notes or {}),
    )


def measure_memory_isolated(
    cell: BenchmarkCell,
    request: Mapping[str, Any],
    *,
    workspace: Path,
    timeout: float = 1800.0,
) -> BenchmarkResult:
    """Measure peak allocation for one cell in a fresh process.

    In-process RSS delta under-reports peak allocation once CPython's allocator
    holds a warm arena -- a repeat can satisfy a multi-megabyte NumPy allocation
    with no RSS growth at all. Under-reporting makes a memory guardrail look
    safer than it is, so memory cells get a cold process, measured once, with no
    warmup. See ``_isolated``.
    """
    import subprocess

    payload = dict(request)
    payload["workspace"] = str(workspace)
    command = [
        sys.executable,
        "-m",
        "smftools.machine_learning.benchmarks._isolated",
        json.dumps(payload),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            cell=cell,
            state="error",
            seconds=(),
            peak_rss_deltas=(),
            error=f"isolated measurement exceeded {timeout}s",
        )

    stdout = completed.stdout.strip().splitlines()
    record: dict[str, Any] | None = None
    for line in reversed(stdout):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = candidate
        break

    if record is None or "peak_rss_delta" not in record:
        detail = record.get("error") if isinstance(record, dict) else completed.stderr.strip()
        return BenchmarkResult(
            cell=cell,
            state="error",
            seconds=(),
            peak_rss_deltas=(),
            error=f"isolated measurement produced no result: {detail or 'no output'}",
        )

    measured_cell = BenchmarkCell(
        sweep=cell.sweep,
        name=cell.name,
        parameters={**dict(cell.parameters), "bytes_per_row": record.get("bytes_per_row")},
        estimated_bytes=int(record["estimated_bytes"]),
    )
    return BenchmarkResult(
        cell=measured_cell,
        state="refused" if record.get("refused") else "supported",
        seconds=(),
        peak_rss_deltas=(int(record["peak_rss_delta"]),),
        rows=record.get("split_counts", {}).get(request.get("split", "train")),
        refusal_message=record.get("refusal_message"),
        notes={
            "isolated": True,
            "baseline_rss": record.get("baseline_rss"),
            "split_counts": record.get("split_counts", {}),
        },
    )


def measure_memory_repeated(
    cell: BenchmarkCell,
    request: Mapping[str, Any],
    *,
    workspace: Path,
    repeats: int = 5,
    timeout: float = 1800.0,
) -> BenchmarkResult:
    """Run ``repeats`` independent cold processes and aggregate their peaks.

    Each repeat is a fresh interpreter, so the repeats are genuinely independent
    samples rather than successive reads of one warming process. Cold-process
    RSS is page-granular and lumpy, so a single sample cannot support a claim
    about the estimator; this is the function that produces publishable numbers.

    The reported ``headroom`` uses the **maximum** observed peak, not the median.
    A memory bound is a worst-case claim: an estimator that covers the typical
    run but not the worst run is not a bound.
    """
    if repeats < 1:
        raise BenchmarkError("repeats must be at least 1")

    peaks: list[int] = []
    estimate: int | None = None
    rows: int | None = None
    refusal_message: str | None = None
    refused = False
    bytes_per_row: int | None = None

    for index in range(repeats):
        attempt = measure_memory_isolated(
            cell,
            request,
            workspace=workspace / f"repeat-{index}",
            timeout=timeout,
        )
        if attempt.state == "error":
            return BenchmarkResult(
                cell=cell,
                state="error",
                seconds=(),
                peak_rss_deltas=tuple(peaks),
                error=f"repeat {index}: {attempt.error}",
            )
        peaks.extend(attempt.peak_rss_deltas)
        estimate = attempt.cell.estimated_bytes
        bytes_per_row = attempt.cell.parameters.get("bytes_per_row")
        rows = attempt.rows
        refused = refused or attempt.state == "refused"
        refusal_message = refusal_message or attempt.refusal_message

    measured_cell = BenchmarkCell(
        sweep=cell.sweep,
        name=cell.name,
        parameters={**dict(cell.parameters), "bytes_per_row": bytes_per_row, "repeats": repeats},
        estimated_bytes=estimate,
    )
    worst = max(peaks) if peaks else None
    return BenchmarkResult(
        cell=measured_cell,
        state="refused" if refused else "supported",
        seconds=(),
        peak_rss_deltas=tuple(peaks),
        rows=rows,
        refusal_message=refusal_message,
        notes={
            "isolated": True,
            "repeats": repeats,
            "worst_peak_bytes": worst,
            "worst_headroom": (worst / estimate) if (worst and estimate) else None,
            "prewarm": bool(request.get("prewarm")),
        },
    )


def capture_environment() -> EnvironmentRecord:
    """Record interpreter, library, device, and host state for reproducibility."""
    import numpy as np

    sklearn_version: str | None
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:  # noqa: BLE001 - sklearn is core but must not break capture
        sklearn_version = None

    torch_version: str | None
    devices: list[str] = ["cpu"]
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            devices.append("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:  # noqa: BLE001 - torch is optional at capture time
        torch_version = None

    return EnvironmentRecord(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        total_memory_bytes=int(psutil.virtual_memory().total),
        numpy=np.__version__,
        sklearn=sklearn_version,
        torch=torch_version,
        torch_devices=tuple(devices),
        git_commit=_git_commit(),
    )


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except Exception:  # noqa: BLE001 - benchmarks must run outside a checkout
        return None
    return result.stdout.strip() or None


def write_results(
    path: str | Path,
    results: list[BenchmarkResult],
    environment: EnvironmentRecord,
) -> Path:
    """Write one JSONL file: an environment header line, then one line per cell."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record": "environment", **environment.to_dict()}) + "\n")
        for result in results:
            handle.write(json.dumps({"record": "cell", **result.to_dict()}) + "\n")
    return destination
