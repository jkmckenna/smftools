"""Performance and scalability qualification for the ML data plane (ML-700).

The memory limits enforced by :mod:`smftools.machine_learning.data.partition_dataset`
are analytic: ``_bytes_per_row`` applies a fixed ``2x`` transient allowance and
``estimate_materialization_bytes`` a further ``3x``. Those constants are the only
thing standing between an approved read and an out-of-memory kill, and nothing
compares them to a measured peak. This package measures that gap.

The headline quantity is the *headroom ratio* ``peak_rss_delta / estimated_bytes``.
It must stay at or below ``1.0`` for the estimate to be a genuine upper bound; a
very small ratio instead means the estimator is over-conservative and refuses
workloads that would have fit.

Nothing here is imported by pipeline or CLI code. Benchmarks are operator-invoked
through :mod:`smftools.machine_learning.benchmarks.sweeps`, with a small
representative subset asserted by ``tests/integration/machine_learning``.
"""

from __future__ import annotations

from .harness import (
    BenchmarkCell,
    BenchmarkResult,
    EnvironmentRecord,
    capture_environment,
    measure,
    write_results,
)

__all__ = [
    "BenchmarkCell",
    "BenchmarkResult",
    "EnvironmentRecord",
    "capture_environment",
    "measure",
    "write_results",
]
