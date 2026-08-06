"""Cold-allocator memory measurement runner (ML-700).

Invoked as ``python -m smftools.machine_learning.benchmarks._isolated <json>``.

Why a subprocess exists at all: RSS delta measured in-process systematically
*under*-reports peak allocation once CPython's allocator holds a warm arena.
A warmed-up repeat can satisfy a multi-megabyte NumPy allocation without the
process RSS growing at all, so the measured peak collapses toward zero and the
estimator looks far safer than it is. Under-reporting is the dangerous
direction for a guardrail, so memory cells are measured once, in a fresh
process, with no warmup.

Timing is not measured here -- a cold process pays import and page-cache costs
that would swamp the signal. ``harness.measure`` keeps that job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(json.dumps({"error": "expected one JSON argument"}))
        return 2

    request = json.loads(argv[1])
    mode = request["mode"]
    workspace = Path(request["workspace"])

    # Imported inside main so the JSON error path above stays cheap.
    from ..data.partition_dataset import MLMemoryBudgetError, PartitionReadPolicy
    from .fixtures import FixtureSpec, build_fixture
    from .harness import _RSSWatchdog

    spec = FixtureSpec(
        n_rows=request["n_rows"],
        n_positions=request["n_positions"],
        n_channels=request["n_channels"],
        n_partitions=request.get("n_partitions", 1),
        seed=request.get("seed", 0),
    )

    policy = None
    if request.get("max_materialization_bytes") is not None:
        policy = PartitionReadPolicy(
            max_materialization_bytes=int(request["max_materialization_bytes"])
        )

    # Init-separation control. One-time costs -- Zarr codec registration, AnnData
    # machinery, pandas index types -- are paid by the first read in a process
    # and would otherwise be attributed to the measured cell's peak, inflating
    # the apparent fixed overhead. A discarded read against a trivial store pays
    # them before the baseline is taken. Requested explicitly so the difference
    # between prewarmed and cold runs is itself measurable.
    if request.get("prewarm"):
        probe_spec = FixtureSpec(n_rows=10, n_positions=8, n_channels=spec.n_channels)
        probe = build_fixture(workspace / "_prewarm", probe_spec)
        probe.dataset.materialize("train")
        del probe

    built = build_fixture(workspace, spec, policy=policy)
    plan = built.plan
    split = request.get("split", "train")

    if mode == "materialize":
        estimate = plan.estimate_materialization_bytes(split)

        def operation() -> None:
            built.dataset.materialize(split)

    elif mode == "iter_batches":
        estimate = plan.estimate_batch_bytes(plan.effective_batch_size)

        def operation() -> None:
            for batch in built.dataset.iter_batches(split):
                # Touch values so lazy decoding cannot be elided.
                int(batch.values.shape[0])

    elif mode == "trajectory":
        # Per-batch RSS after each decoded batch. Peak RSS is a high-water mark
        # and therefore useless for deciding whether a streaming read is
        # bounded: freed pages are not returned to the OS, so the mark rises
        # with the *number* of allocations even when one batch is ever live.
        # The trajectory separates the two -- accumulation grows linearly at
        # roughly the batch size forever, while an allocator arena decelerates
        # toward a plateau.
        import gc

        import psutil

        estimate = plan.estimate_batch_bytes(plan.effective_batch_size)
        process = psutil.Process()
        gc.collect()
        trajectory_baseline = int(process.memory_info().rss)
        trajectory: list[int] = []
        for batch in built.dataset.iter_batches(split):
            int(batch.values.shape[0])
            trajectory.append(int(process.memory_info().rss) - trajectory_baseline)
        gc.collect()
        print(
            json.dumps(
                {
                    "peak_rss_delta": max(trajectory) if trajectory else 0,
                    "baseline_rss": trajectory_baseline,
                    "estimated_bytes": estimate,
                    "bytes_per_row": plan.bytes_per_row,
                    "effective_batch_size": plan.effective_batch_size,
                    "split_counts": dict(built.split_counts),
                    "refused": False,
                    "refusal_message": None,
                    "prewarm": bool(request.get("prewarm")),
                    "trajectory": trajectory,
                    "rss_after_collect": int(process.memory_info().rss) - trajectory_baseline,
                }
            )
        )
        return 0

    else:
        print(json.dumps({"error": f"unknown mode {mode!r}"}))
        return 2

    refused = False
    message = None
    with _RSSWatchdog() as watchdog:
        try:
            operation()
        except MLMemoryBudgetError as exc:
            refused = True
            message = str(exc)

    print(
        json.dumps(
            {
                "peak_rss_delta": watchdog.delta,
                "baseline_rss": watchdog.baseline,
                "estimated_bytes": estimate,
                "bytes_per_row": plan.bytes_per_row,
                "effective_batch_size": plan.effective_batch_size,
                "split_counts": dict(built.split_counts),
                "refused": refused,
                "refusal_message": message,
                "prewarm": bool(request.get("prewarm")),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main(sys.argv))
