"""ML-700 sweeps.

Implemented here:

- :func:`sweep_memory_calibration` (Sweep A) -- the core deliverable. Measures
  peak RSS against the analytic estimate for bounded batch reads and for full
  materialization, producing the headroom distribution.
- :func:`sweep_refusal_boundary` -- confirms ``MLMemoryBudgetError`` fires
  exactly where the closed form predicts, and that refusal happens at preflight
  without allocating.
- :func:`sweep_worker_scaling` (Sweep C) -- per-shard throughput and bounded
  per-worker peak.

Sweep B (backend throughput across sklearn/Torch and cpu/mps) and Sweep D
(explanation chunk sizing) are specified in ``dev/in-progress/ml700_benchmark_plan.md``
and are not implemented yet.

Refusal cells deliberately run against a *reduced* ``max_materialization_bytes``
rather than building a 50,000-row store. The refusal predicate is
``estimate > budget``; scaling the budget down exercises the identical code path
and the identical formula at a fraction of the cost, and the boundary is asserted
against the closed form rather than against a hardcoded row count.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from smftools.machine_learning.data.partition_dataset import (
    MLMemoryBudgetError,
    PartitionReadPolicy,
)

from .fixtures import FixtureSpec, build_fixture
from .harness import (
    BenchmarkCell,
    BenchmarkResult,
    measure,
    measure_memory_repeated,
)

# Sweep A levels. Rows straddle the materialization frontier; positions span a
# single locus through a full amplicon panel.
CALIBRATION_ROWS = (500, 5_000, 50_000)
CALIBRATION_POSITIONS = (500, 1_000, 5_000, 20_000)
CALIBRATION_CHANNELS = (1, 2)

# The CI subset: small enough to run on every push, wide enough to catch a
# changed estimator constant.
CI_ROWS = (500,)
CI_POSITIONS = (500, 1_000)
CI_CHANNELS = (1, 2)

WORKER_COUNTS = (1, 2, 4, 8)


def _consume_batches(dataset, split: str, **kwargs) -> int:
    rows = 0
    for batch in dataset.iter_batches(split, **kwargs):
        # Touch the values so lazy decoding cannot be skipped.
        rows += int(batch.values.shape[0])
    return rows


def sweep_memory_calibration(
    *,
    rows: Sequence[int] = CALIBRATION_ROWS,
    positions: Sequence[int] = CALIBRATION_POSITIONS,
    channels: Sequence[int] = CALIBRATION_CHANNELS,
    repeats: int = 3,
    warmup: int = 1,
    root: Path | None = None,
) -> list[BenchmarkResult]:
    """Measure peak RSS against the analytic estimate for reads and materialization."""
    results: list[BenchmarkResult] = []
    for n_rows in rows:
        for n_positions in positions:
            for n_channels in channels:
                spec = FixtureSpec(
                    n_rows=n_rows,
                    n_positions=n_positions,
                    n_channels=n_channels,
                )
                results.extend(
                    _calibration_cells(
                        spec,
                        repeats=repeats,
                        warmup=warmup,
                        root=root,
                    )
                )
    return results


def _calibration_cells(
    spec: FixtureSpec,
    *,
    repeats: int,
    warmup: int,
    root: Path | None,
) -> list[BenchmarkResult]:
    with _workspace(root) as workspace:
        built = build_fixture(workspace, spec)
        plan = built.plan
        train_rows = built.split_counts.get("train", 0)

        batch_estimate = plan.estimate_batch_bytes(plan.effective_batch_size)
        batch_cell = BenchmarkCell(
            sweep="A",
            name=f"batch_read_{spec.label}",
            parameters={
                "n_rows": spec.n_rows,
                "n_positions": spec.n_positions,
                "n_channels": spec.n_channels,
                "n_partitions": spec.n_partitions,
                "batch_size": plan.effective_batch_size,
                "bytes_per_row": plan.bytes_per_row,
                "mode": "iter_batches",
            },
            estimated_bytes=batch_estimate,
        )
        batch_result = measure(
            batch_cell,
            lambda: _consume_batches(built.dataset, "train"),
            repeats=repeats,
            warmup=warmup,
            rows=train_rows,
            notes={"split_counts": dict(built.split_counts)},
        )

        materialization_estimate = plan.estimate_materialization_bytes("train")
        budget = plan.policy.max_materialization_bytes
        refuses = materialization_estimate > budget
        materialize_cell = BenchmarkCell(
            sweep="A",
            name=f"materialize_{spec.label}",
            parameters={
                "n_rows": spec.n_rows,
                "n_positions": spec.n_positions,
                "n_channels": spec.n_channels,
                "n_partitions": spec.n_partitions,
                "bytes_per_row": plan.bytes_per_row,
                "budget_bytes": budget,
                "mode": "materialize",
            },
            estimated_bytes=materialization_estimate,
        )
        materialize_result = measure(
            materialize_cell,
            lambda: built.dataset.materialize("train"),
            repeats=repeats,
            warmup=0 if refuses else warmup,
            rows=train_rows,
            expect_refusal=MLMemoryBudgetError if refuses else None,
            notes={"predicted_refusal": refuses},
        )

    return [batch_result, materialize_result]


def sweep_memory_repeated(
    *,
    rows: Sequence[int] = CI_ROWS,
    positions: Sequence[int] = CI_POSITIONS,
    channels: Sequence[int] = CI_CHANNELS,
    modes: Sequence[str] = ("iter_batches", "materialize"),
    repeats: int = 5,
    prewarm: bool = True,
    root: Path | None = None,
) -> list[BenchmarkResult]:
    """Sweep A with N independent cold processes per cell.

    This is the publishable variant. :func:`sweep_memory_calibration` measures
    timing in-process and is retained for throughput; its memory readings are
    biased low by allocator warmth and must not be quoted as limits.
    """
    results: list[BenchmarkResult] = []
    with _workspace(root) as workspace:
        for n_rows in rows:
            for n_positions in positions:
                for n_channels in channels:
                    for mode in modes:
                        label = f"rows{n_rows}_pos{n_positions}_ch{n_channels}"
                        cell = BenchmarkCell(
                            sweep="A",
                            name=f"{mode}_{label}",
                            parameters={
                                "n_rows": n_rows,
                                "n_positions": n_positions,
                                "n_channels": n_channels,
                                "mode": mode,
                                "prewarm": prewarm,
                            },
                        )
                        results.append(
                            measure_memory_repeated(
                                cell,
                                {
                                    "mode": mode,
                                    "n_rows": n_rows,
                                    "n_positions": n_positions,
                                    "n_channels": n_channels,
                                    "prewarm": prewarm,
                                },
                                workspace=workspace / f"{mode}-{label}",
                                repeats=repeats,
                            )
                        )
    return results


def sweep_bounded_batch_memory(
    *,
    row_counts: Sequence[int] = (500, 1_000, 2_000, 4_000),
    n_positions: int = 500,
    n_channels: int = 1,
    repeats: int = 5,
    prewarm: bool = True,
    root: Path | None = None,
) -> list[BenchmarkResult]:
    """Test the bounded-memory claim directly: does batch-read peak track total rows?

    ``iter_batches`` is supposed to hold at most one decoded batch regardless of
    how many rows the split contains. If that holds, peak RSS stays flat as
    ``row_counts`` grows and the batch estimate remains a valid bound at any
    scale. If peak instead grows with total rows, something is accumulating
    across batches and the whole "stream instead of materialize" guidance is
    unsound -- which is the single most consequential thing this sweep can find.
    """
    results: list[BenchmarkResult] = []
    with _workspace(root) as workspace:
        for n_rows in row_counts:
            cell = BenchmarkCell(
                sweep="A-bounded",
                name=f"iter_batches_rows{n_rows}_pos{n_positions}_ch{n_channels}",
                parameters={
                    "n_rows": n_rows,
                    "n_positions": n_positions,
                    "n_channels": n_channels,
                    "mode": "iter_batches",
                    "prewarm": prewarm,
                },
            )
            results.append(
                measure_memory_repeated(
                    cell,
                    {
                        "mode": "iter_batches",
                        "n_rows": n_rows,
                        "n_positions": n_positions,
                        "n_channels": n_channels,
                        "prewarm": prewarm,
                    },
                    workspace=workspace / f"bounded-{n_rows}",
                    repeats=repeats,
                )
            )
    return results


def measure_batch_trajectory(
    *,
    n_rows: int = 32_000,
    n_positions: int = 500,
    n_channels: int = 1,
    prewarm: bool = True,
    root: Path | None = None,
    timeout: float = 3600.0,
    mode: str = "trajectory",
    backend: str | None = None,
    max_epochs: int = 3,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Run one long streaming read and decide bounded vs accumulating.

    Returns per-quartile growth in bytes/batch. The verdict rests on the shape,
    not the magnitude:

    - **accumulating** -- growth stays near the batch size in every quartile.
      One batch per iteration is being retained.
    - **bounded** -- growth decelerates toward zero. The allocator arena is
      reaching steady state and live memory is bounded.

    A long run matters because a short one cannot distinguish a plateau from the
    early part of a linear rise.
    """
    import json
    import subprocess
    import sys

    with _workspace(root) as workspace:
        payload = {
            "mode": mode,
            "n_rows": n_rows,
            "n_positions": n_positions,
            "n_channels": n_channels,
            "prewarm": prewarm,
            "workspace": str(workspace),
            "max_epochs": max_epochs,
        }
        if backend is not None:
            payload["backend"] = backend
        if batch_size is not None:
            payload["batch_size"] = batch_size
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "smftools.machine_learning.benchmarks._isolated",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    record = None
    for line in reversed(completed.stdout.strip().splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if record is None or "trajectory" not in record:
        return {"error": completed.stderr.strip()[-2000:] or "no trajectory produced"}

    trajectory = record["trajectory"]
    n = len(trajectory)
    if n < 8:
        return {"error": f"trajectory too short to analyse ({n} batches)"}

    quartiles = []
    for index in range(4):
        start = (n * index) // 4
        end = (n * (index + 1)) // 4 - 1
        span = max(1, end - start)
        quartiles.append((trajectory[end] - trajectory[start]) / span)

    batch_bytes = record["estimated_bytes"]
    last = quartiles[-1]
    first = quartiles[0]
    plateau = trajectory[-1]

    # The criterion is plateau-relative, not batch-estimate-relative. An earlier
    # version compared final-quartile growth to the data batch estimate, which
    # is the wrong scale for a fit: a Torch working set is dominated by model
    # activations, so a genuinely flat trajectory was reported as
    # "accumulating" simply because the model is larger than one data batch.
    # What distinguishes bounded from accumulating is whether growth has stopped
    # relative to where it settled, plus deceleration from the first quartile --
    # real accumulation cannot decelerate, since it allocates afresh every pass.
    tail_fraction = (last * max(1, n // 4)) / plateau if plateau > 0 else 0.0
    verdict = (
        "bounded" if tail_fraction < 0.05 and (first <= 0 or last < 0.5 * first) else "accumulating"
    )

    return {
        "mode": mode,
        "backend": record.get("backend"),
        "plateau_bytes": plateau,
        "tail_growth_fraction": tail_fraction,
        "plateau_over_batch_estimate": (plateau / batch_bytes) if batch_bytes else None,
        "batches": n,
        "batch_estimate_bytes": batch_bytes,
        "effective_batch_size": record["effective_batch_size"],
        "train_rows": record["split_counts"].get("train"),
        "quartile_bytes_per_batch": quartiles,
        "final_rss_delta": trajectory[-1],
        "rss_after_collect": record.get("rss_after_collect"),
        "verdict": verdict,
    }


def sweep_refusal_boundary(
    *,
    n_positions: int = 1_000,
    n_channels: int = 1,
    n_rows: int = 200,
    root: Path | None = None,
) -> list[BenchmarkResult]:
    """Confirm the refusal predicate fires exactly at the closed-form boundary.

    Builds one fixture, then probes it with two budgets derived from its own
    measured ``bytes_per_row``: one a single byte above the estimate (must be
    approved) and one a single byte below (must be refused). This pins the
    boundary to the formula rather than to a remembered row count.
    """
    spec = FixtureSpec(n_rows=n_rows, n_positions=n_positions, n_channels=n_channels)
    results: list[BenchmarkResult] = []

    with _workspace(root) as workspace:
        probe = build_fixture(workspace / "probe", spec)
        estimate = probe.plan.estimate_materialization_bytes("train")
        train_rows = probe.split_counts.get("train", 0)

        for label, budget, expect in (
            ("at_budget", estimate, None),
            ("above_budget", estimate + 1, None),
            ("below_budget", estimate - 1, MLMemoryBudgetError),
        ):
            built = build_fixture(
                workspace / label,
                spec,
                policy=PartitionReadPolicy(max_materialization_bytes=budget),
            )
            cell = BenchmarkCell(
                sweep="refusal",
                name=f"{label}_{spec.label}",
                parameters={
                    "n_rows": spec.n_rows,
                    "n_positions": spec.n_positions,
                    "n_channels": spec.n_channels,
                    "budget_bytes": budget,
                    "estimate_bytes": estimate,
                    "bytes_per_row": built.plan.bytes_per_row,
                },
                estimated_bytes=estimate,
            )
            results.append(
                measure(
                    cell,
                    lambda dataset=built.dataset: dataset.materialize("train"),
                    repeats=1,
                    warmup=0,
                    rows=train_rows,
                    expect_refusal=expect,
                )
            )
    return results


def sweep_worker_scaling(
    *,
    n_rows: int = 5_000,
    n_positions: int = 1_000,
    n_channels: int = 1,
    worker_counts: Iterable[int] = WORKER_COUNTS,
    repeats: int = 3,
    warmup: int = 1,
    root: Path | None = None,
) -> list[BenchmarkResult]:
    """Measure single-shard throughput and peak as ``num_workers`` rises.

    Shards are read in-process one at a time: this measures the *per-shard*
    bound, which is what determines whether N real workers fit in memory
    simultaneously. It deliberately does not spawn processes -- process pool
    behavior is `memory_guard`'s domain and is already covered there.
    """
    results: list[BenchmarkResult] = []
    spec = FixtureSpec(n_rows=n_rows, n_positions=n_positions, n_channels=n_channels)

    with _workspace(root) as workspace:
        built = build_fixture(workspace, spec)
        train_rows = built.split_counts.get("train", 0)

        for num_workers in worker_counts:
            cell = BenchmarkCell(
                sweep="C",
                name=f"worker_shard_{num_workers}_{spec.label}",
                parameters={
                    "n_rows": spec.n_rows,
                    "n_positions": spec.n_positions,
                    "n_channels": spec.n_channels,
                    "num_workers": num_workers,
                    "batch_size": built.plan.effective_batch_size,
                },
                estimated_bytes=built.plan.estimate_batch_bytes(built.plan.effective_batch_size),
            )
            results.append(
                measure(
                    cell,
                    lambda workers=num_workers: _consume_batches(
                        built.dataset, "train", worker_id=0, num_workers=workers
                    ),
                    repeats=repeats,
                    warmup=warmup,
                    rows=max(1, train_rows // num_workers),
                    notes={"total_train_rows": train_rows},
                )
            )
    return results


class _workspace:
    """Yield ``root`` if given, else a temporary directory removed on exit."""

    def __init__(self, root: Path | None) -> None:
        self._root = root
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            return self._root
        self._tmp = tempfile.TemporaryDirectory(prefix="ml700-")
        return Path(self._tmp.name)

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def measure_explanation_chunking(
    *,
    n_rows: int = 400,
    n_positions: int = 400,
    n_channels: int = 1,
    example_batch_sizes: Sequence[int] = (1, 8, 64, 512),
    method: str = "Saliency",
    root: Path | None = None,
) -> list[dict[str, object]]:
    """Sweep D: what does ``example_batch_size`` buy, and what does it cost?

    Attribution runs a forward and backward pass per chunk, so the chunk size
    trades wall time against peak memory in the same way a training batch does.
    Neither term is modelled by any budget in the ML data plane -- see the Torch
    entry in ``tests/acceptance/ml_scale_thresholds.json`` -- so the guidance
    published for chunk sizing has to be measured rather than derived.

    Runs in-process: attribution is a single bounded pass, so the warm-arena
    under-reporting that forced cold subprocesses for repeated reads does not
    apply in the same way. Peak is still the high-water mark and is reported as
    such.
    """
    import gc
    import time

    import numpy as np
    import psutil

    from smftools.machine_learning.artifacts import ExplanationMaskPolicy, ExplanationTarget
    from smftools.machine_learning.interpretability import (
        ExplanationDecisionProvenance,
        InterpretabilityRequest,
        explain_torch_model,
    )
    from smftools.machine_learning.models.registry import BUILTIN_MODEL_REGISTRY
    from smftools.machine_learning.training.torch_backend import (
        TorchTrainingConfig,
        fit_torch_partition_model_streaming,
    )

    results: list[dict[str, object]] = []
    process = psutil.Process()

    with _workspace(root) as workspace:
        built = build_fixture(
            workspace,
            FixtureSpec(
                n_rows=n_rows,
                n_positions=n_positions,
                n_channels=n_channels,
                imbalanced=True,
            ),
        )
        resolved = BUILTIN_MODEL_REGISTRY.resolve(
            "residual_dilated_cnn", input_schema=built.plan.dataset.input_schema
        )
        trained = fit_torch_partition_model_streaming(
            built.dataset,
            resolved,
            training_config=TorchTrainingConfig(max_epochs=1, device="cpu", batch_size=32),
        )
        data = built.dataset.materialize("test")
        mask_kinds = tuple(
            mask.kind
            for mask in trained.model.input_schema.masks
            if mask.kind in trained.model.architecture.capabilities.supported_mask_kinds
        )

        for size in example_batch_sizes:
            request = InterpretabilityRequest.create(
                method=method,
                model_id="a" * 64,
                dataset_snapshot_id=built.plan.dataset.snapshot_id,
                input_schema_hash=trained.model.input_schema.schema_hash,
                split_role=data.split,
                cohort=f"{data.split}-natural",
                observation_uids=data.molecule_uids,
                target=ExplanationTarget(
                    output_name="activity_target_logit", class_id=1, class_name="active"
                ),
                baseline=None,
                layer=None,
                mask_policy=ExplanationMaskPolicy.create(
                    mask_kinds=mask_kinds,
                    handling=(
                        "forward masks through the model and zero invalid input attributions"
                    ),
                ),
                decision=ExplanationDecisionProvenance("fixed"),
                parameters={"absolute": False, "example_batch_size": size},
                random_seed=13,
            )
            gc.collect()
            baseline_rss = int(process.memory_info().rss)
            start = time.perf_counter()
            result = explain_torch_model(trained.model, data, request)
            elapsed = time.perf_counter() - start
            peak = int(process.memory_info().rss) - baseline_rss
            values = np.asarray(result.values)
            results.append(
                {
                    "method": method,
                    "example_batch_size": size,
                    "n_rows_explained": int(values.shape[0]),
                    "n_positions": n_positions,
                    "seconds": elapsed,
                    "rss_delta_bytes": max(0, peak),
                    "attribution_checksum": float(np.abs(values).sum()),
                }
            )
            del result, values
            gc.collect()
    return results
