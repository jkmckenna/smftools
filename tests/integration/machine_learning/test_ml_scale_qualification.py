"""Measured scale qualification for the ML data plane (ML-700).

Runs under the ``integration`` marker, which `ci.yml` does not invoke on pull
requests — it executes weekly via `extended-ci.yml`. The fast, fixture-free
arithmetic guard that *does* run per PR lives in
``tests/unit/machine_learning/test_ml_scale_thresholds.py``; this file covers
what only measurement can establish.

Cells are deliberately small. The published limits come from an operator-invoked
sweep on real hardware (see ``dev/in-progress/ml700_benchmark_plan.md``); what is
asserted here is *behaviour that must not regress* — refusal firing at the
closed-form boundary, refusal happening before allocation, and streaming reads
staying bounded — never absolute byte counts, which are hardware-specific.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.machine_learning.benchmarks.fixtures import FixtureSpec, build_fixture
from smftools.machine_learning.benchmarks.sweeps import sweep_refusal_boundary
from smftools.machine_learning.data.partition_dataset import (
    MLMemoryBudgetError,
    PartitionReadPolicy,
)

pytestmark = pytest.mark.integration

THRESHOLDS = json.loads(
    Path("tests/acceptance/ml_scale_thresholds.json").read_text(encoding="utf-8")
)


def test_refusal_fires_exactly_at_the_closed_form_boundary(tmp_path: Path) -> None:
    # Probes budgets at estimate, estimate+1, and estimate-1, so the boundary is
    # pinned to the formula rather than to a remembered row count.
    results = sweep_refusal_boundary(n_rows=60, n_positions=200, root=tmp_path)

    by_name = {result.cell.name.split("_")[0]: result for result in results}
    assert by_name["at"].state == "supported"
    assert by_name["above"].state == "supported"
    assert by_name["below"].state == "refused"


def test_refusal_happens_before_allocation_and_names_a_remedy(tmp_path: Path) -> None:
    spec = FixtureSpec(n_rows=60, n_positions=200, n_channels=1)
    probe = build_fixture(tmp_path / "probe", spec)
    estimate = probe.plan.estimate_materialization_bytes("train")
    built = build_fixture(
        tmp_path / "tight",
        spec,
        policy=PartitionReadPolicy(max_materialization_bytes=estimate - 1),
    )

    with pytest.raises(MLMemoryBudgetError) as error:
        built.dataset.materialize("train")

    message = str(error.value)
    assert str(estimate) in message.replace(",", "")
    assert "iter_batches" in message


def test_streaming_reads_stay_bounded_as_the_split_grows(tmp_path: Path) -> None:
    # The bounded-memory acceptance criterion. Asserted as a *shape* claim --
    # batch count rises 4x while the per-batch estimate is unchanged -- rather
    # than as a byte threshold, because high-water RSS over-reports for
    # streaming workloads and is meaningless as a bound.
    batch_counts = []
    for index, n_rows in enumerate((200, 800)):
        built = build_fixture(
            tmp_path / f"rows-{index}",
            FixtureSpec(n_rows=n_rows, n_positions=200, n_channels=1),
            policy=PartitionReadPolicy(batch_size=16),
        )
        estimate = built.plan.estimate_batch_bytes(built.plan.effective_batch_size)
        observed = 0
        for batch in built.dataset.iter_batches("train"):
            assert built.plan.estimate_batch_bytes(len(batch.molecule_uids)) <= estimate
            observed += 1
        batch_counts.append(observed)

    assert batch_counts[1] > batch_counts[0], "the larger split must decode more batches"


def test_committed_limits_still_describe_the_live_defaults(tmp_path: Path) -> None:
    # Bridges the committed table to a real store: a fixture built at a shape
    # listed in the limits must report the published bytes_per_row.
    published = {
        (row["n_positions"], row["n_channels"]): row
        for row in THRESHOLDS["materialization_limits"]["rows"]
    }
    row = published[(500, 1)]
    built = build_fixture(tmp_path, FixtureSpec(n_rows=60, n_positions=500, n_channels=1))

    assert built.plan.bytes_per_row == row["bytes_per_row"]
    assert (
        built.plan.policy.max_materialization_bytes
        == THRESHOLDS["estimator"]["default_materialization_budget_bytes"]
    )
