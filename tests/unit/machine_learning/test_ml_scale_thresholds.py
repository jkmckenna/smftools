"""Per-PR guard for the committed ML scale limits (ML-700).

Deliberately fast and fixture-free: it recomputes every published limit from the
estimator itself and compares. Changing ``_bytes_per_row`` or a default budget
moves refusal boundaries package-wide, and this fails immediately rather than
waiting for the weekly measured suite in
``tests/integration/machine_learning/test_ml_scale_qualification.py``.

It asserts arithmetic and structure only. Nothing here measures memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.machine_learning.data.partition_dataset import (
    DEFAULT_BATCH_MEMORY_BYTES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MATERIALIZATION_MEMORY_BYTES,
    _bytes_per_row,
)

THRESHOLDS_PATH = Path("tests/acceptance/ml_scale_thresholds.json")

VALID_STATES = {
    "supported",
    "refused above the ceiling",
    "supported, but memory is NOT modelled",
}


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def test_thresholds_file_declares_its_schema(thresholds: dict) -> None:
    assert thresholds["schema_version"] == 1
    assert thresholds["work_package"] == "ML-700"
    assert thresholds["source"]


def test_published_estimator_constants_match_the_implementation(thresholds: dict) -> None:
    estimator = thresholds["estimator"]

    assert estimator["transient_multiplier"] == 2
    assert estimator["materialization_multiplier"] == 3
    assert estimator["default_materialization_budget_bytes"] == DEFAULT_MATERIALIZATION_MEMORY_BYTES
    assert estimator["default_batch_budget_bytes"] == DEFAULT_BATCH_MEMORY_BYTES
    assert estimator["default_batch_size"] == DEFAULT_BATCH_SIZE


def test_every_published_limit_recomputes_from_the_estimator(thresholds: dict) -> None:
    # The published ceilings are not transcribed numbers to be trusted; they are
    # derivable, and this is the derivation. A changed estimator constant fails
    # here on the pull request that changes it.
    budget = DEFAULT_MATERIALIZATION_MEMORY_BYTES
    rows = thresholds["materialization_limits"]["rows"]
    assert rows, "published limits must not be empty"

    for row in rows:
        expected_bytes_per_row = _bytes_per_row(row["n_positions"], row["n_channels"], labeled=True)
        assert row["bytes_per_row"] == expected_bytes_per_row, (
            f"bytes_per_row for {row['n_positions']}x{row['n_channels']} is stale; "
            "the estimator changed and the published limits were not regenerated"
        )
        assert row["max_materializable_train_rows"] == budget // (3 * expected_bytes_per_row)
        assert row["max_rows_per_batch"] == DEFAULT_BATCH_MEMORY_BYTES // expected_bytes_per_row


def test_published_limits_cover_both_modality_shapes(thresholds: dict) -> None:
    channels = {row["n_channels"] for row in thresholds["materialization_limits"]["rows"]}

    assert channels == {1, 2}, "limits must cover deaminase (1) and conversion (2) shapes"


def test_taxonomy_assigns_every_workload_a_recognised_state(thresholds: dict) -> None:
    taxonomy = thresholds["workload_taxonomy"]
    assert taxonomy

    for entry in taxonomy:
        assert entry["state"] in VALID_STATES, entry["state"]
        assert entry["evidence"].strip()
        assert entry["guidance"].strip()


def test_taxonomy_covers_both_training_backends_and_both_read_modes(thresholds: dict) -> None:
    workloads = " ".join(entry["workload"] for entry in thresholds["workload_taxonomy"])

    assert "iter_batches" in workloads
    assert "materialize" in workloads
    assert "sklearn" in workloads
    assert "Torch" in workloads


def test_torch_memory_is_declared_unmodelled_rather_than_silently_omitted(
    thresholds: dict,
) -> None:
    # Torch process memory is dominated by activations, which no budget models.
    # A limits document that quietly omitted the dominant term would be worse
    # than one naming it, so the omission is asserted to stay explicit and
    # attributed to a work package.
    torch_entry = next(
        entry for entry in thresholds["workload_taxonomy"] if "Torch" in entry["workload"]
    )
    assert torch_entry["state"] == "supported, but memory is NOT modelled"
    assert torch_entry["ceiling"] == "unknown"

    unmodelled = thresholds["unmodelled"]
    assert unmodelled, "the unmodelled register must not be empty while Torch is unmodelled"
    owners = {item["owner"] for item in unmodelled}
    assert "ML-205" in owners


def test_regression_thresholds_are_ordered_and_documented(thresholds: dict) -> None:
    limits = thresholds["regression_thresholds"]

    assert limits["materialization_worst_headroom_max"] == 1.0
    assert 0 < limits["materialization_headroom_soft_min"] < 1.0
    assert 0 < limits["streaming_tail_growth_fraction_max"] < 1.0
    assert limits["wall_time_regression_max_fraction"] > 0
    # The notes carry the measurement caveats. Without them a future reader
    # would reasonably threshold streaming on high-water RSS, which this
    # package established is meaningless as a bound.
    assert len(limits["notes"]) >= 4


def test_baselines_are_marked_hardware_specific(thresholds: dict) -> None:
    note = thresholds["baseline_environment"]["note"]

    assert "ratio" in note.lower()


def test_explanation_chunking_guidance_is_published_with_its_caveats(thresholds: dict) -> None:
    # example_batch_size reads like a performance knob but changes attribution
    # values by roughly 1e-3 relative. The guidance has to say so, or a user
    # tuning it for speed silently perturbs the results being compared.
    chunking = thresholds["explanation_chunking"]

    assert chunking["timing"], "chunk-size timing must be published"
    assert any(entry["relative_to_best"] == 1.0 for entry in chunking["timing"]), (
        "the timing table must identify which chunk size was fastest"
    )
    largest = max(chunking["timing"], key=lambda entry: entry["example_batch_size"])
    assert largest["relative_to_best"] > 1.0, (
        "the largest chunk must not be recorded as fastest; the measured optimum is interior "
        "and the guidance depends on that"
    )
    guidance = " ".join(chunking["guidance"]).lower()
    assert "scientific parameter" in guidance
    assert "hold it fixed" in guidance
    # Memory across chunk sizes was not measured reliably; the caveat must stay
    # visible so nobody quotes the sequential in-process numbers as limits.
    assert any("memory" in caveat.lower() for caveat in chunking["caveats"])


def test_worker_scaling_guidance_separates_sharding_cost_from_wall_clock(thresholds: dict) -> None:
    # Sharding being free is a claim about arithmetic overhead, not about
    # parallel speedup. The distinction has to survive in the published text,
    # or "sharding is free" will be read as "N workers are N times faster".
    scaling = thresholds["worker_scaling"]
    shards = scaling["shards"]

    assert [entry["num_workers"] for entry in shards] == [1, 2, 4, 8]
    total = shards[0]["shard_rows"]
    for entry in shards:
        assert entry["shard_rows"] * entry["num_workers"] == total, (
            "shards must partition the split exactly"
        )
    guidance = " ".join(scaling["guidance"]).lower()
    assert "does not establish" in guidance
    assert "contend" in guidance
