"""MLX-06: a project plan's train job binds to real stores and trains per fold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.ml_project import (
    EXPERIMENTS,
    N_POSITIONS,
    READS_PER_BARCODE,
    make_ml_project,
)
from tests.fixtures.ml_project import ml_plan as _plan

from smftools.machine_learning.data.partition_dataset import PartitionReadPolicy
from smftools.machine_learning.orchestration import bind_ml_job, run_bound_train_job
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_ml_project(tmp_path)


def test_bound_job_has_one_fold_per_held_out_experiment(project: Path) -> None:
    bound = bind_ml_job(_plan(), "train", project_dir=project)

    assert len(bound.snapshot.observations) == len(EXPERIMENTS) * 2 * READS_PER_BARCODE
    experiment_of = {item.molecule_uid: item.experiment_uid for item in bound.snapshot.observations}
    held_out = []
    for fold in bound.folds:
        roles: dict[str, set[str]] = {}
        for uid, role in fold.resolution.assignments.items():
            roles.setdefault(role, set()).add(experiment_of[uid])
        assert len(roles["test"]) == 1, fold.fold_name
        assert not roles["test"] & roles["train"]
        held_out.extend(roles["test"])
    assert sorted(held_out) == sorted(set(experiment_of.values()))


def test_bound_job_trains_and_evaluates_every_fold(project: Path) -> None:
    bound = bind_ml_job(_plan(), "train", project_dir=project)
    runs = run_bound_train_job(bound)

    assert [run.fold_name for run in runs] == [fold.fold_name for fold in bound.folds]
    for run in runs:
        assert run.training.n_training_observations == 2 * 2 * READS_PER_BARCODE
        assert run.predictions.n_observations == 2 * READS_PER_BARCODE
        metrics = {
            metric.name: metric.value
            for metric in run.evaluation.metrics
            if metric.modality is None
        }
        # The signal separates the classes almost perfectly.
        assert metrics["average_precision"] > 0.95, run.fold_name


def test_snapshot_identity_is_stable_across_binds(project: Path) -> None:
    first = bind_ml_job(_plan(), "train", project_dir=project)
    second = bind_ml_job(_plan(), "train", project_dir=project)
    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
    assert [fold.split.split_id for fold in first.folds] == [
        fold.split.split_id for fold in second.folds
    ]


# --- MLX-09: partition-major reads (F67) -----------------------------------


def test_batches_read_one_partition_at_a_time(project: Path) -> None:
    bound = bind_ml_job(
        _plan(), "train", project_dir=project, policy=PartitionReadPolicy(batch_size=8)
    )
    fold = bound.folds[0]
    partition_of = {entry.read_id: entry.read_key[0] for entry in fold.dataset.plan.entries}
    batches = list(fold.dataset.iter_batches("train"))

    assert all(entry.read_key for entry in fold.dataset.plan.entries)
    # Each batch comes from one store partition (8 divides the 24 reads per barcode).
    assert all(len({partition_of[read_id] for read_id in batch.read_ids}) == 1 for batch in batches)
    read = [read_id for batch in batches for read_id in batch.read_ids]
    expected = [entry.read_id for entry in fold.dataset.plan.entries_for("train")]
    assert sorted(read) == sorted(expected) and len(read) == len(set(read))


def test_materialized_split_keeps_manifest_order(project: Path) -> None:
    bound = bind_ml_job(
        _plan(), "train", project_dir=project, policy=PartitionReadPolicy(batch_size=8)
    )
    dataset = bound.folds[0].dataset
    canonical = dataset.plan.entries_for("train")
    assert [entry.read_id for entry in dataset.plan.read_order("train")] != [
        entry.read_id for entry in canonical
    ]

    data = dataset.materialize("train")

    assert list(data.molecule_uids) == [entry.molecule_uid for entry in canonical]
    assert list(data.labels) == [entry.class_id for entry in canonical]


def test_block_reads_match_batch_reads_with_fewer_store_reads(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smftools.machine_learning.data.partition_dataset as reader

    calls = []
    original = reader.materialize

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "materialize", counting)

    def read(max_block_bytes: int):
        policy = PartitionReadPolicy(batch_size=8, max_block_bytes=max_block_bytes)
        dataset = bind_ml_job(_plan(), "train", project_dir=project, policy=policy).folds[0].dataset
        calls.clear()
        batches = list(dataset.iter_batches("train"))
        return batches, len(calls)

    per_batch, batch_reads = read(1)  # a block of one batch
    blocked, block_reads = read(1024**3)

    assert block_reads < batch_reads
    assert [batch.read_ids for batch in blocked] == [batch.read_ids for batch in per_batch]
    for left, right in zip(blocked, per_batch, strict=True):
        np.testing.assert_array_equal(left.values, right.values)
        np.testing.assert_array_equal(left.labels, right.labels)
        np.testing.assert_array_equal(left.padding_mask, right.padding_mask)


def test_test_role_is_predicted_in_batches_not_materialized(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smftools.machine_learning.data.partition_dataset as reader

    def refuse(self, split):
        raise AssertionError("the runner must not materialize a split")

    monkeypatch.setattr(reader.PartitionDataset, "materialize", refuse)
    bound = bind_ml_job(
        _plan(), "train", project_dir=project, policy=PartitionReadPolicy(batch_size=8)
    )
    runs = run_bound_train_job(bound)

    for run, fold in zip(runs, bound.folds, strict=True):
        expected = [entry.molecule_uid for entry in fold.dataset.plan.read_order("test")]
        assert list(run.predictions.molecule_uids) == expected


# --- MLX-02: position masks ------------------------------------------------


def _masked_plan(include, exclude=()):
    document = _plan().to_dict()
    document["datasets"]["reads"]["positions"] = {
        "include": [list(window) for window in include],
        "exclude": [list(window) for window in exclude],
    }
    return parse_ml_plan(document)


def test_masked_dataset_holds_only_kept_positions(project: Path) -> None:
    policy = PartitionReadPolicy(batch_size=8)
    full = bind_ml_job(_plan(), "train", project_dir=project, policy=policy).folds[0].dataset
    masked_plan = _masked_plan([(0, N_POSITIONS)], exclude=[(10, 30)])
    masked = bind_ml_job(masked_plan, "train", project_dir=project, policy=policy).folds[0].dataset

    kept = list(range(0, 10)) + list(range(30, N_POSITIONS))
    assert list(masked.plan.coordinates) == kept
    assert masked.plan.dataset.input_schema.n_positions == len(kept)
    for full_batch, masked_batch in zip(
        full.iter_batches("train"), masked.iter_batches("train"), strict=True
    ):
        np.testing.assert_array_equal(masked_batch.values, full_batch.values[:, kept])
        np.testing.assert_array_equal(masked_batch.observed_mask, full_batch.observed_mask[:, kept])


def test_trained_model_sees_only_kept_positions(project: Path) -> None:
    kept = [*range(0, 5), *range(35, N_POSITIONS)]
    runs = run_bound_train_job(
        bind_ml_job(_masked_plan([(0, 5), (35, N_POSITIONS)]), "train", project_dir=project)
    )
    for run in runs:
        transform = run.training.model.transform
        assert list(transform.coordinates) == kept
        positions = {int(name.rsplit("@", 1)[1]) for name in transform.feature_names}
        assert positions == set(kept)  # no signal or indicator column for a masked position
        metrics = {m.name: m.value for m in run.evaluation.metrics if m.modality is None}
        assert metrics["average_precision"] > 0.9


def test_workers_split_blocks_so_each_is_read_once(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F69: sharding by batch put every block on every worker.
    import smftools.machine_learning.data.partition_dataset as reader

    calls = []
    original = reader.materialize

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "materialize", counting)
    # Blocks of two batches, so a block holds batches of different workers.
    probe = bind_ml_job(_plan(), "train", project_dir=project).folds[0].dataset.plan
    # No row cache: this counts store decodes per pass (MLR-06's cache would
    # serve the second pass from memory).
    policy = PartitionReadPolicy(
        batch_size=8, max_block_bytes=probe.bytes_per_row * 8 * 2, row_cache_bytes=0
    )
    dataset = bind_ml_job(_plan(), "train", project_dir=project, policy=policy).folds[0].dataset

    calls.clear()
    alone = [batch.read_ids for batch in dataset.iter_batches("train")]
    reads_alone = len(calls)
    calls.clear()
    shared = [
        batch.read_ids
        for worker in range(3)
        for batch in dataset.iter_batches("train", worker_id=worker, num_workers=3)
    ]

    assert len(calls) == reads_alone
    assert sorted(shared) == sorted(alone)
