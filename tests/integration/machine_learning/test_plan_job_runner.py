"""MLX-06: a project plan's train job binds to real stores and trains per fold."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.data.partition_dataset import PartitionReadPolicy
from smftools.machine_learning.orchestration import bind_ml_job, run_bound_train_job
from smftools.machine_learning.plan import parse_ml_plan
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry

pytestmark = pytest.mark.integration

N_POSITIONS = 40
READS_PER_BARCODE = 24
EXPERIMENTS = ("exp_a", "exp_b", "exp_c")


def _signal(rng: np.random.Generator, active: bool) -> np.ndarray:
    """Active reads are accessible over the first half, inactive over the second."""
    calls = np.zeros((READS_PER_BARCODE, N_POSITIONS), dtype=np.float32)
    half = N_POSITIONS // 2
    window = slice(0, half) if active else slice(half, N_POSITIONS)
    calls[:, window] = rng.random((READS_PER_BARCODE, half)) < 0.8
    calls[rng.random(calls.shape) < 0.1] = np.nan  # unobserved
    return calls


def _write_experiment(root: Path, experiment_id: str, rng: np.random.Generator) -> dict:
    run_root = root / experiment_id
    preprocess = run_root / "preprocess_adata_outputs"
    read_ids = [
        f"{experiment_id}_{barcode}_{index}"
        for barcode in ("b1", "b2")
        for index in range(READS_PER_BARCODE)
    ]
    barcodes = ["barcode01"] * READS_PER_BARCODE + ["barcode02"] * READS_PER_BARCODE
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["chr1+"] * len(read_ids)),
            "Sample": pd.Categorical(barcodes),
        },
        index=read_ids,
    )
    source = ad.AnnData(
        X=np.vstack([_signal(rng, active=True), _signal(rng, active=False)]), obs=obs
    )
    source.var_names = [str(position) for position in range(N_POSITIONS)]
    source.var["chr1+_C_site"] = True
    paths = write_experiment_store(
        source, preprocess, experiment=experiment_id, modality="deaminase"
    )

    experiment_uid = str(uuid4())
    uids = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    molecule_index = run_root / "molecule_index"
    read_index = preprocess / "read_index"
    for directory in (molecule_index, read_index):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "experiment_uid": experiment_uid,
            "read_id": read_ids,
            "Reference_strand": "chr1+",
            "Sample": barcodes,
            "Barcode": barcodes,
            "activity": ["active"] * READS_PER_BARCODE + ["inactive"] * READS_PER_BARCODE,
            # A raw molecule index records each read's aligned span.
            "reference_start": 0,
            "reference_end": N_POSITIONS,
        }
    ).to_parquet(molecule_index / "part.parquet", index=False)
    # As the pipeline's read index: which store partition holds each read.
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "group_path": [f"store/chr1/{barcode}" for barcode in barcodes],
            "group_row": list(range(READS_PER_BARCODE)) * 2,
        }
    ).to_parquet(read_index / "part.parquet", index=False)
    # The written-store catalog as partitioned preprocess writes it (F66).
    pd.DataFrame(
        {"task_id": ["t0"], "reference": ["chr1+"], "layers": [[]], "has_x": [True]}
    ).to_parquet(preprocess / "catalog.parquet", index=False)
    raw = run_root / "raw_outputs"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "spine.h5ad").touch()
    pd.DataFrame({"reference": ["chr1+"], "max_end": [N_POSITIONS]}).to_parquet(
        raw / "interval_catalog.parquet", index=False
    )
    return {
        "path": str(run_root),
        "name": experiment_id,
        "experiment_uid": experiment_uid,
        "modality": "deaminase",
        "schema_version": 1,
        "spines": {"raw": str(raw / "spine.h5ad"), "preprocess": str(paths["spine"])},
        "references": {"chr1+": "reference-uid"},
        "n_reads": len(read_ids),
        "status": "active",
        "catalogs": {
            "interval_catalog.parquet": str(raw / "interval_catalog.parquet"),
            "molecule_index": str(molecule_index),
            "preprocess_read_index": str(read_index),
        },
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    entries = {name: _write_experiment(tmp_path / "runs", name, rng) for name in EXPERIMENTS}
    root = tmp_path / "project"
    init_project(root)
    registry = load_registry(root)
    registry["experiments"] = entries
    save_registry(root, registry)
    ReferenceRegistry(canonical_names={"reference-uid": "locus"}).save(
        root / "reference_registry.yaml"
    )
    return root


def _plan():
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "channels": [
                        {
                            "name": "accessibility",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "deaminase",
                                    "stage": "preprocess",
                                    "layer": "X",
                                    "site_context": "C",
                                }
                            ],
                        }
                    ],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                        "positive_class": "active",
                    },
                }
            },
            "splits": {
                "by_experiment": {
                    "strategy": "leave_one_group_out",
                    "group_by": ["experiment_uid"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "by_experiment",
                    "models": ["nb"],
                }
            },
        }
    )


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
