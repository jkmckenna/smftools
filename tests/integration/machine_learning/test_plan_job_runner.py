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
    pd.DataFrame({"molecule_uid": uids}).to_parquet(read_index / "part.parquet", index=False)
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
