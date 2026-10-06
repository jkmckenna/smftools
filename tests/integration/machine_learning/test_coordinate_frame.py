"""MLX-03: molecules of a deletion allele placed in the intact allele's coordinates."""

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
from smftools.machine_learning.selection import MLSelectionError, plan_ml_dataset
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry

pytestmark = pytest.mark.integration

FRAME_LENGTH = 40
DELETED = (10, 20)  # frame positions the deletion allele lacks
DELETION_LENGTH = FRAME_LENGTH - (DELETED[1] - DELETED[0])
READS = 24
EXPERIMENTS = ("exp_a", "exp_b", "exp_c")


def _frame_position(source: int) -> int:
    return source if source < DELETED[0] else source + (DELETED[1] - DELETED[0])


def _write_experiment(root: Path, experiment_id: str, rng: np.random.Generator) -> dict:
    run_root = root / experiment_id
    preprocess = run_root / "preprocess_adata_outputs"
    read_ids = [
        f"{experiment_id}_{kind}_{index}" for kind in ("intact", "del") for index in range(READS)
    ]
    references = ["chr1+"] * READS + ["del+"] * READS
    calls = np.full((2 * READS, FRAME_LENGTH), np.nan, dtype=np.float32)
    # Intact reads (active): accessible over frame [0, 10).
    calls[:READS, :FRAME_LENGTH] = 0
    calls[:READS, :10] = rng.random((READS, 10)) < 0.8
    # Deletion reads (inactive): own coordinates [0, 30); accessible at source
    # [20, 30), which is frame [30, 40).
    calls[READS:, :DELETION_LENGTH] = 0
    calls[READS:, 20:DELETION_LENGTH] = rng.random((READS, 10)) < 0.8
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(references),
            "Sample": pd.Categorical(["barcode01"] * (2 * READS)),
        },
        index=read_ids,
    )
    source = ad.AnnData(X=calls, obs=obs)
    source.var_names = [str(position) for position in range(FRAME_LENGTH)]
    source.var["chr1+_C_site"] = True
    source.var["del+_C_site"] = np.arange(FRAME_LENGTH) < DELETION_LENGTH
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
            "Reference_strand": references,
            "Sample": "barcode01",
            "Barcode": "barcode01",
            "activity": ["active"] * READS + ["inactive"] * READS,
            "reference_start": 0,
            "reference_end": [FRAME_LENGTH] * READS + [DELETION_LENGTH] * READS,
        }
    ).to_parquet(molecule_index / "part.parquet", index=False)
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "group_path": [f"store/{reference}" for reference in references],
            "group_row": list(range(READS)) * 2,
        }
    ).to_parquet(read_index / "part.parquet", index=False)
    pd.DataFrame(
        {
            "task_id": ["t0", "t1"],
            "reference": ["chr1+", "del+"],
            "layers": [[], []],
            "has_x": [True, True],
        }
    ).to_parquet(preprocess / "catalog.parquet", index=False)
    raw = run_root / "raw_outputs"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "spine.h5ad").touch()
    pd.DataFrame(
        {"reference": ["chr1+", "del+"], "max_end": [FRAME_LENGTH, DELETION_LENGTH]}
    ).to_parquet(raw / "interval_catalog.parquet", index=False)
    return {
        "path": str(run_root),
        "name": experiment_id,
        "experiment_uid": experiment_uid,
        "modality": "deaminase",
        "schema_version": 1,
        "spines": {"raw": str(raw / "spine.h5ad"), "preprocess": str(paths["spine"])},
        "references": {"chr1+": "uid-intact", "del+": "uid-deletion"},
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
    ReferenceRegistry(canonical_names={"uid-intact": "intact", "uid-deletion": "deletion"}).save(
        root / "reference_registry.yaml"
    )
    maps = root / "ml" / "maps"
    maps.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_position": range(DELETION_LENGTH),
            "frame_position": [_frame_position(p) for p in range(DELETION_LENGTH)],
        }
    ).to_parquet(maps / "deletion_to_intact.parquet", index=False)
    return root


def _plan(*, exclude=(DELETED,), references=("intact", "deletion"), maps=None):
    dataset = {
        "modalities": ["deaminase"],
        "references": list(references),
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
        "coordinate_frame": {
            "reference": "intact",
            "maps": maps or {"deletion": "ml/maps/deletion_to_intact.parquet"},
        },
    }
    if exclude:
        dataset["positions"] = {
            "include": [[0, FRAME_LENGTH]],
            "exclude": [list(window) for window in exclude],
        }
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {"reads": dataset},
            "splits": {
                "by_experiment": {"strategy": "leave_one_group_out", "group_by": ["experiment_uid"]}
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


def test_deletion_reads_land_at_their_frame_positions(project: Path) -> None:
    bound = bind_ml_job(
        _plan(), "train", project_dir=project, policy=PartitionReadPolicy(batch_size=8)
    )
    dataset = bound.folds[0].dataset
    coordinates = list(dataset.plan.coordinates)
    assert coordinates == [*range(0, 10), *range(20, FRAME_LENGTH)]
    column = {position: index for index, position in enumerate(coordinates)}
    deletion_frame_signal = [column[p] for p in range(30, FRAME_LENGTH)]
    deletion_frame_quiet = [column[p] for p in range(20, 30)]

    seen = 0
    for batch in dataset.iter_batches("train"):
        for row, read_id in enumerate(batch.read_ids):
            if "_del_" not in read_id:
                continue
            seen += 1
            values = batch.values[row, :, 0]
            # Source [20, 30) carries the signal and must sit at frame [30, 40).
            assert np.nansum(values[deletion_frame_signal]) > 0
            assert np.nansum(values[deletion_frame_quiet]) == 0
            assert not batch.padding_mask[row, deletion_frame_signal].any()
    assert seen > 0


def test_positions_a_reference_lacks_are_refused(project: Path) -> None:
    with pytest.raises(MLSelectionError, match=r"\(10, 20\)\] have no counterpart on 'deletion'"):
        plan_ml_dataset(_plan(exclude=()), "reads", project_dir=project)


def test_references_outside_the_frame_are_refused(project: Path) -> None:
    plan = _plan(maps={"other": "ml/maps/deletion_to_intact.parquet"})
    with pytest.raises(MLSelectionError, match="neither uses as frame nor maps"):
        plan_ml_dataset(plan, "reads", project_dir=project)


def test_selection_identity_includes_the_map(project: Path) -> None:
    first = plan_ml_dataset(_plan(), "reads", project_dir=project)
    path = project / "ml" / "maps" / "deletion_to_intact.parquet"
    table = pd.read_parquet(path)
    table.loc[table["source_position"] == 0, "frame_position"] = 0  # unchanged values,
    table.assign(note="rewritten").to_parquet(path, index=False)  # different file
    second = plan_ml_dataset(_plan(), "reads", project_dir=project)
    assert first.membership_fingerprint == second.membership_fingerprint
    assert first.selection_id != second.selection_id


def test_mixed_reference_job_trains_over_shared_positions(project: Path) -> None:
    runs = run_bound_train_job(bind_ml_job(_plan(), "train", project_dir=project))
    for run in runs:
        assert run.training.model.transform.n_positions == FRAME_LENGTH - (DELETED[1] - DELETED[0])
        metrics = {m.name: m.value for m in run.evaluation.metrics if m.modality is None}
        assert metrics["average_precision"] > 0.9


def test_reads_outside_the_window_are_padding_not_a_design_conflict(project: Path) -> None:
    # F68: a read that does not reach the window is padded there; it must not
    # make the batch's design mask disagree.
    entry = load_registry(project)["experiments"]["exp_a"]
    index_path = Path(entry["catalogs"]["molecule_index"]) / "part.parquet"
    index = pd.read_parquet(index_path)
    number = index["read_id"].str.rsplit("_", n=1).str[1].astype(int)
    short = index["read_id"].str.contains("_intact_") & (number < 4)
    index.loc[short, "reference_end"] = 5
    index.to_parquet(index_path, index=False)
    short_reads = set(index.loc[short, "read_id"])

    plan = _plan().to_dict()
    plan["datasets"]["reads"]["positions"] = {"include": [[20, FRAME_LENGTH]], "exclude": []}
    bound = bind_ml_job(
        parse_ml_plan(plan), "train", project_dir=project, policy=PartitionReadPolicy(batch_size=64)
    )
    seen = 0
    for fold in bound.folds:
        for split in ("train", "test"):
            for batch in fold.dataset.iter_batches(split):
                for row, read_id in enumerate(batch.read_ids):
                    if read_id in short_reads:
                        seen += 1
                        # Ends at 5, window starts at 20: padded throughout.
                        assert batch.padding_mask[row].all()
                        assert not batch.observed_mask[row].any()
    assert seen
