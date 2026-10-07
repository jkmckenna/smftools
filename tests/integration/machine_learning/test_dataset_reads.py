"""MLX-10: whole-dataset reads, and channels on an HMM stage in generation layout."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.generation import publish_canonical_spine, staged_generation
from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_read import relative_uns_path
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.orchestration import bind_ml_dataset
from smftools.machine_learning.plan import parse_ml_plan
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

pytestmark = pytest.mark.integration

N_POSITIONS = 30
READS = 12
EXPERIMENTS = ("exp_a", "exp_b")
# C sites at every other position, so site-restricted and every-position
# reads differ (`RPG-01`).
C_SITES = np.arange(N_POSITIONS) % 2 == 0


def _store_source(read_ids, barcodes, x, layers=None) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["chr1+"] * len(read_ids)),
            "Sample": pd.Categorical(barcodes),
        },
        index=read_ids,
    )
    source = ad.AnnData(X=x, obs=obs, layers=layers or {})
    source.var_names = [str(position) for position in range(N_POSITIONS)]
    source.var["chr1+_C_site"] = C_SITES
    return source


def _write_experiment(root: Path, experiment_id: str, rng: np.random.Generator) -> dict:
    run_root = root / experiment_id
    read_ids = [f"{experiment_id}_{barcode}_{i}" for barcode in ("b1", "b2") for i in range(READS)]
    barcodes = ["barcode01"] * READS + ["barcode02"] * READS
    calls = (rng.random((len(read_ids), N_POSITIONS)) < 0.5).astype(np.float32)
    footprint = (rng.random((len(read_ids), N_POSITIONS)) < 0.3).astype(np.float32)

    preprocess = run_root / "preprocess_adata_outputs"
    pre_paths = write_experiment_store(
        _store_source(read_ids, barcodes, calls),
        preprocess,
        experiment=experiment_id,
        modality="deaminase",
    )
    experiment_uid = str(uuid4())
    uids = [molecule_uid(experiment_uid, read_id) for read_id in read_ids]
    partition = {
        "molecule_uid": uids,
        "group_path": [f"store/{b}" for b in barcodes],
        "group_row": list(range(READS)) * 2,
    }
    (preprocess / "read_index").mkdir(parents=True)
    pd.DataFrame(partition).to_parquet(preprocess / "read_index" / "part.parquet", index=False)
    pd.DataFrame(
        {"task_id": ["t"], "reference": ["chr1+"], "layers": [[]], "has_x": [True]}
    ).to_parquet(preprocess / "catalog.parquet", index=False)
    # QC flags live only in preprocess obs; one read per barcode fails.
    pd.DataFrame(
        {"read_id": read_ids, "passes_qc": [i % READS != 0 for i in range(len(read_ids))]}
    ).to_parquet(preprocess / "stage_obs.parquet", index=False)

    # The HMM stage as the pipeline publishes it: everything inside a
    # generation, a canonical spine at the stage root.
    hmm = run_root / "hmm_adata_outputs"
    with staged_generation(hmm, run_root=run_root) as staged:
        staged.record_manifest({"kind": "hmm"})
    generation = staged.final_dir
    hmm_paths = write_experiment_store(
        _store_source(
            read_ids, barcodes, np.zeros_like(calls), layers={"C_all_footprint_features": footprint}
        ),
        generation,
        experiment=experiment_id,
        modality="deaminase",
    )
    (generation / "read_index").mkdir()
    pd.DataFrame(partition).to_parquet(generation / "read_index" / "part.parquet", index=False)
    pd.DataFrame(
        {
            "task_id": ["t"],
            "reference": ["chr1+"],
            "layers": [["C_all_footprint_features"]],
            "has_x": [True],
        }
    ).to_parquet(generation / "catalog.parquet", index=False)
    generation_spine = Path(hmm_paths["spine"])
    # As the pipeline's canonical spine: partitions resolve in the generation.
    spine, _ = safe_read_h5ad(generation_spine, verbose=False)
    spine.uns["source_base_dir"] = relative_uns_path(generation, run_root)
    safe_write_h5ad(spine, generation_spine, backup=False, verbose=False)
    publish_canonical_spine(generation_spine, hmm / "spine.h5ad")
    assert not (hmm / "read_index").exists()  # only inside the generation

    (run_root / "molecule_index").mkdir()
    pd.DataFrame(
        {
            "molecule_uid": uids,
            "experiment_uid": experiment_uid,
            "read_id": read_ids,
            "Reference_strand": "chr1+",
            "Sample": barcodes,
            "Barcode": barcodes,
            "reference_start": 0,
            "reference_end": N_POSITIONS,
        }
    ).to_parquet(run_root / "molecule_index" / "part.parquet", index=False)
    raw = run_root / "raw_outputs"
    raw.mkdir()
    (raw / "spine.h5ad").touch()
    pd.DataFrame({"reference": ["chr1+"], "max_end": [N_POSITIONS]}).to_parquet(
        raw / "interval_catalog.parquet", index=False
    )
    return {
        "entry": {
            "path": str(run_root),
            "name": experiment_id,
            "experiment_uid": experiment_uid,
            "modality": "deaminase",
            "schema_version": 1,
            "spines": {
                "raw": str(raw / "spine.h5ad"),
                "preprocess": str(pre_paths["spine"]),
                "hmm": str(hmm / "spine.h5ad"),
            },
            "references": {"chr1+": "uid"},
            "n_reads": len(read_ids),
            "status": "active",
            "catalogs": {
                "interval_catalog.parquet": str(raw / "interval_catalog.parquet"),
                "molecule_index": str(run_root / "molecule_index"),
                "preprocess_read_index": str(preprocess / "read_index"),
            },
        },
        "calls": dict(zip(read_ids, calls, strict=True)),
        "footprint": dict(zip(read_ids, footprint, strict=True)),
    }


@pytest.fixture
def project(tmp_path: Path):
    rng = np.random.default_rng(0)
    written = {name: _write_experiment(tmp_path / "runs", name, rng) for name in EXPERIMENTS}
    root = tmp_path / "project"
    init_project(root)
    registry = load_registry(root)
    registry["experiments"] = {name: item["entry"] for name, item in written.items()}
    save_registry(root, registry)
    ReferenceRegistry(canonical_names={"uid": "locus"}).save(root / "reference_registry.yaml")
    calls = {k: v for item in written.values() for k, v in item["calls"].items()}
    footprint = {k: v for item in written.values() for k, v in item["footprint"].items()}
    return root, calls, footprint


def _plan(footprint_context: str = "C", calls_context: str = "C"):
    def channel(name, stage, layer, context="C"):
        return {
            "name": name,
            "biological_role": "accessibility",
            "sources": [
                {"modality": "deaminase", "stage": stage, "layer": layer, "site_context": context}
            ],
        }

    # Datasets only: no split, model or job.
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase"],
                    "references": ["locus"],
                    "channels": [
                        channel("C", "preprocess", "X", calls_context),
                        channel("footprint", "hmm", "C_all_footprint_features", footprint_context),
                    ],
                }
            },
            "splits": {},
            "models": {},
            "jobs": {},
        }
    )


def test_hmm_channel_reads_through_its_generation(project) -> None:
    root, calls, footprint = project
    bound = bind_ml_dataset(_plan(), "reads", project_dir=root, group_by=["Barcode"])

    seen = []
    for batch in bound.iter_batches():
        for row, read_id in enumerate(batch.read_ids):
            seen.append(read_id)
            np.testing.assert_array_equal(batch.values[row, :, 0], calls[read_id])
            np.testing.assert_array_equal(batch.values[row, :, 1], footprint[read_id])
    assert sorted(seen) == sorted(calls)
    assert len(seen) == len(set(seen))
    assert set(bound.identity["Barcode"]) == {"barcode01", "barcode02"}


def test_whole_dataset_materializes_in_manifest_order(project) -> None:
    root, calls, _ = project
    bound = bind_ml_dataset(_plan(), "reads", project_dir=root)
    data = bound.materialize()
    assert list(data.molecule_uids) == [item.molecule_uid for item in bound.snapshot.observations]
    assert len(data.molecule_uids) == len(calls)


def test_registry_records_the_hmm_read_index_in_its_generation(project) -> None:
    from smftools.project.registry import _discover_catalogs

    root, _, _ = project
    entry = load_registry(root)["experiments"]["exp_a"]
    spines = {stage: Path(path) for stage, path in entry["spines"].items()}
    catalogs = _discover_catalogs(spines, root)
    hmm_index = catalogs["hmm_read_index"]
    assert "/generations/" in hmm_index and hmm_index.endswith("read_index")


def test_a_plan_may_declare_datasets_only() -> None:
    plan = _plan()
    assert not plan.splits and not plan.models and not plan.jobs
    assert parse_ml_plan(plan.to_dict()).plan_hash == plan.plan_hash


def test_whole_dataset_reads_split_across_workers(project) -> None:
    root, calls, _ = project
    bound = bind_ml_dataset(_plan(), "reads", project_dir=root)
    shards = [
        read_id
        for worker in range(2)
        for batch in bound.iter_batches(worker_id=worker, num_workers=2)
        for read_id in batch.read_ids
    ]
    assert sorted(shards) == sorted(calls)


def test_every_position_channel_observes_the_dense_layer(project) -> None:
    """`RPG-01`: site_context "all" observes every covered position of an HMM layer."""
    root, _, footprint = project
    dense = bind_ml_dataset(_plan(footprint_context="all"), "reads", project_dir=root)
    sites = bind_ml_dataset(_plan(), "reads", project_dir=root)
    for every, at_sites in zip(dense.iter_batches(), sites.iter_batches(), strict=True):
        assert list(every.molecule_uids) == list(at_sites.molecule_uids)
        np.testing.assert_array_equal(every.values, at_sites.values)
        for row, read_id in enumerate(every.read_ids):
            covered = np.isfinite(footprint[read_id])
            np.testing.assert_array_equal(every.observed_mask[row, :, 1], covered)
            np.testing.assert_array_equal(at_sites.observed_mask[row, :, 1], covered & C_SITES)
            # The site-call channel is unchanged.
            np.testing.assert_array_equal(
                every.observed_mask[row, :, 0], at_sites.observed_mask[row, :, 0]
            )


def test_every_position_is_refused_for_site_calls(project) -> None:
    from smftools.machine_learning.selection import MLSelectionError

    root, _, _ = project
    with pytest.raises(MLSelectionError, match="holds site calls"):
        bind_ml_dataset(_plan(calls_context="all"), "reads", project_dir=root)


def test_qc_filters_apply_to_a_dataset_reading_only_derived_stages(project) -> None:
    """`F73`: QC flags come from preprocess obs even when no channel reads preprocess."""
    root, calls, _ = project
    document = _plan(footprint_context="all").to_dict()
    dataset = document["datasets"]["reads"]
    dataset["channels"] = [c for c in dataset["channels"] if c["name"] == "footprint"]
    dataset["filters"] = {"passes_qc": True}
    bound = bind_ml_dataset(parse_ml_plan(document), "reads", project_dir=root)
    kept = set(bound.identity["read_id"])
    failing = {read_id for read_id in calls if read_id.endswith("_0")}
    assert kept == set(calls) - failing and failing


def test_a_small_dataset_still_reaches_every_worker(project) -> None:
    """`F74`: a dataset that fits one memory-sized block is split across workers."""
    from smftools.machine_learning.data.partition_dataset import PartitionReadPolicy

    root, calls, _ = project
    bound = bind_ml_dataset(
        _plan(), "reads", project_dir=root, policy=PartitionReadPolicy(batch_size=4)
    )
    single = [batch.read_ids for batch in bound.iter_batches()]
    per_worker = [
        [batch.read_ids for batch in bound.iter_batches(worker_id=worker, num_workers=4)]
        for worker in range(4)
    ]
    assert all(per_worker), "every worker gets at least one block"
    # Same batches, only distributed: blocks stay whole batches.
    assert sorted(b for shard in per_worker for b in shard) == sorted(single)
    assert sorted(r for batch in single for r in batch) == sorted(calls)
