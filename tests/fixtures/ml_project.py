"""A small registered project for ML integration tests (`MLX-06`, `MLR-01`).

Three experiments, two barcodes each: barcode01 reads are "active"
(accessible over the first half of the window), barcode02 "inactive" (the
second half); 10 % of calls unobserved. `ml_plan` trains naive Bayes with
leave-one-experiment-out.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.plan import parse_ml_plan
from smftools.project.reference_registry import ReferenceRegistry
from smftools.project.registry import init_project, load_registry, save_registry

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


def make_ml_project(tmp_path: Path) -> Path:
    """The registered project under ``tmp_path / "project"``."""
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


def ml_plan(models: dict | None = None):
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
            "models": models or {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "by_experiment",
                    "models": sorted(models or {"nb": None}),
                }
            },
        }
    )
