from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import (
    ExperimentPartitionSource,
    PartitionDataset,
    PartitionReadPolicy,
    build_partition_data_plan,
)
from smftools.machine_learning.manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    GenomicInterval,
    SourceArtifactReference,
    SplitManifest,
)
from smftools.machine_learning.plan import parse_ml_plan
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

pytestmark = pytest.mark.integration

DEAMINASE_UID = "12345678-1234-5678-1234-567812345678"
CONVERSION_UID = "87654321-4321-8765-4321-876543218765"


def _write_source(
    root: Path,
    *,
    experiment_uid: str,
    modality: str,
) -> tuple[Path, tuple[str, ...]]:
    n_positions = 6 if modality == "deaminase" else 4
    read_ids = (f"{modality}-read-a", f"{modality}-read-b")
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(["ref_top", "ref_top"]),
            "Sample": pd.Categorical(["sample-a", "sample-b"]),
        },
        index=read_ids,
    )
    if modality == "deaminase":
        layers = {
            "C_site_binary": np.asarray(
                [
                    [np.nan, 1.0, 0.0, np.nan, 1.0, 0.0],
                    [np.nan, 0.0, 1.0, np.nan, np.nan, 1.0],
                ],
                dtype=np.float32,
            )
        }
    else:
        layers = {
            "GpC_site_binary": np.asarray(
                [[np.nan, 1.0, np.nan, 0.0], [np.nan, 0.0, np.nan, np.nan]],
                dtype=np.float32,
            ),
            "CpG_site_binary": np.asarray(
                [[np.nan, np.nan, 1.0, np.nan], [np.nan, np.nan, 0.0, np.nan]],
                dtype=np.float32,
            ),
        }
    source = ad.AnnData(
        X=np.zeros((2, n_positions), dtype=np.float32),
        obs=obs,
        layers=layers,
    )
    source.var_names = [str(position) for position in range(n_positions)]
    source.var["ref_top_C_site"] = [position in {1, 2, 4, 5} for position in range(n_positions)]
    source.var["ref_top_GpC_site"] = [position in {1, 3} for position in range(n_positions)]
    source.var["ref_top_CpG_site"] = [position in {2, 4} for position in range(n_positions)]
    paths = write_experiment_store(
        source,
        root,
        experiment=f"{modality}-experiment",
        modality=modality,
    )
    spine, _ = safe_read_h5ad(paths["spine"], verbose=False)
    spine.obs["reference_start"] = (
        np.asarray([0, 5], dtype=np.int64)
        if modality == "deaminase"
        else np.zeros(spine.n_obs, dtype=np.int64)
    )
    spine.obs["reference_end"] = (
        np.asarray([n_positions, 6], dtype=np.int64)
        if modality == "deaminase"
        else np.full(spine.n_obs, n_positions, dtype=np.int64)
    )
    safe_write_h5ad(spine, paths["spine"], backup=False, verbose=False)
    return paths["spine"], read_ids


def _dataset(tmp_path: Path) -> PartitionDataset:
    deaminase_spine, deaminase_reads = _write_source(
        tmp_path / "deaminase",
        experiment_uid=DEAMINASE_UID,
        modality="deaminase",
    )
    conversion_spine, conversion_reads = _write_source(
        tmp_path / "conversion",
        experiment_uid=CONVERSION_UID,
        modality="conversion",
    )
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {
                "reads": {
                    "modalities": ["deaminase", "conversion"],
                    "channel_policy": "union",
                    "channels": [
                        {
                            "name": "accessibility",
                            "biological_role": "accessibility",
                            "sources": [
                                {
                                    "modality": "deaminase",
                                    "stage": "preprocess",
                                    "layer": "C_site_binary",
                                    "site_context": "C",
                                },
                                {
                                    "modality": "conversion",
                                    "stage": "preprocess",
                                    "layer": "GpC_site_binary",
                                    "site_context": "GpC",
                                },
                            ],
                        },
                        {
                            "name": "endogenous_methylation",
                            "biological_role": "endogenous_methylation",
                            "sources": [
                                {
                                    "modality": "conversion",
                                    "stage": "preprocess",
                                    "layer": "CpG_site_binary",
                                    "site_context": "CpG",
                                }
                            ],
                        },
                    ],
                    "labels": {
                        "column": "activity",
                        "classes": {"inactive": 0, "active": 1},
                    },
                }
            },
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["sample-a", "sample-b"],
                    "validation_groups": ["sample-c"],
                    "test_groups": ["sample-d"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                }
            },
        }
    )
    dataset_spec = plan.datasets["reads"]
    experiment_records = (
        ("deaminase-experiment", DEAMINASE_UID, "deaminase", deaminase_reads),
        ("conversion-experiment", CONVERSION_UID, "conversion", conversion_reads),
    )
    observations = []
    sources = []
    for experiment_id, experiment_uid, modality, read_ids in experiment_records:
        sources.append(
            ExperimentSource(
                experiment_id=experiment_id,
                experiment_uid=experiment_uid,
                modality=modality,
                stage="preprocess",
                stage_generation_id="generation-1",
                membership_fingerprint=f"{modality}-members",
                feature_fingerprint=f"{modality}-features",
                artifacts=(
                    SourceArtifactReference(
                        artifact_id=f"{modality}-spine",
                        kind="spine",
                        relative_path="preprocess_outputs/spine.h5ad",
                        sha256=("a" if modality == "deaminase" else "b") * 64,
                    ),
                ),
            )
        )
        observations.extend(
            DatasetObservation(
                molecule_uid=molecule_uid(experiment_uid, read_id),
                experiment_uid=experiment_uid,
                read_id=read_id,
                sample_id=f"sample-{'a' if index == 0 else 'b'}",
                reference="locus",
                modality=modality,
                class_id=index,
                group_values={},
            )
            for index, read_id in enumerate(read_ids)
        )
    snapshot = DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="project",
            set_name=None,
            dataset_name="reads",
            plan_hash=plan.plan_hash,
            samples=("sample-a", "sample-b"),
            references=("locus",),
            intervals=(GenomicInterval("locus", 1, 5),),
            filters={},
        ),
        input_schema=InputSchema.from_dataset(
            dataset_spec,
            reference="locus",
            n_positions=4,
        ),
        label_schema=LabelSchema.from_plan_label(dataset_spec.labels),
        sources=sources,
        observations=observations,
    )
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("sample_id",),
        assignments={item.molecule_uid: "train" for item in observations},
    )
    read_plan = build_partition_data_plan(
        snapshot,
        split,
        [
            ExperimentPartitionSource(
                experiment_uid=DEAMINASE_UID,
                modality="deaminase",
                stage_spines={"preprocess": deaminase_spine},
            ),
            ExperimentPartitionSource(
                experiment_uid=CONVERSION_UID,
                modality="conversion",
                stage_spines={"preprocess": conversion_spine},
            ),
        ],
        policy=PartitionReadPolicy(batch_size=2),
    )
    return PartitionDataset(read_plan)


def test_partition_batches_align_channels_labels_coordinates_and_masks(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    batches = tuple(dataset.iter_batches("train"))

    assert len(batches) == 2
    assert all(batch.values.shape == (2, 4, 2) for batch in batches)
    assert all(
        batch.channel_names == ("accessibility", "endogenous_methylation") for batch in batches
    )
    by_read = {
        read_id: (batch, row) for batch in batches for row, read_id in enumerate(batch.read_ids)
    }

    deaminase, row = by_read["deaminase-read-a"]
    np.testing.assert_array_equal(deaminase.coordinates, [1, 2, 3, 4])
    np.testing.assert_allclose(
        deaminase.values[row, :, 0],
        [1.0, 0.0, np.nan, 1.0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(deaminase.availability_mask[row], [True, False])
    np.testing.assert_array_equal(deaminase.design_mask[row, :, 0], [True, True, False, True])
    np.testing.assert_array_equal(deaminase.observed_mask[row, :, 0], [True, True, False, True])
    np.testing.assert_array_equal(deaminase.padding_mask[row], [False, False, False, False])

    outside_interval, row = by_read["deaminase-read-b"]
    assert np.isnan(outside_interval.values[row]).all()
    assert not outside_interval.observed_mask[row].any()
    assert outside_interval.padding_mask[row].all()

    conversion, row = by_read["conversion-read-a"]
    np.testing.assert_array_equal(conversion.availability_mask[row], [True, True])
    np.testing.assert_array_equal(conversion.design_mask[row, :, 0], [True, False, True, False])
    np.testing.assert_array_equal(conversion.design_mask[row, :, 1], [False, True, False, False])
    np.testing.assert_array_equal(conversion.padding_mask[row], [False, False, False, True])
    assert np.isnan(conversion.values[row, 3]).all()


def test_worker_shards_are_deterministic_without_duplicates_or_omissions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    first = tuple(dataset.iter_batches("train", worker_id=0, num_workers=2))
    second = tuple(dataset.iter_batches("train", worker_id=1, num_workers=2))
    repeated = tuple(dataset.iter_batches("train", worker_id=0, num_workers=2))
    combined = [uid for batch in (*first, *second) for uid in batch.molecule_uids]

    assert [batch.molecule_uids for batch in first] == [batch.molecule_uids for batch in repeated]
    assert len(combined) == len(set(combined)) == 4
    assert set(combined) == {entry.molecule_uid for entry in dataset.plan.entries}


def test_bounded_materializer_preserves_manifest_order_and_sklearn_view(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    materialized = dataset.materialize("train")

    assert materialized.molecule_uids == tuple(
        entry.molecule_uid for entry in dataset.plan.entries_for("train")
    )
    assert materialized.values.shape == (4, 4, 2)
    assert materialized.X.shape == (4, 8)
    np.testing.assert_array_equal(
        materialized.y,
        [entry.class_id for entry in dataset.plan.entries_for("train")],
    )
