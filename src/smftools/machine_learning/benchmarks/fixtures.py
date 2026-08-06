"""Parameterized real partition stores for ML-700.

Generalized from ``tests/integration/machine_learning/test_partition_dataset.py``.
Deliberately writes through :func:`smftools.informatics.partition_store.write_experiment_store`
rather than fabricating arrays, so benchmarks measure genuine Zarr projection and
AnnData conversion costs -- the things the analytic estimator is trying to bound.

Splits are resolved through the real :class:`SplitManifest` machinery on whole
sample groups. Group-disjointness is enforced there, so a benchmark physically
cannot assemble a leaky split to make itself faster.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.molecule_identity import molecule_uid
from smftools.informatics.partition_store import write_experiment_store
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.data.partition_dataset import (
    ExperimentPartitionSource,
    MLPartitionDataPlan,
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

# The physical reference strand written into the store. Design-site var columns
# are looked up as f"{Reference_strand}_{site_context}_site", so this name and
# the var column prefixes must agree.
PHYSICAL_REFERENCE = "ref_top"

# The logical reference carried by the dataset manifest. Decoupled from the
# physical strand on purpose -- partition_dataset prefers obs Reference_strand
# when resolving design columns.
LOGICAL_REFERENCE = "locus"

# Fraction of sample groups assigned to each role. Groups, never rows.
TRAIN_GROUPS = 6
VALIDATION_GROUPS = 2
TEST_GROUPS = 2
TOTAL_GROUPS = TRAIN_GROUPS + VALIDATION_GROUPS + TEST_GROUPS


@dataclass(frozen=True)
class FixtureSpec:
    """One benchmark fixture shape."""

    n_rows: int
    n_positions: int
    n_channels: int
    n_partitions: int = 1
    seed: int = 0
    missing_fraction: float = 0.3

    def __post_init__(self) -> None:
        if self.n_rows < TOTAL_GROUPS:
            raise ValueError(f"n_rows must be at least {TOTAL_GROUPS} to fill every split role")
        if self.n_positions < 1:
            raise ValueError("n_positions must be positive")
        if self.n_channels not in {1, 2}:
            raise ValueError("n_channels must be 1 (deaminase) or 2 (conversion)")
        if self.n_partitions < 1:
            raise ValueError("n_partitions must be positive")
        if self.n_rows % self.n_partitions:
            raise ValueError("n_rows must divide evenly across partitions")

    @property
    def modality(self) -> str:
        return "deaminase" if self.n_channels == 1 else "conversion"

    @property
    def label(self) -> str:
        return (
            f"rows{self.n_rows}_pos{self.n_positions}_ch{self.n_channels}_part{self.n_partitions}"
        )


@dataclass(frozen=True)
class BuiltFixture:
    """A ready partition dataset plus the manifests behind it."""

    spec: FixtureSpec
    dataset: PartitionDataset
    plan: MLPartitionDataPlan
    snapshot: DatasetSnapshotManifest
    split: SplitManifest
    split_counts: Mapping[str, int]


def _site_columns(n_positions: int, n_channels: int) -> dict[str, np.ndarray]:
    positions = np.arange(n_positions)
    if n_channels == 1:
        # Every third position is a C site: dense enough to exercise projection,
        # sparse enough to leave genuine non-design gaps.
        return {f"{PHYSICAL_REFERENCE}_C_site": positions % 3 == 1}
    return {
        f"{PHYSICAL_REFERENCE}_GpC_site": positions % 4 == 1,
        f"{PHYSICAL_REFERENCE}_CpG_site": positions % 4 == 2,
    }


def _layers(
    rng: np.random.Generator,
    n_rows: int,
    n_positions: int,
    n_channels: int,
    missing_fraction: float,
) -> dict[str, np.ndarray]:
    def draw() -> np.ndarray:
        values = rng.integers(0, 2, size=(n_rows, n_positions)).astype(np.float32)
        missing = rng.random(size=(n_rows, n_positions)) < missing_fraction
        values[missing] = np.nan
        return values

    if n_channels == 1:
        return {"C_site_binary": draw()}
    return {"GpC_site_binary": draw(), "CpG_site_binary": draw()}


def _write_partition(
    root: Path,
    *,
    experiment: str,
    modality: str,
    read_ids: Sequence[str],
    sample_ids: Sequence[str],
    n_positions: int,
    n_channels: int,
    rng: np.random.Generator,
    missing_fraction: float,
) -> Path:
    n_rows = len(read_ids)
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical([PHYSICAL_REFERENCE] * n_rows),
            "Sample": pd.Categorical(list(sample_ids)),
        },
        index=list(read_ids),
    )
    source = ad.AnnData(
        X=np.zeros((n_rows, n_positions), dtype=np.float32),
        obs=obs,
        layers=_layers(rng, n_rows, n_positions, n_channels, missing_fraction),
    )
    source.var_names = [str(position) for position in range(n_positions)]
    for column, values in _site_columns(n_positions, n_channels).items():
        source.var[column] = values

    paths = write_experiment_store(source, root, experiment=experiment, modality=modality)
    spine, _ = safe_read_h5ad(paths["spine"], verbose=False)
    spine.obs["reference_start"] = np.zeros(spine.n_obs, dtype=np.int64)
    spine.obs["reference_end"] = np.full(spine.n_obs, n_positions, dtype=np.int64)
    safe_write_h5ad(spine, paths["spine"], backup=False, verbose=False)
    return paths["spine"]


def _plan_document(modality: str, n_channels: int) -> dict[str, Any]:
    if n_channels == 1:
        channels = [
            {
                "name": "accessibility",
                "biological_role": "accessibility",
                "sources": [
                    {
                        "modality": "deaminase",
                        "stage": "preprocess",
                        "layer": "C_site_binary",
                        "site_context": "C",
                    }
                ],
            }
        ]
    else:
        channels = [
            {
                "name": "accessibility",
                "biological_role": "accessibility",
                "sources": [
                    {
                        "modality": "conversion",
                        "stage": "preprocess",
                        "layer": "GpC_site_binary",
                        "site_context": "GpC",
                    }
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
        ]
    return {
        "schema_version": 1,
        "scope": {"kind": "project"},
        "datasets": {
            "reads": {
                "modalities": [modality],
                # Each fixture holds one modality across N partitions, so the
                # union policy (reserved for multi-modality datasets) does not
                # apply here.
                "channel_policy": "single_modality",
                "channels": channels,
                "labels": {"column": "activity", "classes": {"inactive": 0, "active": 1}},
            }
        },
        "splits": {
            "groups": {
                "strategy": "explicit_groups",
                "group_by": ["sample_id"],
                "train_groups": [f"sample-{index}" for index in range(TRAIN_GROUPS)],
                "validation_groups": [
                    f"sample-{index}"
                    for index in range(TRAIN_GROUPS, TRAIN_GROUPS + VALIDATION_GROUPS)
                ],
                "test_groups": [
                    f"sample-{index}"
                    for index in range(TRAIN_GROUPS + VALIDATION_GROUPS, TOTAL_GROUPS)
                ],
            }
        },
        "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
        "jobs": {
            "train": {"action": "train", "dataset": "reads", "split": "groups", "models": ["nb"]}
        },
    }


def _role_for_group(group_index: int) -> str:
    if group_index < TRAIN_GROUPS:
        return "train"
    if group_index < TRAIN_GROUPS + VALIDATION_GROUPS:
        return "validation"
    return "test"


def build_fixture(
    root: Path,
    spec: FixtureSpec,
    *,
    policy: PartitionReadPolicy | None = None,
) -> BuiltFixture:
    """Write ``spec`` as real partition stores and bind a read plan over them."""
    rng = np.random.default_rng(spec.seed)
    modality = spec.modality
    rows_per_partition = spec.n_rows // spec.n_partitions

    experiment_uids = [
        str(uuid.UUID(int=(spec.seed << 16) + index)) for index in range(spec.n_partitions)
    ]

    spines: list[Path] = []
    observations: list[DatasetObservation] = []
    sources: list[ExperimentSource] = []
    partition_sources: list[ExperimentPartitionSource] = []

    for partition, experiment_uid in enumerate(experiment_uids):
        read_ids = [f"p{partition}-read-{index:07d}" for index in range(rows_per_partition)]
        # Group assignment is round-robin over sample groups so every partition
        # contributes to every role; the role itself is a property of the group.
        group_indices = [
            (partition * rows_per_partition + index) % TOTAL_GROUPS
            for index in range(rows_per_partition)
        ]
        sample_ids = [f"sample-{group}" for group in group_indices]

        spine = _write_partition(
            root / f"partition-{partition}",
            experiment=f"{modality}-experiment-{partition}",
            modality=modality,
            read_ids=read_ids,
            sample_ids=sample_ids,
            n_positions=spec.n_positions,
            n_channels=spec.n_channels,
            rng=rng,
            missing_fraction=spec.missing_fraction,
        )
        spines.append(spine)

        sources.append(
            ExperimentSource(
                experiment_id=f"{modality}-experiment-{partition}",
                experiment_uid=experiment_uid,
                modality=modality,
                stage="preprocess",
                stage_generation_id="generation-1",
                membership_fingerprint=f"{modality}-{partition}-members",
                feature_fingerprint=f"{modality}-{partition}-features",
                artifacts=(
                    SourceArtifactReference(
                        artifact_id=f"{modality}-{partition}-spine",
                        kind="spine",
                        relative_path="preprocess_outputs/spine.h5ad",
                        sha256=f"{partition:064d}",
                    ),
                ),
            )
        )
        partition_sources.append(
            ExperimentPartitionSource(
                experiment_uid=experiment_uid,
                modality=modality,
                stage_spines={"preprocess": spine},
            )
        )
        observations.extend(
            DatasetObservation(
                molecule_uid=molecule_uid(experiment_uid, read_id),
                experiment_uid=experiment_uid,
                read_id=read_id,
                sample_id=sample_id,
                reference=LOGICAL_REFERENCE,
                modality=modality,
                # Label is a deterministic function of the group, never of the
                # split role, so no benchmark can shortcut prediction.
                class_id=group % 2,
                group_values={},
            )
            for read_id, sample_id, group in zip(read_ids, sample_ids, group_indices, strict=True)
        )

    plan_document = parse_ml_plan(_plan_document(modality, spec.n_channels))
    dataset_spec = plan_document.datasets["reads"]

    snapshot = DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="benchmark",
            set_name=None,
            dataset_name="reads",
            plan_hash=plan_document.plan_hash,
            samples=tuple(f"sample-{index}" for index in range(TOTAL_GROUPS)),
            references=(LOGICAL_REFERENCE,),
            intervals=(GenomicInterval(LOGICAL_REFERENCE, 0, spec.n_positions),),
            filters={},
        ),
        input_schema=InputSchema.from_dataset(
            dataset_spec,
            reference=LOGICAL_REFERENCE,
            n_positions=spec.n_positions,
        ),
        label_schema=LabelSchema.from_plan_label(dataset_spec.labels),
        sources=sources,
        observations=observations,
    )

    assignments = {
        observation.molecule_uid: _role_for_group(int(str(observation.sample_id).rsplit("-", 1)[1]))
        for observation in observations
    }
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("sample_id",),
        assignments=assignments,
    )

    read_plan = build_partition_data_plan(
        snapshot,
        split,
        partition_sources,
        policy=policy or PartitionReadPolicy(),
    )

    counts: dict[str, int] = {}
    for role in assignments.values():
        counts[role] = counts.get(role, 0) + 1

    return BuiltFixture(
        spec=spec,
        dataset=PartitionDataset(read_plan),
        plan=read_plan,
        snapshot=snapshot,
        split=split,
        split_counts=counts,
    )
