from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    SourceArtifactReference,
)
from smftools.machine_learning.plan import parse_ml_plan
from smftools.machine_learning.selection import MLDataSelectionPlan
from smftools.machine_learning.splitting import (
    MLSplitPlanningError,
    plan_ml_splits,
)

pytestmark = pytest.mark.unit

EXP_A = "12345678-1234-5678-1234-567812345678"


def _plan(
    split: dict,
    *,
    modalities: tuple[str, ...] = ("deaminase",),
    scope: str = "project",
):
    dataset: dict = {
        "modalities": list(modalities),
        "labels": {
            "column": "activity",
            "classes": {"inactive": 0, "active": 1},
        },
    }
    if len(modalities) > 1:
        dataset.update(
            {
                "channel_policy": "union",
                "channels": [
                    {
                        "name": "accessibility",
                        "biological_role": "accessibility",
                        "sources": [
                            {
                                "modality": modality,
                                "stage": "preprocess",
                                "layer": (
                                    "C_site_binary"
                                    if modality == "deaminase"
                                    else "GpC_site_binary"
                                ),
                                "site_context": "C" if modality == "deaminase" else "GpC",
                            }
                            for modality in modalities
                        ],
                    }
                ],
            }
        )
    return parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": scope},
            "datasets": {"reads": dataset},
            "splits": {"groups": split},
            "models": {"baseline": {"backend": "sklearn", "family": "bernoulli_nb"}},
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["baseline"],
                }
            },
        }
    )


def _identity(
    *,
    n_groups: int = 6,
    classes_by_group: dict[int, tuple[int, ...]] | None = None,
    modality_by_class: dict[int, str] | None = None,
) -> pd.DataFrame:
    rows = []
    classes_by_group = classes_by_group or {index: (0, 1, 0, 1) for index in range(n_groups)}
    modality_by_class = modality_by_class or {}
    for group_index in range(n_groups):
        sample = f"s{group_index + 1}"
        for read_index, class_id in enumerate(classes_by_group[group_index]):
            read_id = f"{sample}-read-{read_index}"
            rows.append(
                {
                    "molecule_uid": molecule_uid(EXP_A, read_id),
                    "experiment_uid": EXP_A,
                    "read_id": read_id,
                    "experiment_id": "exp-a",
                    "sample_id": sample,
                    "Sample": sample,
                    "reference": "locus",
                    "physical_reference": "chr1+",
                    "modality": modality_by_class.get(class_id, "deaminase"),
                    "class_id": class_id,
                }
            )
    return pd.DataFrame(rows)


def _selection(plan, identity: pd.DataFrame) -> MLDataSelectionPlan:
    return MLDataSelectionPlan(
        schema_version=1,
        selection_id="selection-1",
        dataset_name="reads",
        plan_hash=plan.plan_hash,
        scope_kind=plan.scope.kind,
        scope_id="project",
        set_name=None,
        channel_policy=plan.datasets["reads"].channel_policy,
        channel_names=tuple(channel.name for channel in plan.datasets["reads"].channels),
        group_by=tuple(plan.splits["groups"].group_by),
        sources=(),
        identity_table=identity,
        membership_fingerprint="members",
        feature_fingerprint="features",
        n_observations=len(identity),
        n_features=100,
        estimated_materialization_bytes=len(identity) * 600,
        class_counts=dict(Counter(identity["class_id"].astype(str))),
        modality_counts=dict(Counter(identity["modality"])),
        sample_counts=dict(Counter(identity["sample_id"])),
    )


def _snapshot(plan, identity: pd.DataFrame) -> DatasetSnapshotManifest:
    dataset = plan.datasets["reads"]
    input_schema = InputSchema.from_dataset(
        dataset,
        reference="locus",
        n_positions=100,
    )
    label_schema = LabelSchema.from_plan_label(dataset.labels)
    source = ExperimentSource(
        experiment_id="exp-a",
        experiment_uid=EXP_A,
        modality="deaminase",
        stage="preprocess",
        stage_generation_id="generation-1",
        membership_fingerprint="members",
        feature_fingerprint="features",
        artifacts=(
            SourceArtifactReference(
                artifact_id="molecule-index",
                kind="molecule_index",
                relative_path="molecule_index/part.parquet",
                sha256="a" * 64,
            ),
        ),
    )
    observations = tuple(
        DatasetObservation(
            molecule_uid=row.molecule_uid,
            experiment_uid=row.experiment_uid,
            read_id=row.read_id,
            sample_id=row.sample_id,
            reference=row.reference,
            modality=row.modality,
            class_id=int(row.class_id),
            group_values={"Sample": row.Sample},
        )
        for row in identity.itertuples(index=False)
    )
    return DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="project",
            set_name=None,
            dataset_name="reads",
            plan_hash=plan.plan_hash,
            samples=tuple(sorted(set(identity["sample_id"]))),
            references=("locus",),
            intervals=(),
            filters={},
        ),
        input_schema=input_schema,
        label_schema=label_schema,
        sources=(source,),
        observations=observations,
    )


def test_explicit_groups_preserve_exact_assignments_and_build_manifest() -> None:
    plan = _plan(
        {
            "strategy": "explicit_groups",
            "group_by": ["experiment_uid", "Sample"],
            "train_groups": ["exp-a/s1"],
            "validation_groups": ["exp-a/s2"],
            "test_groups": ["exp-a/s3"],
        }
    )
    identity = _identity(n_groups=3)
    selection = _selection(plan, identity)

    resolution = plan_ml_splits(plan, "groups", selection)[0]

    roles_by_sample = {
        sample: {resolution.assignments[molecule_uid] for molecule_uid in rows["molecule_uid"]}
        for sample, rows in identity.groupby("Sample")
    }
    assert roles_by_sample == {
        "s1": {"train"},
        "s2": {"validation"},
        "s3": {"test"},
    }
    assert resolution.locked_roles == ("test", "validation")
    manifest = resolution.to_manifest(_snapshot(plan, identity))
    manifest.validate_against(_snapshot(plan, identity))
    snapshot = _snapshot(plan, identity)
    changed_observations = (
        replace(snapshot.observations[0], class_id=1 - snapshot.observations[0].class_id),
    ) + snapshot.observations[1:]
    changed_snapshot = DatasetSnapshotManifest.create(
        selection=snapshot.selection,
        input_schema=snapshot.input_schema,
        label_schema=snapshot.label_schema,
        sources=snapshot.sources,
        observations=changed_observations,
    )
    with pytest.raises(MLSplitPlanningError, match="labels, modalities, or grouping"):
        resolution.to_manifest(changed_snapshot)


@pytest.mark.parametrize("scope", ["experiment", "project"])
def test_seeded_stratified_groups_are_reproducible_and_disjoint(
    scope: str,
) -> None:
    plan = _plan(
        {
            "strategy": "stratified_group",
            "group_by": ["Sample"],
            "fractions": {"train": 0.5, "validation": 0.25, "test": 0.25},
            "seed": 17,
        },
        scope=scope,
    )
    identity = _identity(n_groups=8)
    selection = _selection(plan, identity)

    first = plan_ml_splits(plan, "groups", selection)[0]
    second = plan_ml_splits(plan, "groups", selection)[0]

    assert first.assignments == second.assignments
    assert first.group_assignments == second.group_assignments
    assert {summary.split for summary in first.summaries} == {
        "train",
        "validation",
        "test",
    }
    assert {summary.split: summary.n_groups for summary in first.summaries} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert all(set(summary.counts_by_class) == {0, 1} for summary in first.summaries)
    for _, rows in identity.groupby("Sample"):
        assert len({first.assignments[item] for item in rows["molecule_uid"]}) == 1


def test_impossible_stratification_fails_instead_of_dropping_classes() -> None:
    plan = _plan(
        {
            "strategy": "stratified_group",
            "group_by": ["Sample"],
            "seed": 2,
        }
    )
    classes = {
        0: (0, 1),
        1: (0, 1),
        2: (0, 0),
        3: (0, 0),
    }
    selection = _selection(plan, _identity(n_groups=4, classes_by_group=classes))

    with pytest.raises(MLSplitPlanningError, match="one biological group per split"):
        plan_ml_splits(plan, "groups", selection)


def test_explicit_split_rejects_a_role_without_all_classes() -> None:
    plan = _plan(
        {
            "strategy": "explicit_groups",
            "group_by": ["Sample"],
            "train_groups": ["s1"],
            "validation_groups": ["s2"],
            "test_groups": ["s3"],
        }
    )
    classes = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 1),
    }

    with pytest.raises(MLSplitPlanningError, match="every split role"):
        plan_ml_splits(
            plan,
            "groups",
            _selection(plan, _identity(n_groups=3, classes_by_group=classes)),
        )


def test_absent_class_by_modality_cells_are_explicitly_reported() -> None:
    plan = _plan(
        {
            "strategy": "stratified_group",
            "group_by": ["Sample"],
            "seed": 5,
        },
        modalities=("deaminase", "conversion"),
    )
    identity = _identity(
        n_groups=6,
        modality_by_class={0: "deaminase", 1: "conversion"},
    )

    resolution = plan_ml_splits(plan, "groups", _selection(plan, identity))[0]

    assert any("dataset has absent class-by-modality cells" in item for item in resolution.warnings)
    assert any(cell.n_observations == 0 for cell in resolution.class_by_modality)
    assert {
        (cell.class_id, cell.modality)
        for cell in resolution.class_by_modality
        if cell.split == "train"
    } == {
        (0, "conversion"),
        (0, "deaminase"),
        (1, "conversion"),
        (1, "deaminase"),
    }


def test_leave_one_group_out_produces_one_locked_test_fold_per_group() -> None:
    plan = _plan(
        {
            "strategy": "leave_one_group_out",
            "group_by": ["Sample"],
            "seed": 11,
        }
    )
    identity = _identity(n_groups=3)

    folds = plan_ml_splits(plan, "groups", _selection(plan, identity))

    assert len(folds) == 3
    assert [fold.fold_name for fold in folds] == [
        "holdout=s1",
        "holdout=s2",
        "holdout=s3",
    ]
    assert all(fold.locked_roles == ("test",) for fold in folds)
    assert all({summary.split for summary in fold.summaries} == {"train", "test"} for fold in folds)
