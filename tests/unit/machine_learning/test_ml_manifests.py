from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from smftools.informatics.molecule_identity import molecule_uid
from smftools.machine_learning.contracts import InputSchema, LabelSchema
from smftools.machine_learning.manifests import (
    DatasetObservation,
    DatasetSelection,
    DatasetSnapshotManifest,
    ExperimentSource,
    GenomicInterval,
    MLManifestError,
    SourceArtifactReference,
    SplitManifest,
    StaleDatasetSourceError,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit

EXP_1 = "12345678-1234-5678-1234-567812345678"
EXP_2 = "87654321-4321-6789-4321-678987654321"


def _dataset_spec(*, modalities: tuple[str, ...] = ("deaminase",)):
    dataset: dict = {
        "modalities": list(modalities),
        "labels": {
            "column": "activity",
            "classes": {"inactive": 0, "active": 1},
            "positive_class": "active",
        },
    }
    if len(modalities) > 1:
        dataset.update(
            {
                "channel_policy": "union",
                "channels": [
                    {
                        "name": "deaminase_accessibility",
                        "biological_role": "accessibility",
                        "sources": [
                            {
                                "modality": "deaminase",
                                "stage": "preprocess",
                                "layer": "C_site_binary",
                                "site_context": "C",
                            }
                        ],
                    },
                    {
                        "name": "direct_accessibility",
                        "biological_role": "accessibility",
                        "sources": [
                            {
                                "modality": "direct",
                                "stage": "preprocess",
                                "layer": "A_site_binary",
                                "site_context": "A",
                            }
                        ],
                    },
                ],
            }
        )
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {"reads": dataset},
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["Sample"],
                    "train_groups": ["s1"],
                    "validation_groups": ["s2"],
                    "test_groups": ["s3"],
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
    return plan.datasets["reads"]


def _schemas(*, modalities: tuple[str, ...] = ("deaminase",)):
    dataset = _dataset_spec(modalities=modalities)
    return (
        InputSchema.from_dataset(dataset, reference="Nkg2a", n_positions=100),
        LabelSchema.from_plan_label(dataset.labels),
    )


def _source(
    experiment_uid: str,
    experiment_id: str,
    *,
    modality: str = "deaminase",
    generation: str = "generation-1",
) -> ExperimentSource:
    return ExperimentSource(
        experiment_id=experiment_id,
        experiment_uid=experiment_uid,
        modality=modality,
        stage="preprocess",
        stage_generation_id=generation,
        membership_fingerprint=f"{experiment_id}-members",
        feature_fingerprint=f"{experiment_id}-features",
        artifacts=(
            SourceArtifactReference(
                artifact_id=f"{experiment_id}-spine",
                kind="spine",
                relative_path="stores/preprocess.spine.parquet",
                sha256="a" * 64,
            ),
        ),
    )


def _observation(
    experiment_uid: str,
    read_id: str,
    sample: str,
    class_id: int,
    *,
    modality: str = "deaminase",
) -> DatasetObservation:
    return DatasetObservation(
        molecule_uid=molecule_uid(experiment_uid, read_id),
        experiment_uid=experiment_uid,
        read_id=read_id,
        sample_id=sample,
        reference="Nkg2a",
        modality=modality,
        class_id=class_id,
        group_values={"Sample": sample, "donor": f"donor-{sample}"},
    )


def _snapshot() -> DatasetSnapshotManifest:
    input_schema, label_schema = _schemas()
    return DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="project-a",
            set_name="training",
            dataset_name="activity",
            plan_hash="1" * 64,
            samples=("s2", "s1"),
            references=("Nkg2a",),
            intervals=(GenomicInterval("Nkg2a", 0, 100),),
            filters={"mapping_quality_min": 20, "flags": ["primary"]},
        ),
        input_schema=input_schema,
        label_schema=label_schema,
        sources=(_source(EXP_2, "exp-2"), _source(EXP_1, "exp-1")),
        observations=(
            _observation(EXP_2, "read-4", "s2", 1),
            _observation(EXP_1, "read-2", "s1", 1),
            _observation(EXP_2, "read-3", "s2", 0),
            _observation(EXP_1, "read-1", "s1", 0),
        ),
    )


def _assignments(snapshot: DatasetSnapshotManifest) -> dict[str, str]:
    return {
        item.molecule_uid: "train" if item.sample_id == "s1" else "validation"
        for item in snapshot.observations
    }


def test_dataset_snapshot_is_order_independent_and_round_trips() -> None:
    snapshot = _snapshot()
    reordered = DatasetSnapshotManifest.create(
        selection=snapshot.selection,
        input_schema=snapshot.input_schema,
        label_schema=snapshot.label_schema,
        sources=tuple(reversed(snapshot.sources)),
        observations=tuple(reversed(snapshot.observations)),
    )

    assert reordered.snapshot_id == snapshot.snapshot_id
    assert reordered.membership_digest == snapshot.membership_digest
    assert DatasetSnapshotManifest.from_dict(snapshot.to_dict()) == snapshot
    assert snapshot.canonical_json() == reordered.canonical_json()


def test_selection_and_membership_have_distinct_identity_effects() -> None:
    snapshot = _snapshot()
    changed_selection = DatasetSnapshotManifest.create(
        selection=replace(snapshot.selection, filters={"mapping_quality_min": 30}),
        input_schema=snapshot.input_schema,
        label_schema=snapshot.label_schema,
        sources=snapshot.sources,
        observations=snapshot.observations,
    )
    changed_observations = DatasetSnapshotManifest.create(
        selection=snapshot.selection,
        input_schema=snapshot.input_schema,
        label_schema=snapshot.label_schema,
        sources=snapshot.sources,
        observations=snapshot.observations[:-1],
    )

    assert changed_selection.membership_digest == snapshot.membership_digest
    assert changed_selection.snapshot_id != snapshot.snapshot_id
    assert changed_observations.membership_digest != snapshot.membership_digest
    assert changed_observations.snapshot_id != snapshot.snapshot_id


def test_snapshot_summary_records_sample_modality_and_class_counts() -> None:
    summary = _snapshot().summary

    assert summary.n_observations == 4
    assert summary.n_experiments == 2
    assert [(item.value, item.count) for item in summary.counts_by_sample] == [
        ("s1", 2),
        ("s2", 2),
    ]
    assert [(item.value, item.count) for item in summary.counts_by_modality] == [("deaminase", 4)]
    assert [(item.value, item.count) for item in summary.counts_by_class] == [
        ("active", 2),
        ("inactive", 2),
    ]


def test_snapshot_rejects_tampered_serialized_identity() -> None:
    raw = _snapshot().to_dict()
    raw["observations"][0]["sample_id"] = "s2"
    raw["observations"][0]["group_values"]["Sample"] = "s2"

    with pytest.raises(MLManifestError, match="membership_digest"):
        DatasetSnapshotManifest.from_dict(raw)


def test_source_change_is_detected_without_opening_artifacts() -> None:
    snapshot = _snapshot()
    current = list(snapshot.sources)
    current[0] = replace(current[0], stage_generation_id="generation-2")

    with pytest.raises(StaleDatasetSourceError, match="sources are stale"):
        snapshot.assert_sources_current(current)

    snapshot.assert_sources_current(tuple(reversed(snapshot.sources)))


def test_source_artifact_reference_must_be_portable() -> None:
    with pytest.raises(MLManifestError, match="portable POSIX path"):
        SourceArtifactReference(
            artifact_id="spine",
            kind="spine",
            relative_path="/tmp/project/spine.parquet",
            sha256="a" * 64,
        )


def test_observation_identity_and_modality_are_validated() -> None:
    snapshot = _snapshot()

    with pytest.raises(MLManifestError, match="stable experiment_uid/read_id"):
        replace(snapshot.observations[0], molecule_uid="wrong")

    with pytest.raises(MLManifestError, match="modality differs"):
        DatasetSnapshotManifest.create(
            selection=snapshot.selection,
            input_schema=snapshot.input_schema,
            label_schema=snapshot.label_schema,
            sources=snapshot.sources,
            observations=(replace(snapshot.observations[0], modality="direct"),)
            + snapshot.observations[1:],
        )


def test_mixed_modality_snapshot_preserves_experiment_modality() -> None:
    input_schema, label_schema = _schemas(modalities=("deaminase", "direct"))
    snapshot = DatasetSnapshotManifest.create(
        selection=DatasetSelection(
            scope_kind="project",
            scope_id="project-a",
            set_name=None,
            dataset_name="mixed",
            plan_hash="2" * 64,
            samples=("s1", "s2"),
            references=("Nkg2a",),
            intervals=(),
            filters={},
        ),
        input_schema=input_schema,
        label_schema=label_schema,
        sources=(
            _source(EXP_1, "exp-1"),
            _source(EXP_2, "exp-2", modality="direct"),
        ),
        observations=(
            _observation(EXP_1, "read-1", "s1", 0),
            _observation(EXP_2, "read-2", "s2", 1, modality="direct"),
        ),
    )

    assert {item.modality for item in snapshot.sources} == {"deaminase", "direct"}
    assert {item.value for item in snapshot.summary.counts_by_modality} == {
        "deaminase",
        "direct",
    }


def test_split_is_order_independent_and_round_trips() -> None:
    snapshot = _snapshot()
    assignments = _assignments(snapshot)
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("experiment_uid", "Sample"),
        assignments=assignments,
    )
    reversed_assignments = dict(reversed(tuple(assignments.items())))
    reordered = SplitManifest.create(
        dataset=snapshot,
        group_by=("experiment_uid", "Sample"),
        assignments=reversed_assignments,
    )

    assert reordered.split_id == split.split_id
    assert reordered.membership_digest == split.membership_digest
    assert SplitManifest.from_dict(split.to_dict(), dataset=snapshot) == split
    split.validate_against(snapshot)


def test_split_requires_exact_dataset_coverage() -> None:
    snapshot = _snapshot()
    assignments = _assignments(snapshot)
    assignments.pop(next(iter(assignments)))

    with pytest.raises(MLManifestError, match="cover the dataset exactly"):
        SplitManifest.create(
            dataset=snapshot,
            group_by=("Sample",),
            assignments=assignments,
        )


def test_split_rejects_biological_group_leakage() -> None:
    snapshot = _snapshot()
    assignments = _assignments(snapshot)
    s1_uids = [item.molecule_uid for item in snapshot.observations if item.sample_id == "s1"]
    assignments[s1_uids[0]] = "test"

    with pytest.raises(MLManifestError, match="occurs in both"):
        SplitManifest.create(
            dataset=snapshot,
            group_by=("Sample",),
            assignments=assignments,
        )


def test_split_summaries_are_auditable_by_group_class_and_modality() -> None:
    snapshot = _snapshot()
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("Sample",),
        assignments=_assignments(snapshot),
    )

    train = next(item for item in split.summaries if item.split == "train")
    validation = next(item for item in split.summaries if item.split == "validation")
    assert (train.n_observations, train.n_groups) == (2, 1)
    assert [(item.value, item.count) for item in train.counts_by_class] == [
        ("active", 1),
        ("inactive", 1),
    ]
    assert [(item.value, item.count) for item in validation.counts_by_modality] == [
        ("deaminase", 2)
    ]


def test_split_grouping_contract_changes_identity() -> None:
    snapshot = _snapshot()
    assignments = _assignments(snapshot)
    by_sample = SplitManifest.create(
        dataset=snapshot,
        group_by=("Sample",),
        assignments=assignments,
    )
    by_experiment_and_sample = SplitManifest.create(
        dataset=snapshot,
        group_by=("experiment_uid", "Sample"),
        assignments=assignments,
    )

    assert by_sample.split_id != by_experiment_and_sample.split_id
    assert by_sample.membership_digest != by_experiment_and_sample.membership_digest


def test_split_rejects_tampered_summary() -> None:
    snapshot = _snapshot()
    split = SplitManifest.create(
        dataset=snapshot,
        group_by=("Sample",),
        assignments=_assignments(snapshot),
    )
    raw = copy.deepcopy(split.to_dict())
    raw["summaries"][0]["n_observations"] += 1

    with pytest.raises(MLManifestError, match="serialized fields"):
        SplitManifest.from_dict(raw, dataset=snapshot)
