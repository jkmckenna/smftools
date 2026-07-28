import json
import shutil
from dataclasses import replace

import pandas as pd

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.pipeline import (
    PROJECT_COMMAND_CLASSES,
    PROJECT_EMBEDDING_FEATURE_NODE,
    PROJECT_EMBEDDING_NODE,
    PROJECT_MATERIALIZATION_NODE,
    PROJECT_SAMPLE_ANALYSIS_NODE,
    PROJECT_SELECTION_NODE,
    ChannelFingerprint,
    DependencyResultIdentity,
    ProjectSourceSnapshot,
    build_project_plan,
    build_project_source_snapshot,
    node_result_from_inputs,
    project_node_inputs,
    project_node_specs,
)
from smftools.project.registry import add_experiment, init_project

SEQUENCE = "ACGTACGT"


def _make_raw_experiment(path, *, experiment, reference_strand, reference_id):
    rows = [
        {
            "read_id": read_id,
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": "bc01",
            "barcode": "bc01",
            "strand": "top",
            "mapping_direction": "fwd",
            "reference_start": 0,
            "cigar": "8M",
            "aligned_length": 8,
            "sequence": [0] * 8,
            "quality": [30] * 8,
            "mismatch": [4] * 8,
            "modification_signal": [0.5] * 8,
        }
        for read_id in ("shared-read", "unique-read")
    ]
    write_raw_store(
        pd.DataFrame(rows),
        path,
        reference_lengths={reference_strand: 8},
        extra_uns={
            "experiment": experiment,
            "modality": "direct",
            "reference_uids": {reference_strand: reference_id},
        },
    )


def _make_project(tmp_path):
    reference_id = reference_uid(SEQUENCE, 8)
    _make_raw_experiment(
        tmp_path / "expA",
        experiment="expA",
        reference_strand="geneA_top",
        reference_id=reference_id,
    )
    _make_raw_experiment(
        tmp_path / "expB",
        experiment="expB",
        reference_strand="geneB_top",
        reference_id=reference_id,
    )
    project = tmp_path / "project"
    init_project(project)
    add_experiment(project, tmp_path / "expA")
    add_experiment(project, tmp_path / "expB")
    return project, reference_id


def _snapshot(*, membership="membership", features="features", reporting="reporting"):
    return ProjectSourceSnapshot(
        snapshot_id=f"{membership}:{features}:{reporting}",
        registry_fingerprint="registry",
        membership_fingerprint=membership,
        feature_fingerprint=features,
        variant_reporting_fingerprint=reporting,
        members=(),
    )


def _request():
    return {
        "project_identity": "project",
        "canonical_reference": "reference",
        "set_name": None,
        "modality": None,
        "experiments": None,
        "stage": "preprocess",
        "layers": ["C_site_binary"],
        "start": None,
        "end": None,
        "read_metrics": False,
        "partitioned": False,
    }


def _completed_results(snapshot):
    specs = {spec.analysis_id: spec for spec in project_node_specs()}
    inputs = project_node_inputs(_request(), snapshot)
    results = {}
    for node_id in (
        PROJECT_SELECTION_NODE,
        PROJECT_MATERIALIZATION_NODE,
        PROJECT_SAMPLE_ANALYSIS_NODE,
        PROJECT_EMBEDDING_FEATURE_NODE,
        PROJECT_EMBEDDING_NODE,
    ):
        spec = specs[node_id]
        dependencies = []
        for dependency_id in spec.dependencies:
            dependency = results[dependency_id]
            wanted = {
                channel.channel_id
                for channel in spec.consumed_channels
                if channel.analysis_id == dependency_id
            }
            dependencies.append(
                DependencyResultIdentity(
                    dependency_id,
                    dependency.result_id,
                    tuple(
                        channel
                        for channel in dependency.produced_channels
                        if channel.channel_id in wanted
                    ),
                )
            )
        node_inputs = replace(inputs[node_id], dependency_results=tuple(dependencies))
        if node_id == PROJECT_SELECTION_NODE:
            channels = (
                ChannelFingerprint(
                    "project.selection.membership",
                    1,
                    snapshot.membership_fingerprint,
                ),
                ChannelFingerprint(
                    "project.selection.features",
                    1,
                    snapshot.feature_fingerprint,
                ),
            )
        else:
            channels = tuple(
                ChannelFingerprint(channel.channel_id, channel.schema_version, f"{node_id}:output")
                for channel in spec.produced_channels
            )
        results[node_id] = node_result_from_inputs(
            spec,
            node_inputs,
            result_id=f"result:{node_id}",
            produced_channels=channels,
        )
    return results


def test_project_commands_have_explicit_semantic_classes():
    assert PROJECT_COMMAND_CLASSES["add"] == "registry_mutation"
    assert PROJECT_COMMAND_CLASSES["materialize"] == "analysis_node"
    assert PROJECT_COMMAND_CLASSES["plan"] == "read_only_consumer"
    assert PROJECT_COMMAND_CLASSES["export-latent"] == "read_only_consumer"


def test_source_snapshot_is_order_invariant_and_read_ids_are_namespaced(tmp_path):
    project, reference_id = _make_project(tmp_path)
    first = build_project_source_snapshot(project, reference_id)
    registry_path = project / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry["experiments"] = dict(reversed(list(registry["experiments"].items())))
    registry_path.write_text(json.dumps(registry))
    second = build_project_source_snapshot(project, reference_id)

    assert second.to_dict() == first.to_dict()
    assert len(first.members) == 2
    assert first.members[0]["membership_fingerprint"] != first.members[1]["membership_fingerprint"]


def test_reporting_only_change_does_not_invalidate_unsubscribed_consumers(tmp_path):
    old = _snapshot(reporting="old-report")
    current = _completed_results(old)
    new = _snapshot(reporting="new-report")

    for target in ("materialization", "sample-analysis", "embedding"):
        plan = build_project_plan(
            tmp_path,
            target,
            _request(),
            snapshot=new,
            current_results=current,
        )
        assert all(decision.state.value == "compatible" for decision in plan.decisions)


def test_membership_and_feature_changes_explain_dependent_recomputation(tmp_path):
    current = _completed_results(_snapshot())
    for changed in (
        _snapshot(membership="passes-dedup-changed"),
        _snapshot(features="feature-channel-changed"),
    ):
        for target in ("materialization", "sample-analysis", "embedding"):
            plan = build_project_plan(
                tmp_path,
                target,
                _request(),
                snapshot=changed,
                current_results=current,
            )
            decisions = {decision.analysis_id: decision for decision in plan.decisions}
            assert decisions[PROJECT_SELECTION_NODE].state.value == "stale_input"
            assert decisions[plan.requested_target].state.value == "dependent_recompute"
            assert PROJECT_SELECTION_NODE in decisions[plan.requested_target].invalidated_by or any(
                decision.state.value == "dependent_recompute" for decision in plan.decisions[1:]
            )


def test_relocated_project_and_experiments_preserve_source_snapshot(tmp_path):
    project, reference_id = _make_project(tmp_path / "original")
    before = build_project_source_snapshot(project, reference_id)
    relocated_root = tmp_path / "relocated"
    shutil.copytree(tmp_path / "original", relocated_root)

    after = build_project_source_snapshot(relocated_root / "project", reference_id)
    assert after.to_dict() == before.to_dict()
