"""Project adapters for channel-sensitive, read-only semantic planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..informatics.molecule_identity import molecule_uid
from .analysis_registry import AnalysisRegistry
from .compatibility import SemanticPlanner, node_result_from_inputs
from .semantic_graph import (
    AnalysisScope,
    ArtifactIdentity,
    ArtifactValidation,
    ChannelDependency,
    ChannelFingerprint,
    ChannelSpec,
    NodeInputs,
    NodeResult,
    SemanticNodeSpec,
    SemanticPlan,
)

PROJECT_SELECTION_NODE = "project.genomic_selection"
PROJECT_MATERIALIZATION_NODE = "project.materialization"
PROJECT_SAMPLE_ANALYSIS_NODE = "project.sample_analysis"
PROJECT_EMBEDDING_FEATURE_NODE = "project.embedding.feature_matrix"
PROJECT_EMBEDDING_NODE = "project.embedding.generation"

PROJECT_TARGETS = {
    "selection": PROJECT_SELECTION_NODE,
    "materialization": PROJECT_MATERIALIZATION_NODE,
    "sample-analysis": PROJECT_SAMPLE_ANALYSIS_NODE,
    "embedding": PROJECT_EMBEDDING_NODE,
}

PROJECT_COMMAND_CLASSES = {
    "init": "registry_mutation",
    "add": "registry_mutation",
    "remove": "registry_mutation",
    "materialize": "analysis_node",
    "sample-analysis": "analysis_node",
    "embedding": "analysis_node",
    "list": "read_only_consumer",
    "plan": "read_only_consumer",
    "sample-store-list": "read_only_consumer",
    "export-fastq": "read_only_consumer",
    "export-latent": "read_only_consumer",
}

_VALIDATOR_ID = "project.semantic"
_MEMBERSHIP_CHANNEL = "project.selection.membership"
_FEATURE_CHANNEL = "project.selection.features"
_VARIANT_REPORTING_CHANNEL = "project.selection.variant_reporting"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectSourceSnapshot:
    """Frozen active-registry selection and independent source channels."""

    snapshot_id: str
    registry_fingerprint: str
    membership_fingerprint: str
    feature_fingerprint: str
    variant_reporting_fingerprint: str
    members: tuple[Mapping[str, Any], ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "registry_fingerprint": self.registry_fingerprint,
            "membership_fingerprint": self.membership_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "variant_reporting_fingerprint": self.variant_reporting_fingerprint,
            "members": [dict(member) for member in self.members],
        }


def project_node_specs() -> tuple[SemanticNodeSpec, ...]:
    """Return project analysis contracts without registry mutation utilities."""
    selection = SemanticNodeSpec(
        analysis_id=PROJECT_SELECTION_NODE,
        scope=AnalysisScope.PROJECT_ANALYSIS,
        produced_channels=(
            ChannelSpec(_MEMBERSHIP_CHANNEL, 1),
            ChannelSpec(_FEATURE_CHANNEL, 1),
        ),
        semantic_config_keys=("selection",),
        task_scope="project_selection",
        validator_id=_VALIDATOR_ID,
    )
    materialization = SemanticNodeSpec(
        analysis_id=PROJECT_MATERIALIZATION_NODE,
        scope=AnalysisScope.PROJECT_ANALYSIS,
        dependencies=(PROJECT_SELECTION_NODE,),
        consumed_channels=(
            ChannelDependency(PROJECT_SELECTION_NODE, _MEMBERSHIP_CHANNEL, 1),
            ChannelDependency(PROJECT_SELECTION_NODE, _FEATURE_CHANNEL, 1),
        ),
        produced_channels=(ChannelSpec("project.materialized", 1),),
        semantic_config_keys=("projection",),
        task_scope="project_selection",
        validator_id=_VALIDATOR_ID,
    )
    sample_analysis = SemanticNodeSpec(
        analysis_id=PROJECT_SAMPLE_ANALYSIS_NODE,
        scope=AnalysisScope.PROJECT_ANALYSIS,
        dependencies=(PROJECT_SELECTION_NODE,),
        consumed_channels=(
            ChannelDependency(PROJECT_SELECTION_NODE, _MEMBERSHIP_CHANNEL, 1),
            ChannelDependency(PROJECT_SELECTION_NODE, _FEATURE_CHANNEL, 1),
        ),
        produced_channels=(ChannelSpec("project.sample_analysis", 1),),
        semantic_config_keys=("sample_analysis",),
        task_scope="experiment_reference_sample",
        validator_id=_VALIDATOR_ID,
    )
    embedding_features = SemanticNodeSpec(
        analysis_id=PROJECT_EMBEDDING_FEATURE_NODE,
        scope=AnalysisScope.PROJECT_ANALYSIS,
        dependencies=(PROJECT_SELECTION_NODE,),
        consumed_channels=(
            ChannelDependency(PROJECT_SELECTION_NODE, _MEMBERSHIP_CHANNEL, 1),
            ChannelDependency(PROJECT_SELECTION_NODE, _FEATURE_CHANNEL, 1),
        ),
        produced_channels=(ChannelSpec("project.embedding.features", 1),),
        semantic_config_keys=("feature_definition",),
        task_scope="project_selection",
        validator_id=_VALIDATOR_ID,
    )
    embedding = SemanticNodeSpec(
        analysis_id=PROJECT_EMBEDDING_NODE,
        scope=AnalysisScope.PROJECT_ANALYSIS,
        dependencies=(PROJECT_EMBEDDING_FEATURE_NODE,),
        consumed_channels=(
            ChannelDependency(
                PROJECT_EMBEDDING_FEATURE_NODE,
                "project.embedding.features",
                1,
            ),
        ),
        produced_channels=(ChannelSpec("project.embedding", 1),),
        semantic_config_keys=("embedding_definition",),
        task_scope="project_selection",
        validator_id=_VALIDATOR_ID,
    )
    return selection, materialization, sample_analysis, embedding_features, embedding


def _preprocess_channels(spine_path: Path) -> dict[str, str]:
    """Read fine-grained preprocess channel identities when available."""
    from ..preprocessing.preprocess_generation import resolve_current_preprocess_generation
    from ..preprocessing.semantic_upgrade import (
        PREPROCESS_TASKS_NODE,
        PREPROCESS_VARIANT_EVIDENCE_NODE,
        PREPROCESS_VARIANT_METRICS_NODE,
        load_preprocess_node_results,
    )

    current = resolve_current_preprocess_generation(spine_path.parent)
    if current is None:
        return {}
    results = load_preprocess_node_results(current[1])

    def fingerprint(analysis_id: str, channel_id: str) -> str | None:
        result = results.get(analysis_id)
        channel = None if result is None else result.channel(channel_id)
        return None if channel is None else channel.fingerprint

    channels = {
        "features": fingerprint(PREPROCESS_TASKS_NODE, "derived_partitions"),
        "variant_evidence": fingerprint(
            PREPROCESS_VARIANT_EVIDENCE_NODE,
            "variant_evidence",
        ),
        "variant_metrics": fingerprint(
            PREPROCESS_VARIANT_METRICS_NODE,
            "variant_cohort_metrics",
        ),
    }
    return {key: value for key, value in channels.items() if value is not None}


def project_source_member_record(member: Mapping[str, Any]) -> dict[str, Any]:
    """Return relocation-safe source-channel identities for one selected member."""
    from ..informatics.experiment_manifest import read_experiment_manifest
    from ..informatics.partition_read import load_spine

    spine_path = Path(member["spine_path"])
    spine = load_spine(spine_path, verbose=False)
    experiment_uid = str(member["experiment_uid"])
    references = set(map(str, member["reference_strands"]))
    obs = spine.obs
    reference_column = "Reference_strand" if "Reference_strand" in obs else "reference"
    selected = obs.loc[obs[reference_column].astype(str).isin(references)].copy()
    read_ids = selected.get("read_id", selected.index.to_series()).astype(str)
    sample_column = next(
        (column for column in ("Sample", "sample", "Barcode", "barcode") if column in selected),
        None,
    )
    membership_rows = []
    for index, read_id in zip(selected.index, read_ids, strict=True):
        membership_rows.append(
            {
                "molecule_uid": molecule_uid(experiment_uid, read_id),
                "reference": str(selected.at[index, reference_column]),
                "sample": str(selected.at[index, sample_column])
                if sample_column is not None
                else "",
                "passes_qc": bool(selected.at[index, "passes_qc"])
                if "passes_qc" in selected
                else None,
                "passes_dedup": bool(selected.at[index, "passes_dedup"])
                if "passes_dedup" in selected
                else None,
            }
        )
    membership_rows.sort(key=lambda row: row["molecule_uid"])

    stage = str(member["stage"])
    run_root = Path(member["spine_path"]).parent.parent
    stage_entry = read_experiment_manifest(run_root).get("stages", {}).get(stage, {})
    if not isinstance(stage_entry, Mapping):
        stage_entry = {}
    fine_channels = _preprocess_channels(spine_path) if stage == "preprocess" else {}
    feature_identity = fine_channels.get("features")
    if feature_identity is None:
        feature_identity = str(
            stage_entry.get("semantic_channel_fingerprint")
            or stage_entry.get("generation_id")
            or _file_sha256(spine_path)
        )
    reporting_identity = _stable_hash(
        {
            key: fine_channels.get(key)
            for key in ("variant_evidence", "variant_metrics")
            if key in fine_channels
        }
    )
    return {
        "experiment": str(member["experiment"]),
        "experiment_uid": experiment_uid,
        "stage": stage,
        "stage_generation_id": stage_entry.get("generation_id"),
        "reference_strands": sorted(references),
        "membership_fingerprint": _stable_hash(membership_rows),
        "feature_fingerprint": feature_identity,
        "variant_reporting_fingerprint": reporting_identity,
        "n_molecules": len(membership_rows),
    }


def build_project_source_snapshot(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
) -> ProjectSourceSnapshot:
    """Resolve and freeze selected active experiment channels without writes."""
    from ..project.set_store import resolve_set_members

    members = resolve_set_members(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    records = tuple(
        sorted(
            (project_source_member_record(member) for member in members),
            key=lambda record: (record["experiment_uid"], record["stage"]),
        )
    )
    registry_payload = [
        {
            key: record[key]
            for key in (
                "experiment",
                "experiment_uid",
                "stage",
                "reference_strands",
            )
        }
        for record in records
    ]
    registry_fingerprint = _stable_hash(registry_payload)
    membership_fingerprint = _stable_hash(
        [(record["experiment_uid"], record["membership_fingerprint"]) for record in records]
    )
    feature_fingerprint = _stable_hash(
        [(record["experiment_uid"], record["feature_fingerprint"]) for record in records]
    )
    variant_fingerprint = _stable_hash(
        [(record["experiment_uid"], record["variant_reporting_fingerprint"]) for record in records]
    )
    snapshot_id = _stable_hash(
        {
            "registry": registry_fingerprint,
            "membership": membership_fingerprint,
            "features": feature_fingerprint,
            "variant_reporting": variant_fingerprint,
        }
    )
    return ProjectSourceSnapshot(
        snapshot_id=snapshot_id,
        registry_fingerprint=registry_fingerprint,
        membership_fingerprint=membership_fingerprint,
        feature_fingerprint=feature_fingerprint,
        variant_reporting_fingerprint=variant_fingerprint,
        members=records,
    )


def project_node_inputs(
    request: Mapping[str, Any],
    snapshot: ProjectSourceSnapshot,
) -> dict[str, NodeInputs]:
    """Build deterministic project-node inputs for one resolved request."""
    selection = {
        key: request.get(key)
        for key in ("canonical_reference", "set_name", "modality", "experiments", "stage")
    }
    scope = f"project:{request.get('project_identity', 'registry')}"
    task_digest = _stable_hash(selection)
    return {
        PROJECT_SELECTION_NODE: NodeInputs(
            semantic_config={"selection": selection},
            input_artifacts=(
                ArtifactIdentity("project.registry.snapshot", snapshot.registry_fingerprint),
                ArtifactIdentity("project.membership", snapshot.membership_fingerprint),
                ArtifactIdentity("project.features", snapshot.feature_fingerprint),
            ),
            logical_scope_identity=scope,
            logical_task_plan_digest=task_digest,
        ),
        PROJECT_MATERIALIZATION_NODE: NodeInputs(
            semantic_config={
                "projection": {
                    key: request.get(key)
                    for key in ("layers", "start", "end", "read_metrics", "partitioned")
                }
            },
            logical_scope_identity=scope,
            logical_task_plan_digest=task_digest,
        ),
        PROJECT_SAMPLE_ANALYSIS_NODE: NodeInputs(
            semantic_config={"sample_analysis": request.get("sample_analysis", {})},
            logical_scope_identity=scope,
            logical_task_plan_digest=task_digest,
        ),
        PROJECT_EMBEDDING_FEATURE_NODE: NodeInputs(
            semantic_config={"feature_definition": request.get("feature_definition", {})},
            logical_scope_identity=scope,
            logical_task_plan_digest=task_digest,
        ),
        PROJECT_EMBEDDING_NODE: NodeInputs(
            semantic_config={"embedding_definition": request.get("embedding_definition", {})},
            logical_scope_identity=scope,
            logical_task_plan_digest=task_digest,
        ),
    }


def _selection_result(
    spec: SemanticNodeSpec,
    inputs: NodeInputs,
    snapshot: ProjectSourceSnapshot,
) -> NodeResult:
    result_identity = _stable_hash(
        {
            "registry": snapshot.registry_fingerprint,
            "membership": snapshot.membership_fingerprint,
            "features": snapshot.feature_fingerprint,
        }
    )
    return node_result_from_inputs(
        spec,
        inputs,
        result_id=f"selection:{result_identity}",
        produced_channels=(
            ChannelFingerprint(_MEMBERSHIP_CHANNEL, 1, snapshot.membership_fingerprint),
            ChannelFingerprint(_FEATURE_CHANNEL, 1, snapshot.feature_fingerprint),
        ),
    )


def build_project_plan(
    project_dir: str | Path,
    target: str,
    request: Mapping[str, Any],
    *,
    snapshot: ProjectSourceSnapshot | None = None,
    current_results: Mapping[str, NodeResult] | None = None,
) -> SemanticPlan:
    """Return a deterministic project plan and perform no writes."""
    normalized_target = str(target).strip().lower()
    if normalized_target not in PROJECT_TARGETS:
        raise ValueError(
            f"unknown project target {target!r}; expected one of {sorted(PROJECT_TARGETS)}"
        )
    snapshot = snapshot or build_project_source_snapshot(
        project_dir,
        str(request["canonical_reference"]),
        set_name=request.get("set_name"),
        modality=request.get("modality"),
        experiments=request.get("experiments"),
        stage=request.get("stage"),
    )
    specs = project_node_specs()
    spec_by_id = {spec.analysis_id: spec for spec in specs}
    registry = AnalysisRegistry(
        specs,
        validators={_VALIDATOR_ID: lambda _result: ArtifactValidation(True)},
    )
    inputs = project_node_inputs(request, snapshot)
    current = dict(current_results or {})
    current.setdefault(
        PROJECT_SELECTION_NODE,
        _selection_result(
            spec_by_id[PROJECT_SELECTION_NODE],
            inputs[PROJECT_SELECTION_NODE],
            snapshot,
        ),
    )
    return SemanticPlanner(registry).plan(
        PROJECT_TARGETS[normalized_target],
        inputs_by_node=inputs,
        current_results=current,
    )


def format_project_plan(plan: SemanticPlan) -> str:
    """Render a compact deterministic human-readable project plan."""
    lines = [
        f"Project target: {plan.requested_target}",
        f"Graph definition version: {plan.graph_definition_version}",
        "",
    ]
    lines.extend(
        f"{index:>2}. {decision.state.value:<23} {decision.analysis_id} — {decision.reason}"
        for index, decision in enumerate(plan.decisions, start=1)
    )
    return "\n".join(lines)
