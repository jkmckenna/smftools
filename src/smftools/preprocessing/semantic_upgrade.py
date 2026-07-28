"""Semantic planning records for immutable preprocess generations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..informatics.experiment_manifest import artifact_record
from ..pipeline import (
    AnalysisRegistry,
    AnalysisScope,
    ArtifactIdentity,
    ArtifactRecord,
    ArtifactValidation,
    ChannelDependency,
    ChannelFingerprint,
    ChannelSpec,
    DependencyResultIdentity,
    NodeInputs,
    NodeResult,
    PlanState,
    SemanticNodeSpec,
    SemanticPlan,
    SemanticPlanner,
    node_result_from_inputs,
)

PREPROCESS_TASKS_NODE = "preprocess.tasks"
PREPROCESS_VARIANT_REFERENCE_NODE = "preprocess.variant_reference"
PREPROCESS_VARIANT_EVIDENCE_NODE = "preprocess.variant_evidence"
PREPROCESS_REDUCERS_NODE = "preprocess.reducers"
PREPROCESS_PLOTS_NODE = "preprocess.plots"

_TASK_CONFIG_KEYS = {
    "binarize_on_fixed_methlyation_threshold",
    "bypass_append_base_context",
    "bypass_clean_nan",
    "clean_nan_layers",
    "fit_j_threshold",
    "fit_position_methylation_thresholds",
    "infer_on_percentile_sample_methylation_fitting",
    "inference_variable_sample_methylation_fitting",
    "mod_target_bases",
    "negative_control_sample_methylation_fitting",
    "output_binary_layer_name",
    "positive_control_sample_methylation_fitting",
    "reference_column",
    "sample_column",
    "smf_modality",
}
_PLOT_CONFIG_KEYS = {
    "bypass_complexity_analysis",
    "duplicate_detection_distance_threshold",
    "emit_automated_plots",
    "plot_allow_unanalyzed_gaps",
    "plot_regions_bed",
    "plot_subsample_seed",
    "preprocess_plot_complexity_bootstraps",
    "preprocess_plot_max_heatmap_positions",
    "preprocess_plot_max_heatmap_reads",
    "sample_name_col_for_plotting",
}
_REDUCER_CONFIG_KEYS = {
    "bypass_filter_reads_on_cigar_indels",
    "bypass_filter_reads_on_length_quality_mapping",
    "bypass_filter_reads_on_modification_thresholds",
    "bypass_flag_duplicate_reads",
    "bypass_label_deaminase_pcr_chimeras",
    "deaminase_chimera_max_single_strand_fraction",
    "deaminase_chimera_min_events_per_span",
    "deaminase_chimera_min_segment_purity",
    "duplicate_detection_chunk_presort_metric",
    "duplicate_detection_demux_types_to_use",
    "duplicate_detection_distance_threshold",
    "duplicate_detection_do_hierarchical",
    "duplicate_detection_do_pca",
    "duplicate_detection_hierarchical_linkage",
    "duplicate_detection_hierarchical_max_representatives",
    "duplicate_detection_keep_best_metric",
    "duplicate_detection_max_reads_per_window",
    "duplicate_detection_max_rounds",
    "duplicate_detection_min_overlapping_positions",
    "duplicate_detection_min_progress_rounds_before_stop",
    "duplicate_detection_n_permutation_passes",
    "duplicate_detection_pca_center",
    "duplicate_detection_pca_n_components",
    "duplicate_detection_permutation_seed",
    "duplicate_detection_round_shuffle_seed",
    "duplicate_detection_site_types",
    "duplicate_detection_window_size_for_hamming_neighbors",
    "mapped_len_filter_thresholds",
    "mapped_len_to_read_len_ratio_filter_thresholds",
    "mapped_len_to_ref_ratio_filter_thresholds",
    "max_internal_deletion_length",
    "max_internal_insertion_length",
    "min_valid_fraction_positions_in_read_vs_ref",
    "position_max_nan_threshold",
    "reference_column",
    "read_len_filter_thresholds",
    "read_len_to_ref_ratio_filter_thresholds",
    "read_mapping_quality_filter_thresholds",
    "read_mod_filtering_a_thresholds",
    "read_mod_filtering_c_thresholds",
    "read_mod_filtering_cpg_thresholds",
    "read_mod_filtering_gpc_thresholds",
    "read_mod_filtering_use_other_c_as_background",
    "read_quality_filter_thresholds",
    "sample_name_col_for_plotting",
    "smf_modality",
    "mod_target_bases",
    "variant_analysis_mode",
}
_VARIANT_REFERENCE_CONFIG_KEYS = {
    "references_to_align_for_variant_annotation",
}
_VARIANT_EVIDENCE_CONFIG_KEYS = {
    "variant_analysis_mode",
}
_NON_SEMANTIC_CONFIG_KEYS = {
    "device",
    "emit_log_file",
    "emit_perf_log",
    "log_level",
    "max_memory_gb",
    "max_memory_percent",
    "memory_reserve_gb",
    "output_directory",
    "perf_log_sample_interval_seconds",
    "plot_threads_fraction",
    "target_task_memory_mb",
    "threads",
}

_NODE_ARTIFACTS = {
    PREPROCESS_TASKS_NODE: ("store", "task_catalog", "catalog", "read_index"),
    PREPROCESS_VARIANT_REFERENCE_NODE: ("variant_reference_catalog",),
    PREPROCESS_VARIANT_EVIDENCE_NODE: (
        "variant_task_store",
        "variant_task_catalog",
        "variant_obs",
        "variant_read_index",
        "variant_generation_manifest",
    ),
    PREPROCESS_REDUCERS_NODE: ("var", "obs", "stage_obs", "spine", "manifest"),
    PREPROCESS_PLOTS_NODE: ("plots", "plot_catalog"),
}


def _cfg_values(cfg: Any) -> dict[str, Any]:
    values = dict(cfg.to_dict()) if hasattr(cfg, "to_dict") else dict(vars(cfg))
    return {
        key: value
        for key, value in values.items()
        if key not in _NON_SEMANTIC_CONFIG_KEYS
        and not key.startswith("force_redo_")
        and not key.endswith("_max_workers")
    }


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _config_fingerprints(cfg: Any) -> dict[str, str]:
    values = _cfg_values(cfg)
    plots = {key: values[key] for key in sorted(_PLOT_CONFIG_KEYS) if key in values}
    reducers = {key: values[key] for key in sorted(_REDUCER_CONFIG_KEYS) if key in values}
    tasks = {key: values[key] for key in sorted(_TASK_CONFIG_KEYS) if key in values}
    variant_reference = {
        key: values[key] for key in sorted(_VARIANT_REFERENCE_CONFIG_KEYS) if key in values
    }
    variant_evidence = {
        key: values[key] for key in sorted(_VARIANT_EVIDENCE_CONFIG_KEYS) if key in values
    }
    return {
        PREPROCESS_TASKS_NODE: _sha256_payload(tasks),
        PREPROCESS_VARIANT_REFERENCE_NODE: _sha256_payload(variant_reference),
        PREPROCESS_VARIANT_EVIDENCE_NODE: _sha256_payload(variant_evidence),
        PREPROCESS_REDUCERS_NODE: _sha256_payload(reducers),
        PREPROCESS_PLOTS_NODE: _sha256_payload(plots),
    }


def preprocess_stage_compute_config(cfg: Any) -> dict[str, str]:
    """Return preprocess compute identities without plot or downstream config."""
    fingerprints = _config_fingerprints(cfg)
    return {
        "tasks": fingerprints[PREPROCESS_TASKS_NODE],
        "reducers": fingerprints[PREPROCESS_REDUCERS_NODE],
    }


def preprocess_node_specs(
    *,
    variant_enabled: bool = False,
    algorithm_versions: Mapping[str, str] | None = None,
    additional_specs: tuple[SemanticNodeSpec, ...] = (),
) -> tuple[SemanticNodeSpec, ...]:
    """Return the versioned semantic contracts for partitioned preprocessing."""
    versions = dict(algorithm_versions or {})
    variant_specs = (
        SemanticNodeSpec(
            analysis_id=PREPROCESS_VARIANT_REFERENCE_NODE,
            scope=AnalysisScope.EXPERIMENT_STAGE,
            produced_channels=(ChannelSpec("variant_reference_catalog", 1),),
            semantic_config_keys=("config_fingerprint",),
            algorithm_version=versions.get(PREPROCESS_VARIANT_REFERENCE_NODE, "1"),
            output_schema_version=1,
            validator_id="preprocess.variant_reference",
        ),
        SemanticNodeSpec(
            analysis_id=PREPROCESS_VARIANT_EVIDENCE_NODE,
            scope=AnalysisScope.EXPERIMENT_STAGE,
            dependencies=(PREPROCESS_VARIANT_REFERENCE_NODE,),
            consumed_channels=(
                ChannelDependency(
                    PREPROCESS_VARIANT_REFERENCE_NODE,
                    "variant_reference_catalog",
                    1,
                ),
            ),
            produced_channels=(ChannelSpec("variant_evidence", 1),),
            semantic_config_keys=("config_fingerprint",),
            algorithm_version=versions.get(PREPROCESS_VARIANT_EVIDENCE_NODE, "1"),
            output_schema_version=1,
            validator_id="preprocess.variant_evidence",
        ),
    )
    reducer_dependencies = (PREPROCESS_TASKS_NODE,)
    reducer_channels = (ChannelDependency(PREPROCESS_TASKS_NODE, "derived_partitions", 1),)
    if variant_enabled:
        reducer_dependencies += (PREPROCESS_VARIANT_EVIDENCE_NODE,)
        reducer_channels += (
            ChannelDependency(PREPROCESS_VARIANT_EVIDENCE_NODE, "variant_evidence", 1),
        )
    builtins = (
        SemanticNodeSpec(
            analysis_id=PREPROCESS_TASKS_NODE,
            scope=AnalysisScope.EXPERIMENT_STAGE,
            produced_channels=(ChannelSpec("derived_partitions", 1),),
            semantic_config_keys=("config_fingerprint",),
            algorithm_version=versions.get(PREPROCESS_TASKS_NODE, "1"),
            output_schema_version=1,
            validator_id="preprocess.tasks",
        ),
        SemanticNodeSpec(
            analysis_id=PREPROCESS_REDUCERS_NODE,
            scope=AnalysisScope.EXPERIMENT_STAGE,
            dependencies=reducer_dependencies,
            consumed_channels=reducer_channels,
            produced_channels=(ChannelSpec("filtered_spine", 1),),
            semantic_config_keys=("config_fingerprint",),
            algorithm_version=versions.get(PREPROCESS_REDUCERS_NODE, "1"),
            output_schema_version=1,
            validator_id="preprocess.reducers",
        ),
        SemanticNodeSpec(
            analysis_id=PREPROCESS_PLOTS_NODE,
            scope=AnalysisScope.EXPERIMENT_ANALYSIS,
            dependencies=(PREPROCESS_REDUCERS_NODE,),
            consumed_channels=(ChannelDependency(PREPROCESS_REDUCERS_NODE, "filtered_spine", 1),),
            produced_channels=(ChannelSpec("plots", 1),),
            semantic_config_keys=("config_fingerprint",),
            algorithm_version=versions.get(PREPROCESS_PLOTS_NODE, "1"),
            output_schema_version=1,
            validator_id="preprocess.plots",
        ),
    )
    return (builtins[:1] + (variant_specs if variant_enabled else ()) + builtins[1:]) + tuple(
        additional_specs
    )


def preprocess_node_inputs(
    source_path: str | Path,
    cfg: Any,
    run_root: str | Path,
) -> dict[str, NodeInputs]:
    """Build deterministic node inputs without writing preprocess artifacts."""
    source_path = Path(source_path)
    source_available = source_path.is_file()
    if source_available:
        from ..cli.helpers import stage_input_artifact_ids

        source_artifacts = tuple(
            ArtifactIdentity(f"source:{index}", identity)
            for index, identity in enumerate(
                stage_input_artifact_ids(
                    run_root,
                    source_path,
                    include_region_catalogs=True,
                )
            )
        )
    else:
        source_artifacts = (ArtifactIdentity("source:0", "unavailable"),)
    fingerprints = _config_fingerprints(cfg)
    logical_digest = hashlib.sha256(
        json.dumps([source.to_dict() for source in source_artifacts], sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    variant_enabled = str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report"
    analysis_ids = [PREPROCESS_TASKS_NODE]
    if variant_enabled:
        analysis_ids.extend([PREPROCESS_VARIANT_REFERENCE_NODE, PREPROCESS_VARIANT_EVIDENCE_NODE])
    analysis_ids.extend([PREPROCESS_REDUCERS_NODE, PREPROCESS_PLOTS_NODE])
    source_nodes = {PREPROCESS_TASKS_NODE}
    if variant_enabled:
        source_nodes.add(PREPROCESS_VARIANT_REFERENCE_NODE)
    return {
        analysis_id: NodeInputs(
            semantic_config={"config_fingerprint": fingerprints[analysis_id]},
            input_artifacts=source_artifacts if analysis_id in source_nodes else (),
            logical_scope_identity="preprocess:experiment",
            logical_task_plan_digest=logical_digest,
            unavailable_inputs=(
                () if source_available or analysis_id not in source_nodes else ("source_spine",)
            ),
        )
        for analysis_id in analysis_ids
    }


def _node_result_from_dict(payload: Mapping[str, Any]) -> NodeResult:
    def channel(value: Mapping[str, Any]) -> ChannelFingerprint:
        return ChannelFingerprint(
            str(value["channel_id"]),
            int(value["schema_version"]),
            str(value["fingerprint"]),
        )

    return NodeResult(
        analysis_id=str(payload["analysis_id"]),
        result_id=str(payload["result_id"]),
        algorithm_version=str(payload["algorithm_version"]),
        output_schema_version=int(payload["output_schema_version"]),
        compatibility_key=str(payload["compatibility_key"]),
        semantic_config_hash=str(payload["semantic_config_hash"]),
        input_artifacts=tuple(
            ArtifactIdentity(str(item["artifact_id"]), str(item["checksum"]))
            for item in payload.get("input_artifacts", ())
        ),
        dependency_results=tuple(
            DependencyResultIdentity(
                str(item["analysis_id"]),
                str(item["result_id"]),
                tuple(channel(value) for value in item.get("channel_fingerprints", ())),
            )
            for item in payload.get("dependency_results", ())
        ),
        logical_scope_identity=str(payload["logical_scope_identity"]),
        logical_task_plan_digest=str(payload["logical_task_plan_digest"]),
        produced_channels=tuple(channel(item) for item in payload.get("produced_channels", ())),
        artifacts=tuple(
            ArtifactRecord(
                str(item["artifact_id"]),
                str(item["relative_path"]),
                str(item["checksum"]),
                int(item["schema_version"]),
                str(item["kind"]),
            )
            for item in payload.get("artifacts", ())
        ),
        state=str(payload.get("state", "complete")),
        reused_from_generation_id=payload.get("reused_from_generation_id"),
        started_at=payload.get("started_at"),
        completed_at=payload.get("completed_at"),
        execution_provenance=tuple(
            (str(key), str(value))
            for key, value in dict(payload.get("execution_provenance", {})).items()
        ),
        schema_version=int(payload.get("schema_version", 1)),
    )


def load_preprocess_node_results(manifest: Mapping[str, Any]) -> dict[str, NodeResult]:
    """Load node results, returning no reusable results for legacy manifests."""
    payload = manifest.get("node_results")
    if not isinstance(payload, list):
        return {}
    results = [_node_result_from_dict(item) for item in payload if isinstance(item, Mapping)]
    by_id = {result.analysis_id: result for result in results}
    if len(by_id) != len(results):
        raise ValueError("preprocess manifest declares duplicate node results")
    return by_id


def _artifact_validator(generation_root: Path, expected_artifact_ids: tuple[str, ...]):
    def validate(result: NodeResult) -> ArtifactValidation:
        artifact_ids = {artifact.artifact_id for artifact in result.artifacts}
        if artifact_ids != set(expected_artifact_ids):
            return ArtifactValidation(
                False,
                "node_artifact_set_changed",
                f"node artifact IDs differ: expected {sorted(expected_artifact_ids)}, "
                f"found {sorted(artifact_ids)}",
            )
        for artifact in result.artifacts:
            path = (generation_root / artifact.relative_path).resolve()
            if not path.is_relative_to(generation_root.resolve()) or not path.exists():
                return ArtifactValidation(
                    False,
                    "missing_node_artifact",
                    f"node artifact is missing: {artifact.artifact_id}",
                )
            actual = artifact_record(path, generation_root, checksum=True)
            if str(actual["sha256"]) != artifact.checksum:
                return ArtifactValidation(
                    False,
                    "corrupt_node_artifact",
                    f"node artifact checksum differs: {artifact.artifact_id}",
                )
            if str(actual["kind"]) != artifact.kind:
                return ArtifactValidation(
                    False,
                    "invalid_node_artifact_kind",
                    f"node artifact kind differs: {artifact.artifact_id}",
                )
        return ArtifactValidation(True)

    return validate


def preprocess_registry(
    generation_root: str | Path,
    *,
    algorithm_versions: Mapping[str, str] | None = None,
    additional_specs: tuple[SemanticNodeSpec, ...] = (),
    additional_validators: Mapping[str, Any] | None = None,
    variant_enabled: bool = False,
) -> AnalysisRegistry:
    """Create the preprocess semantic registry bound to one candidate generation."""
    root = Path(generation_root)
    validators = {
        "preprocess.tasks": _artifact_validator(root, _NODE_ARTIFACTS[PREPROCESS_TASKS_NODE]),
        "preprocess.variant_reference": _artifact_validator(
            root, _NODE_ARTIFACTS[PREPROCESS_VARIANT_REFERENCE_NODE]
        ),
        "preprocess.variant_evidence": _artifact_validator(
            root, _NODE_ARTIFACTS[PREPROCESS_VARIANT_EVIDENCE_NODE]
        ),
        "preprocess.reducers": _artifact_validator(root, _NODE_ARTIFACTS[PREPROCESS_REDUCERS_NODE]),
        "preprocess.plots": _artifact_validator(root, _NODE_ARTIFACTS[PREPROCESS_PLOTS_NODE]),
        **dict(additional_validators or {}),
    }
    return AnalysisRegistry(
        preprocess_node_specs(
            variant_enabled=variant_enabled,
            algorithm_versions=algorithm_versions,
            additional_specs=additional_specs,
        ),
        validators=validators,
    )


def plan_preprocess_upgrade(
    source_path: str | Path,
    cfg: Any,
    output_dir: str | Path,
    *,
    current_generation: tuple[Path, Mapping[str, Any]] | None = None,
    algorithm_versions: Mapping[str, str] | None = None,
    force_targets: set[str] | frozenset[str] = frozenset(),
    target: str = PREPROCESS_PLOTS_NODE,
    additional_specs: tuple[SemanticNodeSpec, ...] = (),
    additional_inputs: Mapping[str, NodeInputs] | None = None,
    additional_validators: Mapping[str, Any] | None = None,
) -> SemanticPlan:
    """Plan an upgrade against the current generation without publishing files."""
    output_dir = Path(output_dir)
    if current_generation is None:
        from .preprocess_generation import resolve_current_preprocess_generation

        current_generation = resolve_current_preprocess_generation(output_dir)
    generation_root = current_generation[0] if current_generation else output_dir
    manifest = current_generation[1] if current_generation else {}
    current_results = load_preprocess_node_results(manifest)
    inputs_by_node = preprocess_node_inputs(source_path, cfg, output_dir.parent)
    inputs_by_node.update(dict(additional_inputs or {}))
    plan = SemanticPlanner(
        preprocess_registry(
            generation_root,
            variant_enabled=str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report",
            algorithm_versions=algorithm_versions,
            additional_specs=additional_specs,
            additional_validators=additional_validators,
        )
    ).plan(
        target,
        inputs_by_node=inputs_by_node,
        current_results=current_results,
        current_generation_id=(
            str(manifest.get("generation_id")) if current_generation is not None else None
        ),
    )
    unknown_force_targets = sorted(set(force_targets).difference(plan.topological_order))
    if unknown_force_targets:
        raise ValueError(f"unknown preprocess force targets: {unknown_force_targets}")
    forced_closure: set[str] = set()
    decisions = []
    specs = {
        spec.analysis_id: spec
        for spec in preprocess_node_specs(
            variant_enabled=str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report",
            additional_specs=additional_specs,
        )
    }
    for decision in plan.decisions:
        analysis_id = decision.analysis_id
        dependencies = set(specs[analysis_id].dependencies)
        if analysis_id in force_targets:
            decision = replace(
                decision,
                state=PlanState.STALE_INPUT,
                reason_code="force_target_requested",
                reason="explicit force target requested recomputation",
                selected_result_id=None,
                invalidated_by=(analysis_id,),
            )
            forced_closure.add(analysis_id)
        elif dependencies.intersection(forced_closure):
            changed = tuple(sorted(dependencies.intersection(forced_closure)))
            decision = replace(
                decision,
                state=PlanState.DEPENDENT_RECOMPUTE,
                reason_code="dependency_recompute",
                reason=f"dependencies require recomputation: {list(changed)}",
                selected_result_id=None,
                invalidated_by=changed,
            )
            forced_closure.add(analysis_id)
        decisions.append(decision)
    return replace(plan, decisions=tuple(decisions))


def preprocess_force_targets(cfg: Any) -> frozenset[str]:
    """Translate legacy force flags into semantic node targets."""
    if bool(getattr(cfg, "force_redo_preprocessing", False)):
        return frozenset({PREPROCESS_TASKS_NODE})
    task_flags = (
        "force_redo_append_base_context",
        "force_redo_append_binary_layer_by_base_context",
        "force_redo_clean_nan",
    )
    reducer_flags = (
        "force_redo_add_read_length_and_mapping_qc",
        "force_redo_calculate_read_modification_stats",
        "force_redo_filter_reads_on_cigar_indels",
        "force_redo_filter_reads_on_modification_thresholds",
        "force_redo_flag_duplicate_reads",
    )
    plot_flags = ("force_redo_complexity_analysis",)
    targets: set[str] = set()
    if any(bool(getattr(cfg, flag, False)) for flag in task_flags):
        targets.add(PREPROCESS_TASKS_NODE)
    if any(bool(getattr(cfg, flag, False)) for flag in reducer_flags):
        targets.add(PREPROCESS_REDUCERS_NODE)
    if any(bool(getattr(cfg, flag, False)) for flag in plot_flags):
        targets.add(PREPROCESS_PLOTS_NODE)
    return frozenset(targets)


def build_preprocess_node_results(
    generation_root: str | Path,
    source_path: str | Path,
    cfg: Any,
    *,
    generation_id: str,
    reused_nodes: set[str] | frozenset[str] = frozenset(),
    reused_from_generation_id: str | None = None,
    algorithm_versions: Mapping[str, str] | None = None,
) -> tuple[NodeResult, ...]:
    """Record completed node outputs and exact dependency channel identities."""
    root = Path(generation_root)
    specs = {
        spec.analysis_id: spec
        for spec in preprocess_node_specs(
            variant_enabled=str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report",
            algorithm_versions=algorithm_versions,
        )
    }
    inputs = preprocess_node_inputs(source_path, cfg, root.parents[2])
    results: dict[str, NodeResult] = {}
    analysis_order = [PREPROCESS_TASKS_NODE]
    if str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report":
        analysis_order.extend([PREPROCESS_VARIANT_REFERENCE_NODE, PREPROCESS_VARIANT_EVIDENCE_NODE])
    analysis_order.extend([PREPROCESS_REDUCERS_NODE, PREPROCESS_PLOTS_NODE])
    for analysis_id in analysis_order:
        spec = specs[analysis_id]
        dependencies = tuple(
            DependencyResultIdentity(
                dependency_id,
                results[dependency_id].result_id,
                tuple(
                    results[dependency_id].channel(channel.channel_id)
                    for channel in spec.consumed_channels
                    if channel.analysis_id == dependency_id
                ),
            )
            for dependency_id in spec.dependencies
        )
        node_inputs = NodeInputs(
            semantic_config=inputs[analysis_id].semantic_config,
            input_artifacts=inputs[analysis_id].input_artifacts,
            dependency_results=dependencies,
            logical_scope_identity=inputs[analysis_id].logical_scope_identity,
            logical_task_plan_digest=inputs[analysis_id].logical_task_plan_digest,
        )
        artifacts = tuple(
            ArtifactRecord(
                artifact_id=artifact_id,
                relative_path=str(_artifact_relative_path(artifact_id)),
                checksum=str(
                    artifact_record(
                        root / _artifact_relative_path(artifact_id), root, checksum=True
                    )["sha256"]
                ),
                schema_version=1,
                kind=str(
                    artifact_record(root / _artifact_relative_path(artifact_id), root)["kind"]
                ),
            )
            for artifact_id in _NODE_ARTIFACTS[analysis_id]
        )
        channel_fingerprint = _sha256_payload([artifact.checksum for artifact in artifacts])
        results[analysis_id] = node_result_from_inputs(
            spec,
            node_inputs,
            result_id=f"{generation_id}:{analysis_id}",
            produced_channels=tuple(
                ChannelFingerprint(channel.channel_id, channel.schema_version, channel_fingerprint)
                for channel in spec.produced_channels
            ),
            artifacts=artifacts,
            reused_from_generation_id=(
                reused_from_generation_id if analysis_id in reused_nodes else None
            ),
        )
    return tuple(results[analysis_id] for analysis_id in sorted(results))


def _artifact_relative_path(artifact_id: str) -> Path:
    from .partitioned_executor import (
        PREPROCESS_OBS_SIDECAR,
        PREPROCESS_PARTITION_CATALOG,
        PREPROCESS_SPINE_FILENAME,
        PREPROCESS_STAGE_OBS,
        PREPROCESS_STORE_SUBDIR,
        PREPROCESS_TASK_CATALOG,
        PREPROCESS_VAR_CATALOG,
    )

    return {
        "store": Path(PREPROCESS_STORE_SUBDIR),
        "task_catalog": Path(PREPROCESS_TASK_CATALOG),
        "catalog": Path(PREPROCESS_PARTITION_CATALOG),
        "read_index": Path("read_index"),
        "var": Path(PREPROCESS_VAR_CATALOG),
        "obs": Path(PREPROCESS_OBS_SIDECAR),
        "stage_obs": Path(PREPROCESS_STAGE_OBS),
        "spine": Path(PREPROCESS_SPINE_FILENAME),
        "manifest": Path("sidecar_manifest.json"),
        "plots": Path("plots"),
        "plot_catalog": Path("plots/catalog.parquet"),
        "variant_reference_catalog": Path("variant/reference_catalog.json"),
        "variant_task_store": Path("variant/task_store"),
        "variant_task_catalog": Path("variant/task_catalog.parquet"),
        "variant_obs": Path("variant/variant_obs"),
        "variant_read_index": Path("variant/read_index"),
        "variant_generation_manifest": Path("variant/generation_manifest.json"),
    }[artifact_id]
