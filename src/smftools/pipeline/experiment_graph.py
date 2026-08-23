"""Experiment-stage adapters for the engine-neutral semantic planner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from ..constants import PARTITIONED_STAGE_REQUIRED_ARTIFACTS, PREPROCESS_DIR
from ..informatics.experiment_manifest import read_experiment_manifest, stage_is_complete
from .analysis_registry import AnalysisRegistry, NodeExecutor
from .compatibility import SemanticPlanner, node_result_from_inputs
from .semantic_graph import (
    AnalysisScope,
    ArtifactIdentity,
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
)
from .upgrade_impact import UpgradeImpactReport, build_upgrade_impact

EXPERIMENT_STAGES = ("raw", "preprocess", "spatial", "hmm", "latent")
LEGACY_EXPERIMENT_LEAVES = ("chimeric",)
EXPERIMENT_NODE_IDS = {stage: f"experiment.{stage}.complete" for stage in EXPERIMENT_STAGES}
LEGACY_EXPERIMENT_NODE_IDS = {
    stage: f"experiment.{stage}.legacy" for stage in LEGACY_EXPERIMENT_LEAVES
}
EXPERIMENT_TARGETS = ("raw", "preprocess", "variant", "spatial", "hmm", "latent", "full")

_STAGE_OUTPUT_SCHEMA_VERSIONS = {
    "raw": 3,
    "preprocess": 2,
    "spatial": 3,
    "hmm": 2,
    "latent": 2,
}
# Per-stage algorithm versions.
#
# BUMP POLICY: increment a stage's version whenever a code change alters what
# that stage produces for unchanged inputs and unchanged config. Config changes
# are already covered by ``stage_config_hash`` and input changes by the channel
# fingerprints; this value is the *only* signal that the producing code itself
# changed. Leaving it untouched after a behavioural fix makes the planner report
# a pre-fix generation as ``compatible``, silently serving stale results.
#
# Do not bump for refactors, logging, or performance work that leaves outputs
# byte-identical. A bump invalidates every stored generation of that stage
# across all experiments, and cascades to downstream stages.
_STAGE_ALGORITHM_VERSIONS = {
    # 2: chunked-FASTQ pairing (F20), barcode-from-filename (F21), ragged obs
    #    collapse (F24), and demux status carried into obs (F25, F31).
    "raw": "2",
    # 2: mismatch integer-encoding clustermaps added to the stage output
    #    (EGL-26).
    "preprocess": "2",
    # 2: per-group read caps (EGL-24), locus-plottable regions (F27a, F27b),
    #    and streamed position matrices (F32).
    "spatial": "2",
    "hmm": "1",
    # 2: Leiden clustering over latent embeddings, cluster-block ordering, and
    #    per-unit clustermaps (EGL-28a-d, F29).
    "latent": "2",
}
_STAGE_DEPENDENCIES = {
    "raw": None,
    "preprocess": "raw",
    "spatial": "preprocess",
    "hmm": "spatial",
    "latent": "hmm",
}
_STAGE_FORCE_FLAGS = {
    "raw": ("force_redo_load_adata",),
    "preprocess": (
        "force_redo_preprocessing",
        "force_redo_flag_duplicate_reads",
    ),
    "spatial": ("force_redo_spatial_analyses",),
    "hmm": (
        "force_redo_hmm_fit",
        "force_redo_hmm_apply",
        "force_redo_hmm_plots",
    ),
    "latent": ("force_redo_latent_analyses",),
}
_STAGE_OUTPUT_PATH_ATTRIBUTES = {
    "raw": "raw_spine",
    "preprocess": "preprocess_spine",
    "spatial": "spatial_spine",
    "hmm": "hmm_spine",
    "latent": "latent_spine",
}
_VALIDATOR_ID = "experiment.stage"
_LEGACY_VALIDATOR_ID = "experiment.legacy"


@dataclass(frozen=True)
class ExperimentExecutionResult:
    """Semantic plan and stage results from one experiment target request."""

    plan: SemanticPlan
    stage_results: tuple[tuple[str, Any], ...]
    final_result: Any


@dataclass(frozen=True)
class _ExperimentPlanningContext:
    cfg: Any
    paths: Any
    registry: AnalysisRegistry
    plan: SemanticPlan


def experiment_node_specs() -> tuple[SemanticNodeSpec, ...]:
    """Return the coarse experiment graph and registered legacy leaves."""
    specs: list[SemanticNodeSpec] = []
    for stage in EXPERIMENT_STAGES:
        dependency_stage = _STAGE_DEPENDENCIES[stage]
        dependencies = () if dependency_stage is None else (EXPERIMENT_NODE_IDS[dependency_stage],)
        consumed_channels = ()
        if dependency_stage is not None:
            consumed_channels = (
                ChannelDependency(
                    EXPERIMENT_NODE_IDS[dependency_stage],
                    _stage_channel_id(dependency_stage),
                    _STAGE_OUTPUT_SCHEMA_VERSIONS[dependency_stage],
                ),
            )
        specs.append(
            SemanticNodeSpec(
                analysis_id=EXPERIMENT_NODE_IDS[stage],
                scope=AnalysisScope.EXPERIMENT_STAGE,
                dependencies=dependencies,
                consumed_channels=consumed_channels,
                produced_channels=(
                    ChannelSpec(
                        _stage_channel_id(stage),
                        _STAGE_OUTPUT_SCHEMA_VERSIONS[stage],
                    ),
                ),
                semantic_config_keys=("stage_config_hash",),
                algorithm_version=_STAGE_ALGORITHM_VERSIONS[stage],
                output_schema_version=_STAGE_OUTPUT_SCHEMA_VERSIONS[stage],
                task_scope="experiment",
                validator_id=_VALIDATOR_ID,
            )
        )
    preprocess_id = EXPERIMENT_NODE_IDS["preprocess"]
    preprocess_schema = _STAGE_OUTPUT_SCHEMA_VERSIONS["preprocess"]
    for stage in LEGACY_EXPERIMENT_LEAVES:
        specs.append(
            SemanticNodeSpec(
                analysis_id=LEGACY_EXPERIMENT_NODE_IDS[stage],
                scope=AnalysisScope.EXPERIMENT_STAGE,
                dependencies=(preprocess_id,),
                consumed_channels=(
                    ChannelDependency(
                        preprocess_id,
                        _stage_channel_id("preprocess"),
                        preprocess_schema,
                    ),
                ),
                produced_channels=(ChannelSpec(f"experiment.{stage}.legacy", 1),),
                semantic_config_keys=("stage_config_hash",),
                algorithm_version="legacy-1",
                output_schema_version=1,
                task_scope="experiment",
                validator_id=_LEGACY_VALIDATOR_ID,
            )
        )
    return tuple(specs)


def resolve_experiment_target(cfg: Any, target: str) -> str:
    """Resolve a public experiment target alias to one semantic analysis ID."""
    normalized = str(target).strip().lower()
    if normalized == "full":
        normalized = "latent" if bool(getattr(cfg, "full_run_latent", True)) else "hmm"
    elif normalized == "variant":
        normalized = "preprocess"
    try:
        return EXPERIMENT_NODE_IDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown experiment target {target!r}; expected one of {EXPERIMENT_TARGETS}"
        ) from exc


def experiment_stage_result_metadata(
    stage: str,
    *,
    stage_config_hash: str | None,
    input_artifact_ids: list[str] | tuple[str, ...],
    artifacts: Mapping[str, Any],
    schema_versions: Mapping[str, int],
) -> dict[str, Any]:
    """Return semantic identity fields stored beside a completed coarse stage."""
    if stage not in EXPERIMENT_STAGES:
        return {}
    payload = {
        "analysis_id": EXPERIMENT_NODE_IDS[stage],
        "algorithm_version": _STAGE_ALGORITHM_VERSIONS[stage],
        "output_schema_version": _STAGE_OUTPUT_SCHEMA_VERSIONS[stage],
        "stage_config_hash": stage_config_hash,
        "input_artifact_ids": list(input_artifact_ids),
        "artifacts": dict(artifacts),
        "schema_versions": dict(schema_versions),
    }
    fingerprint = _stable_payload_digest(payload)
    return {
        "semantic_analysis_id": EXPERIMENT_NODE_IDS[stage],
        "semantic_algorithm_version": _STAGE_ALGORITHM_VERSIONS[stage],
        "semantic_output_schema_version": _STAGE_OUTPUT_SCHEMA_VERSIONS[stage],
        "semantic_result_id": f"{stage}:{fingerprint}",
        "semantic_channel_fingerprint": fingerprint,
    }


def build_experiment_plan(
    cfg: Any,
    target: str,
    *,
    paths: Any | None = None,
) -> SemanticPlan:
    """Build a read-only semantic plan from a loaded experiment config."""
    return _build_experiment_context(cfg, target, paths=paths).plan


def _build_experiment_context(
    cfg: Any,
    target: str,
    *,
    paths: Any | None = None,
    executors: Mapping[str, NodeExecutor] | None = None,
) -> _ExperimentPlanningContext:
    """Adapt current experiment lifecycle records into a read-only semantic plan."""
    from ..cli.helpers import get_adata_paths

    paths = get_adata_paths(cfg) if paths is None else paths
    run_root = Path(cfg.output_directory)
    manifest = read_experiment_manifest(run_root)
    specs = experiment_node_specs()
    spec_by_id = {spec.analysis_id: spec for spec in specs}

    def validate_stage(result: NodeResult) -> ArtifactValidation:
        stage = _stage_from_analysis_id(result.analysis_id)
        valid = stage_is_complete(
            run_root,
            stage,
            required_artifacts=PARTITIONED_STAGE_REQUIRED_ARTIFACTS[stage],
        )
        if valid and stage == "preprocess":
            from ..preprocessing.preprocess_generation import (
                PreprocessGenerationError,
                resolve_current_preprocess_generation,
            )

            try:
                current = resolve_current_preprocess_generation(run_root / PREPROCESS_DIR)
            except PreprocessGenerationError:
                valid = False
            else:
                valid = current is not None and current[1].get("generation_id") == (
                    manifest.get("stages", {}).get("preprocess", {}).get("generation_id")
                )
        return ArtifactValidation(
            valid=valid,
            reason_code="stage_artifact_validation_failed",
            reason=f"published {stage} stage artifacts failed lifecycle validation",
        )

    def validate_legacy(result: NodeResult) -> ArtifactValidation:
        stage = _legacy_stage_from_analysis_id(result.analysis_id)
        path = getattr(paths, stage, None)
        valid = path is not None and Path(path).is_file()
        return ArtifactValidation(
            valid=valid,
            reason_code="legacy_artifact_missing",
            reason=f"legacy {stage} output is unavailable",
        )

    registry = AnalysisRegistry(
        specs,
        validators={
            _VALIDATOR_ID: validate_stage,
            _LEGACY_VALIDATOR_ID: validate_legacy,
        },
        executors=executors,
    )
    resolved_target = resolve_experiment_target(cfg, target)
    requested_nodes = registry.dependency_closure((resolved_target,))
    scope_identity = _experiment_scope_identity(cfg, manifest)
    expected_inputs = {
        EXPERIMENT_NODE_IDS[stage]: _expected_stage_inputs(
            cfg,
            paths,
            stage,
            scope_identity=scope_identity,
        )
        for stage in EXPERIMENT_STAGES
        if EXPERIMENT_NODE_IDS[stage] in requested_nodes
    }
    current_results = _current_stage_results(
        cfg,
        manifest,
        specs=spec_by_id,
        scope_identity=scope_identity,
    )
    for stage in EXPERIMENT_STAGES:
        if _stage_is_forced(cfg, stage):
            current_results.pop(EXPERIMENT_NODE_IDS[stage], None)
    plan = SemanticPlanner(registry).plan(
        resolved_target,
        inputs_by_node=expected_inputs,
        current_results=current_results,
        current_generation_id=_current_generation_id(manifest, resolved_target),
    )
    return _ExperimentPlanningContext(
        cfg=cfg,
        paths=paths,
        registry=registry,
        plan=plan,
    )


def plan_experiment(config_path: str | Path, target: str = "full") -> SemanticPlan:
    """Load an experiment config and return its read-only semantic plan."""
    from ..cli.helpers import load_experiment_config

    cfg = load_experiment_config(str(config_path))
    return build_experiment_plan(cfg, target)


def build_experiment_upgrade_impact(
    cfg: Any,
    target: str = "full",
    *,
    paths: Any | None = None,
) -> UpgradeImpactReport:
    """Build upgrade impact with prior stage timings where they are available."""
    plan = build_experiment_plan(cfg, target, paths=paths)
    manifest = read_experiment_manifest(Path(cfg.output_directory))
    stages = manifest.get("stages", {})
    historical_seconds: dict[str, float] = {}
    if isinstance(stages, Mapping):
        for stage, analysis_id in EXPERIMENT_NODE_IDS.items():
            entry = stages.get(stage, {})
            if not isinstance(entry, Mapping):
                continue
            timing_entry = entry
            if entry.get("state") != "complete" and isinstance(
                entry.get("previous_complete"), Mapping
            ):
                timing_entry = entry["previous_complete"]
            timings = timing_entry.get("timings", {})
            elapsed = timings.get("elapsed_seconds") if isinstance(timings, Mapping) else None
            if elapsed is not None:
                historical_seconds[analysis_id] = elapsed
    return build_upgrade_impact(
        plan,
        scope="experiment",
        historical_seconds=historical_seconds,
    )


def plan_experiment_upgrade_impact(
    config_path: str | Path, target: str = "full"
) -> UpgradeImpactReport:
    """Load an experiment config and return installed-code upgrade impact."""
    from ..cli.helpers import load_experiment_config

    cfg = load_experiment_config(str(config_path))
    return build_experiment_upgrade_impact(cfg, target)


def execute_experiment_target(
    config_path: str | Path,
    target: str,
    *,
    cfg: Any | None = None,
    paths: Any | None = None,
    stage_runners: Mapping[str, Any] | None = None,
) -> ExperimentExecutionResult:
    """Execute non-compatible nodes in the semantic plan using existing stage wrappers."""
    from ..cli.helpers import load_experiment_config

    config_path = str(config_path)
    cfg = load_experiment_config(config_path) if cfg is None else cfg
    runners = _default_stage_runners() if stage_runners is None else dict(stage_runners)
    missing_runners = sorted(set(EXPERIMENT_STAGES).difference(runners))
    if missing_runners:
        raise ValueError(f"stage_runners lacks experiment stages: {missing_runners}")
    executors = {
        EXPERIMENT_NODE_IDS[stage]: partial(runners[stage], config_path)
        for stage in EXPERIMENT_STAGES
    }
    context = _build_experiment_context(
        cfg,
        target,
        paths=paths,
        executors=executors,
    )
    blocked = [
        decision
        for decision in context.plan.decisions
        if decision.state is PlanState.BLOCKED_MISSING_INPUT
    ]
    if blocked:
        reasons = "; ".join(f"{decision.analysis_id}: {decision.reason}" for decision in blocked)
        raise RuntimeError(f"experiment target is blocked by missing input: {reasons}")

    results: list[tuple[str, Any]] = []
    for decision in context.plan.decisions:
        stage = _stage_from_analysis_id(decision.analysis_id)
        if decision.state is PlanState.COMPATIBLE:
            result = _compatible_stage_result(stage, context.paths, cfg)
        else:
            executor = context.registry.executor_for(decision.analysis_id)
            if executor is None:
                raise RuntimeError(
                    f"no executor is registered for incompatible node {decision.analysis_id!r}"
                )
            result = executor()
        results.append((stage, result))
    final_result = results[-1][1] if results else None
    return ExperimentExecutionResult(
        plan=context.plan,
        stage_results=tuple(results),
        final_result=final_result,
    )


def format_experiment_plan(plan: SemanticPlan) -> str:
    """Render a compact deterministic human-readable experiment plan."""
    lines = [
        f"Experiment target: {plan.requested_target}",
        f"Graph definition version: {plan.graph_definition_version}",
    ]
    if plan.current_generation_id is not None:
        lines.append(f"Current generation: {plan.current_generation_id}")
    lines.append("")
    for index, decision in enumerate(plan.decisions, start=1):
        lines.append(
            f"{index:>2}. {decision.state.value:<23} {decision.analysis_id} — {decision.reason}"
        )
    return "\n".join(lines)


def _default_stage_runners() -> dict[str, Any]:
    from ..cli.hmm_adata import hmm_adata
    from ..cli.latent_adata import latent_adata
    from ..cli.preprocess_adata import preprocess_adata
    from ..cli.raw_adata import raw_adata
    from ..cli.spatial_adata import spatial_adata

    return {
        "raw": raw_adata,
        "preprocess": preprocess_adata,
        "spatial": spatial_adata,
        "hmm": hmm_adata,
        "latent": latent_adata,
    }


def _stage_channel_id(stage: str) -> str:
    return f"experiment.{stage}.output"


def _stage_from_analysis_id(analysis_id: str) -> str:
    for stage, node_id in EXPERIMENT_NODE_IDS.items():
        if node_id == analysis_id:
            return stage
    raise ValueError(f"analysis_id {analysis_id!r} is not a coarse experiment stage")


def _legacy_stage_from_analysis_id(analysis_id: str) -> str:
    for stage, node_id in LEGACY_EXPERIMENT_NODE_IDS.items():
        if node_id == analysis_id:
            return stage
    raise ValueError(f"analysis_id {analysis_id!r} is not a legacy experiment leaf")


def _experiment_scope_identity(cfg: Any, manifest: Mapping[str, Any]) -> str:
    experiment_uid = manifest.get("experiment_uid")
    if experiment_uid:
        return f"experiment_uid:{experiment_uid}"
    experiment_name = manifest.get("experiment") or getattr(cfg, "experiment_name", None)
    if experiment_name:
        return f"experiment_name:{experiment_name}"
    return f"run:{Path(cfg.output_directory).name}"


def _logical_task_digest(stage: str) -> str:
    return f"experiment-stage:{stage}:logical-v1"


def _stage_is_forced(cfg: Any, stage: str) -> bool:
    return any(bool(getattr(cfg, name, False)) for name in _STAGE_FORCE_FLAGS[stage])


def _expected_stage_inputs(
    cfg: Any,
    paths: Any,
    stage: str,
    *,
    scope_identity: str,
) -> NodeInputs:
    from ..cli.helpers import raw_input_artifact_ids, stage_config_hash, stage_input_artifact_ids

    source_path = _source_path_for_stage(cfg, paths, stage)
    if stage == "raw":
        input_ids = raw_input_artifact_ids(cfg)
    else:
        input_ids = (
            stage_input_artifact_ids(
                cfg.output_directory,
                source_path,
                include_region_catalogs=stage == "latent",
            )
            if source_path is not None
            else []
        )
    return NodeInputs(
        semantic_config={"stage_config_hash": stage_config_hash(cfg, stage)},
        input_artifacts=_artifact_identities(input_ids),
        logical_scope_identity=scope_identity,
        logical_task_plan_digest=_logical_task_digest(stage),
    )


def _source_path_for_stage(cfg: Any, paths: Any, stage: str) -> Path | None:
    if stage == "raw":
        return None
    candidates: tuple[Any, ...]
    if stage == "preprocess":
        candidates = (
            getattr(paths, "spine", None),
            getattr(paths, "raw_spine", None),
            getattr(paths, "raw", None),
        )
    elif stage == "spatial":
        candidates = (
            getattr(paths, "preprocess_spine", None),
            getattr(paths, "pp_dedup", None),
            getattr(paths, "pp", None),
        )
    elif stage == "hmm":
        candidates = (
            getattr(paths, "spatial_spine", None),
            getattr(paths, "preprocess_spine", None),
            getattr(paths, "spatial", None),
            getattr(paths, "pp_dedup", None),
            getattr(paths, "pp", None),
        )
    elif stage == "latent":
        requested = getattr(cfg, "from_adata_stage", None)
        if requested is not None:
            requested = str(requested).strip().lower()
            aliases = {
                "pp": "preprocess",
                "pp_dedup": "preprocess",
                "preprocess_dedup": "preprocess",
            }
            canonical = aliases.get(requested, requested)
            requested_attr = {
                "preprocess": "preprocess_spine",
                "spatial": "spatial_spine",
                "hmm": "hmm_spine",
            }.get(canonical)
            if requested_attr is not None:
                candidates = (getattr(paths, requested_attr, None),)
            else:
                candidates = ()
        else:
            candidates = (
                getattr(paths, "hmm_spine", None),
                getattr(paths, "spatial_spine", None),
                getattr(paths, "preprocess_spine", None),
                getattr(paths, "hmm", None),
                getattr(paths, "spatial", None),
                getattr(paths, "pp_dedup", None),
                getattr(paths, "pp", None),
            )
    else:
        raise ValueError(f"unknown experiment stage {stage!r}")
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _artifact_identities(values: list[str] | tuple[str, ...]) -> tuple[ArtifactIdentity, ...]:
    identities = []
    for index, value in enumerate(values):
        text = str(value)
        if text.startswith("input-manifest:"):
            identities.append(
                ArtifactIdentity("input-manifest", text.removeprefix("input-manifest:"))
            )
        elif text.startswith("alignment-reference-bundle:"):
            identities.append(
                ArtifactIdentity(
                    "alignment-reference-bundle",
                    text.removeprefix("alignment-reference-bundle:"),
                )
            )
        elif text.startswith("source:"):
            _, source_id, checksum = text.split(":", 2)
            identities.append(ArtifactIdentity(f"source:{source_id}", checksum))
        else:
            identities.append(
                ArtifactIdentity(
                    artifact_id=f"stage-input:{index}",
                    checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(identities)


def _current_stage_results(
    cfg: Any,
    manifest: Mapping[str, Any],
    *,
    specs: Mapping[str, SemanticNodeSpec],
    scope_identity: str,
) -> dict[str, NodeResult]:
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        return {}
    current: dict[str, NodeResult] = {}
    for stage in EXPERIMENT_STAGES:
        analysis_id = EXPERIMENT_NODE_IDS[stage]
        spec = specs[analysis_id]
        entry = stages.get(stage)
        if not isinstance(entry, Mapping) or not entry.get("config_hash"):
            continue
        dependencies: list[DependencyResultIdentity] = []
        missing_dependency = False
        for dependency_id in spec.dependencies:
            dependency = current.get(dependency_id)
            if dependency is None:
                missing_dependency = True
                break
            consumed = [
                channel
                for channel in spec.consumed_channels
                if channel.analysis_id == dependency_id
            ]
            fingerprints: list[ChannelFingerprint] = []
            for channel in consumed:
                fingerprint = dependency.channel(channel.channel_id)
                if fingerprint is None or fingerprint.schema_version != channel.schema_version:
                    missing_dependency = True
                    break
                fingerprints.append(fingerprint)
            if missing_dependency:
                break
            dependencies.append(
                DependencyResultIdentity(
                    analysis_id=dependency_id,
                    result_id=dependency.result_id,
                    channel_fingerprints=tuple(fingerprints),
                )
            )
        if missing_dependency:
            continue
        stored_algorithm = str(entry.get("semantic_algorithm_version", spec.algorithm_version))
        stored_output_schema = int(
            entry.get(
                "semantic_output_schema_version",
                _stored_stage_schema_version(stage, entry, spec.output_schema_version),
            )
        )
        stored_spec = replace(
            spec,
            algorithm_version=stored_algorithm,
            output_schema_version=stored_output_schema,
            produced_channels=(ChannelSpec(_stage_channel_id(stage), stored_output_schema),),
        )
        state = entry.get("state")
        if state is None and entry.get("completed_at"):
            state = "complete"
        if state not in {"planned", "running", "complete", "failed"}:
            continue
        inputs = NodeInputs(
            semantic_config={"stage_config_hash": str(entry["config_hash"])},
            input_artifacts=_artifact_identities(
                tuple(map(str, entry.get("input_artifact_ids", [])))
            ),
            dependency_results=tuple(dependencies),
            logical_scope_identity=scope_identity,
            logical_task_plan_digest=_logical_task_digest(stage),
        )
        result_id = str(
            entry.get("semantic_result_id")
            or entry.get("generation_id")
            or f"{stage}:{_stage_entry_digest(stage, entry)}"
        )
        channel_fingerprint = str(
            entry.get("semantic_channel_fingerprint") or _stage_entry_digest(stage, entry)
        )
        current[analysis_id] = node_result_from_inputs(
            stored_spec,
            inputs,
            result_id=result_id,
            produced_channels=(
                ChannelFingerprint(
                    _stage_channel_id(stage),
                    stored_output_schema,
                    channel_fingerprint,
                ),
            ),
            state=str(state),
            started_at=_optional_string(entry.get("started_at")),
            completed_at=_optional_string(entry.get("completed_at")),
            execution_provenance=(
                ("manifest_schema_version", str(manifest.get("schema_version", "legacy"))),
            ),
        )
    return current


def _stored_stage_schema_version(
    stage: str,
    entry: Mapping[str, Any],
    fallback: int,
) -> int:
    schemas = entry.get("schema_versions", {})
    if isinstance(schemas, Mapping) and stage in schemas:
        return int(schemas[stage])
    return int(fallback)


def _stage_entry_digest(stage: str, entry: Mapping[str, Any]) -> str:
    artifacts = entry.get("artifacts", {})
    payload = {
        "stage": stage,
        "config_hash": entry.get("config_hash"),
        "input_artifact_ids": entry.get("input_artifact_ids", []),
        "schema_versions": entry.get("schema_versions", {}),
        "artifacts": artifacts if isinstance(artifacts, Mapping) else {},
        "generation_id": entry.get("generation_id"),
    }
    return _stable_payload_digest(payload)


def _stable_payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _current_generation_id(
    manifest: Mapping[str, Any],
    target_analysis_id: str,
) -> str | None:
    stage = _stage_from_analysis_id(target_analysis_id)
    entry = manifest.get("stages", {}).get(stage, {})
    if not isinstance(entry, Mapping):
        return None
    generation_id = entry.get("generation_id")
    return None if generation_id is None else str(generation_id)


def _compatible_stage_result(stage: str, paths: Any, cfg: Any) -> Any:
    path = getattr(paths, _STAGE_OUTPUT_PATH_ATTRIBUTES[stage], None)
    resolved_path = None if path is None else Path(path)
    if stage == "raw":
        return None, resolved_path, cfg
    if stage == "preprocess":
        return resolved_path, None
    return None, resolved_path
