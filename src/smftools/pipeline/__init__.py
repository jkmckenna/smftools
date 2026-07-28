"""Engine-neutral scientific dependency and compatibility planning."""

from .analysis_registry import AnalysisRegistry, RegistryError
from .compatibility import SemanticPlanner, compatibility_fingerprint, node_result_from_inputs
from .semantic_graph import (
    AnalysisScope,
    ArtifactIdentity,
    ArtifactRecord,
    ArtifactValidation,
    ChannelDependency,
    ChannelFingerprint,
    ChannelSpec,
    CompatibilityFingerprint,
    DependencyResultIdentity,
    NodeInputs,
    NodeResult,
    PlanDecision,
    PlanState,
    SemanticNodeSpec,
    SemanticPlan,
)

__all__ = [
    "AnalysisRegistry",
    "AnalysisScope",
    "ArtifactIdentity",
    "ArtifactRecord",
    "ArtifactValidation",
    "ChannelDependency",
    "ChannelFingerprint",
    "ChannelSpec",
    "CompatibilityFingerprint",
    "DependencyResultIdentity",
    "NodeInputs",
    "NodeResult",
    "PlanDecision",
    "PlanState",
    "RegistryError",
    "SemanticNodeSpec",
    "SemanticPlan",
    "SemanticPlanner",
    "compatibility_fingerprint",
    "node_result_from_inputs",
]
