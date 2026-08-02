"""Backend-neutral explanation contracts; computation adapters are added separately."""

from .artifacts import (
    ExplanationArtifactLayout,
    create_explanation_manifest,
    resolved_explanation_method,
)
from .background import BackgroundReference, sample_training_background
from .classical import explain_sklearn_model
from .contracts import (
    AGGREGATION_REDUCTIONS,
    EXPLANATION_SPLITS,
    INTERPRETABILITY_SCHEMA_VERSION,
    METHOD_CONTRACTS,
    AttributionAggregation,
    AttributionFeature,
    AttributionResult,
    ExplanationDecisionProvenance,
    ExplanationMethodContract,
    InterpretabilityContractError,
    InterpretabilityRequest,
    validate_interpretability_request,
)

__all__ = [
    "AGGREGATION_REDUCTIONS",
    "EXPLANATION_SPLITS",
    "INTERPRETABILITY_SCHEMA_VERSION",
    "METHOD_CONTRACTS",
    "AttributionAggregation",
    "AttributionFeature",
    "AttributionResult",
    "BackgroundReference",
    "ExplanationArtifactLayout",
    "ExplanationDecisionProvenance",
    "ExplanationMethodContract",
    "InterpretabilityContractError",
    "InterpretabilityRequest",
    "create_explanation_manifest",
    "explain_sklearn_model",
    "resolved_explanation_method",
    "sample_training_background",
    "validate_interpretability_request",
]
