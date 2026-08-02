"""Framework-independent orchestration for resolved smftools ML jobs."""

from .actions import (
    SklearnTrainOptions,
    TorchTrainOptions,
    apply_partition_model,
    evaluate_prediction_result,
    explain_partition_model,
    train_partition_model,
)
from .contracts import (
    JobArtifact,
    JobCancellationToken,
    JobDryRun,
    JobExecutionContext,
    JobExecutionOutcome,
    JobOperationResult,
    MLJobCancellationRequested,
    MLJobCancelledError,
    MLJobExecutionError,
    MLJobServiceError,
    ModelMetricCandidate,
    ModelSelectionRequest,
    ResolvedJob,
    ResolvedModelSelection,
)
from .planning import MLWorkflowDryRun, MLWorkflowPlanningError, plan_ml_workflow
from .resolution import resolve_model_selection
from .service import (
    dry_run_job,
    run_apply_job,
    run_evaluate_job,
    run_explain_job,
    run_plot_job,
    run_train_job,
)

__all__ = [
    "JobArtifact",
    "JobCancellationToken",
    "JobDryRun",
    "JobExecutionContext",
    "JobExecutionOutcome",
    "JobOperationResult",
    "MLJobCancellationRequested",
    "MLJobCancelledError",
    "MLJobExecutionError",
    "MLJobServiceError",
    "MLWorkflowDryRun",
    "MLWorkflowPlanningError",
    "ModelMetricCandidate",
    "ModelSelectionRequest",
    "ResolvedJob",
    "ResolvedModelSelection",
    "SklearnTrainOptions",
    "TorchTrainOptions",
    "apply_partition_model",
    "dry_run_job",
    "evaluate_prediction_result",
    "explain_partition_model",
    "resolve_model_selection",
    "plan_ml_workflow",
    "run_apply_job",
    "run_evaluate_job",
    "run_explain_job",
    "run_plot_job",
    "run_train_job",
    "train_partition_model",
]
