"""Read-only explanation layout and manifest conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..artifacts import ArtifactReference, ExplanationManifest, ResolvedDefinition
from ..contracts import InputSchema
from ..workspace import MLRunPaths
from .contracts import (
    METHOD_CONTRACTS,
    AttributionResult,
    InterpretabilityContractError,
    _digest,
    _thaw_json,
)


@dataclass(frozen=True)
class ExplanationArtifactLayout:
    """Deterministic artifact paths for one immutable explanation result."""

    result_id: str
    root: Path
    manifest: Path
    values: Path
    feature_summary: Path
    group_summary: Path
    plots: Path

    def __post_init__(self) -> None:
        _digest(self.result_id, "layout.result_id")
        root = self.root.resolve()
        object.__setattr__(self, "root", root)
        expected = {
            "manifest": root / "manifest.json",
            "values": root / "values.zarr",
            "feature_summary": root / "feature_summary.parquet",
            "group_summary": root / "group_summary.parquet",
            "plots": root / "plots",
        }
        for name, path in expected.items():
            if getattr(self, name).resolve() != path:
                raise InterpretabilityContractError(
                    f"explanation layout {name} must resolve to {path}"
                )
            object.__setattr__(self, name, path)

    @classmethod
    def resolve(
        cls,
        run_paths: MLRunPaths,
        result_id: str,
    ) -> ExplanationArtifactLayout:
        """Resolve the canonical layout without creating any paths."""
        root = run_paths.explanation_dir(result_id)
        return cls(
            result_id=result_id,
            root=root,
            manifest=root / "manifest.json",
            values=root / "values.zarr",
            feature_summary=root / "feature_summary.parquet",
            group_summary=root / "group_summary.parquet",
            plots=root / "plots",
        )


def resolved_explanation_method(result: AttributionResult) -> ResolvedDefinition:
    """Embed all computation choices that affect an explanation into its identity."""
    request = result.request
    contract = METHOD_CONTRACTS[request.method]
    return ResolvedDefinition.create(
        name=request.method,
        version=contract.version,
        parameters={
            "request_id": request.request_id,
            "input_schema_hash": request.input_schema_hash,
            "split_role": request.split_role,
            "layer": request.layer,
            "aggregation": request.aggregation.to_dict(),
            "decision": request.decision.to_dict(),
            "method_parameters": _thaw_json(request.parameters),
            "random_seed": request.random_seed,
            "result_axes": list(result.axes),
            "convergence_delta_recorded": result.convergence_delta is not None,
        },
    )


def create_explanation_manifest(
    result: AttributionResult,
    *,
    input_schema: InputSchema,
    run_id: str,
    workspace_id: str,
    values: ArtifactReference,
    summary: ArtifactReference | None,
    created_at: str,
) -> ExplanationManifest:
    """Create the existing durable manifest from a validated runtime result."""
    result.validate_against(input_schema)
    values_path = PurePosixPath(values.relative_path)
    if values.media_type != "application/vnd.zarr" or values_path.parts[-3:] != (
        "explanations",
        result.result_id,
        "values.zarr",
    ):
        raise InterpretabilityContractError(
            "explanation values must be a Zarr artifact in the canonical result directory"
        )
    if summary is not None:
        summary_path = PurePosixPath(summary.relative_path)
        if summary.media_type != "application/vnd.apache.parquet" or summary_path.parts[-3:] != (
            "explanations",
            result.result_id,
            "feature_summary.parquet",
        ):
            raise InterpretabilityContractError(
                "explanation summary must use the canonical result directory"
            )
    request = result.request
    return ExplanationManifest.create(
        run_id=run_id,
        workspace_id=workspace_id,
        model_id=request.model_id,
        dataset_snapshot_id=request.dataset_snapshot_id,
        cohort=request.cohort,
        n_observations=len(request.observation_uids),
        method=resolved_explanation_method(result),
        target=request.target,
        baseline=request.baseline,
        mask_policy=request.mask_policy,
        feature_axes=tuple(axis for axis in result.axes if axis != "observation"),
        values=values,
        summary=summary,
        created_at=created_at,
    )
