"""Resolve model selectors once to validated immutable model artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..artifacts import (
    ModelManifest,
    resolve_model_alias,
    validate_published_bundle,
)
from ..workspace import MLWorkspace
from .contracts import (
    MLJobServiceError,
    ModelMetricCandidate,
    ModelSelectionRequest,
    ResolvedModelSelection,
)


def _load_exact(workspace: MLWorkspace, model_id: str) -> ModelManifest:
    bundle = validate_published_bundle(
        workspace,
        workspace.model_dir(model_id),
        kind="model",
        expected_id=model_id,
    )
    with (bundle.path / "model_manifest.json").open(encoding="utf-8") as handle:
        return ModelManifest.from_dict(json.load(handle))


def resolve_model_selection(
    workspace: MLWorkspace,
    request: ModelSelectionRequest,
    *,
    candidates: Sequence[ModelMetricCandidate] = (),
) -> ResolvedModelSelection:
    """Resolve an exact ID, mutable alias, or held-out metric selector once.

    Alias and metric selectors are converted to an immutable model ID before a
    job run is created. Best-from-run selection accepts explicit stored metric
    candidates and never scans directories or recomputes model performance.
    """
    if request.kind == "exact":
        assert request.model_id is not None
        manifest = _load_exact(workspace, request.model_id)
        return ResolvedModelSelection(manifest=manifest, selection_kind="exact")
    if request.kind == "alias":
        assert request.alias is not None
        aliased = resolve_model_alias(workspace, request.alias)
        manifest = _load_exact(workspace, aliased.model_id)
        return ResolvedModelSelection(manifest=manifest, selection_kind="alias")

    assert request.source_run_id is not None
    assert request.metric_name is not None
    assert request.direction is not None
    eligible = [
        candidate
        for candidate in candidates
        if candidate.source_run_id == request.source_run_id
        and candidate.metric_name == request.metric_name
        and candidate.cohort == request.cohort
    ]
    if not eligible:
        raise MLJobServiceError("best-from-run selector has no matching validation metrics")
    reverse = request.direction == "maximize"
    ordered = sorted(
        eligible,
        key=lambda item: (
            -item.metric_value if reverse else item.metric_value,
            item.model_id,
        ),
    )
    winner = ordered[0]
    manifest = _load_exact(workspace, winner.model_id)
    if manifest.originating_run_id != winner.source_run_id:
        raise MLJobServiceError(
            "metric candidate source run differs from the model artifact lineage"
        )
    return ResolvedModelSelection(
        manifest=manifest,
        selection_kind="best_from_run",
        metric_name=winner.metric_name,
        metric_value=float(winner.metric_value),
        metric_direction=request.direction,
        metric_cohort=winner.cohort,
    )
