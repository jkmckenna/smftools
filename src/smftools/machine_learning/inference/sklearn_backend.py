"""Backend-neutral application of fitted sklearn models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data.partition_dataset import MLMaterializedPartitionData, MLPartitionBatch
from ..evaluation.contracts import PredictionResult
from ..training.sklearn_backend import FittedSklearnModel

SklearnPredictionResult = PredictionResult


def apply_sklearn_partition_model(
    model: FittedSklearnModel,
    data: MLMaterializedPartitionData | MLPartitionBatch,
    *,
    phase: str | None = None,
    cohort: str | None = None,
    groups: Sequence[str | None] | None = None,
    model_id: str | None = None,
) -> PredictionResult:
    """Transform and apply a fitted sklearn model without mutating input data."""
    data_split = getattr(data, "split", None)
    resolved_phase = phase or ("inference" if data.labels is None else data_split)
    if resolved_phase not in {"train", "validation", "test", "inference"}:
        raise ValueError(f"unsupported prediction phase {resolved_phase!r}")
    features = model.transform.transform(data)
    predictor = model.predictor
    return PredictionResult(
        molecule_uids=tuple(data.molecule_uids),
        class_ids=predictor.predict(features, phase=resolved_phase),
        scores=predictor.predict_scores(features, phase=resolved_phase),
        probabilities=predictor.predict_probabilities(features, phase=resolved_phase),
        class_order=model.label_schema.class_order,
        split=data_split or resolved_phase,
        experiment_uids=tuple(data.experiment_uids),
        modalities=tuple(data.modalities),
        groups=None if groups is None else tuple(groups),
        truth_class_ids=None if data.labels is None else np.asarray(data.labels),
        positive_class=model.label_schema.positive_class,
        cohort=cohort,
        model_id=model_id,
    )
