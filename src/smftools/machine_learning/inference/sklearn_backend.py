"""Backend-neutral application records for fitted sklearn models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.partition_dataset import MLMaterializedPartitionData, MLPartitionBatch
from ..training.sklearn_backend import FittedSklearnModel


@dataclass(frozen=True)
class SklearnPredictionResult:
    """Ordered sklearn predictions for one immutable cohort or batch."""

    molecule_uids: tuple[str, ...]
    class_ids: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    class_order: tuple[str, ...]
    split: str

    def __post_init__(self) -> None:
        n_rows = len(self.molecule_uids)
        n_classes = len(self.class_order)
        for name, shape in (
            ("class_ids", (n_rows,)),
            ("scores", (n_rows, n_classes)),
            ("probabilities", (n_rows, n_classes)),
        ):
            array = np.asarray(getattr(self, name)).copy()
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def apply_sklearn_partition_model(
    model: FittedSklearnModel,
    data: MLMaterializedPartitionData | MLPartitionBatch,
    *,
    phase: str | None = None,
) -> SklearnPredictionResult:
    """Transform and apply a fitted sklearn model without mutating input data."""
    data_split = getattr(data, "split", None)
    resolved_phase = phase or ("inference" if data.labels is None else data_split)
    if resolved_phase not in {"train", "validation", "test", "inference"}:
        raise ValueError(f"unsupported prediction phase {resolved_phase!r}")
    features = model.transform.transform(data)
    predictor = model.predictor
    return SklearnPredictionResult(
        molecule_uids=tuple(data.molecule_uids),
        class_ids=predictor.predict(features, phase=resolved_phase),
        scores=predictor.predict_scores(features, phase=resolved_phase),
        probabilities=predictor.predict_probabilities(features, phase=resolved_phase),
        class_order=model.label_schema.class_order,
        split=data_split or resolved_phase,
    )
