"""Backend-neutral application records for fitted plain-Torch models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.partition_dataset import MLMaterializedPartitionData, MLPartitionBatch
from ..data.transforms import TorchFeatureTransform
from ..evaluation.contracts import PredictionResult
from ..training.torch_backend import FittedTorchModel

TorchPredictionResult = PredictionResult


def _mask_arrays(model: FittedTorchModel, data: object) -> Mapping[str, Any]:
    values = {
        "observed": data.observed_mask,
        "availability": data.availability_mask,
        "design": data.design_mask,
        "padding": data.padding_mask,
    }
    return {
        mask.name: values[mask.kind] for mask in model.input_schema.masks if mask.kind in values
    }


def apply_torch_partition_model(
    model: FittedTorchModel,
    data: MLMaterializedPartitionData | MLPartitionBatch,
    *,
    phase: str | None = None,
    cohort: str | None = None,
    groups: Sequence[str | None] | None = None,
    model_id: str | None = None,
) -> PredictionResult:
    """Transform and apply a fitted plain-Torch model without mutating data."""
    data_split = getattr(data, "split", None)
    if phase is None and data.labels is not None and data_split is None:
        raise ValueError("phase is required for labeled partition batches without a split field")
    resolved_phase = phase or ("inference" if data.labels is None else data_split)
    if resolved_phase not in {"train", "validation", "test", "inference"}:
        raise ValueError(f"unsupported prediction phase {resolved_phase!r}")
    transformed = TorchFeatureTransform(model.transform, device="cpu")(data)
    values = transformed.values.detach().cpu().numpy()
    masks = _mask_arrays(model, data)
    predictor = model.predictor
    return PredictionResult(
        molecule_uids=tuple(data.molecule_uids),
        class_ids=predictor.predict(values, masks=masks, phase=resolved_phase),
        scores=predictor.predict_scores(values, masks=masks, phase=resolved_phase),
        probabilities=predictor.predict_probabilities(
            values,
            masks=masks,
            phase=resolved_phase,
        ),
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
