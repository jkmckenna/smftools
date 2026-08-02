"""Adapters from backend fit results to the shared training-event schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import TrainingEvent, TrainingHistory

if TYPE_CHECKING:
    from ..training.sklearn_backend import FittedSklearnModel
    from ..training.torch_backend import FittedTorchModel


def sklearn_training_history(model: FittedSklearnModel) -> TrainingHistory:
    """Represent a one-shot sklearn fit as one non-epoch event."""
    return TrainingHistory(
        backend="sklearn",
        events=(
            TrainingEvent(
                event_index=0,
                event_type=f"{model.fit_mode}_completed",
                metrics={},
            ),
        ),
    )


def torch_training_history(model: FittedTorchModel) -> TrainingHistory:
    """Convert persisted plain-Torch epoch losses to shared events."""
    return TrainingHistory(
        backend="torch",
        events=tuple(
            TrainingEvent(
                event_index=index,
                event_type="epoch_completed",
                epoch=record.epoch,
                metrics={
                    "train_loss": record.train_loss,
                    "validation_loss": record.validation_loss,
                },
            )
            for index, record in enumerate(model.history)
        ),
    )
