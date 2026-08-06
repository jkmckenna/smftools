from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .sklearn_backend import (
    FittedSklearnModel,
    SklearnTrainingError,
    SklearnTrainingResult,
    fit_sklearn_partition_model,
    fit_sklearn_partition_model_streaming,
)
from .torch_backend import (
    TORCH_TRAINING_CONFIG_VERSION,
    ClassificationTask,
    FittedTorchModel,
    TorchEpochRecord,
    TorchTrainingConfig,
    TorchTrainingError,
    TorchTrainingResult,
    fit_torch_partition_model,
    fit_torch_partition_model_streaming,
)

if TYPE_CHECKING:
    from .train_lightning_model import (
        run_sliding_window_lightning_training,
        train_lightning_model,
    )
    from .train_sklearn_model import run_sliding_window_sklearn_training, train_sklearn_model


_LAZY_EXPORTS = {
    "run_sliding_window_lightning_training": (
        ".train_lightning_model",
        "run_sliding_window_lightning_training",
    ),
    "run_sliding_window_sklearn_training": (
        ".train_sklearn_model",
        "run_sliding_window_sklearn_training",
    ),
    "train_lightning_model": (".train_lightning_model", "train_lightning_model"),
    "train_sklearn_model": (".train_sklearn_model", "train_sklearn_model"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose legacy high-level training entry points."""
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FittedSklearnModel",
    "FittedTorchModel",
    "ClassificationTask",
    "SklearnTrainingError",
    "SklearnTrainingResult",
    "TORCH_TRAINING_CONFIG_VERSION",
    "TorchEpochRecord",
    "TorchTrainingConfig",
    "TorchTrainingError",
    "TorchTrainingResult",
    "fit_sklearn_partition_model",
    "fit_sklearn_partition_model_streaming",
    "fit_torch_partition_model",
    "fit_torch_partition_model_streaming",
    "run_sliding_window_lightning_training",
    "run_sliding_window_sklearn_training",
    "train_lightning_model",
    "train_sklearn_model",
]
