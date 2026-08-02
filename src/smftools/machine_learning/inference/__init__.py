from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ..evaluation import PredictionResult
from .sklearn_backend import SklearnPredictionResult, apply_sklearn_partition_model
from .torch_backend import TorchPredictionResult, apply_torch_partition_model

if TYPE_CHECKING:
    from .lightning_inference import run_lightning_inference
    from .sklearn_inference import run_sklearn_inference
    from .sliding_window_inference import sliding_window_inference


_LAZY_EXPORTS = {
    "run_lightning_inference": (".lightning_inference", "run_lightning_inference"),
    "run_sklearn_inference": (".sklearn_inference", "run_sklearn_inference"),
    "sliding_window_inference": (".sliding_window_inference", "sliding_window_inference"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose legacy inference helpers with optional dependencies."""
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SklearnPredictionResult",
    "TorchPredictionResult",
    "PredictionResult",
    "apply_sklearn_partition_model",
    "apply_torch_partition_model",
    "run_lightning_inference",
    "run_sklearn_inference",
    "sliding_window_inference",
]
