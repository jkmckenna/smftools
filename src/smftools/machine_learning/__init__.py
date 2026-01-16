from __future__ import annotations

from importlib import import_module

_LAZY_MODULES = {
    "data": "smftools.machine_learning.data",
    "evaluation": "smftools.machine_learning.evaluation",
    "inference": "smftools.machine_learning.inference",
    "models": "smftools.machine_learning.models",
    "training": "smftools.machine_learning.training",
    "utils": "smftools.machine_learning.utils",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_MODULES.keys())
