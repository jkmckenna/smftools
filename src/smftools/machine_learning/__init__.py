from __future__ import annotations

from importlib import import_module

# Lazily exposed so that importing ``smftools.machine_learning`` does not pull
# torch, sklearn, and the partition-store stack into every process. The contract
# modules are as public as the subpackages -- a caller writing a plan needs
# ``plan``, and one inspecting provenance needs ``manifests`` and ``artifacts``
# -- so they are declared here rather than left as undeclared-but-required
# imports.
_LAZY_MODULES = {
    "artifacts": "smftools.machine_learning.artifacts",
    "contracts": "smftools.machine_learning.contracts",
    "data": "smftools.machine_learning.data",
    "evaluation": "smftools.machine_learning.evaluation",
    "inference": "smftools.machine_learning.inference",
    "interpretability": "smftools.machine_learning.interpretability",
    "manifests": "smftools.machine_learning.manifests",
    "models": "smftools.machine_learning.models",
    "orchestration": "smftools.machine_learning.orchestration",
    "plan": "smftools.machine_learning.plan",
    "selection": "smftools.machine_learning.selection",
    "splitting": "smftools.machine_learning.splitting",
    "training": "smftools.machine_learning.training",
    "utils": "smftools.machine_learning.utils",
    "workspace": "smftools.machine_learning.workspace",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = sorted(_LAZY_MODULES)
