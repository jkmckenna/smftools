"""smftools"""
from __future__ import annotations

import logging
import warnings
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from .readwrite import adata_to_df, merge_barcoded_anndatas_core, safe_read_h5ad, safe_write_h5ad

package_name = "smftools"
__version__ = version(package_name)

if TYPE_CHECKING:
    from smftools import cli, config, datasets, hmm, informatics, machine_learning, plotting
    from smftools import preprocessing, tools

_LAZY_MODULES = {
    "cli": "smftools.cli",
    "config": "smftools.config",
    "datasets": "smftools.datasets",
    "hmm": "smftools.hmm",
    "inform": "smftools.informatics",
    "ml": "smftools.machine_learning",
    "pl": "smftools.plotting",
    "pp": "smftools.preprocessing",
    "tl": "smftools.tools",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "adata_to_df",
    "inform",
    "ml",
    "pp",
    "tl",
    "pl",
    "datasets",
    "safe_write_h5ad",
    "safe_read_h5ad",
]
