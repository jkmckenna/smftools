"""smftools"""

from __future__ import annotations

import os

# Single-threaded BLAS/OMP/numexpr by default, for this process and every
# child it later forks/spawns. Must run as the literal first statements in
# this file, before the `from .readwrite import ...` a few lines down (which
# pulls in anndata/pandas, and therefore numpy) -- these env vars are read
# once, at BLAS backend init, so anything after the first numpy import is
# too late. This is also why the equivalent fix can't live in cli_entry.py:
# importing that submodule first runs this __init__.py in full (Python
# always imports a package's __init__ before any of its submodules), so by
# the time cli_entry.py's own body executes, numpy is already loaded either
# way.
#
# Without this, task-level (process-count) parallelism
# (memory_guard.run_tasks_parallel, raw ingestion's bucket extraction) and
# BLAS's own internal thread pool compound: each worker independently spins
# up threads across every physical core for its own numpy calls, so N
# concurrent task workers can end up scheduling N x (BLAS's own thread
# count) threads total -- severe CPU oversubscription that makes wall-clock
# time *worse* than fewer, purely task-parallel workers would give.
# Confirmed via real-data testing on this exact codebase: 8 concurrent
# spatial-analysis task workers each independently using ~200% CPU via their
# own BLAS thread pool. `setdefault` (not a hard overwrite) respects an
# explicit user override (e.g. set in the caller's shell environment before
# importing smftools) rather than silently replacing it.
for _blas_thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_thread_var, "1")
del _blas_thread_var

import logging  # noqa: E402
import warnings  # noqa: E402
from importlib import import_module  # noqa: E402
from importlib.metadata import version  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from .readwrite import adata_to_df, safe_read_h5ad, safe_write_h5ad  # noqa: E402

package_name = "smftools"
__version__ = version(package_name)

if TYPE_CHECKING:
    from smftools import (
        cli,
        config,
        datasets,
        hmm,
        informatics,
        machine_learning,
        plotting,
        preprocessing,
        tools,
    )

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
