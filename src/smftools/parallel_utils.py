"""Utilities for safe multiprocessing across macOS and Linux."""

from __future__ import annotations

import os
import sys


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to a concrete positive worker count.

    Parameters
    ----------
    n_jobs:
        Number of workers. Negative values map to ``os.cpu_count()``.
        Zero is treated as 1.
    """
    from .memory_guard import detected_usable_cpu_count

    detected = detected_usable_cpu_count()
    requested = (os.cpu_count() or 1) if n_jobs < 0 else max(1, n_jobs)
    return min(requested, detected)


def configure_worker_threads(n_threads: int = 1) -> None:
    """Cap BLAS/OpenMP/TBB/torch threads inside a worker process.

    Pass as the ``initializer`` to ``ProcessPoolExecutor`` so it runs before
    any task modules are imported::

        ProcessPoolExecutor(
            max_workers=n,
            initializer=configure_worker_threads,
            initargs=(1,),
        )

    This prevents ``n_workers × blas_threads`` CPU over-subscription. Calling
    it inside the worker function body is too late — numpy/scipy import their
    BLAS thread pool when the module is first imported, which happens before
    the function body runs.

    Parameters
    ----------
    n_threads:
        Number of threads to allow per worker. Default 1 (no nested
        parallelism). Increase only if the worker is doing heavy linear
        algebra and you want intra-worker BLAS parallelism.

    Notes
    -----
    Environment variables are set unconditionally (not via ``setdefault``)
    because worker processes inherit the parent's environment. If the user has
    ``OMP_NUM_THREADS=8`` in their shell and smftools spawns 8 workers,
    ``setdefault`` would be a no-op and each worker would use 8 BLAS threads
    (64 total). Unconditional assignment ensures each worker uses exactly
    ``n_threads`` regardless of the inherited environment.

    These must be set *before* numpy/scipy are imported so that OpenBLAS/MKL/
    OpenMP pick them up at library init time. In a ``spawn`` process (macOS
    default, recommended on Linux too) no numpy/scipy has been imported yet
    when the worker first runs, so setting envvars here is safe.

    Heavy libraries are never imported solely by this initializer. If Torch
    or Matplotlib is already loaded, its runtime setting is updated too.
    """
    thread_str = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = thread_str
    os.environ["MKL_NUM_THREADS"] = thread_str
    os.environ["OPENBLAS_NUM_THREADS"] = thread_str
    os.environ["BLIS_NUM_THREADS"] = thread_str
    os.environ["TBB_NUM_THREADS"] = thread_str
    os.environ["NUMEXPR_NUM_THREADS"] = thread_str
    # Apple Accelerate / vecLib (macOS) — does not honour the OpenBLAS/OMP vars
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_str
    os.environ["ACCELERATE_NUM_THREADS"] = thread_str

    # Force matplotlib to the non-GUI Agg backend before any smftools module
    # imports matplotlib.pyplot.  On macOS the default backend is MacOSX (AppKit)
    # which initialises GUI event-loop threads even in non-interactive workers,
    # causing ~200 % CPU per worker process.  This must run before matplotlib is
    # imported, which is guaranteed here because the initializer executes before
    # any task module is imported in a spawn-based worker process.
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is not None:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

    # Importing torch solely from a generic pool initializer adds hundreds of
    # MiB to every worker. Configure its runtime pool only when the worker has
    # already imported it; otherwise OMP/MKL caps above apply at later import.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(n_threads)
