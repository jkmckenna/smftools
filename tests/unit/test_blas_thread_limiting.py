"""Regression coverage for smftools/__init__.py's BLAS thread-limiting fix.

Must run in a fresh subprocess: pytest's own collection (and every other test
module) has almost certainly already imported numpy by the time any test in
this process runs, so checking os.environ in-process would only prove
something about import order in the test suite, not about a real, fresh
`import smftools`.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_smftools_sets_blas_thread_env_vars_to_one():
    code = (
        "import smftools, os, json; "
        "print(json.dumps({v: os.environ.get(v) for v in "
        "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',"
        "'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}))"
    )
    import json

    values = json.loads(_run(code))
    assert values == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_import_smftools_sets_env_vars_before_numpy_is_ever_imported():
    # If any of these were still unset when the interpreter first imports
    # numpy, the fix would be too late (BLAS reads them once, at backend
    # init) -- so this is the actual correctness property that matters, not
    # just "the vars end up set eventually".
    code = (
        "import sys; assert 'numpy' not in sys.modules; "
        "import smftools; "
        "import os; assert os.environ.get('OMP_NUM_THREADS') == '1'; "
        "import numpy; "
        "import threadpoolctl; "
        "info = threadpoolctl.threadpool_info(); "
        "blas = [d for d in info if d.get('user_api') == 'blas']; "
        "assert blas and all(d.get('num_threads') == 1 for d in blas), blas; "
        "print('ok')"
    )
    assert _run(code) == "ok"


def test_explicit_user_env_override_is_respected():
    import json
    import os

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import smftools, os, json; print(json.dumps(os.environ.get('OMP_NUM_THREADS')))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == "4"


@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_blas_threads_limited_in_multiprocessing_worker(start_method, tmp_path):
    # Confirms the fix actually propagates into worker processes started by
    # this codebase's own multiprocessing start methods, not just the main
    # process -- forkserver's warm-template model is exactly what made a
    # naive per-pool ProcessPoolExecutor(initializer=...) approach
    # unreliable (see cli_entry.py / memory_guard.py's history for why).
    script = tmp_path / "worker_check.py"
    script.write_text(
        "import multiprocessing as mp\n"
        "def check(_):\n"
        "    import os, numpy, threadpoolctl\n"
        "    info = threadpoolctl.threadpool_info()\n"
        "    blas = [d for d in info if d.get('user_api') == 'blas']\n"
        "    return os.environ.get('OMP_NUM_THREADS'), [d.get('num_threads') for d in blas]\n"
        "if __name__ == '__main__':\n"
        "    import smftools\n"
        f"    mp.set_start_method({start_method!r})\n"
        "    from concurrent.futures import ProcessPoolExecutor\n"
        "    with ProcessPoolExecutor(max_workers=2) as pool:\n"
        "        for result in pool.map(check, [None, None]):\n"
        "            print(result)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        omp_threads, blas_thread_counts = ast.literal_eval(line)
        assert omp_threads == "1"
        assert blas_thread_counts and all(count == 1 for count in blas_thread_counts)
