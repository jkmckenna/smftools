"""Forked pool workers must not inherit a dead dask thread pool.

Dask's threaded scheduler keeps one module-level thread pool. A worker forked
after the parent used it inherits the pool but none of its threads, and its
first compute (anndata's lazy zarr reads in ``materialize``) waits forever --
this hung CI's Python 3.12 job (`HCE-06`'s end-to-end variant test). The
scenario runs in a subprocess so that a regression fails on the timeout
instead of hanging the suite.
"""

import multiprocessing as mp
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

SCENARIO = textwrap.dedent(
    """
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor

    import dask
    import dask.array as da

    from smftools.parallel_utils import configure_worker_threads


    def slow(block):
        time.sleep(0.005)
        return block


    def dask_sum():
        return int(da.ones(100, chunks=10).sum().compute())


    if __name__ == "__main__":
        with dask.config.set(scheduler="threads"):
            # Start every thread of the parent's pool ...
            blocks = da.ones(400, chunks=1).map_blocks(slow, dtype=float)
            assert int(blocks.sum().compute()) == 400
            # ... then compute in a forked worker.
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("fork"),
                initializer=configure_worker_threads,
                initargs=(1,),
            ) as pool:
                assert pool.submit(dask_sum).result() == 100
        print("ok")
    """
)


@pytest.mark.skipif(
    sys.platform == "win32" or "fork" not in mp.get_all_start_methods(),
    reason="needs the fork start method",
)
def test_a_forked_worker_computes_dask_after_the_parent_did(tmp_path):
    script = tmp_path / "scenario.py"
    script.write_text(SCENARIO)
    try:
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        pytest.fail("a forked worker hung on dask's inherited thread pool")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
