from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from smftools.latent_resource import resolve_latent_operation
from smftools.memory_guard import process_tree_rss_bytes

pytestmark = pytest.mark.integration


def test_latent_resource_decision_uses_live_runtime_headroom():
    # The cap is derived from live RSS rather than hardcoded at 1.0 GiB. A fixed
    # cap made this test depend on how much memory every *earlier* test in the
    # session had left resident: the integration suite reaches roughly 688 MiB
    # before this file is even collected, and adding ML integration coverage
    # pushed Linux CI past the cap, so headroom resolved to zero and the
    # operation was refused.
    #
    # Deriving the cap also strengthens the assertion: a decision that ignored
    # live usage entirely would report headroom equal to the whole cap.
    margin_bytes = 512 * 1024**2
    live_rss = process_tree_rss_bytes()
    cap_bytes = live_rss + margin_bytes

    cfg = SimpleNamespace(
        threads=1,
        max_memory_percent=None,
        max_memory_gb=cap_bytes / 1024**3,
        memory_reserve_gb=0.0,
        latent_run_pca_umap=True,
        latent_run_nmf=True,
        latent_run_cp=False,
        latent_n_pcs=2,
        latent_nmf_components=1,
        latent_knn_neighbors=2,
    )

    decision = resolve_latent_operation(
        cfg,
        "fit",
        requested_reads=3,
        n_positions=10,
        minimum_reads=3,
    )

    headroom = decision.pool_budget["usable_headroom_bytes"]

    assert decision.effective_reads == 3
    assert decision.pool_budget["process_tree_rss_bytes"] > 0
    assert headroom > 0
    assert decision.predicted_peak_bytes > 0
    # Live usage was subtracted from the cap rather than ignored. Deliberately
    # not asserted against the margin: the decision re-reads RSS internally, and
    # that reading legitimately differs from the one above by tens of MiB, so a
    # tight bound would be flaky in exactly the way this test is being fixed for.
    assert headroom < cap_bytes


def test_process_tree_rss_includes_a_live_child():
    baseline = process_tree_rss_bytes()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "data = bytearray(16 * 1024 * 1024); print('ready', flush=True); import time; time.sleep(10)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        try:
            descendant_pids = {process.pid for process in psutil.Process().children(recursive=True)}
        except (psutil.AccessDenied, psutil.Error, OSError) as error:
            pytest.skip(f"descendant enumeration is restricted: {error}")
        if child.pid not in descendant_pids:
            pytest.skip("descendant enumeration is restricted by the runtime sandbox")
        assert process_tree_rss_bytes() >= baseline + 8 * 1024**2
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroup runtime check")
def test_linux_runtime_reports_real_cgroup_activation_or_fallback(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        """
import json
from types import SimpleNamespace
from smftools.memory_guard import activate_resource_envelope, resolve_resource_envelope

cfg = SimpleNamespace(
    threads=2,
    max_memory_percent=None,
    max_memory_gb=2.0,
    memory_reserve_gb=0.0,
    target_task_memory_mb=128,
)
active = activate_resource_envelope(resolve_resource_envelope(cfg, environ={}))
print(json.dumps({
    "active": active.enforcement_active,
    "capability": active.enforcement_capability,
    "mode": active.enforcement_mode,
}))
""".strip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)

    assert record["mode"] in {"cgroup_v2", "advisory"}
    assert record["active"] is (record["mode"] == "cgroup_v2")
    if record["active"]:
        assert record["capability"] == "cgroup_v2"
