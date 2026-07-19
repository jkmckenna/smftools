"""Verify the clustermap plotting pools (spatial + HMM) wire up the
per-worker memory watchdog when given a cfg, and stay unguarded (previous
behavior) when cfg is omitted -- these pools previously bypassed the
memory-guard machinery entirely (dev/pipeline_scaling_audit.md, finding E).
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from smftools.plotting import (  # noqa: E402
    combined_hmm_length_clustermap,
    combined_hmm_raw_clustermap,
    combined_raw_clustermap,
)


def _cfg(**overrides):
    defaults = dict(threads=2, max_memory_percent=60.0, max_memory_gb=None)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _spatial_adata(n_positions=8):
    matrix = np.random.default_rng(0).random((2, n_positions))
    adata = ad.AnnData(X=np.zeros((2, n_positions)))
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = [str(i) for i in range(n_positions)]
    adata.var["R1_C_site"] = [True] * n_positions
    adata.var["R1_GpC_site"] = [True] * n_positions
    adata.var["R1_CpG_site"] = [True] * n_positions
    adata.var["R1_A_site"] = [False] * n_positions
    return adata


def _hmm_adata(n_positions=8):
    adata = _spatial_adata(n_positions)
    matrix = np.random.default_rng(1).random((2, n_positions))
    adata.layers["hmm_combined"] = matrix
    return adata


def _patch_watchdog(monkeypatch):
    calls = []

    def fake_watchdog(pool, per_worker_budget_bytes, *args, **kwargs):
        calls.append(per_worker_budget_bytes)
        return lambda: None

    monkeypatch.setattr("smftools.memory_guard.start_worker_watchdog", fake_watchdog)
    return calls


def test_combined_raw_clustermap_starts_watchdog_when_cfg_given(monkeypatch, tmp_path):
    calls = _patch_watchdog(monkeypatch)
    combined_raw_clustermap(
        _spatial_adata(),
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        n_jobs=2,
        cfg=_cfg(),
    )
    assert calls, "start_worker_watchdog was never called with cfg set"
    assert all(budget > 0 for budget in calls)


def test_combined_raw_clustermap_skips_watchdog_without_cfg(monkeypatch, tmp_path):
    calls = _patch_watchdog(monkeypatch)
    combined_raw_clustermap(
        _spatial_adata(),
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        n_jobs=2,
    )
    assert not calls, "watchdog must stay unguarded when no cfg is passed (previous behavior)"


def test_combined_hmm_raw_clustermap_starts_watchdog_when_cfg_given(monkeypatch, tmp_path):
    calls = _patch_watchdog(monkeypatch)
    combined_hmm_raw_clustermap(
        _hmm_adata(),
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        sort_by="hmm",
        save_path=tmp_path,
        n_jobs=2,
        cfg=_cfg(),
    )
    assert calls, "start_worker_watchdog was never called with cfg set"


def test_combined_hmm_length_clustermap_starts_watchdog_when_cfg_given(monkeypatch, tmp_path):
    calls = _patch_watchdog(monkeypatch)
    adata = _hmm_adata()
    adata.layers["hmm_combined_lengths"] = np.random.default_rng(2).random(adata.shape)
    combined_hmm_length_clustermap(
        adata,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=tmp_path,
        n_jobs=2,
        cfg=_cfg(),
    )
    assert calls, "start_worker_watchdog was never called with cfg set"
