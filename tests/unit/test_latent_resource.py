from types import SimpleNamespace

import pytest

from smftools.latent_resource import (
    LATENT_RESOURCE_ESTIMATOR_VERSION,
    LatentResourceError,
    decide_latent_operation,
    estimate_latent_memory,
    memory_safe_read_count,
)
from smftools.memory_guard import PoolBudget


def _cfg(**overrides):
    values = {
        "latent_run_pca_umap": True,
        "latent_run_nmf": True,
        "latent_run_cp": True,
        "latent_n_pcs": 10,
        "latent_nmf_components": 2,
        "latent_cp_rank": 2,
        "latent_knn_neighbors": 15,
        "umap_layers_to_plot": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _budget(usable_headroom_bytes: int) -> PoolBudget:
    return PoolBudget(
        estimator="test",
        estimator_version="1",
        n_items=1,
        per_item_memory_bytes=1,
        process_tree_rss_bytes=0,
        process_tree_private_bytes=0,
        system_available_bytes=usable_headroom_bytes,
        cgroup_headroom_bytes=None,
        run_headroom_bytes=usable_headroom_bytes,
        usable_headroom_bytes=usable_headroom_bytes,
        max_workers=1,
        max_in_flight=1,
    )


@pytest.mark.parametrize("operation", ["fit", "transform", "cp", "result", "write", "plot"])
def test_latent_estimate_increases_with_reads_and_positions(operation):
    cfg = _cfg()
    small = estimate_latent_memory(cfg, operation, n_reads=10, n_positions=100)
    more_reads = estimate_latent_memory(cfg, operation, n_reads=20, n_positions=100)
    more_positions = estimate_latent_memory(cfg, operation, n_reads=10, n_positions=200)

    assert more_reads.predicted_peak_bytes > small.predicted_peak_bytes
    assert more_positions.predicted_peak_bytes > small.predicted_peak_bytes
    assert small.estimator_version == LATENT_RESOURCE_ESTIMATOR_VERSION


def test_latent_estimate_increases_with_components_and_enabled_algorithms():
    base = estimate_latent_memory(
        _cfg(
            latent_run_pca_umap=False,
            latent_run_nmf=False,
            latent_run_cp=False,
        ),
        "fit",
        n_reads=100,
        n_positions=1000,
    )
    enabled = estimate_latent_memory(
        _cfg(latent_n_pcs=2, latent_nmf_components=1),
        "fit",
        n_reads=100,
        n_positions=1000,
    )
    wider = estimate_latent_memory(
        _cfg(latent_n_pcs=20, latent_nmf_components=5),
        "fit",
        n_reads=100,
        n_positions=1000,
    )

    assert enabled.predicted_peak_bytes > base.predicted_peak_bytes
    assert wider.predicted_peak_bytes > enabled.predicted_peak_bytes


def test_latent_estimate_accounts_for_source_and_fit_dtypes():
    cfg = _cfg()
    float32 = estimate_latent_memory(
        cfg,
        "fit",
        n_reads=100,
        n_positions=1000,
        source_dtype="float32",
        fit_dtype="float32",
    )
    float64 = estimate_latent_memory(
        cfg,
        "fit",
        n_reads=100,
        n_positions=1000,
        source_dtype="float64",
        fit_dtype="float64",
    )

    assert float64.predicted_peak_bytes > float32.predicted_peak_bytes


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_small_envelope_reduces_effective_read_count(operation):
    cfg = _cfg()
    headroom = estimate_latent_memory(
        cfg,
        operation,
        n_reads=20,
        n_positions=1000,
    ).predicted_peak_bytes

    decision = decide_latent_operation(
        cfg,
        operation,
        requested_reads=100,
        n_positions=1000,
        minimum_reads=3 if operation == "fit" else 1,
        pool_budget=_budget(headroom),
    )

    assert 1 <= decision.effective_reads < 100
    assert decision.limiting_operation == operation


def test_minimum_unit_failure_names_estimator_and_operation():
    cfg = _cfg()

    with pytest.raises(
        LatentResourceError,
        match=r"estimator 1.*operation 'fit'.*minimum_reads=3",
    ):
        decide_latent_operation(
            cfg,
            "fit",
            requested_reads=100,
            n_positions=1000,
            minimum_reads=3,
            pool_budget=_budget(1),
        )


def test_memory_safe_count_does_not_change_component_semantics():
    cfg = _cfg(latent_n_pcs=17, latent_nmf_components=4, latent_cp_rank=3)
    headroom = estimate_latent_memory(
        cfg,
        "fit",
        n_reads=25,
        n_positions=500,
    ).predicted_peak_bytes

    effective = memory_safe_read_count(
        cfg,
        "fit",
        requested_reads=100,
        n_positions=500,
        usable_headroom_bytes=headroom,
    )
    estimate = estimate_latent_memory(
        cfg,
        "fit",
        n_reads=effective,
        n_positions=500,
    )

    assert effective == 25
    assert estimate.breakdown_bytes["pca_workspace"] > 0
    assert estimate.breakdown_bytes["nmf_workspace"] > 0


def test_latent_resource_limits_do_not_change_compute_identity():
    from smftools.cli.helpers import stage_config_hash, stage_plot_config_hash

    cfg = _cfg(
        latent_cp_memory_policy="skip",
        latent_plot_max_reads=1000,
        max_memory_gb=64,
    )
    compute_hash = stage_config_hash(cfg, "latent")
    plot_hash = stage_plot_config_hash(cfg, "latent")

    cfg.max_memory_gb = 8
    assert stage_config_hash(cfg, "latent") == compute_hash

    cfg.latent_plot_max_reads = 100
    assert stage_config_hash(cfg, "latent") == compute_hash
    assert stage_plot_config_hash(cfg, "latent") != plot_hash

    cfg.latent_cp_memory_policy = "fail"
    assert stage_config_hash(cfg, "latent") != compute_hash
