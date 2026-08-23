"""Stage reuse must depend only on settings that can change that stage (`F30`).

Every stage except `latent` (allowlisted) and `preprocess` (dedicated function)
hashed *the entire config* minus a short denylist. So a spatial plotting toggle
marked the **raw** store stale, forcing a rebuild from FASTQ, and correcting a
TSS offset -- a pure display projection -- did the same.

The risk here runs one way: a scope that is too *narrow* silently reuses a stage
that should have been recomputed, which is far worse than recomputing one too
often. So these tests weight heavily toward proving that real changes still
invalidate.
"""

from __future__ import annotations

import copy

import pytest

from smftools.cli.helpers import (
    _downstream_config_prefixes,
    resolved_stage_config,
    stage_config_hash,
)
from smftools.config.experiment_config import ExperimentConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def cfg():
    return ExperimentConfig()


def _changes(cfg, stage, **overrides):
    other = copy.copy(cfg)
    for key, value in overrides.items():
        setattr(other, key, value)
    return stage_config_hash(other, stage) != stage_config_hash(cfg, stage)


# --- the guard that matters most: real changes must still invalidate ---------


@pytest.mark.parametrize(
    "stage,key,value",
    [
        ("raw", "smf_modality", "conversion"),
        ("raw", "barcode_kit", "SQK-RBK114-96"),
        ("raw", "analysis_mode", "genome"),
        ("raw", "fastq_auto_pairing", False),
        ("raw", "demux_backend", "smftools"),
        ("raw", "input_already_demuxed", True),
        ("raw", "derive_demux_status_from_sequence", True),
        ("raw", "max_full_matrix_gb", 1.0),
        ("spatial", "spatial_generate_position_matrices", False),
        ("spatial", "autocorr_max_lag", 3),
        ("hmm", "hmm_clustermap_sortby", "zzz"),
        ("latent", "latent_max_fit_reads", 99),
    ],
)
def test_a_stage_is_still_invalidated_by_its_own_settings(cfg, stage, key, value):
    assert _changes(cfg, stage, **{key: value}), (
        f"{key} must still invalidate {stage}; a too-narrow scope silently reuses stale work"
    )


# --- what must no longer invalidate ------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("reindexing_offsets", {"ref_top": -1637}),
        ("reindexing_invert", False),
        ("reindexed_var_suffix", "other"),
    ],
)
def test_display_only_settings_invalidate_no_analysis_stage(cfg, key, value):
    """These change a plot axis, never a stored value."""
    for stage in ("raw", "spatial", "hmm", "latent"):
        assert not _changes(cfg, stage, **{key: value}), f"{key} must not invalidate {stage}"


@pytest.mark.parametrize(
    "key,value",
    [
        ("spatial_generate_position_matrices", False),
        ("hmm_clustermap_sortby", "zzz"),
        ("latent_leiden_resolution", 9.9),
        ("latent_max_fit_reads", 77),
    ],
)
def test_downstream_settings_do_not_invalidate_raw(cfg, key, value):
    """Raw runs first; nothing a later stage declares can change its output."""
    assert not _changes(cfg, "raw", **{key: value})


def test_downstream_settings_do_not_invalidate_spatial(cfg):
    assert not _changes(cfg, "spatial", latent_max_fit_reads=77)
    assert not _changes(cfg, "spatial", hmm_clustermap_sortby="zzz")


def test_upstream_settings_still_invalidate_downstream(cfg):
    """Scoping is one-directional: a later stage still depends on earlier ones."""
    assert _changes(cfg, "spatial", smf_modality="conversion")
    assert _changes(cfg, "hmm", smf_modality="conversion")


# --- the ordering helper ------------------------------------------------------


def test_downstream_prefixes_follow_stage_order():
    assert _downstream_config_prefixes("raw") == ("preprocess_", "spatial_", "hmm_", "latent_")
    assert _downstream_config_prefixes("hmm") == ("latent_",)
    assert _downstream_config_prefixes("latent") == ()


def test_unknown_stage_has_no_downstream_prefixes():
    """`full` and `None` must not accidentally drop everything."""
    assert _downstream_config_prefixes("full") == ()
    assert _downstream_config_prefixes("nonsense") == ()


def test_an_unscoped_stage_still_hashes_broadly(cfg):
    """Only known stages get narrowed; anything else keeps the old behaviour."""
    resolved = resolved_stage_config(cfg, "full")
    assert "spatial_generate_position_matrices" in resolved


def test_raw_scope_is_narrower_than_before_but_not_empty(cfg):
    resolved = resolved_stage_config(cfg, "raw")
    assert len(resolved) > 100, "raw must still depend on most of its own settings"
    assert not any(key.startswith(("spatial_", "hmm_", "latent_")) for key in resolved)
    assert "reindexing_offsets" not in resolved
    assert "barcode_kit" in resolved
