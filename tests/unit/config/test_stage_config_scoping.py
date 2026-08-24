"""Config keys must only invalidate the stages they actually affect (`F46`, `F47`)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from smftools.cli.helpers import load_experiment_config, resolved_stage_config, stage_config_hash

pytestmark = pytest.mark.unit

#: Settings owned by a stage after raw. None of these change a single byte of
#: the raw store, and raw is the most expensive stage to rebuild.
DOWNSTREAM_OF_RAW = (
    "bypass_deamination_segmentation",
    "bypass_label_deaminase_pcr_chimeras",
    "bypass_flag_duplicate_reads",
    "bypass_hmm_fit",
    "bypass_spatial_autocorr_calculations",
    "bypass_matrix_corr_plotting",
    "deaminase_segment_penalty_scale",
    "deaminase_chimera_min_events_per_span",
    "read_len_filter_thresholds",
    "read_quality_filter_thresholds",
    "variant_chimera_min_adjacent_sites",
)

#: Settings raw genuinely depends on. Excluding any of these would let a raw
#: store be reused when it should have been rebuilt -- silently wrong, unlike
#: the merely-wasteful direction this scoping errs toward.
RAW_CRITICAL = (
    "input_already_demuxed",
    "barcode_kit",
    "demux_backend",
    "derive_demux_status_from_sequence",
    "spike_in_references",
    "skip_unclassified",
    "barcode_end_score_threshold",
    "smf_modality",
    "conversion_types",
    "analysis_mode",
    "aligner",
    "raw_parquet_shard_size",
)


@pytest.fixture
def cfg(tmp_path):
    config = tmp_path / "experiment_config.csv"
    config.write_text("variable,value,help,options,type\nexperiment_name,x,,,str\n")
    return load_experiment_config(str(config))


@pytest.mark.parametrize("key", DOWNSTREAM_OF_RAW)
def test_downstream_settings_do_not_invalidate_raw(cfg, key):
    """A preprocess or plotting tweak must not force a re-extraction.

    Raw carried 279 semantic keys, including every downstream bypass and filter
    threshold, so changing a read-length filter rebuilt the raw store from
    FASTQ -- 36 minutes on a real run.
    """
    assert key in vars(cfg), f"{key} is no longer a config field; update this test"
    assert key not in resolved_stage_config(cfg, "raw")


@pytest.mark.parametrize("key", RAW_CRITICAL)
def test_raw_still_depends_on_its_own_settings(cfg, key):
    """The dangerous direction: excluding a key raw really needs.

    That would reuse a stale raw store and report it as compatible, which is
    the failure `F33`, `F36` and `F41` each produced in a different place.
    """
    assert key in vars(cfg), f"{key} is no longer a config field; update this test"
    assert key in resolved_stage_config(cfg, "raw")


@pytest.mark.parametrize(
    ("stage", "key"),
    [
        ("spatial", "bypass_spatial_autocorr_calculations"),
        ("hmm", "bypass_hmm_fit"),
    ],
)
def test_a_stage_still_owns_its_own_settings(cfg, stage, key):
    """Scoping must not strip a stage's own keys from its own hash."""
    assert key in resolved_stage_config(cfg, stage)


def test_segmentation_settings_invalidate_preprocess_but_not_raw(cfg):
    """`EGL-25`'s bypass flag was in no stage's fingerprint at all (`F47`).

    Only the scalar chimera keys were fingerprinted, so toggling segmentation
    silently reused a preprocess generation built the other way.
    """
    other = replace(cfg, bypass_deamination_segmentation=not cfg.bypass_deamination_segmentation)

    assert stage_config_hash(cfg, "preprocess") != stage_config_hash(other, "preprocess")
    assert stage_config_hash(cfg, "raw") == stage_config_hash(other, "raw")


def test_segment_batch_size_is_scheduling_not_semantics(cfg):
    """Batches are cut within shards and each read is segmented independently."""
    other = replace(cfg, deaminase_segment_batch_reads=cfg.deaminase_segment_batch_reads * 2)

    assert stage_config_hash(cfg, "preprocess") == stage_config_hash(other, "preprocess")
