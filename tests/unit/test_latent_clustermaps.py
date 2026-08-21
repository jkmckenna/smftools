"""Latent-ordered clustermaps of projected layers (`EGL-28c`).

Three views of the same molecules in one figure, rows ordered by the latent
clustering. The failure modes worth pinning are the ones that still render:
panels drifting out of row alignment, the raw panel drawn over positions where
accessibility is undefined, and the length layers being paired with the wrong
HMM generation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.tools.latent_clustermaps import (
    _panel_specs,
    display_positions,
    gather_hmm_length_layers,
    length_layer_names,
    mod_site_mask,
    resolve_feature_prefix,
    resolve_hmm_generation,
)

pytestmark = pytest.mark.unit


# --- layer naming ------------------------------------------------------------


def test_prefix_follows_the_target_base_not_the_modality_name():
    """A config targeting an unusual base must still resolve."""
    assert resolve_feature_prefix(SimpleNamespace(mod_target_bases=["C"])) == "C_"
    assert resolve_feature_prefix(SimpleNamespace(mod_target_bases=["GpC", "CpG"])) == "GpC_"


def test_prefix_falls_back_to_the_modality_when_bases_are_absent():
    assert resolve_feature_prefix(SimpleNamespace(smf_modality="conversion")) == "GpC_"
    assert resolve_feature_prefix(SimpleNamespace(smf_modality="deaminase")) == "C_"


def test_length_layer_names_match_the_hmm_stage():
    assert length_layer_names("C_") == (
        "C_all_footprint_features_lengths",
        "C_all_accessible_features_lengths",
    )


# --- the raw panel's columns -------------------------------------------------


def test_mod_site_mask_selects_only_target_base_positions():
    """Accessibility is undefined between sites.

    Drawn over every position, the informative columns go sub-pixel and the
    panel reads as vertical noise -- which is what the first version did.
    """
    var = pd.DataFrame(
        {
            "ref_top_C_site": [True, False, True, False, True],
            "position_in_ref_top": [True] * 5,
        }
    )
    assert mod_site_mask(var, "ref_top", ["C"]).tolist() == [True, False, True, False, True]


def test_mod_site_mask_respects_reference_membership():
    """A site column can be True at a position belonging to another reference."""
    var = pd.DataFrame(
        {
            "ref_top_C_site": [True, True, True],
            "position_in_ref_top": [True, False, True],
        }
    )
    assert mod_site_mask(var, "ref_top", ["C"]).tolist() == [True, False, True]


def test_missing_site_columns_yield_no_mask():
    """Better to draw everything and say so than to silently drop all columns."""
    assert mod_site_mask(pd.DataFrame({"other": [1, 2]}), "ref_top", ["C"]) is None


def test_a_mask_selecting_nothing_is_treated_as_absent():
    var = pd.DataFrame({"ref_top_C_site": [False, False], "position_in_ref_top": [True, True]})
    assert mod_site_mask(var, "ref_top", ["C"]) is None


# --- coordinates -------------------------------------------------------------


def test_display_positions_use_the_reindexed_column_when_present():
    """`EGL-23`: a panel that ignores this runs backwards next to its neighbours."""
    var = pd.DataFrame({"ref_top_reindexed": [-10, -9, -8]})
    cfg = SimpleNamespace(reindexed_var_suffix="reindexed")
    assert display_positions(var, "ref_top", cfg, [0, 1, 2]).tolist() == [-10, -9, -8]


def test_display_positions_fall_back_to_stored_coordinates():
    cfg = SimpleNamespace(reindexed_var_suffix="reindexed")
    assert display_positions(pd.DataFrame(), "ref_top", cfg, [0, 1, 2]).tolist() == [0, 1, 2]


# --- panels ------------------------------------------------------------------


def _grids(n_rows=4, n_cols=6):
    rng = np.random.default_rng(0)
    return {
        "C_all_footprint_features_lengths": rng.uniform(0, 100, size=(n_rows, n_cols)),
        "C_all_accessible_features_lengths": rng.uniform(0, 50, size=(n_rows, n_cols)),
    }


def test_panels_share_the_row_axis_even_when_widths_differ():
    """The shared row is the whole point: one row, one molecule, three views."""
    raw = (np.zeros((4, 3), dtype=float), np.array([1, 2, 3]))
    panels = _panel_specs(raw, _grids(), length_layer_names("C_"), np.arange(6))

    assert len(panels) == 3
    assert {panel["matrix"].shape[0] for panel in panels} == {4}
    assert [panel["matrix"].shape[1] for panel in panels] == [3, 6, 6]


def test_each_panel_carries_its_own_axis_labels():
    """Widths differ, so a single shared label array would mislabel two panels."""
    raw = (np.zeros((4, 3), dtype=float), np.array([1, 2, 3]))
    panels = _panel_specs(raw, _grids(), length_layer_names("C_"), np.arange(6))
    for panel in panels:
        assert len(panel["positions"]) == panel["matrix"].shape[1]


def test_a_missing_length_layer_is_omitted_rather_than_faked():
    grids = {"C_all_footprint_features_lengths": np.zeros((4, 6))}
    panels = _panel_specs(
        (np.zeros((4, 3)), np.arange(3)), grids, length_layer_names("C_"), np.arange(6)
    )
    assert [panel["name"] for panel in panels][1:] == ["all footprint features lengths"]


def test_length_panels_are_clipped_against_the_heavy_tail():
    """One 881 bp feature would flatten everything else on a max-scaled bar."""
    values = np.zeros((4, 6))
    values[0, 0] = 10_000.0
    panels = _panel_specs(
        (np.zeros((4, 3)), np.arange(3)),
        {"C_all_footprint_features_lengths": values},
        length_layer_names("C_"),
        np.arange(6),
    )
    assert panels[1]["vmax"] < 10_000.0


# --- cross-stage join --------------------------------------------------------


def test_hmm_generation_comes_from_the_lineage_not_from_current(tmp_path):
    """`current.json` can have advanced since this latent generation published.

    Pairing these labels with a different HMM generation would draw feature
    lengths from a different analysis beside them, and nothing would say so.
    """
    lineage = tmp_path / "hmm_adata_outputs" / "generations" / "aaa"
    lineage.mkdir(parents=True)
    (lineage / "task_catalog.parquet").write_bytes(b"")
    spine = SimpleNamespace(
        uns={"hmm_catalog": "hmm_adata_outputs/generations/aaa/task_catalog.parquet"}
    )
    assert resolve_hmm_generation(spine, tmp_path) == lineage


def test_an_unresolvable_hmm_pointer_returns_none(tmp_path):
    spine = SimpleNamespace(uns={"hmm_catalog": "hmm_adata_outputs/generations/gone/x.parquet"})
    assert resolve_hmm_generation(spine, tmp_path) is None


def test_no_hmm_pointer_returns_none():
    assert resolve_hmm_generation(SimpleNamespace(uns={}), "/tmp") is None


def _write_hmm_shard(root, name, read_ids, values):
    import anndata as ad

    group = root / name
    adata = ad.AnnData(
        X=np.zeros((len(read_ids), values.shape[1]), dtype=np.float32),
        obs=pd.DataFrame(index=list(read_ids)),
    )
    adata.layers["C_all_footprint_features_lengths"] = values
    adata.write_zarr(group)
    return group


def test_length_layers_are_gathered_across_shards(tmp_path):
    """The HMM stage splits by barcode where latent splits by reference.

    On the DAF pilot that is 32 shards against 2 units, so a single-shard read
    would silently cover a fraction of the molecules.
    """
    import anndata as ad  # noqa: F401

    generation = tmp_path / "gen"
    generation.mkdir()
    _write_hmm_shard(generation, "s0", ["r0", "r1"], np.full((2, 3), 1.0, dtype=np.float32))
    _write_hmm_shard(generation, "s1", ["r2"], np.full((1, 3), 2.0, dtype=np.float32))
    pd.DataFrame({"reference": ["ref_top", "ref_top"], "group_path": ["s0", "s1"]}).to_parquet(
        generation / "task_catalog.parquet"
    )

    grids = gather_hmm_length_layers(
        generation,
        reference="ref_top",
        read_ids=["r0", "r1", "r2"],
        layer_names=("C_all_footprint_features_lengths",),
    )

    values = grids["C_all_footprint_features_lengths"]
    assert values.shape == (3, 3)
    assert values[0, 0] == 1.0 and values[2, 0] == 2.0


def test_molecules_absent_from_the_hmm_generation_stay_as_rows(tmp_path):
    """Dropping them would change which molecules the panel shows.

    The row must stay, blank, so it remains aligned with the two panels beside
    it -- a shifted row order is invisible in the output and wrong everywhere.
    """
    generation = tmp_path / "gen"
    generation.mkdir()
    _write_hmm_shard(generation, "s0", ["r0"], np.full((1, 3), 1.0, dtype=np.float32))
    pd.DataFrame({"reference": ["ref_top"], "group_path": ["s0"]}).to_parquet(
        generation / "task_catalog.parquet"
    )

    grids = gather_hmm_length_layers(
        generation,
        reference="ref_top",
        read_ids=["r0", "missing"],
        layer_names=("C_all_footprint_features_lengths",),
    )

    values = grids["C_all_footprint_features_lengths"]
    assert values.shape[0] == 2
    assert np.isnan(values[1]).all()


def test_shards_for_other_references_are_ignored(tmp_path):
    generation = tmp_path / "gen"
    generation.mkdir()
    _write_hmm_shard(generation, "s0", ["r0"], np.full((1, 3), 5.0, dtype=np.float32))
    pd.DataFrame({"reference": ["other_ref"], "group_path": ["s0"]}).to_parquet(
        generation / "task_catalog.parquet"
    )
    assert (
        gather_hmm_length_layers(
            generation,
            reference="ref_top",
            read_ids=["r0"],
            layer_names=("C_all_footprint_features_lengths",),
        )
        == {}
    )


# --- config ------------------------------------------------------------------


def test_config_knob_defaults_on():
    from smftools.config.experiment_config import ExperimentConfig

    assert ExperimentConfig().latent_plot_clustermaps is True
    cfg, _ = ExperimentConfig.from_var_dict({"latent_plot_clustermaps": "FALSE"})
    assert cfg.latent_plot_clustermaps is False


def test_clusters_category_exists_for_latent():
    from smftools.cli.stage_artifacts import STAGE_PLOT_CATEGORIES

    assert "clusters" in STAGE_PLOT_CATEGORIES["latent"]
