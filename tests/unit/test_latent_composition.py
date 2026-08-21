"""Stacked Leiden composition barplots (`EGL-28d`).

These are read alongside the clustermaps -- "which cluster is that block" next
to "how much of each sample is that cluster" -- so the colour mapping has to be
shared, and a proportion has to carry the count it was computed over.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.plotting.latent_plotting import cluster_color_map, plot_leiden_composition
from smftools.tools.latent_clustermaps import resolve_composition_groups

pytestmark = pytest.mark.unit


# --- grouping ----------------------------------------------------------------


def test_grouping_defaults_to_the_plotting_sample_column():
    obs = pd.DataFrame({"Barcode": ["a", "b", "a", "b"]})
    cfg = SimpleNamespace(sample_name_col_for_plotting="Barcode")
    assert resolve_composition_groups(obs, cfg) == ["Barcode"]


def test_configured_columns_are_used_when_present():
    """This is how the per-biorep breakdown appears without a code change.

    No biorep column exists in obs or config today, so hard-coding one would
    produce a plot that never renders.
    """
    obs = pd.DataFrame({"Barcode": ["a", "b"], "biorep": ["r1", "r2"]})
    cfg = SimpleNamespace(latent_composition_group_columns=["Barcode", "biorep"])
    assert resolve_composition_groups(obs, cfg) == ["Barcode", "biorep"]


def test_absent_columns_are_skipped_not_fatal():
    obs = pd.DataFrame({"Barcode": ["a", "b"]})
    cfg = SimpleNamespace(latent_composition_group_columns=["Barcode", "biorep"])
    assert resolve_composition_groups(obs, cfg) == ["Barcode"]


def test_single_valued_columns_are_dropped():
    """A stacked bar chart with one bar is the cluster strip again, with more ink.

    `Sample` holds one value per run on these datasets, so without this the
    default grouping would produce exactly that.
    """
    obs = pd.DataFrame({"Sample": ["run"] * 4, "Barcode": ["a", "b", "a", "b"]})
    cfg = SimpleNamespace(latent_composition_group_columns=["Sample", "Barcode"])
    assert resolve_composition_groups(obs, cfg) == ["Barcode"]


# --- colour sharing ----------------------------------------------------------


def test_cluster_colours_are_numeric_aware():
    """`2` must land where a reader expects, not where a string sort puts it."""
    colors = cluster_color_map([str(index) for index in range(12)])
    assert list(colors) == [str(index) for index in range(12)]


def test_the_same_labels_always_get_the_same_colours():
    """A cluster must be one colour across the clustermap and every barplot."""
    labels = np.array(["0", "1", "2", "1"])
    assert cluster_color_map(labels) == cluster_color_map(labels[::-1])


# --- the plot ----------------------------------------------------------------


def _labels_and_groups():
    labels = np.array(["0"] * 30 + ["1"] * 30)
    groups = np.array(["s1"] * 15 + ["s2"] * 15 + ["s1"] * 5 + ["s2"] * 25)
    return labels, groups


def test_composition_reports_group_sizes(tmp_path):
    """A proportion over eleven molecules and one over two thousand look alike.

    The count is what decides whether a compositional shift is worth believing,
    so it is reported rather than left to be inferred from the bar.
    """
    labels, groups = _labels_and_groups()
    result = plot_leiden_composition(labels, groups, save_path=tmp_path / "c.png", min_group_size=1)
    assert result["group_sizes"] == {"s1": 20, "s2": 40}
    assert result["n_groups"] == 2
    assert result["n_clusters"] == 2


def test_small_groups_are_omitted(tmp_path):
    labels = np.array(["0"] * 20 + ["1"] * 3)
    groups = np.array(["big"] * 20 + ["tiny"] * 3)
    result = plot_leiden_composition(
        labels, groups, save_path=tmp_path / "c.png", min_group_size=10
    )
    assert result["n_groups"] == 1
    assert "tiny" not in result["group_sizes"]


def test_no_group_large_enough_draws_nothing(tmp_path):
    """Returning None beats an empty axes that looks like a real result."""
    labels = np.array(["0", "1"])
    groups = np.array(["a", "b"])
    assert (
        plot_leiden_composition(labels, groups, save_path=tmp_path / "c.png", min_group_size=10)
        is None
    )
    assert not list(tmp_path.glob("*.png"))


def test_the_figure_reaches_disk(tmp_path):
    labels, groups = _labels_and_groups()
    result = plot_leiden_composition(labels, groups, save_path=tmp_path / "c.png", min_group_size=1)
    assert (tmp_path / "c.png").is_file()
    assert result["output_path"].endswith("c.png")


def test_an_externally_supplied_colour_map_is_honoured(tmp_path):
    """The caller passes the embedding's map so all its figures agree."""
    labels, groups = _labels_and_groups()
    colors = cluster_color_map(np.array(["0", "1", "2"]))
    result = plot_leiden_composition(
        labels, groups, save_path=tmp_path / "c.png", min_group_size=1, color_map=colors
    )
    # Cluster "2" is absent from this unit; it must not create an empty stack.
    assert result["n_clusters"] == 2


def test_counts_mode_reports_the_same_sizes(tmp_path):
    labels, groups = _labels_and_groups()
    result = plot_leiden_composition(
        labels, groups, save_path=tmp_path / "c.png", min_group_size=1, normalize=False
    )
    assert result["group_sizes"] == {"s1": 20, "s2": 40}


# --- config ------------------------------------------------------------------


def test_config_knobs():
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.latent_plot_composition is True
    assert cfg.latent_composition_group_columns == []
    assert cfg.latent_composition_min_group_size == 10

    parsed, _ = ExperimentConfig.from_var_dict(
        {
            "latent_composition_group_columns": "Barcode,biorep",
            "latent_composition_min_group_size": "25",
        }
    )
    assert parsed.latent_composition_group_columns == ["Barcode", "biorep"]
    assert parsed.latent_composition_min_group_size == 25
