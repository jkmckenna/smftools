"""Per-strategy Leiden resolution defaults (`EGL-28`).

Chosen against a target of roughly 4-10 clusters, measured by sweeping 13
resolutions across all 24 embeddings of the DAF and EMseq pilots. The values
are empirical, so what these tests protect is not the numbers themselves but
the two ways they can silently stop applying: the dataclass drifting from
`from_var_dict`, and a user override being merged with the defaults instead of
replacing them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smftools.config.experiment_config import (
    _DEFAULT_LEIDEN_RESOLUTION_BY_STRATEGY,
    ExperimentConfig,
)
from smftools.tools.latent_clustering import resolve_parameter

pytestmark = pytest.mark.unit


def test_defaults_are_the_measured_values():
    assert ExperimentConfig().latent_leiden_resolution_by_strategy == {
        "pca": 0.5,
        "umap": 0.01,
        "nmf": 0.01,
        "cp": 0.0005,
    }


def test_a_single_shared_resolution_would_not_do():
    """The reason per-strategy values exist at all.

    PCA needs a resolution 50x UMAP's to reach a comparable cluster count: at
    0.1 PCA collapsed to 1-2 clusters on every pilot unit while UMAP gave
    10-18. If these ever converge to one number, the measurement that justified
    splitting them has been lost.
    """
    resolutions = _DEFAULT_LEIDEN_RESOLUTION_BY_STRATEGY
    assert resolutions["pca"] > resolutions["umap"] * 10


def test_dataclass_and_from_var_dict_agree():
    """The `F18` shape: two copies of a default, free to drift apart."""
    parsed, _ = ExperimentConfig.from_var_dict({})
    assert (
        parsed.latent_leiden_resolution_by_strategy
        == ExperimentConfig().latent_leiden_resolution_by_strategy
    )


def test_an_override_replaces_the_defaults_rather_than_merging():
    """A user setting one strategy must not silently inherit the other three.

    Merging would mean a config that looks like it pins every resolution is
    quietly still using shipped values for the strategies it omits.
    """
    parsed, _ = ExperimentConfig.from_var_dict(
        {"latent_leiden_resolution_by_strategy": {"pca": 0.9}}
    )
    assert parsed.latent_leiden_resolution_by_strategy == {"pca": 0.9}


def test_the_defaults_reach_the_clustering_code():
    """Config and consumer are wired through the same key, not two spellings."""
    cfg = ExperimentConfig()
    assert resolve_parameter(cfg, "pca", "leiden_resolution", 0.1) == 0.5
    assert resolve_parameter(cfg, "umap", "leiden_resolution", 0.1) == 0.01


def test_a_strategy_without_an_override_falls_back_to_the_shared_knob():
    cfg = SimpleNamespace(
        latent_leiden_resolution=0.25, latent_leiden_resolution_by_strategy={"pca": 0.5}
    )
    assert resolve_parameter(cfg, "somethingelse", "leiden_resolution", 0.1) == 0.25


# --- which strategies get figures --------------------------------------------


def test_clustermap_strategies_default_to_pca_and_umap():
    """Clustering runs for every embedding; only the *figures* are bounded.

    12 embeddings per unit renders 24 clustermaps per pilot run, and the
    multiplication to watch is references x regions.
    """
    assert ExperimentConfig().latent_clustermap_strategies == ["pca", "umap"]


def test_clustermap_strategies_survive_the_config_path():
    parsed, _ = ExperimentConfig.from_var_dict({})
    assert parsed.latent_clustermap_strategies == ["pca", "umap"]


def test_clustermap_strategies_are_overridable():
    parsed, _ = ExperimentConfig.from_var_dict({"latent_clustermap_strategies": "pca,nmf"})
    assert parsed.latent_clustermap_strategies == ["pca", "nmf"]


def test_clustering_is_not_narrowed_by_the_plot_selection():
    """Selecting figures must not stop the other embeddings being clustered.

    The labels are stored per embedding and are useful beyond plotting, so
    narrowing the plot set must not narrow the analysis.
    """
    cfg = ExperimentConfig()
    assert cfg.latent_cluster_embeddings is True
    assert set(cfg.latent_leiden_resolution_by_strategy) > set(cfg.latent_clustermap_strategies)
