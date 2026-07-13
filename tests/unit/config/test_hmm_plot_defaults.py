from smftools.config.experiment_config import ExperimentConfig


def test_hmm_plot_defaults_include_merged_footprint_layers():
    cfg = ExperimentConfig()

    assert "all_footprint_features_merged" in cfg.hmm_clustermap_feature_layers
    assert "all_footprint_features_merged" in cfg.hmm_clustermap_length_layers
