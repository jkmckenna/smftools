from smftools.config.experiment_config import ExperimentConfig


def test_spatial_read_metric_ranges_parse_from_csv_values(tmp_path):
    input_bam = tmp_path / "input.bam"
    input_bam.write_text("stub")
    cfg, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_bam),
            "output_directory": str(tmp_path / "outputs"),
            "experiment_name": "spatial_read_metrics",
            "spatial_save_read_autocorrelation": "True",
            "spatial_compute_read_lomb_scargle": "True",
            "spatial_plot_read_lomb_scargle": "True",
            "spatial_plot_read_metric_clustermaps": "True",
            "hmm_execution_mode": "partitioned",
            "spatial_lomb_scargle_period_range_bp": "[80, 400]",
            "spatial_lomb_scargle_peak_range_bp": "150, 250",
            "spatial_lomb_scargle_poly_degree": "2",
            "spatial_lomb_scargle_min_sites": "40",
        },
        merge_with_defaults=False,
    )

    assert cfg.spatial_save_read_autocorrelation is True
    assert cfg.spatial_compute_read_lomb_scargle is True
    assert cfg.spatial_plot_read_lomb_scargle is True
    assert cfg.spatial_plot_read_metric_clustermaps is True
    assert cfg.hmm_execution_mode == "partitioned"
    assert cfg.spatial_lomb_scargle_period_range_bp == [80.0, 400.0]
    assert cfg.spatial_lomb_scargle_peak_range_bp == [150.0, 250.0]
    assert cfg.spatial_lomb_scargle_poly_degree == 2
    assert cfg.spatial_lomb_scargle_min_sites == 40
