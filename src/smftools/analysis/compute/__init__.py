"""
smftools.analysis.compute — pure statistical functions.

Array-in, result-out. No file I/O, no AnnData dependency (except ep_classification).

Modules
-------
autocorrelation         binary_autocorrelation_with_spacing(), compute_replicate_curve()
pearson                 nan_pearson_matrix(), make_ticks()
hmm_features            extract_intervals_from_row()
ep_classification       classify_position(), add_ep_obs_columns()
ls_periodicity          analyze_ls_periodicity(), analyze_fft_periodicity()
dimensionality_reduction  run_pipeline(), coverage_filter(), make_features_acf()
read_cache              load_layer(), load_var_info(), load_obs_metadata(), is_cached()
"""
