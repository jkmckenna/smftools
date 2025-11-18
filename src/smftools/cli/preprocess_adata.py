def preprocess_adata(config_path):
    """
    High-level function to call for preprocessing an adata object. 
    Command line accesses this through smftools preprocess <config_path>

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        (pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path)
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad, make_dirs, add_or_update_column_in_csv
    from .load_adata import load_adata

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc

    import os
    from importlib import resources
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")

    ################################### 1) Load existing  ###################################
    initial_adata, initial_adata_path, bam_files, cfg = load_adata(config_path)

    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    output_directory = Path(cfg.output_directory)  # Path to the output directory to make for the analysis. Necessary.

    # Make initial output directory
    make_dirs([output_directory])

    # Preprocessed adata path info
    pp_adata_basename = initial_adata_path.name.split(".")[0] + '_preprocessed.h5ad.gz'
    pp_adata_path = initial_adata_path.parent / pp_adata_basename

    # Preprocessed duplicate removed adata path info
    pp_dup_rem_adata_basename = pp_adata_path.name.split(".")[0] + '_duplicates_removed.h5ad.gz'
    pp_dup_rem_adata_path = pp_adata_path.parent / pp_dup_rem_adata_basename

    if initial_adata:
        # This happens on first run of the load pipeline
        adata = initial_adata

    else:
        # If an anndata is saved, check which stages of the anndata are available
        initial_version_available = initial_adata_path.exists()
        preprocessed_version_available = pp_adata_path.exists()
        preprocessed_dup_removed_version_available = pp_dup_rem_adata_path.exists()

        if cfg.force_redo_preprocessing:
            print(f"Forcing full redo of preprocessing workflow, starting from earliest stage adata available.")
            if initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            elif preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path)
            else:
                print(f"Can not redo preprocessing when there is no adata available.")
                return
        elif cfg.force_redo_flag_duplicate_reads:
            print(f"Forcing redo of duplicate detection workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
                return
        elif cfg.force_redo_basic_analyses:
            print(f"Forcing redo of basic analysis workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
        elif preprocessed_version_available or preprocessed_dup_removed_version_available:
            print(f"Preprocessed anndatas found: {pp_dup_rem_adata_path} and {pp_adata_path}")
            return (None, pp_adata_path, None, pp_dup_rem_adata_path)
        elif initial_version_available:
            adata, load_report = safe_read_h5ad(initial_adata_path)
        else:
            print(f"No adata available.")
            return
            
    ######### Begin Preprocessing #########
    pp_dir = output_directory / "preprocessed"

    ## Load sample sheet metadata based on barcode mapping ##
    if cfg.sample_sheet_path:
        from ..preprocessing import load_sample_sheet
        load_sample_sheet(adata, 
                          cfg.sample_sheet_path, 
                          mapping_key_column=cfg.sample_sheet_mapping_column, 
                          as_category=True,
                          force_reload=cfg.force_reload_sample_sheet)
    else:
        pass
    
    # Adding read length, read quality, reference length, mapped_length, and mapping quality metadata to adata object.
    pp_length_qc_dir = pp_dir / "01_Read_length_and_quality_QC_metrics"
    from ..preprocessing import add_read_length_and_mapping_qc
    from ..informatics.bam_functions import extract_read_features_from_bam  
    add_read_length_and_mapping_qc(adata, bam_files, 
                                   extract_read_features_from_bam_callable=extract_read_features_from_bam, 
                                   bypass=cfg.bypass_add_read_length_and_mapping_qc,
                                   force_redo=cfg.force_redo_add_read_length_and_mapping_qc)

    adata.obs['Raw_modification_signal'] =  np.nansum(adata.X, axis=1)

    if pp_length_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        print( f'{pp_length_qc_dir} already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(adata,
                                pp_length_qc_dir, 
                                obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)

    # Filter on read length, read quality, reference length, mapped_length, and mapping quality metadata.
    from ..preprocessing import filter_reads_on_length_quality_mapping
    print(adata.shape)
    adata = filter_reads_on_length_quality_mapping(adata, 
                                                         filter_on_coordinates=cfg.read_coord_filter,
                                                         read_length=cfg.read_len_filter_thresholds,
                                                         length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds, 
                                                         read_quality=cfg.read_quality_filter_thresholds,
                                                         mapping_quality=cfg.read_mapping_quality_filter_thresholds,
                                                         bypass=None,
                                                         force_redo=None)
    print(adata.shape)

    pp_length_qc_dir = pp_dir / "02_Read_length_and_quality_QC_metrics_post_filtering"

    if pp_length_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        print( f'{pp_length_qc_dir} already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(adata,
                                pp_length_qc_dir, 
                                obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)
        
    ############## Binarize direct modcall data and store in new layer. Clean nans and store as new layers with various nan replacement strategies ##########
    from ..preprocessing import clean_NaN
    if smf_modality == 'direct':
        from ..preprocessing import calculate_position_Youden, binarize_on_Youden, binarize_adata
        native = True
        if cfg.fit_position_methylation_thresholds:
            # Calculate positional methylation thresholds for mod calls
            calculate_position_Youden(adata, 
                                    positive_control_sample=cfg.positive_control_sample_methylation_fitting, 
                                    negative_control_sample=cfg.negative_control_sample_methylation_fitting, 
                                    J_threshold=cfg.fit_j_threshold, 
                                    obs_column=cfg.reference_column, 
                                    infer_on_percentile=cfg.infer_on_percentile_sample_methylation_fitting, 
                                    inference_variable=cfg.inference_variable_sample_methylation_fitting, 
                                    save=False, 
                                    output_directory=''
                                    )
            # binarize the modcalls based on the determined thresholds
            binarize_on_Youden(adata, 
                            obs_column=cfg.reference_column,
                            output_layer_name=cfg.output_binary_layer_name
                            )
        else:
            binarize_adata(adata, source="X", target_layer=cfg.output_binary_layer_name, threshold=cfg.binarize_on_fixed_methlyation_threshold)

        clean_NaN(adata, 
                  layer=cfg.output_binary_layer_name,
                  bypass=cfg.bypass_clean_nan, 
                  force_redo=cfg.force_redo_clean_nan
                  )
    else:
        native = False
        clean_NaN(adata, 
                  bypass=cfg.bypass_clean_nan, 
                  force_redo=cfg.force_redo_clean_nan
                  )

    ############### Add base context to each position for each Reference_strand and calculate read level methylation/deamination stats ###############
    from ..preprocessing import append_base_context, append_binary_layer_by_base_context
    # Additionally, store base_context level binary modification arrays in adata.obsm
    append_base_context(adata, 
                        obs_column=cfg.reference_column, 
                        use_consensus=False, 
                        native=native, 
                        mod_target_bases=cfg.mod_target_bases,
                        bypass=cfg.bypass_append_base_context,
                        force_redo=cfg.force_redo_append_base_context)
    
    adata = append_binary_layer_by_base_context(adata, 
                                                cfg.reference_column, 
                                                smf_modality,
                                                bypass=cfg.bypass_append_binary_layer_by_base_context,
                                                force_redo=cfg.force_redo_append_binary_layer_by_base_context)
    
    ############### Optional inversion of the adata along positions axis ###################
    if cfg.invert_adata:
        from ..preprocessing import invert_adata
        adata = invert_adata(adata)

    ############### Calculate read methylation/deamination statistics for specific base contexts defined above ###############
    from ..preprocessing import calculate_read_modification_stats
    calculate_read_modification_stats(adata, 
                                      cfg.reference_column, 
                                      cfg.sample_column,
                                      cfg.mod_target_bases,
                                      bypass=cfg.bypass_calculate_read_modification_stats,
                                      force_redo=cfg.force_redo_calculate_read_modification_stats)
    
    ### Make a dir for outputting sample level read modification metrics before filtering ###
    pp_meth_qc_dir = pp_dir / "03_read_modification_QC_metrics"

    if pp_meth_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        print(f'{pp_meth_qc_dir} already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in cfg.mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in cfg.mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(adata, 
                                pp_meth_qc_dir, obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ##### Optionally filter reads on modification metrics
    from ..preprocessing import filter_reads_on_modification_thresholds
    adata = filter_reads_on_modification_thresholds(adata, 
                                                          smf_modality=smf_modality,
                                                          mod_target_bases=cfg.mod_target_bases,
                                                          gpc_thresholds=cfg.read_mod_filtering_gpc_thresholds, 
                                                          cpg_thresholds=cfg.read_mod_filtering_cpg_thresholds,
                                                          any_c_thresholds=cfg.read_mod_filtering_any_c_thresholds,
                                                          a_thresholds=cfg.read_mod_filtering_a_thresholds,
                                                          use_other_c_as_background=cfg.read_mod_filtering_use_other_c_as_background,
                                                          min_valid_fraction_positions_in_read_vs_ref=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                          bypass=cfg.bypass_filter_reads_on_modification_thresholds,
                                                          force_redo=cfg.force_redo_filter_reads_on_modification_thresholds)
    
    pp_meth_qc_dir = pp_dir / "04_read_modification_QC_metrics_post_filtering"
    
    if pp_meth_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        print(f'{pp_meth_qc_dir} already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in cfg.mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in cfg.mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(adata, 
                                pp_meth_qc_dir, obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)
        
    ############### Calculate positional coverage in dataset ###############
    from ..preprocessing import calculate_coverage
    calculate_coverage(adata, 
                       obs_column=cfg.reference_column, 
                       position_nan_threshold=0.1)

    ############### Duplicate detection for conversion/deamination SMF ###############
    if smf_modality != 'direct':
        from ..preprocessing import flag_duplicate_reads, calculate_complexity_II
        references = adata.obs[cfg.reference_column].cat.categories

        var_filters_sets =[]
        for ref in references:
            for site_type in cfg.duplicate_detection_site_types:
                var_filters_sets += [[f"{ref}_{site_type}_site", f"position_in_{ref}"]]

        pp_dup_qc_dir = pp_dir / "05_read_duplication_QC_metrics"

        make_dirs([pp_dup_qc_dir])

        # Flag duplicate reads and plot duplicate detection QC
        adata_unique, adata = flag_duplicate_reads(adata, 
                                                    var_filters_sets, 
                                                    distance_threshold=cfg.duplicate_detection_distance_threshold, 
                                                    obs_reference_col=cfg.reference_column, 
                                                    sample_col=cfg.sample_name_col_for_plotting,
                                                    output_directory=pp_dup_qc_dir,
                                                    metric_keys=cfg.hamming_vs_metric_keys,
                                                    keep_best_metric=cfg.duplicate_detection_keep_best_metric,
                                                    bypass=cfg.bypass_flag_duplicate_reads,
                                                    force_redo=cfg.force_redo_flag_duplicate_reads,
                                                    window_size=cfg.duplicate_detection_window_size_for_hamming_neighbors,
                                                    min_overlap_positions=cfg.duplicate_detection_min_overlapping_positions,
                                                    do_pca=cfg.duplicate_detection_do_pca,
                                                    pca_n_components=50,
                                                    pca_center=True,
                                                    do_hierarchical=cfg.duplicate_detection_do_hierarchical,
                                                    hierarchical_linkage=cfg.duplicate_detection_hierarchical_linkage,
                                                    hierarchical_metric="euclidean",
                                                    hierarchical_window=cfg.duplicate_detection_window_size_for_hamming_neighbors
                                                    )
        
        # Use the flagged duplicate read groups and perform complexity analysis
        complexity_outs = os.path.join(pp_dup_qc_dir, "sample_complexity_analyses")
        make_dirs([complexity_outs])
        calculate_complexity_II(
            adata=adata,
            output_directory=complexity_outs,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            cluster_col='sequence__merged_cluster_id',
            plot=True,
            save_plot=True,   # set False to display instead
            n_boot=30,
            n_depths=12,
            random_state=42,
            csv_summary=True,
            bypass=cfg.bypass_complexity_analysis,
            force_redo=cfg.force_redo_complexity_analysis
        )

    else:
        adata_unique = adata

    ########################################################################################################################

    ############################################### Save preprocessed adata with duplicate detection ###############################################
    from ..readwrite import safe_write_h5ad
    if not pp_adata_path.exists() or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata.')
        if ".gz" == pp_adata_path.suffix:
            safe_write_h5ad(adata, pp_adata_path, compression='gzip', backup=True)
        else:
            pp_adata_path = pp_adata_path.with_name(pp_adata_path.name + '.gz')
            safe_write_h5ad(adata, pp_adata_path, compression='gzip', backup=True)

    if not pp_dup_rem_adata_path.exists() or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata with duplicates removed.')
        if ".gz" == pp_dup_rem_adata_path.suffix:
            safe_write_h5ad(adata_unique, pp_dup_rem_adata_path, compression='gzip', backup=True) 
        else:
            pp_adata_path = pp_dup_rem_adata_path.with_name(pp_dup_rem_adata_path.name + '.gz')
            safe_write_h5ad(adata_unique, pp_dup_rem_adata_path, compression='gzip', backup=True)
    ########################################################################################################################

    add_or_update_column_in_csv(cfg.summary_file, "pp_adata", pp_adata_path)
    add_or_update_column_in_csv(cfg.summary_file, "pp_dedup_adata", pp_dup_rem_adata_path)

    return (adata, pp_adata_path, adata_unique, pp_dup_rem_adata_path)