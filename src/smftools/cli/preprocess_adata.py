from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.constants import LOGGING_DIR, PREPROCESS_DIR
from smftools.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def preprocess_adata(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path], Optional[ad.AnnData], Optional[Path]]:
    """
    CLI-facing wrapper for preprocessing.

    Called by: `smftools preprocess <config_path>`

    - Ensure a raw AnnData exists (or some later-stage AnnData) via `load_adata`.
    - Determine which AnnData stages exist (raw, pp, pp_dedup, spatial, hmm).
    - Respect cfg flags (force_redo_preprocessing, force_redo_flag_duplicate_reads).
    - Decide what starting AnnData to load (or whether to early-return).
    - Call `preprocess_adata_core(...)` when appropriate.

    Returns
    -------
    pp_adata : AnnData | None
        Preprocessed AnnData (may be None if we skipped work).
    pp_adata_path : Path | None
        Path to preprocessed AnnData.
    pp_dedup_adata : AnnData | None
        Preprocessed, duplicate-removed AnnData.
    pp_dedup_adata_path : Path | None
        Path to preprocessed, duplicate-removed AnnData.
    """
    from ..readwrite import safe_read_h5ad
    from .helpers import get_adata_paths, load_experiment_config

    # 1) Ensure config is loaded and at least *some* AnnData stage exists
    cfg = load_experiment_config(config_path)

    # 2) Compute canonical paths
    paths = get_adata_paths(cfg)
    raw_path = paths.raw
    pp_path = paths.pp
    pp_dedup_path = paths.pp_dedup

    raw_exists = raw_path.exists()
    pp_exists = pp_path.exists()
    pp_dedup_exists = pp_dedup_path.exists()

    # Helper: read from disk
    def _load(path: Path):
        adata, _ = safe_read_h5ad(path)
        return adata

    # -----------------------------
    # Case A: full redo of preprocessing
    # -----------------------------
    if getattr(cfg, "force_redo_preprocessing", False):
        logger.info("Forcing full redo of preprocessing workflow.")
        if pp_exists:
            adata = _load(pp_path)
            source_path = pp_path
        elif raw_exists:
            adata = _load(raw_path)
            source_path = raw_path
        else:
            logger.error("Cannot redo preprocessing: no AnnData available at any stage.")
            return (None, None, None, None)

        pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path = preprocess_adata_core(
            adata=adata,
            cfg=cfg,
            pp_adata_path=pp_path,
            pp_dup_rem_adata_path=pp_dedup_path,
            source_adata_path=source_path,
            config_path=config_path,
        )
        return pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path

    # -----------------------------
    # Case B: redo duplicate detection only
    # -----------------------------
    if getattr(cfg, "force_redo_flag_duplicate_reads", False):
        logger.info(
            "Forcing redo of duplicate detection workflow, starting from the preprocessed AnnData "
            "if available. Otherwise, will use the raw AnnData."
        )
        if pp_exists:
            adata = _load(pp_path)
            source_path = pp_path
        elif raw_exists:
            adata = _load(raw_path)
            source_path = raw_path
        else:
            logger.error(
                "Cannot redo duplicate detection: no compatible AnnData available "
                "(need at least raw or preprocessed)."
            )
            return (None, None, None, None)

        pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path = preprocess_adata_core(
            adata=adata,
            cfg=cfg,
            pp_adata_path=pp_path,
            pp_dup_rem_adata_path=pp_dedup_path,
            source_adata_path=source_path,
            config_path=config_path,
        )
        return pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path

    # -----------------------------
    # Case C: normal behavior (no explicit redo flags)
    # -----------------------------

    # If pp_dedup exists, just return paths (no recomputation)
    if pp_dedup_exists:
        logger.info(
            f"Skipping preprocessing. Preprocessed deduplicated AnnData found: {pp_dedup_path}"
        )
        return (None, pp_path, None, pp_dedup_path)

    # If pp exists but pp_dedup does not, load pp and run core
    if pp_exists:
        logger.info(f"Preprocessed AnnData found: {pp_path}")
        adata = _load(pp_path)
        source_path = pp_path
        pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path = preprocess_adata_core(
            adata=adata,
            cfg=cfg,
            pp_adata_path=pp_path,
            pp_dup_rem_adata_path=pp_dedup_path,
            source_adata_path=source_path,
            config_path=config_path,
        )
        return pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path

    # Otherwise, fall back to raw (if available)
    if raw_exists:
        adata = _load(raw_path)
        source_path = raw_path
        pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path = preprocess_adata_core(
            adata=adata,
            cfg=cfg,
            pp_adata_path=pp_path,
            pp_dup_rem_adata_path=pp_dedup_path,
            source_adata_path=source_path,
            config_path=config_path,
        )
        return pp_adata, pp_adata_path, pp_dedup_adata, pp_dedup_adata_path

    logger.error("No AnnData available at any stage for preprocessing.")
    return (None, None, None, None)


def preprocess_adata_core(
    adata: ad.AnnData,
    cfg,
    pp_adata_path: Path,
    pp_dup_rem_adata_path: Path,
    source_adata_path: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Path, ad.AnnData, Path]:
    """
    Core preprocessing pipeline.

    Assumes:
    - `adata` is an AnnData object at some stage (raw/pp/etc.) to start preprocessing from.
    - `cfg` is the ExperimentConfig containing all thresholds & options.
    - `pp_adata_path` and `pp_dup_rem_adata_path` are the target output paths for
      preprocessed and preprocessed+deduplicated AnnData.

    Does NOT:
    - Decide which stage to load from (that's the wrapper's job).
    - Decide whether to skip entirely; it always runs its steps, but individual
      sub-steps may skip based on `cfg.bypass_*` or directory existence.

    Returns
    -------
    pp_adata : AnnData
        Preprocessed AnnData (with QC filters, binarization, etc.).
    pp_adata_path : Path
        Path where pp_adata was written.
    pp_dedup_adata : AnnData
        Preprocessed AnnData with duplicate reads removed (for non-direct SMF).
    pp_dup_rem_adata_path : Path
        Path where pp_dedup_adata was written.
    """
    from datetime import datetime
    from pathlib import Path

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        plot_mismatch_base_frequency_by_position,
        plot_read_qc_histograms,
        plot_read_span_quality_clustermaps,
        plot_sequence_integer_encoding_clustermaps,
    )
    from ..preprocessing import (
        append_base_context,
        append_binary_layer_by_base_context,
        append_mismatch_frequency_sites,
        append_sequence_mismatch_annotations,
        binarize_adata,
        binarize_on_Youden,
        calculate_complexity_II,
        calculate_coverage,
        calculate_position_Youden,
        calculate_read_modification_stats,
        clean_NaN,
        filter_reads_on_length_quality_mapping,
        filter_reads_on_modification_thresholds,
        flag_duplicate_reads,
        invert_adata,
        load_sample_sheet,
        reindex_references_adata,
    )
    from ..readwrite import make_dirs
    from .helpers import write_gz_h5ad

    ################################### 1) Load existing  ###################################
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")

    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality  # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    output_directory = Path(
        cfg.output_directory
    )  # Path to the output directory to make for the analysis. Necessary.
    preprocess_directory = output_directory / PREPROCESS_DIR
    logging_directory = preprocess_directory / LOGGING_DIR

    make_dirs([output_directory, preprocess_directory])

    if cfg.emit_log_file:
        log_file = logging_directory / f"{date_str}_{time_str}_log.log"
        make_dirs([logging_directory])
    else:
        log_file = None

    setup_logging(level=log_level, log_file=log_file, reconfigure=log_file is not None)

    ######### Begin Preprocessing #########
    ## Load sample sheet metadata based on barcode mapping ##
    if getattr(cfg, "sample_sheet_path", None):
        load_sample_sheet(
            adata,
            cfg.sample_sheet_path,
            mapping_key_column=cfg.sample_sheet_mapping_column,
            as_category=True,
            force_reload=cfg.force_reload_sample_sheet,
        )
    else:
        pass

    # Adding read length, read quality, reference length, mapped_length, and mapping quality metadata to adata object.
    pp_length_qc_dir = preprocess_directory / "01_Read_length_and_quality_QC_metrics"

    if pp_length_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        logger.debug(f"{pp_length_qc_dir} already exists. Skipping read level QC plotting.")
    else:
        make_dirs([preprocess_directory, pp_length_qc_dir])
        plot_read_qc_histograms(
            adata,
            pp_length_qc_dir,
            cfg.obs_to_plot_pp_qc,
            sample_key=cfg.sample_name_col_for_plotting,
            rows_per_fig=cfg.rows_per_qc_histogram_grid,
        )

    # Filter on read length, read quality, reference length, mapped_length, and mapping quality metadata.
    print(adata.shape)
    adata = filter_reads_on_length_quality_mapping(
        adata,
        filter_on_coordinates=cfg.read_coord_filter,
        read_length=cfg.read_len_filter_thresholds,
        length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds,
        read_quality=cfg.read_quality_filter_thresholds,
        mapping_quality=cfg.read_mapping_quality_filter_thresholds,
        bypass=None,
        force_redo=None,
    )
    print(adata.shape)

    pp_length_qc_dir = preprocess_directory / "02_Read_length_and_quality_QC_metrics_post_filtering"

    if pp_length_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        logger.debug(f"{pp_length_qc_dir} already exists. Skipping read level QC plotting.")
    else:
        make_dirs([preprocess_directory, pp_length_qc_dir])
        plot_read_qc_histograms(
            adata,
            pp_length_qc_dir,
            cfg.obs_to_plot_pp_qc,
            sample_key=cfg.sample_name_col_for_plotting,
            rows_per_fig=cfg.rows_per_qc_histogram_grid,
        )

    ############## Binarize direct modcall data and store in new layer. Clean nans and store as new layers with various nan replacement strategies ##########
    if smf_modality == "direct":
        native = True
        if cfg.fit_position_methylation_thresholds:
            pp_Youden_dir = preprocess_directory / "02B_Position_wide_Youden_threshold_performance"
            make_dirs([pp_Youden_dir])
            # Calculate positional methylation thresholds for mod calls
            calculate_position_Youden(
                adata,
                positive_control_sample=cfg.positive_control_sample_methylation_fitting,
                negative_control_sample=cfg.negative_control_sample_methylation_fitting,
                J_threshold=cfg.fit_j_threshold,
                ref_column=cfg.reference_column,
                sample_column=cfg.sample_column,
                infer_on_percentile=cfg.infer_on_percentile_sample_methylation_fitting,
                inference_variable=cfg.inference_variable_sample_methylation_fitting,
                save=True,
                output_directory=pp_Youden_dir,
            )
            # binarize the modcalls based on the determined thresholds
            binarize_on_Youden(
                adata,
                ref_column=cfg.reference_column,
                output_layer_name=cfg.output_binary_layer_name,
            )
        else:
            binarize_adata(
                adata,
                source="X",
                target_layer=cfg.output_binary_layer_name,
                threshold=cfg.binarize_on_fixed_methlyation_threshold,
            )

        clean_NaN(
            adata,
            layer=cfg.output_binary_layer_name,
            bypass=cfg.bypass_clean_nan,
            force_redo=cfg.force_redo_clean_nan,
        )
    else:
        native = False
        clean_NaN(adata, bypass=cfg.bypass_clean_nan, force_redo=cfg.force_redo_clean_nan)

    ############### Calculate positional coverage by reference set in dataset ###############
    calculate_coverage(
        adata,
        ref_column=cfg.reference_column,
        position_nan_threshold=cfg.position_max_nan_threshold,
        smf_modality=smf_modality,
        target_layer=cfg.output_binary_layer_name,
    )

    ############### Add base context to each position for each Reference_strand and calculate read level methylation/deamination stats ###############
    append_base_context(
        adata,
        ref_column=cfg.reference_column,
        use_consensus=False,
        native=native,
        mod_target_bases=cfg.mod_target_bases,
        bypass=cfg.bypass_append_base_context,
        force_redo=cfg.force_redo_append_base_context,
    )

    ############### Calculate read methylation/deamination statistics for specific base contexts defined by append_base_context ###############
    calculate_read_modification_stats(
        adata,
        cfg.reference_column,
        cfg.sample_column,
        cfg.mod_target_bases,
        bypass=cfg.bypass_calculate_read_modification_stats,
        force_redo=cfg.force_redo_calculate_read_modification_stats,
        smf_modality=cfg.smf_modality,
    )

    ### Make a dir for outputting sample level read modification metrics before filtering ###
    pp_meth_qc_dir = preprocess_directory / "03_read_modification_QC_metrics"

    if pp_meth_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        logger.debug(
            f"{pp_meth_qc_dir} already exists. Skipping read level methylation QC plotting."
        )
    else:
        make_dirs([preprocess_directory, pp_meth_qc_dir])
        obs_to_plot = ["Raw_modification_signal"]
        if any(base in cfg.mod_target_bases for base in ["GpC", "CpG", "C"]):
            obs_to_plot += [
                "Fraction_GpC_site_modified",
                "Fraction_CpG_site_modified",
                "Fraction_other_C_site_modified",
                "Fraction_C_site_modified",
            ]
        if "A" in cfg.mod_target_bases:
            obs_to_plot += ["Fraction_A_site_modified"]
        plot_read_qc_histograms(
            adata,
            pp_meth_qc_dir,
            obs_to_plot,
            sample_key=cfg.sample_name_col_for_plotting,
            rows_per_fig=cfg.rows_per_qc_histogram_grid,
        )

    ##### Optionally filter reads on modification metrics
    adata = filter_reads_on_modification_thresholds(
        adata,
        smf_modality=smf_modality,
        mod_target_bases=cfg.mod_target_bases,
        gpc_thresholds=cfg.read_mod_filtering_gpc_thresholds,
        cpg_thresholds=cfg.read_mod_filtering_cpg_thresholds,
        any_c_thresholds=cfg.read_mod_filtering_c_thresholds,
        a_thresholds=cfg.read_mod_filtering_a_thresholds,
        use_other_c_as_background=cfg.read_mod_filtering_use_other_c_as_background,
        min_valid_fraction_positions_in_read_vs_ref=cfg.min_valid_fraction_positions_in_read_vs_ref,
        bypass=cfg.bypass_filter_reads_on_modification_thresholds,
        force_redo=cfg.force_redo_filter_reads_on_modification_thresholds,
    )

    pp_meth_qc_dir = preprocess_directory / "04_read_modification_QC_metrics_post_filtering"

    if pp_meth_qc_dir.is_dir() and not cfg.force_redo_preprocessing:
        logger.debug(
            f"{pp_meth_qc_dir} already exists. Skipping read level methylation QC plotting."
        )
    else:
        make_dirs([preprocess_directory, pp_meth_qc_dir])
        obs_to_plot = ["Raw_modification_signal"]
        if any(base in cfg.mod_target_bases for base in ["GpC", "CpG", "C"]):
            obs_to_plot += [
                "Fraction_GpC_site_modified",
                "Fraction_CpG_site_modified",
                "Fraction_other_C_site_modified",
                "Fraction_C_site_modified",
            ]
        if "A" in cfg.mod_target_bases:
            obs_to_plot += ["Fraction_A_site_modified"]
        plot_read_qc_histograms(
            adata,
            pp_meth_qc_dir,
            obs_to_plot,
            sample_key=cfg.sample_name_col_for_plotting,
            rows_per_fig=cfg.rows_per_qc_histogram_grid,
        )

    ############### Calculate final positional coverage by reference set in dataset after filtering reads ###############
    calculate_coverage(
        adata,
        ref_column=cfg.reference_column,
        position_nan_threshold=cfg.position_max_nan_threshold,
        smf_modality=smf_modality,
        target_layer=cfg.output_binary_layer_name,
        force_redo=True,
    )

    ############### Add base context to each position for each Reference_strand and calculate read level methylation/deamination stats after filtering reads ###############
    # Additionally, store base_context level binary modification arrays in adata.obsm
    append_base_context(
        adata,
        ref_column=cfg.reference_column,
        use_consensus=False,
        native=native,
        mod_target_bases=cfg.mod_target_bases,
        bypass=cfg.bypass_append_base_context,
        force_redo=True,
    )

    # Add site type binary modification layers for valid coverage sites
    adata = append_binary_layer_by_base_context(
        adata,
        cfg.reference_column,
        smf_modality,
        bypass=cfg.bypass_append_binary_layer_by_base_context,
        force_redo=cfg.force_redo_append_binary_layer_by_base_context,
        from_valid_sites_only=True,
    )

    ############### Reference variant position annotation ###########################
    seq1_col, seq2_col = getattr(cfg, "references_to_align_for_variant_annotation", [None, None])
    if seq1_col and seq2_col:
        append_sequence_mismatch_annotations(
            adata,
            seq1_col,
            seq2_col
        )


    ############### Duplicate detection for conversion/deamination SMF ###############
    if smf_modality != "direct":
        references = adata.obs[cfg.reference_column].cat.categories

        var_filters_sets = []
        for ref in references:
            for site_type in cfg.duplicate_detection_site_types:
                var_filters_sets += [[f"{ref}_{site_type}_site", f"position_in_{ref}"]]

        pp_dup_qc_dir = preprocess_directory / "05_read_duplication_QC_metrics"

        make_dirs([pp_dup_qc_dir])

        # Flag duplicate reads and plot duplicate detection QC
        adata_unique, adata = flag_duplicate_reads(
            adata,
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
            hierarchical_window=cfg.duplicate_detection_window_size_for_hamming_neighbors,
            demux_types=cfg.duplicate_detection_demux_types_to_use,
            demux_col="demux_type",
        )

        # Use the flagged duplicate read groups and perform complexity analysis
        complexity_outs = pp_dup_qc_dir / "sample_complexity_analyses"
        make_dirs([complexity_outs])
        calculate_complexity_II(
            adata=adata,
            output_directory=complexity_outs,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            cluster_col="sequence__merged_cluster_id",
            plot=True,
            save_plot=True,  # set False to display instead
            n_boot=30,
            n_depths=12,
            random_state=42,
            csv_summary=True,
            bypass=cfg.bypass_complexity_analysis,
            force_redo=cfg.force_redo_complexity_analysis,
        )

    else:
        adata_unique = adata
    ########################################################################################################################

    # -----------------------------
    # Optional inversion along positions axis
    # -----------------------------
    if getattr(cfg, "invert_adata", False):
        adata = invert_adata(adata)

    # -----------------------------
    # Optional reindexing by reference
    # -----------------------------
    reindex_references_adata(
        adata,
        reference_col=cfg.reference_column,
        offsets=cfg.reindexing_offsets,
        new_col=cfg.reindexed_var_suffix,
    )

    ############################################### Append mismatch frequency per position ###############################################
    append_mismatch_frequency_sites(
        adata_unique,
        ref_column=cfg.reference_column,
        mismatch_layer=cfg.mismatch_frequency_layer,
        read_span_layer=cfg.mismatch_frequency_read_span_layer,
        mismatch_frequency_range=cfg.mismatch_frequency_range,
        bypass=cfg.bypass_append_mismatch_frequency_sites,
        force_redo=cfg.force_redo_append_mismatch_frequency_sites,
    )

    ############################################### Plot mismatch base frequencies ###############################################
    if cfg.mismatch_frequency_layer not in adata_unique.layers:
        logger.debug(
            "Mismatch layer '%s' not found; skipping mismatch base frequency plots.",
            cfg.mismatch_frequency_layer,
        )
    elif not adata_unique.uns.get("mismatch_integer_encoding_map"):
        logger.debug("Mismatch encoding map not found; skipping mismatch base frequency plots.")
    else:
        pp_mismatch_base_freq_dir = preprocess_directory / "deduplicated" / "08_mismatch_base_frequency_plots"
        if pp_mismatch_base_freq_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{pp_mismatch_base_freq_dir} already exists. Skipping mismatch base frequency plots."
            )
        else:
            make_dirs([pp_mismatch_base_freq_dir])
            plot_mismatch_base_frequency_by_position(
                adata_unique,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                mismatch_layer=cfg.mismatch_frequency_layer,
                read_span_layer=cfg.mismatch_frequency_read_span_layer,
                exclude_mod_sites=True, #cfg.mismatch_base_frequency_exclude_mod_sites,
                mod_site_bases=cfg.mod_target_bases,
                save_path=pp_mismatch_base_freq_dir,
                plot_zscores=True,
            )

    ############################################### Plot integer sequence encoding clustermaps ###############################################
    if "sequence_integer_encoding" not in adata.layers:
        logger.debug(
            "sequence_integer_encoding layer not found; skipping integer encoding clustermaps."
        )
    else:
        pp_seq_clustermap_dir = preprocess_directory / "06_sequence_integer_encoding_clustermaps"
        if pp_seq_clustermap_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{pp_seq_clustermap_dir} already exists. Skipping sequence integer encoding clustermaps."
            )
        else:
            make_dirs([pp_seq_clustermap_dir])
            plot_sequence_integer_encoding_clustermaps(
                adata,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                demux_types=cfg.clustermap_demux_types_to_plot,
                min_quality=None,
                min_length=None,
                min_mapped_length_to_reference_length_ratio=None,
                sort_by="none",
                max_unknown_fraction=0.5,
                save_path=pp_seq_clustermap_dir,
                show_position_axis=True,
            )

        if "mismatch_integer_encoding" in adata.layers:
            pp_mismatch_clustermap_dir = (
                preprocess_directory / "06_mismatch_integer_encoding_clustermaps_no_mod_sites"
            )
            if pp_mismatch_clustermap_dir.is_dir() and not cfg.force_redo_preprocessing:
                logger.debug(
                    f"{pp_mismatch_clustermap_dir} already exists. Skipping mismatch clustermaps without mod sites."
                )
            else:
                make_dirs([pp_mismatch_clustermap_dir])
                plot_sequence_integer_encoding_clustermaps(
                    adata,
                    sample_col=cfg.sample_name_col_for_plotting,
                    reference_col=cfg.reference_column,
                    demux_types=cfg.clustermap_demux_types_to_plot,
                    min_quality=None,
                    min_length=None,
                    min_mapped_length_to_reference_length_ratio=None,
                    sort_by="none",
                    max_unknown_fraction=0.5,
                    save_path=pp_mismatch_clustermap_dir,
                    show_position_axis=True,
                    exclude_mod_sites=True,
                    mod_site_bases=cfg.mod_target_bases,
                )

        pp_dedup_seq_clustermap_dir = (
            preprocess_directory / "deduplicated" / "06_sequence_integer_encoding_clustermaps"
        )
        if pp_dedup_seq_clustermap_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{pp_dedup_seq_clustermap_dir} already exists. Skipping sequence integer encoding clustermaps."
            )
        else:
            make_dirs([pp_dedup_seq_clustermap_dir])
            plot_sequence_integer_encoding_clustermaps(
                adata_unique,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                demux_types=cfg.clustermap_demux_types_to_plot,
                min_quality=None,
                min_length=None,
                min_mapped_length_to_reference_length_ratio=None,
                sort_by="none",
                max_unknown_fraction=0.5,
                save_path=pp_dedup_seq_clustermap_dir,
                show_position_axis=True,
            )

        if "mismatch_integer_encoding" in adata_unique.layers:
            pp_dedup_mismatch_clustermap_dir = (
                preprocess_directory
                / "deduplicated"
                / "06_mismatch_integer_encoding_clustermaps_no_mod_sites"
            )
            if pp_dedup_mismatch_clustermap_dir.is_dir() and not cfg.force_redo_preprocessing:
                logger.debug(
                    f"{pp_dedup_mismatch_clustermap_dir} already exists. "
                    "Skipping mismatch clustermaps without mod sites."
                )
            else:
                make_dirs([pp_dedup_mismatch_clustermap_dir])
                plot_sequence_integer_encoding_clustermaps(
                    adata_unique,
                    sample_col=cfg.sample_name_col_for_plotting,
                    reference_col=cfg.reference_column,
                    demux_types=cfg.clustermap_demux_types_to_plot,
                    min_quality=None,
                    min_length=None,
                    min_mapped_length_to_reference_length_ratio=None,
                    sort_by="none",
                    max_unknown_fraction=0.5,
                    save_path=pp_dedup_mismatch_clustermap_dir,
                    show_position_axis=True,
                    exclude_mod_sites=True,
                    mod_site_bases=cfg.mod_target_bases,
                )

    ############################################### Plot read span mask + base quality clustermaps ###############################################
    quality_layer = None
    if "base_quality_scores" in adata.layers:
        quality_layer = "base_quality_scores"
    elif "base_qualities" in adata.layers:
        quality_layer = "base_qualities"

    if "read_span_mask" not in adata.layers or quality_layer is None:
        logger.debug(
            "read_span_mask and base quality layers not found; skipping read span/base quality clustermaps."
        )
    else:
        pp_span_quality_dir = preprocess_directory / "07_read_span_quality_clustermaps"
        if pp_span_quality_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{pp_span_quality_dir} already exists. Skipping read span/base quality clustermaps."
            )
        else:
            make_dirs([pp_span_quality_dir])
            plot_read_span_quality_clustermaps(
                adata,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                quality_layer=quality_layer,
                read_span_layer="read_span_mask",
                demux_types=cfg.clustermap_demux_types_to_plot,
                save_path=pp_span_quality_dir,
                show_position_axis=True,
                max_nan_fraction=0.5,
            )

        pp_dedup_span_quality_dir = (
            preprocess_directory / "deduplicated" / "07_read_span_quality_clustermaps"
        )
        if pp_dedup_span_quality_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{pp_dedup_span_quality_dir} already exists. Skipping read span/base quality clustermaps."
            )
        elif quality_layer in adata_unique.layers and "read_span_mask" in adata_unique.layers:
            make_dirs([pp_dedup_span_quality_dir])
            plot_read_span_quality_clustermaps(
                adata_unique,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                quality_layer=quality_layer,
                read_span_layer="read_span_mask",
                demux_types=cfg.clustermap_demux_types_to_plot,
                save_path=pp_dedup_span_quality_dir,
                show_position_axis=True,
                max_nan_fraction=0.5,
            )

    ############################################### Save preprocessed adata with duplicate detection ###############################################
    if not pp_adata_path.exists() or cfg.force_redo_preprocessing:
        logger.info("Saving preprocessed adata.")
        record_smftools_metadata(
            adata,
            step_name="preprocess",
            cfg=cfg,
            config_path=config_path,
            input_paths=[source_adata_path] if source_adata_path else None,
            output_path=pp_adata_path,
        )
        write_gz_h5ad(adata, pp_adata_path)

    if not pp_dup_rem_adata_path.exists() or cfg.force_redo_preprocessing:
        logger.info("Saving preprocessed adata with duplicates removed.")
        record_smftools_metadata(
            adata_unique,
            step_name="preprocess",
            cfg=cfg,
            config_path=config_path,
            input_paths=[pp_adata_path],
            output_path=pp_dup_rem_adata_path,
        )
        write_gz_h5ad(adata_unique, pp_dup_rem_adata_path)

    ########################################################################################################################

    return (adata, pp_adata_path, adata_unique, pp_dup_rem_adata_path)
