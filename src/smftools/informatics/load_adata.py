## load_adata

def load_adata(config_path):
    """
    High-level function to call for converting raw sequencing data to an adata object. 
    Works for nanopore pod5, fast5, and unaligned modBAM data types for direct SMF workflows.
    Works for nanopore pod5, fast5, unaligned BAM for conversion SMF workflows.
    Also works for illumina fastq and unaligned BAM for conversion SMF workflows.

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        None
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad
    from .config import LoadExperimentConfig, ExperimentConfig
    from .helpers import discover_input_files, make_dirs, concatenate_fastqs_to_bam, extract_read_features_from_bam
    from .fast5_to_pod5 import fast5_to_pod5
    from .subsample_fasta_from_bed import subsample_fasta_from_bed

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc

    import os
    from importlib import resources
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")


    ################################### 1) General params and input organization ###################################

    # Load experiment config parameters into global variables
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("informatics/config")
    cfg, report = ExperimentConfig.from_var_dict(loader.var_dict, date_str=date_str, defaults_dir=defaults_dir)

    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    input_data_path = cfg.input_data_path  # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = cfg.output_directory  # Path to the output directory to make for the analysis. Necessary.
    fasta = cfg.fasta  # Path to reference FASTA. Necessary.

    bam_suffix = cfg.bam_suffix
    split_dir = cfg.split_dir
    strands = cfg.strands

    # General config variable init - Optional user passed inputs for enzyme base specificity
    mod_target_bases = cfg.mod_target_bases  # Nucleobases of interest that may be modified. ['GpC', 'CpG', 'C', 'A']

    # Conversion/deamination specific variable init
    conversion_types = cfg.conversion_types  # 5mC
    conversions = cfg.conversions

    # Common Anndata accession params
    reference_column = cfg.reference_column

    # Make initial output directory
    make_dirs([output_directory])
    os.chdir(output_directory)

    # Define the pathname to split BAMs into later during demultiplexing.
    split_path = os.path.join(output_directory, split_dir)

    # If conversion_types is passed:
    if conversion_types:
        conversions += conversion_types

    # Detect the input filetype
    if Path(input_data_path).is_file():
        input_data_filetype = '.' + os.path.basename(input_data_path).split('.')[1].lower()
        input_is_pod5 = input_data_filetype in ['.pod5','.p5']
        input_is_fast5 = input_data_filetype in ['.fast5','.f5']
        input_is_fastq = input_data_filetype in ['.fastq', '.fq']
        input_is_bam = input_data_filetype == bam_suffix
        if input_is_fastq:
            fastq_paths = [input_data_path]
    elif Path(input_data_path).is_dir():
        found = discover_input_files(input_data_path, bam_suffix=bam_suffix, recursive=cfg.recursive_input_search)

        input_is_pod5 = found["input_is_pod5"]
        input_is_fast5 = found["input_is_fast5"]
        input_is_fastq = found["input_is_fastq"]
        input_is_bam = found["input_is_bam"]

        pod5_paths = found["pod5_paths"]
        fast5_paths = found["fast5_paths"]
        fastq_paths = found["fastq_paths"]
        bam_paths = found["bam_paths"]

        print(f"Found {found['all_files_searched']} files; fastq={len(fastq_paths)}, bam={len(bam_paths)}, pod5={len(pod5_paths)}, fast5={len(fast5_paths)}")

    # If the input files are not pod5 files, and they are fast5 files, convert the files to a pod5 file before proceeding.
    if input_is_fast5 and not input_is_pod5:
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        output_pod5 = os.path.join(output_directory, 'FAST5s_to_POD5.pod5')
        print(f'Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}')
        fast5_to_pod5(input_data_path, output_pod5)
        # Reassign the pod5_dir variable to point to the new pod5 file.
        input_data_path = output_pod5
        input_is_pod5 = True
        input_is_fast5 = False
    
    # If the input is a fastq or a directory of fastqs, concatenate them into an unaligned BAM and save the barcode
    elif input_is_fastq:
        output_bam = os.path.join(output_directory, 'FASTQs_concatenated_into_BAM.bam')

        concatenate_fastqs_to_bam(
            fastq_paths,
            output_bam,
            barcode_tag='BC',
            gzip_suffixes=('.gz','.gzip'),
            barcode_map=cfg.fastq_barcode_map,
            add_read_group=True,
            rg_sample_field=None,
            progress=False,
            auto_pair=cfg.fastq_auto_pairing)

        input_data_path = output_bam
        input_is_bam = True
        input_is_fastq = False

    # Determine if the input data needs to be basecalled
    if input_is_pod5:
        basecall = True
    elif input_is_bam:
        basecall = False
    else:
        print('Error, can not find input bam or pod5')

    # Generate the base name of the unaligned bam without the .bam suffix
    if basecall:
        model_basename = os.path.basename(cfg.model)
        model_basename = model_basename.replace('.', '_')
        if smf_modality == 'direct':
            mod_string = "_".join(cfg.mod_list)
            bam=f"{output_directory}/{model_basename}_{mod_string}_calls"
        else:
            bam=f"{output_directory}/{model_basename}_canonical_basecalls"
    else:
        bam_base=os.path.basename(input_data_path).split('.bam')[0]
        bam=os.path.join(output_directory, bam_base)

    # Generate path names for the unaligned, aligned, as well as the aligned/sorted bam.
    unaligned_output = bam + bam_suffix
    aligned_BAM=f"{bam}_aligned"
    aligned_output = aligned_BAM + bam_suffix
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix

    # Naming of the demultiplexed output directory
    if cfg.barcode_both_ends:
        split_dir = split_dir + '_both_ends_barcoded'
    else:
        split_dir = split_dir + '_at_least_one_end_barcoded'

    # Direct methylation detection SMF specific parameters
    if smf_modality == 'direct':
        mod_bed_dir=f"{split_dir}/split_mod_beds"
        mod_tsv_dir=f"{split_dir}/split_mod_tsvs"
        bam_qc_dir = f"{split_dir}/bam_qc"
        mod_map = {'6mA': '6mA', '5mC_5hmC': '5mC'}
        mods = [mod_map[mod] for mod in cfg.mod_list]

    os.chdir(output_directory)
    ########################################################################################################################

    ################################### 2) FASTA Handling ###################################
    from .helpers import generate_converted_FASTA, get_chromosome_lengths

    # If fasta_regions_of_interest bed is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if cfg.fasta_regions_of_interest and '.bed' in cfg.fasta_regions_of_interest:
        fasta_basename = os.path.basename(fasta).split('.fa')[0]
        bed_basename_minus_suffix = os.path.basename(cfg.fasta_regions_of_interest).split('.bed')[0]
        output_FASTA = fasta_basename + '_subsampled_by_' + bed_basename_minus_suffix + '.fasta'
        subsample_fasta_from_bed(fasta, cfg.fasta_regions_of_interest, output_directory, output_FASTA)
        fasta = os.path.join(output_directory, output_FASTA)

    # For conversion style SMF, make a converted reference FASTA
    if smf_modality == 'conversion':
        fasta_basename = os.path.basename(fasta)
        converted_FASTA_basename = fasta_basename.split('.fa')[0]+'_converted.fasta'
        converted_FASTA = os.path.join(output_directory, converted_FASTA_basename)
        if 'converted.fa' in fasta:
            print(fasta + ' is already converted. Using existing converted FASTA.')
            converted_FASTA = fasta
        elif os.path.exists(converted_FASTA):
            print(converted_FASTA + ' already exists. Using existing converted FASTA.')
        else:
            generate_converted_FASTA(fasta, conversion_types, strands, converted_FASTA)
        fasta = converted_FASTA

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)
    ########################################################################################################################

    ################################### 3) Basecalling ###################################
    from .helpers import modcall, canoncall
    # 1) Basecall using dorado
    if basecall and cfg.sequencer == 'ont':
        if os.path.exists(unaligned_output):
            print(unaligned_output + ' already exists. Using existing basecalled BAM.')
        elif smf_modality != 'direct':
            canoncall(cfg.model_dir, cfg.model, input_data_path, cfg.barcode_kit, bam, bam_suffix, cfg.barcode_both_ends, cfg.trim, cfg.device)
        else:
            modcall(cfg.model_dir, cfg.model, input_data_path, cfg.barcode_kit, cfg.mod_list, bam, bam_suffix, cfg.barcode_both_ends, cfg.trim, cfg.device)
    elif basecall:
        print(f"Basecalling is currently only supported for ont sequencers and not pacbio.")
    else:
        pass
    ########################################################################################################################

    ################################### 4) Alignment and sorting #############################################
    from .helpers import align_and_sort_BAM, aligned_BAM_to_bed
    # 3) Align the BAM to the reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    if os.path.exists(aligned_output) and os.path.exists(aligned_sorted_output):
        print(aligned_sorted_output + ' already exists. Using existing aligned/sorted BAM.')
    else:
        align_and_sort_BAM(fasta, unaligned_output, bam_suffix, output_directory, cfg.make_bigwigs, cfg.threads, cfg.aligner, cfg.aligner_args)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(output_directory, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for ' + aligned_sorted_output)
    else:
        aligned_BAM_to_bed(aligned_output, output_directory, fasta, cfg.make_bigwigs, cfg.threads)
    ########################################################################################################################

    ################################### 5) Demultiplexing ######################################################################
    from .helpers import demux_and_index_BAM
    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if os.path.isdir(split_dir):
        print(split_dir + ' already exists. Using existing demultiplexed BAMs.')
        bam_files = os.listdir(split_dir)
        bam_files = [os.path.join(split_dir, file) for file in bam_files if '.bam' in file and '.bai' not in file and 'unclassified' not in file]
        bam_files.sort()
    else:
        make_dirs([split_dir])
        bam_files = demux_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, cfg.barcode_kit, cfg.barcode_both_ends, cfg.trim, fasta, cfg.make_bigwigs, cfg.threads)
        # split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, output_directory, converted_FASTA) # deprecated, just use dorado demux

    # Make beds and provide basic histograms
    bed_dir = os.path.join(split_dir, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for demultiplexed bams')
    else:
        for bam in bam_files:
            aligned_BAM_to_bed(bam, split_dir, fasta, cfg.make_bigwigs, cfg.threads)
    ########################################################################################################################

    ################################### 6) SAMTools based BAM QC ######################################################################
    from .helpers import bam_qc
    # 5) Samtools QC metrics on split BAM files
    bam_qc_dir = f"{split_dir}/bam_qc"
    if os.path.isdir(bam_qc_dir):
        print(bam_qc_dir + ' already exists. Using existing BAM QC calculations.')
    else:
        make_dirs([bam_qc_dir])
        bam_qc(bam_files, bam_qc_dir, cfg.threads, modality=smf_modality)
    ######################################################################################################################## 

    ################################### 7) AnnData loading ######################################################################
    if smf_modality != 'direct':
        from .helpers import converted_BAM_to_adata_II
        # 6) Take the converted BAM and load it into an adata object.
        if smf_modality == 'deaminase':
            deaminase_footprinting = True
        else:
            deaminase_footprinting = False
        final_adata, final_adata_path = converted_BAM_to_adata_II(fasta, 
                                                                  split_dir, 
                                                                  cfg.mapping_threshold, 
                                                                  cfg.experiment_name, 
                                                                  conversion_types, 
                                                                  bam_suffix, 
                                                                  cfg.device, 
                                                                  cfg.threads, 
                                                                  deaminase_footprinting,
                                                                  delete_intermediates=cfg.delete_intermediate_hdfs) 
    else:
        if os.path.isdir(mod_bed_dir):
            print(mod_bed_dir + ' already exists, skipping making modbeds')
        else:
            from .helpers import modQC, make_modbed, extract_mods, modkit_extract_to_adata
            make_dirs([mod_bed_dir])  
            modQC(aligned_sorted_output, cfg.thresholds) # get QC metrics for mod calls
            make_modbed(aligned_sorted_output, cfg.thresholds, mod_bed_dir) # Generate bed files of position methylation summaries for every sample

            make_dirs([mod_tsv_dir])  
            extract_mods(cfg.thresholds, mod_tsv_dir, split_dir, bam_suffix, cfg.skip_unclassified, cfg.threads) # Extract methylations calls for split BAM files into split TSV files

            #6 Load the modification data from TSVs into an adata object
            final_adata, final_adata_path = modkit_extract_to_adata(fasta, 
                                                                    split_dir, 
                                                                    cfg.mapping_threshold, 
                                                                    cfg.experiment_name, 
                                                                    mods, 
                                                                    cfg.batch_size, 
                                                                    mod_tsv_dir, 
                                                                    cfg.delete_batch_hdfs, 
                                                                    cfg.threads)

    ########################################################################################################################

    ############################################### 8) Basic Read quality metrics: Read length, read quality, mapping quality, etc #################################################

    # Preprocessed adata path info
    pp_adata_basename = os.path.basename(final_adata_path).split('.h5ad')[0] + '_preprocessed.h5ad.gz'
    pp_adata_path = os.path.join(os.path.dirname(final_adata_path), pp_adata_basename)
    pp_backup_dir=os.path.join(os.path.dirname(pp_adata_path), 'pp_adata_accessory_data')

    pp_dup_rem_adata_basename = os.path.basename(pp_adata_basename).split('.h5ad')[0] + '_duplicates_removed.h5ad.gz'
    pp_dup_rem_adata_path = os.path.join(os.path.dirname(pp_adata_path), pp_dup_rem_adata_basename)
    pp_dup_rem_backup_dir=os.path.join(os.path.dirname(pp_dup_rem_adata_path), 'pp_dup_rem_adata_accessory_data')

    analyzed_adata_basename = os.path.basename(pp_dup_rem_adata_path).split('.h5ad')[0] + '_analyzed_I.h5ad.gz'
    analyzed_adata_path = os.path.join(os.path.dirname(pp_dup_rem_adata_path), analyzed_adata_basename)
    analyzed_backup_dir=os.path.join(os.path.dirname(pp_dup_rem_adata_path), 'duplicate_removed_analyzed_adata_I_accessory_data')

    if final_adata:
        # This happens on first run before any adata is ever saved
        pass
    else:
        # If an anndata is saved, first check if there is already a preprocessed version.
        preprocessed_version_available = os.path.exists(pp_adata_path)
        preprocessed_dup_removed_version_available = os.path.exists(pp_dup_rem_adata_path)
        preprocessed_dup_removed_analyzed_I_version_available = os.path.exists(analyzed_adata_path)

        if preprocessed_dup_removed_analyzed_I_version_available and not cfg.force_redo_preprocessing:
            final_adata, load_report = safe_read_h5ad(analyzed_adata_path, backup_dir=analyzed_backup_dir)
        elif preprocessed_dup_removed_version_available and not cfg.force_redo_preprocessing:
            final_adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path, backup_dir=pp_dup_rem_backup_dir)
        # Use the preprocessed version if it exists and isn't overrode.
        elif preprocessed_version_available and not cfg.force_redo_preprocessing:
            final_adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
        # Load the non preprocessed anndata otherwise
        else:
            backup_dir=os.path.join(os.path.dirname(final_adata_path), 'adata_accessory_data')
            final_adata, load_report = safe_read_h5ad(final_adata_path, backup_dir=backup_dir)


    ## Load sample sheet metadata based on barcode mapping ##
    if cfg.sample_sheet_path:
        from ..preprocessing import load_sample_sheet
        load_sample_sheet(final_adata, 
                          cfg.sample_sheet_path, 
                          mapping_key_column=cfg.sample_sheet_mapping_column, 
                          as_category=True,
                          force_reload=cfg.force_reload_sample_sheet)
    else:
        pass

    # Adding read length, read quality, reference length, mapped_length, and mapping quality metadata to adata object.
    from ..preprocessing import add_read_length_and_mapping_qc
    add_read_length_and_mapping_qc(final_adata, bam_files, 
                                   extract_read_features_from_bam_callable=extract_read_features_from_bam, 
                                   bypass=cfg.bypass_add_read_length_and_mapping_qc,
                                   force_redo=cfg.force_redo_add_read_length_and_mapping_qc)


    final_adata.obs['Raw_modification_signal'] = final_adata.X.sum(axis=1)

    pp_dir = f"{split_dir}/preprocessed"
    pp_qc_dir = f"{pp_dir}/QC_metrics"
    pp_length_qc_dir = f"{pp_qc_dir}/01_Read_QC_metrics"

    if os.path.isdir(pp_length_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(final_adata, pp_length_qc_dir, obs_to_plot, sample_key=cfg.sample_name_col_for_plotting, rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ## Read length, quality, and mapping filtering
    from ..preprocessing import filter_reads_on_length_quality_mapping
    print(final_adata.shape)
    final_adata = filter_reads_on_length_quality_mapping(final_adata, 
                                                         filter_on_coordinates=cfg.read_coord_filter,
                                                         read_length=cfg.read_len_filter_thresholds,
                                                         length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds, 
                                                         read_quality=cfg.read_quality_filter_thresholds,
                                                         mapping_quality=cfg.read_mapping_quality_filter_thresholds,
                                                         bypass=None,
                                                         force_redo=None)
    print(final_adata.shape)

    pp_length_qc_dir = f"{pp_qc_dir}/02_Read_QC_metrics_post_filtering"
    if os.path.isdir(pp_length_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(final_adata, pp_length_qc_dir, obs_to_plot, sample_key=cfg.sample_name_col_for_plotting, rows_per_fig=cfg.rows_per_qc_histogram_grid)
    ########################################################################################################################

    ############################################### 9) Basic Preprocessing ###############################################

    ############## Binarize direct modcall data and store in new layer. Clean nans and store as new layers with various nan replacement strategies ##########
    from ..preprocessing import clean_NaN
    if smf_modality == 'direct':
        from ..preprocessing import calculate_position_Youden, binarize_on_Youden
        native = True
        # Calculate positional methylation thresholds for mod calls
        calculate_position_Youden(final_adata, positive_control_sample=None, negative_control_sample=None, J_threshold=0.5, 
                                  obs_column=reference_column, infer_on_percentile=10, inference_variable='Raw_modification_signal', save=False, output_directory='')
        # binarize the modcalls based on the determined thresholds
        binarize_on_Youden(final_adata, obs_column=reference_column)
        clean_NaN(final_adata, layer='binarized_methylation', bypass=cfg.bypass_clean_nan, force_redo=cfg.force_redo_clean_nan)
    else:
        native = False
        clean_NaN(final_adata, bypass=cfg.bypass_clean_nan, force_redo=cfg.force_redo_clean_nan)

    ############### Add base context to each position for each Reference_strand and calculate read level methylation/deamination stats ###############
    from ..preprocessing import append_base_context, append_binary_layer_by_base_context
    # Additionally, store base_context level binary modification arrays in adata.obsm
    append_base_context(final_adata, 
                        obs_column=reference_column, 
                        use_consensus=False, 
                        native=native, 
                        mod_target_bases=mod_target_bases,
                        bypass=cfg.bypass_append_base_context,
                        force_redo=cfg.force_redo_append_base_context)
    
    final_adata = append_binary_layer_by_base_context(final_adata, 
                                                      reference_column, 
                                                      smf_modality,
                                                      bypass=cfg.bypass_append_binary_layer_by_base_context,
                                                      force_redo=cfg.force_redo_append_binary_layer_by_base_context)
    
    ############### Optional inversion of the adata along positions axis ###################
    if cfg.invert_adata:
        from ..preprocessing import invert_adata
        final_adata = invert_adata(final_adata)

    ############### Calculate read methylation/deamination statistics for specific base contexts defined above ###############
    from ..preprocessing import calculate_read_modification_stats
    calculate_read_modification_stats(final_adata, 
                                      reference_column, 
                                      cfg.sample_column,
                                      mod_target_bases,
                                      bypass=cfg.bypass_calculate_read_modification_stats,
                                      force_redo=cfg.force_redo_calculate_read_modification_stats)

    ### Make a dir for outputting sample level read modification metrics before filtering ###
    pp_dir = f"{split_dir}/preprocessed"
    pp_qc_dir = f"{pp_dir}/QC_metrics"
    pp_meth_qc_dir = f"{pp_qc_dir}/03_read_methylation_QC"

    if os.path.isdir(pp_meth_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(final_adata, pp_meth_qc_dir, obs_to_plot, sample_key=cfg.sample_name_col_for_plotting, rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ##### Optionally filter reads on modification metrics
    from ..preprocessing import filter_reads_on_modification_thresholds
    final_adata = filter_reads_on_modification_thresholds(final_adata, 
                                                          smf_modality=smf_modality,
                                                          mod_target_bases=mod_target_bases,
                                                          gpc_thresholds=cfg.read_mod_filtering_gpc_thresholds, 
                                                          cpg_thresholds=cfg.read_mod_filtering_cpg_thresholds,
                                                          any_c_thresholds=cfg.read_mod_filtering_any_c_thresholds,
                                                          a_thresholds=cfg.read_mod_filtering_a_thresholds,
                                                          use_other_c_as_background=cfg.read_mod_filtering_use_other_c_as_background,
                                                          min_valid_fraction_positions_in_read_vs_ref=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                          bypass=cfg.bypass_filter_reads_on_modification_thresholds,
                                                          force_redo=cfg.force_redo_filter_reads_on_modification_thresholds)

    ## Plot post filtering read methylation metrics
    pp_meth_qc_dir = f"{pp_qc_dir}/04_read_methylation_QC_post_filtering"

    if os.path.isdir(pp_meth_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(final_adata, pp_meth_qc_dir, obs_to_plot, sample_key=cfg.sample_name_col_for_plotting, rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ############### Calculate positional coverage in dataset ###############
    from ..preprocessing import calculate_coverage
    calculate_coverage(final_adata, obs_column=reference_column, position_nan_threshold=0.1)

    ############### Duplicate detection for conversion/deamination SMF ###############
    if smf_modality != 'direct':
        from ..preprocessing import flag_duplicate_reads, calculate_complexity_II
        references = final_adata.obs[reference_column].cat.categories

        var_filters_sets =[]
        for ref in references:
            for site_type in cfg.duplicate_detection_site_types:
                var_filters_sets += [[f"{ref}_{site_type}_site", f"position_in_{ref}"]]

        pp_dir = f"{split_dir}/preprocessed"
        pp_qc_dir = f"{pp_dir}/QC_metrics"
        pp_dup_qc_dir = f"{pp_qc_dir}/06_read_duplication_QC"

        make_dirs([pp_dir, pp_qc_dir, pp_dup_qc_dir])

        ## Will need to improve here. Should do this for each barcode and then concatenate. rather than all at once ###
        final_adata_unique, final_adata = flag_duplicate_reads(final_adata, 
                                                            var_filters_sets, 
                                                            distance_threshold=cfg.duplicate_detection_distance_threshold, 
                                                            obs_reference_col=reference_column, 
                                                            sample_col=cfg.sample_name_col_for_plotting,
                                                            output_directory=pp_dup_qc_dir,
                                                            metric_keys=cfg.hamming_vs_metric_keys,
                                                            keep_best_metric='read_quality',
                                                            bypass=cfg.bypass_flag_duplicate_reads,
                                                            force_redo=cfg.force_redo_flag_duplicate_reads)
        
        calculate_complexity_II(
            adata=final_adata,
            output_directory=pp_dup_qc_dir,
            sample_col=cfg.sample_name_col_for_plotting,
            cluster_col='sequence__merged_cluster_id',
            plot=True,
            save_plot=True,   # set False to display instead
            n_boot=30,
            n_depths=12,
            random_state=42,
            csv_summary=True,
        )

    else:
        final_adata_unique = final_adata

    ########################################################################################################################

    ############################################### Save preprocessed adata with duplicate detection ###############################################
    from ..readwrite import safe_write_h5ad
    if not os.path.exists(pp_adata_path) or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata post duplicate detection.')
        if ".gz" in pp_adata_path:
            safe_write_h5ad(final_adata, f"{pp_adata_path}", compression='gzip', backup=True, backup_dir=pp_backup_dir)
        else:
            safe_write_h5ad(final_adata, f"{pp_adata_path}.gz", compression='gzip', backup=True, backup_dir=pp_backup_dir)

    if not os.path.exists(pp_dup_rem_adata_path) or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata with duplicates removed.')
        if ".gz" in pp_dup_rem_adata_path:
            safe_write_h5ad(final_adata_unique, f"{pp_dup_rem_adata_path}", compression='gzip', backup=True, backup_dir=pp_dup_rem_backup_dir) 
        else:
            safe_write_h5ad(final_adata_unique, f"{pp_dup_rem_adata_path}.gz", compression='gzip', backup=True, backup_dir=pp_dup_rem_backup_dir)
    ########################################################################################################################

    ############################################### Basic Analyses ###############################################
    if smf_modality != 'direct':
        if smf_modality == 'conversion':
            deaminase = False
        else:
            deaminase = True
        references = final_adata.obs[reference_column].cat.categories

        pp_dir = f"{split_dir}/preprocessed"
        pp_clustermap_dir = f"{pp_dir}/07_clustermaps"
        pp_umap_dir = f"{pp_dir}/08_umaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(final_adata, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         reference_col=cfg.reference_column,
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=25, 
                                                         min_length=500, 
                                                         min_mapped_length_to_reference_length_ratio=0.8,
                                                         min_position_valid_fraction=0.5,
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        ## Basic PCA/UMAP
        if os.path.isdir(pp_umap_dir):
            print(pp_umap_dir + ' already exists. Skipping UMAP plotting.')
        else:
            from ..tools import calculate_umap
            make_dirs([pp_dir, pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            final_adata = calculate_umap(final_adata, layer=cfg.layer_for_umap_plotting, var_filters=var_filters, n_pcs=10, knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(final_adata, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            save = 'umap_plot.png'
            sc.pl.umap(final_adata, color=['leiden', cfg.sample_name_col_for_plotting], show=False, save=save)

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_clustermap_dir = f"{pp_dir}/07_clustermaps"
        pp_umap_dir = f"{pp_dir}/08_umaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(final_adata_unique, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=25, 
                                                         min_length=500, 
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        ## Basic PCA/UMAP
        if os.path.isdir(pp_umap_dir):
            print(pp_umap_dir + ' already exists. Skipping UMAP plotting.')
        else:
            from ..tools import calculate_umap
            make_dirs([pp_dir, pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            final_adata = calculate_umap(final_adata_unique, layer=cfg.layer_for_umap_plotting, var_filters=var_filters, n_pcs=10, knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(final_adata_unique, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            save = 'umap_plot.png'
            sc.pl.umap(final_adata_unique, color=['leiden', cfg.sample_name_col_for_plotting], show=False, save=save)


    ########################################################################################################################

    ############################################### Spatial autocorrelation analyses ###############################################
    from ..tools.read_stats import binary_autocorrelation_with_spacing
    if smf_modality != 'direct':
        pp_dir = f"{split_dir}/preprocessed"
        pp_autocorr_dir = f"{pp_dir}/09_autocorrelations"

        if os.path.isdir(pp_autocorr_dir):
            print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
        else:
            positions = final_adata.var_names.astype(int).values
            for site_type in cfg.autocorr_site_types:
                X = final_adata.layers[f"{site_type}_site_binary"]

                autocorr_matrix = np.array([
                    binary_autocorrelation_with_spacing(row, positions, max_lag=cfg.autocorr_max_lag)
                    for row in X
                ])

                final_adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
                final_adata.uns[f"{site_type}_spatial_autocorr_lags"] = np.arange(cfg.autocorr_max_lag + 1)

            if os.path.isdir(pp_autocorr_dir):
                print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
            else:
                from ..plotting import plot_spatial_autocorr_grid
                make_dirs([pp_dir, pp_autocorr_dir])

                plot_spatial_autocorr_grid(final_adata, pp_autocorr_dir, site_types=cfg.autocorr_site_types, 
                                           sample_col=cfg.sample_name_col_for_plotting, window=cfg.autocorr_rolling_window_size, 
                                           rows_per_fig=cfg.rows_per_qc_autocorr_grid)

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_autocorr_dir = f"{pp_dir}/09_autocorrelations"

        if os.path.isdir(pp_autocorr_dir):
            print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
        else:
            positions = final_adata_unique.var_names.astype(int).values
            for site_type in cfg.autocorr_site_types:
                X = final_adata_unique.layers[f"{site_type}_site_binary"]

                autocorr_matrix = np.array([
                    binary_autocorrelation_with_spacing(row, positions, max_lag=cfg.autocorr_max_lag)
                    for row in X
                ])

                final_adata_unique.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
                final_adata_unique.uns[f"{site_type}_spatial_autocorr_lags"] = np.arange(cfg.autocorr_max_lag + 1)

            if os.path.isdir(pp_autocorr_dir):
                print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
            else:
                from ..plotting import plot_spatial_autocorr_grid
                make_dirs([pp_autocorr_dir, pp_autocorr_dir])

                plot_spatial_autocorr_grid(final_adata_unique, pp_autocorr_dir, site_types=cfg.autocorr_site_types, 
                                           sample_col=cfg.sample_name_col_for_plotting, window=cfg.autocorr_rolling_window_size, 
                                           rows_per_fig=cfg.rows_per_qc_autocorr_grid)

    else:
        pass
    ########################################################################################################################

    ############################################### Save basic analysis adata ###############################################
    from ..readwrite import safe_write_h5ad
    final_adata = final_adata_unique

    if not os.path.exists(analyzed_adata_path) or cfg.force_redo_preprocessing:
        print('Saving basic analyzed adata')
        if ".gz" in final_adata_path:
            safe_write_h5ad(final_adata, f"{analyzed_adata_path}", compression='gzip', backup=True, backup_dir=analyzed_backup_dir)
        else:
            safe_write_h5ad(final_adata, f"{analyzed_adata_path}.gz", compression='gzip', backup=True, backup_dir=analyzed_backup_dir)
    ########################################################################################################################

    ############################################### HMM based feature annotations ###############################################
    # need to add the option here to do a per sample HMM fit/inference
    if not (cfg.bypass_hmm_fit and cfg.bypass_hmm_apply):
        from ..hmm.HMM import HMM

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        hmm_dir = f"{pp_dir}/10_hmm_models"

        if os.path.isdir(hmm_dir):
            print(hmm_dir + ' already exists.')
        else:
            make_dirs([pp_dir, hmm_dir])

        samples = final_adata.obs[cfg.sample_name_col_for_plotting].cat.categories
        references = final_adata.obs[reference_column].cat.categories
        mod_sites = mod_target_bases

        for sample in samples:
            for ref in references:
                mask = (final_adata.obs[cfg.sample_name_col_for_plotting] == sample) & (final_adata.obs[reference_column] == ref)
                subset = final_adata[mask].copy()
                if subset.shape[0] > 1:
                    for mod_site in mod_sites:
                        mod_label = {'C': 'any_C'}.get(mod_site, mod_site)
                        hmm_path = os.path.join(hmm_dir, f"{sample}_{ref}_{mod_label}_hmm_model.pth")
                        if os.path.exists(hmm_path) and not cfg.force_redo_hmm_fit:
                            hmm = HMM.load(hmm_path)
                        else:
                            print(f"Fitting HMM for {sample} {ref} {mod_label}")
                            hmm = HMM(n_states=cfg.hmm_n_states, 
                                    init_start=cfg.hmm_init_start_probs,
                                    init_trans=cfg.hmm_init_transition_probs,
                                    init_emission=cfg.hmm_init_emission_probs,
                                    smf_modality=smf_modality)
                            
                            hmm.fit(subset.obsm[f'{ref}_{mod_label}_site'])
                            hmm.save(hmm_path)

                        if not cfg.bypass_hmm_apply or cfg.force_redo_hmm_apply:
                            print(f"Apply HMM for {sample} {ref} {mod_label}")
                            hmm.annotate_adata(subset, 
                                            reference_column, 
                                            layer=None, 
                                            footprints=True, 
                                            accessible_patches=True, 
                                            cpg=True, 
                                            methbases=[mod_label])

    else:
        pass
    
    pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
    hmm_dir = f"{pp_dir}/11_hmm_clustermaps"

    if os.path.isdir(hmm_dir):
        print(hmm_dir + ' already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])

        from ..plotting import combined_hmm_raw_clustermap
        combined_hmm_raw_clustermap(
        final_adata,
        sample_col=cfg.sample_name_col_for_plotting,
        reference_col=reference_column,
        hmm_feature_layer="any_C_large_accessible_patch",
        layer_gpc="nan0_0minus1",
        layer_cpg="nan0_0minus1",
        layer_any_c="nan0_0minus1",
        cmap_hmm="tab10",
        cmap_gpc="coolwarm",
        cmap_cpg="viridis",
        cmap_any_c='coolwarm',
        min_quality=20,
        min_length=200,
        min_mapped_length_to_reference_length_ratio=0.8,
        min_position_valid_fraction=0.5,
        sample_mapping=None,
        save_path=hmm_dir,
        normalize_hmm=False,
        sort_by="gpc",  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
        bins=None,
        deaminase=True,
        min_signal=0
        )

    pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
    hmm_dir = f"{pp_dir}/12_hmm_bulk_traces"

    if os.path.isdir(hmm_dir):
        print(hmm_dir + ' already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_layers_rolling_by_sample_ref
        saved = plot_hmm_layers_rolling_by_sample_ref(
            final_adata,
            layers=final_adata.uns['hmm_appended_layers'],   # replace with your actual layer names
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=reference_column,
            window=101,
            rows_per_page=4,
            figsize_per_cell=(4,2.5),
            output_dir=hmm_dir,
            save=True,
            show_raw=True
        )

    ########################################################################################################################

    ############################################### MultiQC HTML Report ###############################################
    from .helpers import run_multiqc
    # multiqc ###
    if os.path.isdir(f"{split_dir}/multiqc"):
        print(f"{split_dir}/multiqc" + ' already exists, skipping multiqc')
    else:
        run_multiqc(split_dir, f"{split_dir}/multiqc")
    ########################################################################################################################