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
    from .readwrite import safe_read_h5ad, safe_write_h5ad
    from .config import LoadExperimentConfig, ExperimentConfig
    from .informatics.helpers import discover_input_files, make_dirs, concatenate_fastqs_to_bam, extract_read_features_from_bam
    from .informatics.fast5_to_pod5 import fast5_to_pod5
    from .informatics.subsample_fasta_from_bed import subsample_fasta_from_bed

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
    defaults_dir = resources.files("smftools").joinpath("config")
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
        # Output file for FASTQ concatenation.
        output_bam = os.path.join(output_directory, 'FASTQs_concatenated_into_BAM.bam')

        summary = concatenate_fastqs_to_bam(
            fastq_paths,
            output_bam,
            barcode_tag='BC',
            gzip_suffixes=('.gz','.gzip'),
            barcode_map=cfg.fastq_barcode_map,
            add_read_group=True,
            rg_sample_field=None,
            progress=False,
            auto_pair=cfg.fastq_auto_pairing)
        
        print(f"Found the following barcodes: {summary['barcodes']}")

        # Set the input data path to the concatenated BAM.
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
    from .informatics.helpers import generate_converted_FASTA, get_chromosome_lengths

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
            generate_converted_FASTA(fasta, conversions, strands, converted_FASTA)
        fasta = converted_FASTA

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)
    ########################################################################################################################

    ################################### 3) Basecalling ###################################
    from .informatics.helpers import modcall, canoncall
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
    from .informatics.helpers import align_and_sort_BAM, aligned_BAM_to_bed
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
    from .informatics.helpers import demux_and_index_BAM, split_and_index_BAM
    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if os.path.isdir(split_dir):
        print(split_dir + ' already exists. Using existing demultiplexed BAMs.')
        bam_files = os.listdir(split_dir)
        bam_files = [os.path.join(split_dir, file) for file in bam_files if '.bam' in file and '.bai' not in file and 'unclassified' not in file]
        bam_files.sort()
    else:
        make_dirs([split_dir])
        if cfg.input_already_demuxed:
            split_and_index_BAM(aligned_sorted_BAM, 
                                split_dir, 
                                bam_suffix)
        else:
            bam_files = demux_and_index_BAM(aligned_sorted_BAM, 
                                            split_dir, bam_suffix, 
                                            cfg.barcode_kit, 
                                            cfg.barcode_both_ends, 
                                            cfg.trim, 
                                            fasta, 
                                            cfg.make_bigwigs, 
                                            cfg.threads)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(split_dir, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for demultiplexed bams')
    else:
        for bam in bam_files:
            aligned_BAM_to_bed(bam, split_dir, fasta, cfg.make_bigwigs, cfg.threads)
    ########################################################################################################################

    ################################### 6) SAMTools based BAM QC ######################################################################
    from .informatics.helpers import bam_qc
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
        from .informatics.helpers import converted_BAM_to_adata_II
        # 6) Take the converted BAM and load it into an adata object.
        if smf_modality == 'deaminase':
            deaminase_footprinting = True
        else:
            deaminase_footprinting = False
        raw_adata, raw_adata_path = converted_BAM_to_adata_II(fasta, 
                                                                  split_dir, 
                                                                  cfg.mapping_threshold, 
                                                                  cfg.experiment_name, 
                                                                  conversions, 
                                                                  bam_suffix, 
                                                                  cfg.device, 
                                                                  cfg.threads, 
                                                                  deaminase_footprinting,
                                                                  delete_intermediates=cfg.delete_intermediate_hdfs) 
    else:
        if os.path.isdir(mod_bed_dir):
            print(mod_bed_dir + ' already exists, skipping making modbeds')
        else:
            from .informatics.helpers import modQC, make_modbed, extract_mods, modkit_extract_to_adata
            make_dirs([mod_bed_dir])  

            modQC(aligned_sorted_output, 
                  cfg.thresholds) # get QC metrics for mod calls
            
            make_modbed(aligned_sorted_output, 
                        cfg.thresholds, 
                        mod_bed_dir) # Generate bed files of position methylation summaries for every sample

            make_dirs([mod_tsv_dir])  
            extract_mods(cfg.thresholds, 
                         mod_tsv_dir, 
                         split_dir, 
                         bam_suffix, 
                         cfg.skip_unclassified, 
                         cfg.threads) # Extract methylations calls for split BAM files into split TSV files

            #6 Load the modification data from TSVs into an adata object
            raw_adata, raw_adata_path = modkit_extract_to_adata(fasta, 
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

    # Raw adata path info
    raw_backup_dir = os.path.join(os.path.dirname(raw_adata_path), 'adata_accessory_data')

    # Preprocessed adata path info
    pp_adata_basename = os.path.basename(raw_adata_path).split('.h5ad')[0] + '_preprocessed.h5ad.gz'
    pp_adata_path = os.path.join(os.path.dirname(raw_adata_path), pp_adata_basename)
    pp_backup_dir=os.path.join(os.path.dirname(pp_adata_path), 'pp_adata_accessory_data')

    # Preprocessed duplicate removed adata path info
    pp_dup_rem_adata_basename = os.path.basename(pp_adata_basename).split('.h5ad')[0] + '_duplicates_removed.h5ad.gz'
    pp_dup_rem_adata_path = os.path.join(os.path.dirname(pp_adata_path), pp_dup_rem_adata_basename)
    pp_dup_rem_backup_dir=os.path.join(os.path.dirname(pp_dup_rem_adata_path), 'pp_dup_rem_adata_accessory_data')

    # Preprocessed duplicate removed adata with basic analyses appended path info
    basic_analyzed_adata_basename = os.path.basename(pp_dup_rem_adata_path).split('.h5ad')[0] + '_analyzed_I.h5ad.gz'
    basic_analyzed_adata_path = os.path.join(os.path.dirname(pp_dup_rem_adata_path), basic_analyzed_adata_basename)
    basic_analyzed_backup_dir=os.path.join(os.path.dirname(pp_dup_rem_adata_path), 'duplicate_removed_analyzed_adata_I_accessory_data')

    # Preprocessed duplicate removed adata with basic analyses appended path info. With additional HMM feature layers added
    hmm_adata_basename = os.path.basename(basic_analyzed_adata_path).split('.h5ad')[0] + '_hmm.h5ad.gz'
    hmm_adata_path = os.path.join(os.path.dirname(basic_analyzed_adata_path), hmm_adata_basename)
    hmm_backup_dir=os.path.join(os.path.dirname(hmm_adata_path), 'duplicate_removed_analyzed_adata_I_hmm_accessory_data')

    if raw_adata:
        # This happens on first run of the pipeline
        adata = raw_adata
    else:
        # If an anndata is saved, check which stages of the anndata are available
        raw_version_available = os.path.exists(raw_adata_path) and os.path.isdir(raw_backup_dir)
        preprocessed_version_available = os.path.exists(pp_adata_path) and os.path.isdir(pp_backup_dir)
        preprocessed_dup_removed_version_available = os.path.exists(pp_dup_rem_adata_path) and os.path.isdir(pp_dup_rem_backup_dir)
        preprocessed_dup_removed_analyzed_I_version_available = os.path.exists(basic_analyzed_adata_path) and os.path.isdir(basic_analyzed_backup_dir)
        hmm_version_available = os.path.exists(hmm_adata_path) and os.path.isdir(hmm_backup_dir)

        if cfg.force_redo_preprocessing:
            print(f"Forcing full redo of preprocessing workflow, starting from earliest stage adata available.")
            if raw_version_available:
                adata, load_report = safe_read_h5ad(raw_adata_path, backup_dir=raw_backup_dir)
            elif preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path, backup_dir=pp_dup_rem_backup_dir)
            elif preprocessed_dup_removed_analyzed_I_version_available:
                adata, load_report = safe_read_h5ad(basic_analyzed_adata_path, backup_dir=basic_analyzed_backup_dir)
            elif hmm_version_available:
                adata, load_report = safe_read_h5ad(hmm_adata_path, backup_dir=hmm_backup_dir)
            else:
                print(f"Can not redo preprocessing when there is no adata available.")
        elif cfg.force_redo_flag_duplicate_reads:
            print(f"Forcing redo of duplicate detection workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
            elif raw_version_available:
                adata, load_report = safe_read_h5ad(raw_adata_path, backup_dir=raw_backup_dir)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
        elif cfg.force_redo_basic_analyses:
            print(f"Forcing redo of basic analysis workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
            elif raw_version_available:
                adata, load_report = safe_read_h5ad(raw_adata_path, backup_dir=raw_backup_dir)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
        elif hmm_version_available:
            adata, load_report = safe_read_h5ad(hmm_adata_path, backup_dir=hmm_backup_dir)
        elif preprocessed_dup_removed_analyzed_I_version_available:
            adata, load_report = safe_read_h5ad(basic_analyzed_adata_path, backup_dir=basic_analyzed_backup_dir)
        elif preprocessed_dup_removed_version_available:
            adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path, backup_dir=pp_dup_rem_backup_dir)
        elif preprocessed_version_available:
            adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
        elif raw_version_available:
            adata, load_report = safe_read_h5ad(raw_adata_path, backup_dir=raw_backup_dir)
        else:
            print(f"No adata available.")

    ## Load sample sheet metadata based on barcode mapping ##
    if cfg.sample_sheet_path:
        from .preprocessing import load_sample_sheet
        load_sample_sheet(adata, 
                          cfg.sample_sheet_path, 
                          mapping_key_column=cfg.sample_sheet_mapping_column, 
                          as_category=True,
                          force_reload=cfg.force_reload_sample_sheet)
    else:
        pass

    # Adding read length, read quality, reference length, mapped_length, and mapping quality metadata to adata object.
    from .preprocessing import add_read_length_and_mapping_qc
    add_read_length_and_mapping_qc(adata, bam_files, 
                                   extract_read_features_from_bam_callable=extract_read_features_from_bam, 
                                   bypass=cfg.bypass_add_read_length_and_mapping_qc,
                                   force_redo=cfg.force_redo_add_read_length_and_mapping_qc)


    adata.obs['Raw_modification_signal'] =  np.nansum(adata.X, axis=1)

    pp_dir = f"{split_dir}/preprocessed"
    pp_length_qc_dir = f"{pp_dir}/01_Read_length_and_quality_QC_metrics"

    if os.path.isdir(pp_length_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from .plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(adata,
                                pp_length_qc_dir, 
                                obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ## Read length, quality, and mapping filtering
    from .preprocessing import filter_reads_on_length_quality_mapping
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

    pp_length_qc_dir = f"{pp_dir}/02_Read_length_and_quality_QC_metrics_post_filtering"
    if os.path.isdir(pp_length_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from .plotting import plot_read_qc_histograms
        make_dirs([pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(adata, 
                                pp_length_qc_dir, 
                                obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)
    ########################################################################################################################

    ############################################### 9) Basic Preprocessing ###############################################

    ############## Binarize direct modcall data and store in new layer. Clean nans and store as new layers with various nan replacement strategies ##########
    from .preprocessing import clean_NaN
    if smf_modality == 'direct':
        from .preprocessing import calculate_position_Youden, binarize_on_Youden
        native = True
        # Calculate positional methylation thresholds for mod calls
        calculate_position_Youden(adata, 
                                  positive_control_sample=None, 
                                  negative_control_sample=None, 
                                  J_threshold=0.5, 
                                  obs_column=reference_column, 
                                  infer_on_percentile=10, 
                                  inference_variable='Raw_modification_signal', 
                                  save=False, 
                                  output_directory=''
                                  )
        # binarize the modcalls based on the determined thresholds
        binarize_on_Youden(adata, 
                           obs_column=reference_column
                           )
        clean_NaN(adata, 
                  layer='binarized_methylation',
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
    from .preprocessing import append_base_context, append_binary_layer_by_base_context
    # Additionally, store base_context level binary modification arrays in adata.obsm
    append_base_context(adata, 
                        obs_column=reference_column, 
                        use_consensus=False, 
                        native=native, 
                        mod_target_bases=mod_target_bases,
                        bypass=cfg.bypass_append_base_context,
                        force_redo=cfg.force_redo_append_base_context)
    
    adata = append_binary_layer_by_base_context(adata, 
                                                reference_column, 
                                                smf_modality,
                                                bypass=cfg.bypass_append_binary_layer_by_base_context,
                                                force_redo=cfg.force_redo_append_binary_layer_by_base_context)
    
    ############### Optional inversion of the adata along positions axis ###################
    if cfg.invert_adata:
        from .preprocessing import invert_adata
        adata = invert_adata(adata)

    ############### Calculate read methylation/deamination statistics for specific base contexts defined above ###############
    from .preprocessing import calculate_read_modification_stats
    calculate_read_modification_stats(adata, 
                                      reference_column, 
                                      cfg.sample_column,
                                      mod_target_bases,
                                      bypass=cfg.bypass_calculate_read_modification_stats,
                                      force_redo=cfg.force_redo_calculate_read_modification_stats)

    ### Make a dir for outputting sample level read modification metrics before filtering ###
    pp_dir = f"{split_dir}/preprocessed"
    pp_meth_qc_dir = f"{pp_dir}/03_read_modification_QC_metrics"

    if os.path.isdir(pp_meth_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from .plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(adata, 
                                pp_meth_qc_dir, obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ##### Optionally filter reads on modification metrics
    from .preprocessing import filter_reads_on_modification_thresholds
    adata = filter_reads_on_modification_thresholds(adata, 
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
    pp_meth_qc_dir = f"{pp_dir}/04_read_modification_QC_metrics_post_filtering"

    if os.path.isdir(pp_meth_qc_dir) and not cfg.force_redo_preprocessing:
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from .plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(adata, 
                                pp_meth_qc_dir,
                                obs_to_plot, 
                                sample_key=cfg.sample_name_col_for_plotting, 
                                rows_per_fig=cfg.rows_per_qc_histogram_grid)

    ############### Calculate positional coverage in dataset ###############
    from .preprocessing import calculate_coverage
    calculate_coverage(adata, 
                       obs_column=reference_column, 
                       position_nan_threshold=0.1)

    ############### Duplicate detection for conversion/deamination SMF ###############
    if smf_modality != 'direct':
        from .preprocessing import flag_duplicate_reads, calculate_complexity_II
        references = adata.obs[reference_column].cat.categories

        var_filters_sets =[]
        for ref in references:
            for site_type in cfg.duplicate_detection_site_types:
                var_filters_sets += [[f"{ref}_{site_type}_site", f"position_in_{ref}"]]

        pp_dir = f"{split_dir}/preprocessed"
        pp_dup_qc_dir = f"{pp_dir}/05_read_duplication_QC_metrics"

        make_dirs([pp_dir, pp_dup_qc_dir])

        # Flag duplicate reads and plot duplicate detection QC
        adata_unique, adata = flag_duplicate_reads(adata, 
                                                    var_filters_sets, 
                                                    distance_threshold=cfg.duplicate_detection_distance_threshold, 
                                                    obs_reference_col=reference_column, 
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
            ref_col=reference_column,
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
    from .readwrite import safe_write_h5ad
    if not os.path.exists(pp_adata_path) or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata post duplicate detection.')
        if ".gz" in pp_adata_path:
            safe_write_h5ad(adata, f"{pp_adata_path}", compression='gzip', backup=True, backup_dir=pp_backup_dir)
        else:
            safe_write_h5ad(adata, f"{pp_adata_path}.gz", compression='gzip', backup=True, backup_dir=pp_backup_dir)

    if not os.path.exists(pp_dup_rem_adata_path) or cfg.force_redo_preprocessing:
        print('Saving preprocessed adata with duplicates removed.')
        if ".gz" in pp_dup_rem_adata_path:
            safe_write_h5ad(adata_unique, f"{pp_dup_rem_adata_path}", compression='gzip', backup=True, backup_dir=pp_dup_rem_backup_dir) 
        else:
            safe_write_h5ad(adata_unique, f"{pp_dup_rem_adata_path}.gz", compression='gzip', backup=True, backup_dir=pp_dup_rem_backup_dir)
    ########################################################################################################################

    ############################################### Basic Analyses ###############################################
    if smf_modality != 'direct':
        if smf_modality == 'conversion':
            deaminase = False
        else:
            deaminase = True
        references = adata.obs[reference_column].cat.categories

        pp_dir = f"{split_dir}/preprocessed"
        pp_clustermap_dir = f"{pp_dir}/06_clustermaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from .plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(adata, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         reference_col=cfg.reference_column,
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=cfg.read_quality_filter_thresholds[0], 
                                                         min_length=cfg.read_len_filter_thresholds[0], 
                                                         min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[0],
                                                         min_position_valid_fraction=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        # Switch the main adata moving forward to be the one with duplicates removed.
        adata = adata_unique

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_clustermap_dir = f"{pp_dir}/06_clustermaps"
        pp_umap_dir = f"{pp_dir}/07_umaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from .plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(adata, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         reference_col=cfg.reference_column,
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=cfg.read_quality_filter_thresholds[0], 
                                                         min_length=cfg.read_len_filter_thresholds[0], 
                                                         min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[0],
                                                         min_position_valid_fraction=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        ## Basic PCA/UMAP
        if os.path.isdir(pp_umap_dir):
            print(pp_umap_dir + ' already exists. Skipping UMAP plotting.')
        else:
            from .tools import calculate_umap
            make_dirs([pp_dir, pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            adata = calculate_umap(adata, 
                                   layer=cfg.layer_for_umap_plotting, 
                                   var_filters=var_filters, 
                                   n_pcs=10, 
                                   knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(adata, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            sc.settings.figdir = pp_umap_dir
            umap_layers = ['leiden', cfg.sample_name_col_for_plotting]
            umap_layers += cfg.umap_layers_to_plot
            sc.pl.umap(adata, color=umap_layers, show=False, save=True)


    ########################################################################################################################

    ############################################### Spatial autocorrelation analyses ########################################
    from .tools.spatial_autocorrelation import binary_autocorrelation_with_spacing, analyze_autocorr_matrix, bootstrap_periodicity, rolling_autocorr_metrics
    from .plotting import plot_rolling_grid
    import warnings

    pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
    pp_autocorr_dir = f"{pp_dir}/08_autocorrelations"

    if os.path.isdir(pp_autocorr_dir):
        print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
    else:
        positions = adata.var_names.astype(int).values
        lags = np.arange(cfg.autocorr_max_lag + 1)

        # optional: try to parallelize autocorr per-row with joblib
        try:
            from joblib import Parallel, delayed
            _have_joblib = True
        except Exception:
            _have_joblib = False

        for site_type in cfg.autocorr_site_types:
            layer_key = f"{site_type}_site_binary"
            if layer_key not in adata.layers:
                print(f"Layer {layer_key} not found in adata.layers — skipping {site_type}.")
                continue

            X = adata.layers[layer_key]
            if getattr(X, "shape", (0,))[0] == 0:
                print(f"Layer {layer_key} empty — skipping {site_type}.")
                continue

            # compute per-molecule autocorrs (and counts)
            rows = []
            counts = []
            if _have_joblib:
                # parallel map
                def _worker(row):
                    try:
                        ac, cnts = binary_autocorrelation_with_spacing(
                            row, positions, max_lag=cfg.autocorr_max_lag, return_counts=True
                        )
                    except Exception as e:
                        # on error return NaN arrays
                        ac = np.full(cfg.autocorr_max_lag + 1, np.nan, dtype=np.float32)
                        cnts = np.zeros(cfg.autocorr_max_lag + 1, dtype=np.int32)
                    return ac, cnts

                res = Parallel(n_jobs=cfg.n_jobs if hasattr(cfg, "n_jobs") else -1)(
                    delayed(_worker)(X[i]) for i in range(X.shape[0])
                )
                for ac, cnts in res:
                    rows.append(ac)
                    counts.append(cnts)
            else:
                # sequential fallback
                for i in range(X.shape[0]):
                    ac, cnts = binary_autocorrelation_with_spacing(
                        X[i], positions, max_lag=cfg.autocorr_max_lag, return_counts=True
                    )
                    rows.append(ac)
                    counts.append(cnts)

            autocorr_matrix = np.asarray(rows, dtype=np.float32)
            counts_matrix = np.asarray(counts, dtype=np.int32)

            # store raw per-molecule arrays (keep memory format compact)
            adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
            adata.obsm[f"{site_type}_spatial_autocorr_counts"] = counts_matrix
            adata.uns[f"{site_type}_spatial_autocorr_lags"] = lags

            # compute global periodicity metrics across all molecules for this site_type
            try:
                results = analyze_autocorr_matrix(
                    autocorr_matrix, counts_matrix, lags,
                    nrl_search_bp=(120, 260), pad_factor=4, min_count=20, max_harmonics=6
                )
            except Exception as e:
                results = {"error": str(e)}

            # store global metrics (same keys you used)
            global_metrics = {
                "nrl_bp": results.get("nrl_bp", np.nan),
                "xi": results.get("xi", np.nan),
                "snr": results.get("snr", np.nan),
                "fwhm_bp": results.get("fwhm_bp", np.nan),
                "envelope_sample_lags": results.get("envelope_sample_lags", np.array([])).tolist(),
                "envelope_heights": results.get("envelope_heights", np.array([])).tolist(),
                "analyzer_error": results.get("error", None),
            }
            adata.uns[f"{site_type}_spatial_periodicity_metrics"] = global_metrics

            # bootstrap for CI (use a reasonable default; set low only for debugging)
            n_boot = getattr(cfg, "autocorr_bootstrap_n", 200)
            # if user intentionally set very low n_boot in cfg, we keep that; otherwise default 200
            try:
                bs = bootstrap_periodicity(
                    autocorr_matrix, counts_matrix, lags,
                    n_boot=n_boot, nrl_search_bp=(120, 260), pad_factor=4, min_count=20
                )
                adata.uns[f"{site_type}_spatial_periodicity_boot"] = {
                    "nrl_boot": np.asarray(bs["nrl_boot"]).tolist(),
                    "xi_boot": np.asarray(bs["xi_boot"]).tolist(),
                }
            except Exception as e:
                adata.uns[f"{site_type}_spatial_periodicity_boot_error"] = str(e)

            # ----------------------------
            # Compute group-level metrics for plotting (per sample × reference)
            # ----------------------------
            metrics_by_group = {}
            sample_col = cfg.sample_name_col_for_plotting
            ref_col = cfg.reference_strand_col if hasattr(cfg, "reference_strand_col") else "Reference_strand"
            samples = adata.obs[sample_col].astype("category").cat.categories.tolist()
            refs = adata.obs[ref_col].astype("category").cat.categories.tolist()

            # iterate groups and run analyzer on each group's subset; cache errors
            for sample_name in samples:
                sample_mask = (adata.obs[sample_col].values == sample_name)
                # combined group
                mask = sample_mask
                ac_sel = autocorr_matrix[mask, :]
                cnt_sel = counts_matrix[mask, :] if counts_matrix is not None else None
                if ac_sel.size:
                    try:
                        r = analyze_autocorr_matrix(ac_sel, cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                                                    lags, nrl_search_bp=(120,260), pad_factor=4, min_count=10, max_harmonics=6)
                    except Exception as e:
                        r = {"error": str(e)}
                else:
                    r = {"error": "no_data"}
                metrics_by_group[(sample_name, None)] = r

                # per-reference groups
                for ref in refs:
                    mask_ref = sample_mask & (adata.obs[ref_col].values == ref)
                    ac_sel = autocorr_matrix[mask_ref, :]
                    cnt_sel = counts_matrix[mask_ref, :] if counts_matrix is not None else None
                    if ac_sel.size:
                        try:
                            r = analyze_autocorr_matrix(ac_sel, cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                                                        lags, nrl_search_bp=(120,260), pad_factor=4, min_count=10, max_harmonics=6)
                        except Exception as e:
                            r = {"error": str(e)}
                    else:
                        r = {"error": "no_data"}
                    metrics_by_group[(sample_name, ref)] = r

            # persist group metrics
            adata.uns[f"{site_type}_spatial_periodicity_metrics_by_group"] = metrics_by_group

            global_nrl = adata.uns.get(f"{site_type}_spatial_periodicity_metrics", {}).get("nrl_bp", None)

            # configuration / sensible defaults (override in cfg if present)
            rolling_cfg = {
                "window_size": getattr(cfg, "rolling_window_size", getattr(cfg, "autocorr_rolling_window_size", 600)),
                "step": getattr(cfg, "rolling_step", 100),
                "max_lag": getattr(cfg, "rolling_max_lag", cfg.autocorr_max_lag if hasattr(cfg, "autocorr_max_lag") else 500),
                "min_molecules_per_window": getattr(cfg, "rolling_min_molecules_per_window", 10),
                "nrl_search_bp": getattr(cfg, "rolling_nrl_search_bp", (120, 240)),
                "pad_factor": getattr(cfg, "rolling_pad_factor", 4),
                "min_count_for_mean": getattr(cfg, "rolling_min_count_for_mean", 10),
                "max_harmonics": getattr(cfg, "rolling_max_harmonics", 6),
                "n_jobs": getattr(cfg, "rolling_n_jobs", 4),
            }

            write_plots = getattr(cfg, "rolling_write_plots", True)
            write_csvs = getattr(cfg, "rolling_write_csvs", True)
            min_molecules_for_group = getattr(cfg, "rolling_min_molecules_for_group", 30)  # only run rolling if group has >= this many molecules

            rolling_out_dir = os.path.join(pp_autocorr_dir, "rolling_metrics")
            os.makedirs(rolling_out_dir, exist_ok=True)
            # also a per-site subfolder
            site_out_dir = os.path.join(rolling_out_dir, site_type)
            os.makedirs(site_out_dir, exist_ok=True)

            combined_rows = []  # accumulate one row per window for combined CSV
            rolling_results_by_group = {}  # store DataFrame per group in memory (persist later to adata.uns)

            # iterate groups (samples × refs). `samples` and `refs` were computed above.
            for sample_name in samples:
                sample_mask = (adata.obs[sample_col].values == sample_name)
                # first the combined group ("all refs")
                group_masks = [("all", sample_mask)]
                # then per-reference groups
                for ref in refs:
                    ref_mask = sample_mask & (adata.obs[ref_col].values == ref)
                    group_masks.append((ref, ref_mask))

                for ref_label, mask in group_masks:
                    n_group = int(mask.sum())
                    if n_group < min_molecules_for_group:
                        # skip tiny groups
                        if cfg.get("verbosity", 0) if hasattr(cfg, "get") else False:
                            print(f"Skipping rolling for {site_type} {sample_name} {ref_label}: only {n_group} molecules (<{min_molecules_for_group})")
                        # still write an empty CSV row set if desired; here we skip
                        continue

                    # extract group matrix X_group (works with dense or sparse adata.layers)
                    X_group = X[mask, :]
                    # positions already set above
                    try:
                        # call your rolling function (this may be slow; it uses cfg.n_jobs)
                        df_roll = rolling_autocorr_metrics(
                            X_group,
                            positions,
                            site_label=site_type,
                            window_size=rolling_cfg["window_size"],
                            step=rolling_cfg["step"],
                            max_lag=rolling_cfg["max_lag"],
                            min_molecules_per_window=rolling_cfg["min_molecules_per_window"],
                            nrl_search_bp=rolling_cfg["nrl_search_bp"],
                            pad_factor=rolling_cfg["pad_factor"],
                            min_count_for_mean=rolling_cfg["min_count_for_mean"],
                            max_harmonics=rolling_cfg["max_harmonics"],
                            n_jobs=rolling_cfg["n_jobs"],
                            verbose=False,
                            fixed_nrl_bp=global_nrl
                        )
                    except Exception as e:
                        warnings.warn(f"rolling_autocorr_metrics failed for {site_type} {sample_name} {ref_label}: {e}")
                        continue

                    # normalize column names and keep only the compact set you want
                    # keep: center, n_molecules, nrl_bp, snr, xi, fwhm_bp
                    if "center" not in df_roll.columns:
                        # defensive: if the rolling function returned different schema, skip
                        warnings.warn(f"rolling_autocorr_metrics returned unexpected schema for {site_type} {sample_name} {ref_label}")
                        continue

                    compact_df = df_roll[["center", "n_molecules", "nrl_bp", "snr", "xi", "fwhm_bp"]].copy()
                    compact_df["site"] = site_type
                    compact_df["sample"] = sample_name
                    compact_df["reference"] = ref_label if ref_label != "all" else "all"

                    # save per-group CSV
                    if write_csvs:
                        safe_sample = str(sample_name).replace(os.sep, "_")
                        safe_ref = str(ref_label if ref_label != "all" else "all").replace(os.sep, "_")
                        out_csv = os.path.join(site_out_dir, f"{safe_sample}__{safe_ref}__rolling_metrics.csv")
                        try:
                            compact_df.to_csv(out_csv, index=False)
                        except Exception as e:
                            warnings.warn(f"Failed to write rolling CSV {out_csv}: {e}")

                    # save a plot per-group (NRL and SNR vs center)
                    if write_plots:
                        try:
                            # use your plot helper; if it's in a different module, import accordingly
                            from .plotting import plot_rolling_metrics as _plot_roll
                        except Exception:
                            _plot_roll = globals().get("plot_rolling_metrics", None)
                        if _plot_roll is not None:
                            plot_png = os.path.join(site_out_dir, f"{safe_sample}__{safe_ref}__rolling_metrics.png")
                            try:
                                _plot_roll(compact_df, out_png=plot_png,
                                        title=f"{site_type} {sample_name} {ref_label}",
                                        figsize=(10,3.5), dpi=160, show=False)
                            except Exception as e:
                                warnings.warn(f"Failed to create rolling plot for {site_type} {sample_name} {ref_label}: {e}")

                    # store in combined_rows and in-memory dict
                    combined_rows.append(compact_df.assign(site=site_type, sample=sample_name, reference=ref_label))
                    rolling_results_by_group[(sample_name, None if ref_label == "all" else ref_label)] = compact_df

            # persist per-site rolling metrics into adata.uns as dict of DataFrames (or empty dict)
            adata.uns[f"{site_type}_rolling_metrics_by_group"] = rolling_results_by_group

            # write combined CSV for this site across all groups
            if len(combined_rows):
                combined_df_site = pd.concat(combined_rows, ignore_index=True, sort=False)
                combined_out_csv = os.path.join(rolling_out_dir, f"{site_type}__rolling_metrics_combined.csv")
                try:
                    combined_df_site.to_csv(combined_out_csv, index=False)
                except Exception as e:
                    warnings.warn(f"Failed to write combined rolling CSV for {site_type}: {e}")

            rolling_dict = adata.uns[f"{site_type}_rolling_metrics_by_group"]
            plot_out_dir = os.path.join(pp_autocorr_dir, "rolling_plots")
            os.makedirs(plot_out_dir, exist_ok=True)
            pages = plot_rolling_grid(rolling_dict, plot_out_dir, site_type,
                                    rows_per_page=cfg.rows_per_qc_autocorr_grid,
                                    cols_per_page=len(refs),
                                    dpi=160,
                                    metrics=("nrl_bp","snr", "xi"),
                                    per_metric_ylim={"snr": (0, 25)})

            from .plotting import plot_spatial_autocorr_grid
            make_dirs([pp_autocorr_dir, pp_autocorr_dir])

            plot_spatial_autocorr_grid(adata, 
                                        pp_autocorr_dir, 
                                        site_types=cfg.autocorr_site_types, 
                                        sample_col=cfg.sample_name_col_for_plotting, 
                                        window=cfg.autocorr_rolling_window_size, 
                                        rows_per_fig=cfg.rows_per_qc_autocorr_grid)

    ########################################################################################################################

    ############################################### Pearson analyses ########################################
    if smf_modality != 'direct':
        from .tools.position_stats import compute_positionwise_statistics, plot_positionwise_matrices

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_corr_dir = f"{pp_dir}/09_correlation_matrices"

        if os.path.isdir(pp_corr_dir):
            print(pp_corr_dir + ' already exists. Skipping correlation matrix plotting.')
        else:
            compute_positionwise_statistics(
                adata,
                layer="nan0_0minus1",
                methods=cfg.correlation_matrix_types,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_col=reference_column,
                output_key="positionwise_result",
                site_types=cfg.correlation_matrix_site_types,
                encoding="signed",
                max_threads=cfg.threads,
                min_count_for_pairwise=10,
            )
        
            plot_positionwise_matrices(
                adata,
                methods=cfg.correlation_matrix_types,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_col=reference_column,
                figsize_per_cell=(4.0, 3.0),
                dpi=160,
                cmaps=cfg.correlation_matrix_cmaps,
                vmin=None,
                vmax=None,
                output_dir=pp_corr_dir,
                output_key= "positionwise_result"
            )

    ########################################################################################################################

    ############################################### Save basic analysis adata - post preprocessing and duplicate removal ###############################################
    from .readwrite import safe_write_h5ad

    if not os.path.exists(basic_analyzed_adata_path) or cfg.force_redo_preprocessing:
        print('Saving basic analyzed adata post preprocessing and duplicate removal')
        if ".gz" in basic_analyzed_adata_path:
            safe_write_h5ad(adata, f"{basic_analyzed_adata_path}", compression='gzip', backup=True, backup_dir=basic_analyzed_backup_dir)
        else:
            safe_write_h5ad(adata, f"{basic_analyzed_adata_path}.gz", compression='gzip', backup=True, backup_dir=basic_analyzed_backup_dir)
    ########################################################################################################################

    ############################################### HMM based feature annotations ###############################################
    if not (cfg.bypass_hmm_fit and cfg.bypass_hmm_apply):
        from .hmm.HMM import HMM
        from scipy.sparse import issparse, csr_matrix
        import warnings

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        hmm_dir = f"{pp_dir}/10_hmm_models"

        if os.path.isdir(hmm_dir):
            print(hmm_dir + ' already exists.')
        else:
            make_dirs([pp_dir, hmm_dir])

        samples = adata.obs[cfg.sample_name_col_for_plotting].cat.categories
        references = adata.obs[reference_column].cat.categories
        uns_key = "hmm_appended_layers"

        # ensure uns key exists (avoid KeyError later)
        if adata.uns.get(uns_key) is None:
            adata.uns[uns_key] = []

        for sample in samples:
            for ref in references:
                mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (adata.obs[reference_column] == ref)
                subset = adata[mask].copy()
                if subset.shape[0] < 1:
                    continue

                for mod_site in cfg.hmm_methbases:
                    mod_label = {'C': 'any_C'}.get(mod_site, mod_site)
                    hmm_path = os.path.join(hmm_dir, f"{sample}_{ref}_{mod_label}_hmm_model.pth")

                    # ensure the input obsm exists
                    obsm_key = f'{ref}_{mod_label}_site'
                    if obsm_key not in subset.obsm:
                        print(f"Skipping {sample} {ref} {mod_label}: missing obsm '{obsm_key}'")
                        continue

                    # Fit or load model
                    if os.path.exists(hmm_path) and not cfg.force_redo_hmm_fit:
                        hmm = HMM.load(hmm_path)
                        hmm.print_params()
                    else:
                        print(f"Fitting HMM for {sample} {ref} {mod_label}")
                        hmm = HMM.from_config(cfg)
                        # fit expects a list-of-seqs or 2D ndarray in the obsm
                        seqs = subset.obsm[obsm_key]
                        hmm.fit(seqs)
                        hmm.print_params()
                        hmm.save(hmm_path)

                    # Apply / annotate on the subset, then copy layers back to final_adata
                    if (not cfg.bypass_hmm_apply) or cfg.force_redo_hmm_apply:
                        print(f"Applying HMM on subset for {sample} {ref} {mod_label}")
                        # Use the new uns_key argument so subset will record appended layer names
                        # (annotate_adata modifies subset.obs/layers in-place and should write subset.uns[uns_key])
                        hmm.annotate_adata(subset,
                                        obs_column=cfg.reference_column,
                                        layer=cfg.layer_for_umap_plotting,
                                        config=cfg)
                        
                        #to_merge = [("C_all_accessible_features", 80)]
                        to_merge = cfg.hmm_merge_layer_features
                        for layer_to_merge, merge_distance in to_merge:
                            if layer_to_merge:
                                hmm.merge_intervals_in_layer(subset,
                                                            layer=layer_to_merge,
                                                            distance_threshold=merge_distance,
                                                            overwrite=True
                                                            )
                            else:
                                pass

                        # collect appended layers from subset.uns
                        appended = list(subset.uns.get(uns_key, []))
                        print(appended)
                        if len(appended) == 0:
                            # nothing appended for this subset; continue
                            continue

                        # copy each appended layer into adata
                        subset_mask_bool = mask.values if hasattr(mask, "values") else np.asarray(mask)
                        for layer_name in appended:
                            if layer_name not in subset.layers:
                                # defensive: skip
                                warnings.warn(f"Expected layer {layer_name} in subset but not found; skipping copy.")
                                continue
                            sub_layer = subset.layers[layer_name]
                            # ensure final layer exists and assign rows
                            try:
                               hmm._ensure_final_layer_and_assign(adata, layer_name, subset_mask_bool, sub_layer)
                            except Exception as e:
                                warnings.warn(f"Failed to copy layer {layer_name} into adata: {e}", stacklevel=2)
                                # fallback: if dense and small, try to coerce
                                if issparse(sub_layer):
                                    arr = sub_layer.toarray()
                                else:
                                    arr = np.asarray(sub_layer)
                                adata.layers[layer_name] = adata.layers.get(layer_name, np.zeros((adata.shape[0], arr.shape[1]), dtype=arr.dtype))
                                final_idx = np.nonzero(subset_mask_bool)[0]
                                adata.layers[layer_name][final_idx, :] = arr

                        # merge appended layer names into adata.uns
                        existing = list(adata.uns.get(uns_key, []))
                        for ln in appended:
                            if ln not in existing:
                                existing.append(ln)
                        adata.uns[uns_key] = existing

    else:
        pass
    
    ## Save HMM annotated adata
    if not os.path.exists(hmm_adata_path):
        print('Saving hmm analyzed adata post preprocessing and duplicate removal')
        if ".gz" in hmm_adata_path:
            safe_write_h5ad(adata, f"{hmm_adata_path}", compression='gzip', backup=True, backup_dir=hmm_backup_dir)
        else:
            safe_write_h5ad(adata, f"{hmm_adata_path}.gz", compression='gzip', backup=True, backup_dir=hmm_backup_dir)

    ########################################################################################################################

############################################### HMM based feature plotting ###############################################
    
    pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
    hmm_dir = f"{pp_dir}/11_hmm_clustermaps"

    if os.path.isdir(hmm_dir):
        print(hmm_dir + ' already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from .plotting import combined_hmm_raw_clustermap

        for layer in ['C_all_accessible_features', 'C_small_bound_stretch', 'C_medium_bound_stretch', 'C_putative_nucleosome', 'C_all_accessible_features_merged']:
            save_path = os.path.join(hmm_dir, layer)
            make_dirs([save_path])

            combined_hmm_raw_clustermap(
            adata,
            sample_col=cfg.sample_name_col_for_plotting,
            reference_col=reference_column,
            hmm_feature_layer=layer,
            layer_gpc="nan0_0minus1",
            layer_cpg="nan0_0minus1",
            layer_any_c="nan0_0minus1",
            cmap_hmm="coolwarm",
            cmap_gpc="coolwarm",
            cmap_cpg="viridis",
            cmap_any_c='coolwarm',
            min_quality=20,
            min_length=80,
            min_mapped_length_to_reference_length_ratio=0.2,
            min_position_valid_fraction=0.2,
            sample_mapping=None,
            save_path=save_path,
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
        from .plotting import plot_hmm_layers_rolling_by_sample_ref
        saved = plot_hmm_layers_rolling_by_sample_ref(
            adata,
            layers=adata.uns['hmm_appended_layers'],
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=reference_column,
            window=101,
            rows_per_page=4,
            figsize_per_cell=(4,2.5),
            output_dir=hmm_dir,
            save=True,
            show_raw=False
        )

    pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
    hmm_dir = f"{pp_dir}/13_hmm_fragment_distributions"
    if os.path.isdir(hmm_dir):
        print(hmm_dir + ' already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from .plotting import plot_hmm_size_contours

        for layer, max in [('C_all_accessible_features_lengths', 400), ('C_all_footprint_features_lengths', 160), ('C_all_accessible_features_merged_lengths', 800)]:
            save_path = os.path.join(hmm_dir, layer)
            make_dirs([save_path])

            figs = plot_hmm_size_contours(
                adata,
                length_layer=layer,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_obs_col=reference_column,
                rows_per_page=6,
                max_length_cap=max,
                figsize_per_cell=(3.5, 2.2),
                save_path=save_path,
                save_pdf=False,
                save_each_page=True,
                dpi=200,
                smoothing_sigma=None,
                normalize_after_smoothing=False,
                cmap='viridis', 
                log_scale_z=True
            )

    ########################################################################################################################

    ############################################### MultiQC HTML Report ###############################################
    from .informatics.helpers import run_multiqc
    # multiqc ###
    if os.path.isdir(f"{split_dir}/multiqc"):
        print(f"{split_dir}/multiqc" + ' already exists, skipping multiqc')
    else:
        run_multiqc(split_dir, f"{split_dir}/multiqc")
    ########################################################################################################################