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
    # Lazy importing of packages
    from .helpers import LoadExperimentConfig, make_dirs, concatenate_fastqs_to_bam, extract_read_features_from_bam
    from .fast5_to_pod5 import fast5_to_pod5
    from .subsample_fasta_from_bed import subsample_fasta_from_bed
    import os
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")


    ################################### 1) General params and input organization ###################################

    # Default params
    bam_suffix = '.bam' # If different, change from here.
    split_dir = 'demultiplexed_BAMs' # If different, change from here.
    strands = ['bottom', 'top'] # If different, change from here. Having both listed generally doesn't slow things down too much.
    conversions = ['unconverted'] # The name to use for the unconverted files. If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict

    # These below variables will point to default_value if they are empty in the experiment_config.csv or if the variable is fully omitted from the csv.
    default_value = None

    # General config variable init - Necessary user passed inputs
    smf_modality = var_dict.get('smf_modality', default_value) # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    input_data_path = var_dict.get('input_data_path', default_value) # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = var_dict.get('output_directory', default_value) # Path to the output directory to make for the analysis. Necessary.
    fasta = var_dict.get('fasta', default_value) # Path to reference FASTA. Necessary.
    model_dir = var_dict.get('model_dir', default_value) # needed for dorado basecaller if starting from POD5/FAST5.
    barcode_kit = var_dict.get('barcode_kit', default_value) # needed for dorado basecaller

    # General config variable init - Optional user passed inputs
    fasta_regions_of_interest = var_dict.get("fasta_regions_of_interest", default_value) # Path to a bed file listing coordinate regions of interest within the FASTA to include. Optional.
    mapping_threshold = var_dict.get('mapping_threshold', 0.01) # Minimum proportion of mapped reads that need to fall within a region to include in the final AnnData.
    experiment_name = var_dict.get('experiment_name', f"{date_str}_SMF_experiment") # A key term to add to the AnnData file name.
    model = var_dict.get('model', 'hac') # needed for dorado basecaller
    barcode_both_ends = var_dict.get('barcode_both_ends', False) # dorado demultiplexing
    trim = var_dict.get('trim', False) # dorado adapter and barcode removal
    input_already_demuxed = var_dict.get('input_already_demuxed', False) # If the input files are already demultiplexed.
    threads = var_dict.get('threads', default_value) # number of cpu threads available for multiprocessing
    sample_sheet_path = var_dict.get('sample_sheet_path', default_value) # Optional path to a sample sheet with barcode metadata
    aligner = var_dict.get('aligner', 'minimap2') # Aligner to use: dorado, minimap2
    if aligner == 'minimap2':
        aligner_args = var_dict.get('aligner_args', ['-a', '-x', 'map-ont', '--MD', '-Y', '-y', '-N', '5', '--secondary=no']) # Aligner arguments to use for minimap2
    elif aligner == 'dorado':
        aligner_args = var_dict.get('aligner_args', ['--mm2-opts', '-N 5']) # Aligner arguments to use for dorado
    device = var_dict.get('device', 'auto') # What device to use for computation: 'mps', 'cpu', 'auto', etc
    make_bigwigs = var_dict.get('make_bigwigs', False) # Whether to make coverage bigwigs

    # General config variable init - Optional user passed inputs for enzyme base specificity
    mod_target_bases = var_dict.get('mod_target_bases', ['GpC', 'CpG']) # Nucleobases of interest that may be modified. ['GpC', 'CpG', 'C', 'A']

    # Conversion/deamination specific variable init
    conversion_types = var_dict.get('conversion_types', ['5mC']) # 5mC, 

    # Direct methylation specific variable init
    filter_threshold = var_dict.get('filter_threshold', 0.8) # min threshold to call a canononical base
    m6A_threshold = var_dict.get('m6A_threshold', 0.7) # min threshold to call a modified m6a base
    m5C_threshold = var_dict.get('m5C_threshold', 0.7) # min threshold to call a modified 5mC base
    hm5C_threshold = var_dict.get('hm5C_threshold', 0.7) # min threshold to call a modified 5hmC base
    thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
    mod_list = var_dict.get('mod_list', ['5mC_5hmC', '6mA']) # mods to detect
    batch_size = var_dict.get('batch_size', 4) # How many mod TSVs to load into memory at a time when making anndata batches
    skip_unclassified = var_dict.get('skip_unclassified', True)
    delete_batch_hdfs = var_dict.get('delete_batch_hdfs', True)

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
        # Get the file names in the input data dir
        input_files = os.listdir(input_data_path)
        input_is_pod5 = sum([True for file in input_files if '.pod5' in file or '.p5' in file])
        input_is_fast5 = sum([True for file in input_files if '.fast5' in file or '.f5' in file])
        input_is_fastq = sum([True for file in input_files if '.fastq' in file or '.fq' in file])
        input_is_bam = sum([True for file in input_files if bam_suffix in file])
        if input_is_fastq:
            fastq_paths = [os.path.join(input_data_path, file) for file in input_files if '.fastq' in file or '.fq' in file]

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
        concatenate_fastqs_to_bam(fastq_paths, output_bam, barcode_tag='BC', gzip_suffix='.gz')
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
        model_basename = os.path.basename(model)
        model_basename = model_basename.replace('.', '_')
        if smf_modality == 'direct':
            mod_string = "_".join(mod_list)
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
    if barcode_both_ends:
        split_dir = split_dir + '_both_ends_barcoded'
    else:
        split_dir = split_dir + '_at_least_one_end_barcoded'

    # Direct methylation detection SMF specific parameters
    if smf_modality == 'direct':
        mod_bed_dir=f"{split_dir}/split_mod_beds"
        mod_tsv_dir=f"{split_dir}/split_mod_tsvs"
        bam_qc_dir = f"{split_dir}/bam_qc"
        mod_map = {'6mA': '6mA', '5mC_5hmC': '5mC'}
        mods = [mod_map[mod] for mod in mod_list]

    os.chdir(output_directory)
    ########################################################################################################################

    ################################### 2) FASTA Handling ###################################
    from .helpers import generate_converted_FASTA, get_chromosome_lengths

    # If fasta_regions_of_interest bed is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if fasta_regions_of_interest and '.bed' in fasta_regions_of_interest:
        fasta_basename = os.path.basename(fasta).split('.fa')[0]
        bed_basename_minus_suffix = os.path.basename(fasta_regions_of_interest).split('.bed')[0]
        output_FASTA = fasta_basename + '_subsampled_by_' + bed_basename_minus_suffix + '.fasta'
        subsample_fasta_from_bed(fasta, fasta_regions_of_interest, output_directory, output_FASTA)
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
    if basecall:
        if os.path.exists(unaligned_output):
            print(unaligned_output + ' already exists. Using existing basecalled BAM.')
        elif smf_modality != 'direct':
            canoncall(model_dir, model, input_data_path, barcode_kit, bam, bam_suffix, barcode_both_ends, trim, device)
        else:
            modcall(model_dir, model, input_data_path, barcode_kit, mod_list, bam, bam_suffix, barcode_both_ends, trim, device)
    else:
        pass
    ########################################################################################################################

    ################################### 4) Alignment and sorting #############################################
    from .helpers import align_and_sort_BAM, aligned_BAM_to_bed
    # 3) Align the BAM to the reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    if os.path.exists(aligned_output) and os.path.exists(aligned_sorted_output):
        print(aligned_sorted_output + ' already exists. Using existing aligned/sorted BAM.')
    else:
        align_and_sort_BAM(fasta, unaligned_output, bam_suffix, output_directory, make_bigwigs, threads, aligner, aligner_args)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(output_directory, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for ' + aligned_sorted_output)
    else:
        aligned_BAM_to_bed(aligned_output, output_directory, fasta, make_bigwigs, threads)
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
        bam_files = demux_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, barcode_kit, barcode_both_ends, trim, fasta, make_bigwigs, threads)
        # split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, output_directory, converted_FASTA) # deprecated, just use dorado demux

    # Make beds and provide basic histograms
    bed_dir = os.path.join(split_dir, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for demultiplexed bams')
    else:
        for bam in bam_files:
            aligned_BAM_to_bed(bam, split_dir, fasta, make_bigwigs, threads)
    ########################################################################################################################

    ################################### 6) SAMTools based BAM QC ######################################################################
    from .helpers import bam_qc
    # 5) Samtools QC metrics on split BAM files
    bam_qc_dir = f"{split_dir}/bam_qc"
    if os.path.isdir(bam_qc_dir):
        print(bam_qc_dir + ' already exists. Using existing BAM QC calculations.')
    else:
        make_dirs([bam_qc_dir])
        bam_qc(bam_files, bam_qc_dir, threads, modality=smf_modality)
    ######################################################################################################################## 

    ################################### 7) AnnData loading ######################################################################
    if smf_modality != 'direct':
        from .helpers import converted_BAM_to_adata_II
        # 6) Take the converted BAM and load it into an adata object.
        if smf_modality == 'deaminase':
            deaminase_footprinting = True
        else:
            deaminase_footprinting = False
        final_adata, final_adata_path = converted_BAM_to_adata_II(fasta, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix, device, threads, deaminase_footprinting) 
    else:
        if os.path.isdir(mod_bed_dir):
            print(mod_bed_dir + ' already exists, skipping making modbeds')
        else:
            from .helpers import modQC, make_modbed, extract_mods, modkit_extract_to_adata
            make_dirs([mod_bed_dir])  
            modQC(aligned_sorted_output, thresholds) # get QC metrics for mod calls
            make_modbed(aligned_sorted_output, thresholds, mod_bed_dir) # Generate bed files of position methylation summaries for every sample

            make_dirs([mod_tsv_dir])  
            extract_mods(thresholds, mod_tsv_dir, split_dir, bam_suffix, skip_unclassified, threads) # Extract methylations calls for split BAM files into split TSV files

            #6 Load the modification data from TSVs into an adata object
            final_adata, final_adata_path = modkit_extract_to_adata(fasta, split_dir, mapping_threshold, experiment_name, mods, batch_size, mod_tsv_dir, delete_batch_hdfs, threads)

    ## Load sample sheet metadata based on barcode mapping ##
    if sample_sheet_path:
        from ..preprocessing import load_sample_sheet
        load_sample_sheet(final_adata, sample_sheet_path, mapping_key_column='Barcode', as_category=True)
    else:
        pass
    ########################################################################################################################

    ############################################### 8) Basic Read quality metrics: Read length, read quuality, mapping quality, etc #################################################
    if final_adata:
        pass
    else:
        final_adata = ad.read_h5ad(final_adata_path)

    final_adata.obs_names_make_unique()
    cols = final_adata.obs.columns
    for col in cols:
        final_adata.obs[col] = final_adata.obs[col].astype('category')

    # Adding read length, read quality, reference length, mapped_length, and mapping quality metadata to adata object.
    read_metrics = {}
    for bam_file in bam_files:
        bam_read_metrics = extract_read_features_from_bam(bam_file)
        read_metrics.update(bam_read_metrics)

    query_read_length_values = []
    query_read_quality_values = []
    reference_lengths = []
    mapped_lengths = []
    mapping_qualities = []

    # Iterate over each row of the AnnData object
    for obs_name in final_adata.obs_names:
        # Fetch the value from the dictionary using the obs_name as the key
        value = read_metrics.get(obs_name, np.nan)  # Use np.nan if the key is not found
        if type(value) is list:
            query_read_length_values.append(value[0])
            query_read_quality_values.append(value[1])
            reference_lengths.append(value[2])
            mapped_lengths.append(value[3])
            mapping_qualities.append(value[4])
        else:
            query_read_length_values.append(value)
            query_read_quality_values.append(value)
            reference_lengths.append(value) 
            mapped_lengths.append(value)
            mapping_qualities.append(value)
                       
    # Add the new column to adata.obs
    final_adata.obs['read_length'] = query_read_length_values
    final_adata.obs['mapped_length'] = mapped_lengths
    final_adata.obs['reference_length'] = reference_lengths
    final_adata.obs['read_quality'] = query_read_quality_values
    final_adata.obs['mapping_quality'] = mapping_qualities
    final_adata.obs['read_length_to_reference_length_ratio'] = np.array(query_read_length_values) / np.array(reference_lengths)
    final_adata.obs['mapped_length_to_reference_length_ratio'] = np.array(mapped_lengths) / np.array(reference_lengths)
    final_adata.obs['mapped_length_to_read_length_ratio'] = np.array(mapped_lengths) / np.array(query_read_length_values)

    ## Add read level raw modification signal
    final_adata.obs['Raw_modification_signal'] = final_adata.X.sum(axis=1)

    pp_dir = f"{split_dir}/preprocessed"
    pp_qc_dir = f"{pp_dir}/QC_metrics"
    pp_length_qc_dir = f"{pp_qc_dir}/Read_QC_metrics"

    if os.path.isdir(pp_length_qc_dir):
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(final_adata, pp_length_qc_dir, obs_to_plot, sample_key='Barcode')

    ## add optional read length based filtering here.
    from ..preprocessing import filter_reads_on_length
    print(final_adata.shape)
    final_adata = filter_reads_on_length(final_adata, min_length_ratio=0.2, max_length_ratio=1.1)
    print(final_adata.shape)

    pp_length_qc_dir = f"{pp_qc_dir}/Read_QC_metrics_post_filtering"
    if os.path.isdir(pp_length_qc_dir):
        print(pp_length_qc_dir + ' already exists. Skipping read level QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_length_qc_dir])
        obs_to_plot = ['read_length', 'mapped_length','read_quality', 'mapping_quality','mapped_length_to_reference_length_ratio', 'mapped_length_to_read_length_ratio', 'Raw_modification_signal']
        plot_read_qc_histograms(final_adata, pp_length_qc_dir, obs_to_plot, sample_key='Barcode')
    ########################################################################################################################

    ############################################### 9) Basic Preprocessing ###############################################

    ############## Binarize direct modcall data and store in new layer. Clean nans and store as new layers with various nan replacement strategies ##########
    from ..preprocessing import clean_NaN
    if smf_modality == 'direct':
        from ..preprocessing import calculate_position_Youden, binarize_on_Youden
        native = True
        # Calculate positional methylation thresholds for mod calls
        calculate_position_Youden(final_adata, positive_control_sample=None, negative_control_sample=None, J_threshold=0.5, 
                                  obs_column='Reference_strand', infer_on_percentile=10, inference_variable='Raw_modification_signal', save=False, output_directory='')
        # binarize the modcalls based on the determined thresholds
        binarize_on_Youden(final_adata, obs_column='Reference_strand')
        clean_NaN(final_adata, layer='binarized_methylation')
    else:
        native = False
        clean_NaN(final_adata)

    ############### Add base context to each position for each Reference_strand and calculate read level methylation/deamination stats ###############
    from ..preprocessing import append_base_context
    # Additionally, store base_context level binary modification arrays in adata.obsm
    append_base_context(final_adata, obs_column='Reference_strand', use_consensus=False, native=native, mod_target_bases=mod_target_bases)

    ############### Calculate read methylation/deamination statistics for specific base contexts defined above ###############
    from ..preprocessing import calculate_read_modification_stats
    calculate_read_modification_stats(final_adata, "Reference_strand", "Sample", mod_target_bases)

    ### Make a dir for outputting sample level read modification metrics before filtering ###
    pp_dir = f"{split_dir}/preprocessed"
    pp_qc_dir = f"{pp_dir}/QC_metrics"
    pp_meth_qc_dir = f"{pp_qc_dir}/read_methylation_QC"

    if os.path.isdir(pp_meth_qc_dir):
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(final_adata, pp_meth_qc_dir, obs_to_plot, sample_key='Barcode')

    ##### Optionally filter reads on modification metrics
    from ..preprocessing import filter_reads_on_modification_thresholds
    if smf_modality == 'conversion':
        if 'GpC' in mod_target_bases:
            final_adata = filter_reads_on_modification_thresholds(final_adata, gpc_thresholds=[0.025, 0.975], use_other_c_as_background=True)
        if 'CpG' in mod_target_bases:
            final_adata = filter_reads_on_modification_thresholds(final_adata, gpc_thresholds=[0.00, 1])
    elif smf_modality == 'deaminase':
        # Need to add equivalent as above for deamination stats and filtering
        final_adata = filter_reads_on_modification_thresholds(final_adata, any_c_thresholds=[0.025, 0.975])
    elif smf_modality == 'direct':
        if 'A' in mod_target_bases:
            final_adata = filter_reads_on_modification_thresholds(final_adata, a_thresholds=[0.025, 0.975])
        if 'GpC' in mod_target_bases:
            final_adata = filter_reads_on_modification_thresholds(final_adata, gpc_thresholds=[0.025, 0.975])

    ## Plot post filtering read methylation metrics
    pp_meth_qc_dir = f"{pp_qc_dir}/read_methylation_QC_post_filtering"

    if os.path.isdir(pp_meth_qc_dir):
        print(pp_meth_qc_dir + ' already exists. Skipping read level methylation QC plotting.')
    else:
        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_meth_qc_dir])
        obs_to_plot = ['Raw_modification_signal']
        if any(base in mod_target_bases for base in ['GpC', 'CpG', 'C']):
            obs_to_plot += ['Fraction_GpC_site_modified', 'Fraction_CpG_site_modified', 'Fraction_other_C_site_modified', 'Fraction_any_C_site_modified']
        if 'A' in mod_target_bases:
            obs_to_plot += ['Fraction_A_site_modified']
        plot_read_qc_histograms(final_adata, pp_meth_qc_dir, obs_to_plot, sample_key='Barcode')

    ############### Calculate positional coverage in dataset ###############
    from ..preprocessing import calculate_coverage
    calculate_coverage(final_adata, obs_column='Reference_strand', position_nan_threshold=0.1)

    ############### Add layers to adata that are the binary GpC, CpG, any C methylation/deamination patterns ###############
    if smf_modality != 'direct':
        if smf_modality == 'conversion':
            deaminase = False
        else:
            deaminase = True
        references = final_adata.obs['Reference_strand'].cat.categories
        # Step 1: Define reference → GpC and CpG site annotation columns
        reference_to_gpc_column = {reference: f"{reference}_GpC_site" for reference in references}
        reference_to_cpg_column = {reference: f"{reference}_CpG_site" for reference in references}
        reference_to_c_column = {reference: f"{reference}_any_C_site" for reference in references}
        reference_to_other_c_column = {reference: f"{reference}_other_C_site" for reference in references}

        # Step 2: Precompute per-reference var masks
        gpc_var_masks = {
            ref: final_adata.var[col].values.astype(bool)
            for ref, col in reference_to_gpc_column.items()
        }
        cpg_var_masks = {
            ref: final_adata.var[col].values.astype(bool)
            for ref, col in reference_to_cpg_column.items()
        }

        c_var_masks = {
            ref: final_adata.var[col].values.astype(bool)
            for ref, col in reference_to_c_column.items()
        }

        other_c_var_masks = {
            ref: final_adata.var[col].values.astype(bool)
            for ref, col in reference_to_other_c_column.items()
        }

        # Step 3: Build row-level masks
        n_obs, n_vars = final_adata.shape
        gpc_row_mask = np.zeros((n_obs, n_vars), dtype=bool)
        cpg_row_mask = np.zeros((n_obs, n_vars), dtype=bool)
        c_row_mask = np.zeros((n_obs, n_vars), dtype=bool)
        other_c_row_mask = np.zeros((n_obs, n_vars), dtype=bool)

        for ref in reference_to_gpc_column:
            row_indices = final_adata.obs["Reference_strand"] == ref
            gpc_row_mask[row_indices.values, :] = gpc_var_masks[ref]
            cpg_row_mask[row_indices.values, :] = cpg_var_masks[ref]
            c_row_mask[row_indices.values, :] = c_var_masks[ref]
            other_c_row_mask[row_indices.values, :] = other_c_var_masks[ref]

        # Step 4: Mask adata.X
        X = final_adata.X.toarray() if not isinstance(final_adata.X, np.ndarray) else final_adata.X
        masked_gpc_X = np.where(gpc_row_mask, X, np.nan)
        masked_cpg_X = np.where(cpg_row_mask, X, np.nan)
        masked_c_X = np.where(c_row_mask, X, np.nan)
        masked_other_c_X = np.where(other_c_row_mask, X, np.nan)

        # Step 5: Store in layers
        final_adata.layers['GpC_site_binary'] = masked_gpc_X
        final_adata.layers['CpG_site_binary'] = masked_cpg_X
        final_adata.layers['GpC_CpG_combined_site_binary'] = masked_gpc_X + masked_cpg_X
        final_adata.layers['any_C_site_binary'] = masked_c_X
        final_adata.layers['other_C_site_binary'] = masked_other_c_X

        # # Calculate per read metrics:
        # final_adata.obs['Raw_GpC_site_signal'] = np.nansum(final_adata.layers['GpC_site_binary'], axis=1)
        # final_adata.obs['mean_GpC_site_signal_density'] = final_adata.obs['Raw_GpC_site_signal'] / final_adata.obs['read_length']
        # final_adata.obs['Raw_CpG_site_signal'] = np.nansum(final_adata.layers['CpG_site_binary'], axis=1)
        # final_adata.obs['mean_CpG_site_signal_density'] = final_adata.obs['Raw_CpG_site_signal'] / final_adata.obs['read_length']
        # final_adata.obs['Raw_any_C_site_signal'] = np.nansum(final_adata.layers['any_C_site_binary'], axis=1)
        # final_adata.obs['mean_any_C_site_signal_density'] = final_adata.obs['Raw_any_C_site_signal'] / final_adata.obs['read_length']
        # final_adata.obs['Raw_other_C_site_signal'] = np.nansum(final_adata.layers['other_C_site_binary'], axis=1)
        # final_adata.obs['mean_other_C_site_signal_density'] = final_adata.obs['Raw_other_C_site_signal'] / final_adata.obs['read_length']

        # ### Make a dir for outputting sample level QC metrics before preprocessing ###
        # pp_dir = f"{split_dir}/preprocessed"
        # pp_qc_dir = f"{pp_dir}/QC_metrics"
        # if os.path.isdir(pp_qc_dir):
        #     print(pp_qc_dir + ' already exists. Skipping read level QC plotting.')
        # else:
        #     from ..plotting import plot_read_qc_histograms
        #     make_dirs([pp_dir, pp_qc_dir])
        #     obs_to_plot = ['Raw_GpC_site_signal', 'mean_GpC_site_signal_density', 'Raw_CpG_site_signal', 
        #                    'mean_CpG_site_signal_density', 'Raw_any_C_site_signal', 'mean_any_C_site_signal_density', 
        #                    'Raw_other_C_site_signal', 'mean_other_C_site_signal_density']
        #     plot_read_qc_histograms(final_adata, pp_qc_dir, obs_to_plot, sample_key='Barcode')

    ############### Duplicate detection for conversion/deamination SMF ###############
    if smf_modality != 'direct' and 'is_duplicate' not in final_adata.obs.columns:
        from ..preprocessing import flag_duplicate_reads, calculate_complexity_II
        references = final_adata.obs['Reference_strand'].cat.categories
        if smf_modality == 'conversion':
            site_types = ['GpC', 'CpG', 'ambiguous_GpC_CpG']
            distance_threshold = 0.1
        elif smf_modality == 'deaminase':
            site_types = ['any_C']
            distance_threshold = 0.01

        var_filters_sets =[]
        for ref in references:
            for site_type in site_types:
                var_filters_sets += [[f"{ref}_{site_type}_site", f"position_in_{ref}"]]

        pp_dir = f"{split_dir}/preprocessed"
        pp_qc_dir = f"{pp_dir}/QC_metrics"
        pp_dup_qc_dir = f"{pp_qc_dir}/read_duplication_QC"

        from ..plotting import plot_read_qc_histograms
        make_dirs([pp_dir, pp_qc_dir, pp_dup_qc_dir])

        ## Will need to improve here. Should do this for each barcode and then concatenate. rather than all at once ###
        final_adata_unique, final_adata = flag_duplicate_reads(final_adata, 
                                                            var_filters_sets, 
                                                            distance_threshold=distance_threshold, 
                                                            obs_reference_col='Reference_strand', 
                                                            sample_col='Barcode',
                                                            output_directory=pp_dup_qc_dir)

        calculate_complexity_II(
            adata=final_adata,
            output_directory=pp_dup_qc_dir,
            sample_col='Barcode',
            cluster_col='merged_cluster_id',
            plot=True,
            save_plot=True,   # set False to display instead
            n_boot=30,
            n_depths=12,
            random_state=42,
            csv_summary=True,
        )

    ########################################################################################################################

    ############################################### Basic Analyses ###############################################
    if smf_modality != 'direct':
        if smf_modality == 'conversion':
            deaminase = False
        else:
            deaminase = True
        references = final_adata.obs['Reference_strand'].cat.categories

        pp_dir = f"{split_dir}/preprocessed"
        pp_clustermap_dir = f"{pp_dir}/clustermaps"
        pp_umap_dir = f"{pp_dir}/umaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(final_adata, sample_col='Barcode', layer_any_c='nan0_0minus1', 
                                        layer_gpc="nan0_0minus1", layer_cpg="nan0_0minus1", cmap_any_c="coolwarm", 
                                        cmap_gpc="coolwarm", cmap_cpg="viridis", min_quality=25, min_length=500, bins=None,
                                        sample_mapping=None, save_path=pp_clustermap_dir, sort_by='gpc', deaminase=deaminase)
        
        ## Basic PCA/UMAP
        if os.path.isdir(pp_umap_dir):
            print(pp_umap_dir + ' already exists. Skipping UMAP plotting.')
        else:
            from ..tools import calculate_umap
            make_dirs([pp_dir, pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            final_adata = calculate_umap(final_adata, layer='nan_half', var_filters=var_filters, n_pcs=10, knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(final_adata, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            save = 'umap_plot.png'
            sc.pl.umap(final_adata, color=['leiden', 'Sample'], show=False, save=save)

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_clustermap_dir = f"{pp_dir}/clustermaps"
        pp_umap_dir = f"{pp_dir}/umaps"

        # ## Basic clustermap plotting
        if os.path.isdir(pp_clustermap_dir):
            print(pp_clustermap_dir + ' already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(final_adata_unique, sample_col='Barcode', layer_any_c='nan0_0minus1', 
                                        layer_gpc="nan0_0minus1", layer_cpg="nan0_0minus1", cmap_any_c="coolwarm", 
                                        cmap_gpc="coolwarm", cmap_cpg="viridis", min_quality=25, min_length=500, bins=None,
                                        sample_mapping=None, save_path=pp_clustermap_dir, sort_by='gpc', deaminase=deaminase)
        
        ## Basic PCA/UMAP
        if os.path.isdir(pp_umap_dir):
            print(pp_umap_dir + ' already exists. Skipping UMAP plotting.')
        else:
            from ..tools import calculate_umap
            make_dirs([pp_dir, pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            final_adata = calculate_umap(final_adata_unique, layer='nan_half', var_filters=var_filters, n_pcs=10, knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(final_adata_unique, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            save = 'umap_plot.png'
            sc.pl.umap(final_adata_unique, color=['leiden', 'Sample'], show=False, save=save)


    ########################################################################################################################

    ############################################### Spatial analyses ###############################################
    from ..tools.read_stats import binary_autocorrelation_with_spacing
    if smf_modality != 'direct':
        pp_dir = f"{split_dir}/preprocessed"
        pp_autocorr_dir = f"{pp_dir}/autocorrelations"

        positions = final_adata.var_names.astype(int).values
        site_types = ['GpC', 'CpG', 'any_C']
        for site_type in site_types:
            X = final_adata.layers[f"{site_type}_site_binary"]
            max_lag = 500

            autocorr_matrix = np.array([
                binary_autocorrelation_with_spacing(row, positions, max_lag=max_lag)
                for row in X
            ])

            final_adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
            final_adata.uns[f"{site_type}_spatial_autocorr_lags"] = np.arange(max_lag + 1)

        if os.path.isdir(pp_autocorr_dir):
            print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
        else:
            from ..plotting import plot_spatial_autocorr_grid
            make_dirs([pp_dir, pp_autocorr_dir])

            plot_spatial_autocorr_grid(final_adata, pp_autocorr_dir, site_types=site_types, sample_col='Barcode', window=25, rows_per_fig=6)

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = f"{split_dir}/preprocessed_duplicates_removed"
        pp_autocorr_dir = f"{pp_dir}/autocorrelations"

        positions = final_adata_unique.var_names.astype(int).values
        site_types = ['GpC', 'CpG', 'any_C']
        for site_type in site_types:
            X = final_adata_unique.layers[f"{site_type}_site_binary"]
            max_lag = 500

            autocorr_matrix = np.array([
                binary_autocorrelation_with_spacing(row, positions, max_lag=max_lag)
                for row in X
            ])

            final_adata_unique.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
            final_adata_unique.uns[f"{site_type}_spatial_autocorr_lags"] = np.arange(max_lag + 1)

        if os.path.isdir(pp_autocorr_dir):
            print(pp_autocorr_dir + ' already exists. Skipping autocorrelation plotting.')
        else:
            from ..plotting import plot_spatial_autocorr_grid
            make_dirs([pp_autocorr_dir, pp_autocorr_dir])

            plot_spatial_autocorr_grid(final_adata_unique, pp_autocorr_dir, site_types=site_types, sample_col='Barcode', window=25, rows_per_fig=6)

    else:
        pass
    ########################################################################################################################

    ############################################### HMM based feature annotations ###############################################
    # need to add the option here to do a per sample HMM fit/inference
    run_hmm=True
    if run_hmm:
        from ..hmm.HMM import HMM

        samples = final_adata.obs['Barcode'].cat.categories
        references = final_adata.obs['Reference_strand'].cat.categories
        mod_sites = mod_target_bases
        pp_dir = f"{split_dir}/preprocessed"
        hmm_dir = f"{pp_dir}/hmm_models"

        emission_probs=[[0.8, 0.2], [0.2, 0.8]],
        transitions=[[0.9, 0.1], [0.1, 0.9]],
        start_probs=[0.5, 0.5],
        end_probs=[0.5, 0.5]

        if os.path.isdir(hmm_dir):
            print(hmm_dir + ' already exists.')
        else:
            make_dirs([pp_dir, hmm_dir])

        for sample in samples:
            for ref in references:
                mask = (final_adata.obs['Barcode'] == sample) & (final_adata.obs['Reference_strand'] == ref)
                subset = final_adata[mask].copy()
                if subset.shape[0] > 1:
                    for mod_site in mod_sites:
                        mod_label = {'C': 'any_C'}.get(mod_site, mod_site)
                        hmm_path = os.path.join(hmm_dir, f"{sample}_{ref}_{mod_label}_hmm_model.pth")
                        if os.path.exists(hmm_path):
                            hmm = HMM.load(hmm_path)
                        else:
                            print(f"Fitting HMM for {sample} {ref} {mod_label}")
                            hmm = HMM(n_states=2, init_start=start_probs,init_trans=transitions,init_emission=emission_probs)
                            hmm.fit(subset.obsm[f'{ref}_{mod_label}_site'])
                            hmm.save(hmm_path)

                        print(f"Apply HMM for {sample} {ref} {mod_label}")

                        hmm.annotate_adata(subset, 'Reference_strand', layer=None, footprints=True, accessible_patches=True, cpg=True, methbases=[mod_label])

                        # for col in subset.obs.columns:
                        #     if col not in final_adata.obs.columns:
                        #         final_adata.obs[col] = pd.NA
                        #     final_adata.obs.loc[subset.obs_names, col] = subset.obs[col]

                        # for layer in subset.layers:
                        #     if layer not in final_adata.layers:
                        #         final_adata.layers[layer] = np.full_like(final_adata.X, np.nan)
                        #     final_adata.layers[layer][mask] = subset.layers[layer]


    ########################################################################################################################

    ############################################### Save final adata ###############################################
    from ..readwrite import safe_write_h5ad
    print('Saving final adata')
    if ".gz" in final_adata_path:
        safe_write_h5ad(final_adata, f"{final_adata_path}", compression='gzip')
    else:
        safe_write_h5ad(final_adata, f"{final_adata_path}.gz", compression='gzip')
    ########################################################################################################################

    ############################################### MultiQC HTML Report ###############################################
    from .helpers import run_multiqc
    # multiqc ###
    if os.path.isdir(f"{split_dir}/multiqc"):
        print(f"{split_dir}/multiqc" + ' already exists, skipping multiqc')
    else:
        run_multiqc(split_dir, f"{split_dir}/multiqc")
    ########################################################################################################################