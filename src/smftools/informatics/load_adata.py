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
    from .helpers import LoadExperimentConfig, make_dirs, concatenate_fastqs_to_bam
    from .fast5_to_pod5 import fast5_to_pod5
    from .subsample_fasta_from_bed import subsample_fasta_from_bed
    import os
    import numpy as np
    from pathlib import Path

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

    # General config variable init
    smf_modality = var_dict.get('smf_modality', default_value) # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Necessary.
    input_data_path = var_dict.get('input_data_path', default_value) # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = var_dict.get('output_directory', default_value) # Path to the output directory to make for the analysis. Necessary.
    fasta = var_dict.get('fasta', default_value) # Path to reference FASTA.
    fasta_regions_of_interest = var_dict.get("fasta_regions_of_interest", default_value) # Path to a bed file listing coordinate regions of interest within the FASTA to include. Optional.
    mapping_threshold = var_dict.get('mapping_threshold', default_value) # Minimum proportion of mapped reads that need to fall within a region to include in the final AnnData.
    experiment_name = var_dict.get('experiment_name', default_value) # A key term to add to the AnnData file name.
    model = var_dict.get('model', default_value) # needed for dorado basecaller
    barcode_kit = var_dict.get('barcode_kit', default_value) # needed for dorado basecaller
    # Conversion specific variable init
    conversion_types = var_dict.get('conversion_types', default_value)
    # Direct methylation specific variable init
    filter_threshold = var_dict.get('filter_threshold', default_value)
    m6A_threshold = var_dict.get('m6A_threshold', default_value)
    m5C_threshold = var_dict.get('m5C_threshold', default_value)
    hm5C_threshold = var_dict.get('hm5C_threshold', default_value)
    thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
    mod_list = var_dict.get('mod_list', default_value)
    batch_size = var_dict.get('batch_size', default_value)

    # Make initial output directory
    make_dirs([output_directory])
    os.chdir(output_directory)
    # Define the pathname to split BAMs into later during demultiplexing.
    split_path = os.path.join(output_directory, split_dir)

    # If fasta_regions_of_interest is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if fasta_regions_of_interest and '.bed' in fasta_regions_of_interest:
        fasta_basename = os.path.basename(fasta).split('.fa')[0]
        bed_basename_minus_suffix = os.path.basename(fasta_regions_of_interest).split('.bed')[0]
        output_FASTA = fasta_basename + '_subsampled_by_' + bed_basename_minus_suffix + '.fasta'
        subsample_fasta_from_bed(fasta, fasta_regions_of_interest, output_directory, output_FASTA)
        fasta = os.path.join(output_directory, output_FASTA)

    # If conversion_types is passed:
    if conversion_types:
        conversions += conversion_types

    # Get the input filetype
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
    
    elif input_is_fastq:
        output_bam = os.path.join(output_directory, 'FASTQs_concatenated_into_BAM.bam')
        concatenate_fastqs_to_bam(fastq_paths, output_bam, barcode_tag='BC', gzip_suffix='.gz')
        input_data_path = output_bam
        input_is_bam = True
        input_is_fastq = False

    if input_is_pod5:
        basecall = True
    elif input_is_bam:
        basecall = False
    else:
        print('Error, can not find input bam or pod5')

    if smf_modality == 'conversion':
        from .conversion_smf import conversion_smf
        conversion_smf(fasta, output_directory, conversions, strands, model, input_data_path, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix, basecall)
    elif smf_modality == 'direct':
        from .direct_smf import direct_smf
        direct_smf(fasta, output_directory, mod_list, model, thresholds, input_data_path, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size, basecall)
    else:
            print("Error")
