## basecalls_to_adata

def basecalls_to_adata(config_path):
    """
    High-level function to call for loading basecalled SMF data from a BAM file into an adata object. Also works with FASTQ for conversion SMF.

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        None
    """
    from .helpers import LoadExperimentConfig, make_dirs
    from .subsample_fasta_from_bed import subsample_fasta_from_bed
    import os
    import numpy as np
    bam_suffix = '.bam' # If different, change from here.
    split_dir = 'split_BAMs' # If different, change from here.
    strands = ['bottom', 'top'] # If different, change from here. Having both listed generally doesn't slow things down too much.
    conversions = ['unconverted'] # The name to use for the unconverted files. If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict

    # These below variables will point to the value np.nan if they are either empty in the experiment_config.csv or if the variable is fully omitted from the csv.
    default_value = None
    
    conversion_types = var_dict.get('conversion_types', default_value)
    output_directory = var_dict.get('output_directory', default_value)
    smf_modality = var_dict.get('smf_modality', default_value)
    fasta = var_dict.get('fasta', default_value)
    fasta_regions_of_interest = var_dict.get("fasta_regions_of_interest", default_value)
    basecalled_path = var_dict.get('basecalled_path', default_value)
    mapping_threshold = var_dict.get('mapping_threshold', default_value)
    experiment_name = var_dict.get('experiment_name', default_value)
    filter_threshold = var_dict.get('filter_threshold', default_value)
    m6A_threshold = var_dict.get('m6A_threshold', default_value)
    m5C_threshold = var_dict.get('m5C_threshold', default_value)
    hm5C_threshold = var_dict.get('hm5C_threshold', default_value)
    mod_list = var_dict.get('mod_list', default_value)
    batch_size = var_dict.get('batch_size', default_value)
    thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]

    split_path = os.path.join(output_directory, split_dir)

    make_dirs([output_directory])
    os.chdir(output_directory)

    conversions += conversion_types

    # If a bed file is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if fasta_regions_of_interest != None:
        if '.bed' in fasta_regions_of_interest:
            fasta_basename = os.path.basename(fasta)
            bed_basename_minus_suffix = os.path.basename(fasta_regions_of_interest).split('.bed')[0]
            output_FASTA = bed_basename_minus_suffix + '_' + fasta_basename
            subsample_fasta_from_bed(fasta, fasta_regions_of_interest, output_directory, output_FASTA)
            fasta = output_FASTA

    if smf_modality == 'conversion':
        from .bam_conversion import bam_conversion
        bam_conversion(fasta, output_directory, conversions, strands, basecalled_path, split_path, mapping_threshold, experiment_name, bam_suffix)
    elif smf_modality == 'direct':
        if bam_suffix in basecalled_path:
            from .bam_direct import bam_direct
            bam_direct(fasta, output_directory, mod_list, thresholds, basecalled_path, split_path, mapping_threshold, experiment_name, bam_suffix, batch_size)
        else:
            print('basecalls_to_adata function only work with the direct modality when the input filetype is BAM and not FASTQ.')
    else:
        print("Error")