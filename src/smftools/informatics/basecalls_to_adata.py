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
    import os
    bam_suffix = '.bam' # If different, change from here.
    split_dir = 'split_BAMs' # If different, change from here.
    strands = ['bottom', 'top'] # If different, change from here. Having both listed generally doesn't slow things down too much.
    conversions = ['unconverted'] # The name to use for the unconverted files. If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict
    for key, value in var_dict.items():
        globals()[key] = value

    split_path = os.path.join(output_directory, split_dir)
    make_dirs([output_directory, split_path])
    os.chdir(output_directory)

    conversions += conversion_types

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