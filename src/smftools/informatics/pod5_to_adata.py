## pod5_to_adata

def pod5_to_adata(config_path):
    """
    High-level function to call for converting raw sequencing data to an adata object.

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

    conversions += conversion_types

    split_path = os.path.join(output_directory, split_dir)
    make_dirs([output_directory, split_path])
    os.chdir(output_directory)

    if smf_modality == 'conversion':
        from .pod5_conversion import pod5_conversion
        pod5_conversion(fasta, output_directory, conversions, strands, model, pod5_dir, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix)
    elif smf_modality == 'direct':
        from .pod5_direct import pod5_direct
        thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
        pod5_direct(fasta, output_directory, mod_list, model, thresholds, pod5_dir, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size)
    else:
        print("Error")
