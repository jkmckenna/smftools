## bam_to_adata

def bam_to_adata(config_path):
    """
    High-level function to call for loading basecalled SMF data from a BAM file into an adata object.

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        None
    """
    from .helpers import load_experiment_config, make_dirs
    import os
    # Load experiment config parameters into global variables
    experiment_config = load_experiment_config(config_path)
    var_dict = experiment_config.var_dict
    for key, value in var_dict.items():
        globals()[key] = value

    split_path = os.path.join(output_directory, split_dir)
    make_dirs([output_directory, split_path])

    if smf_modality == 'conversion':
        from .bam_conversion import bam_conversion
        bam_conversion(fasta, output_directory, conversion_types, strands, bam_path, split_path, mapping_threshold, experiment_name, bam_suffix)
    elif smf_modality == 'direct':
        from .bam_direct import bam_direct
        bam_direct(fasta, output_directory, mod_list, thresholds, pod5_dir, split_path, mapping_threshold, experiment_name, bam_suffix, batch_size)
    else:
        print("Error")