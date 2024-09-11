## pod5_to_adata

def pod5_to_adata(config_path):
    """
    High-level function to call for converting raw sequencing data to an adata object. Works for pod5 and fast5 data types.

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        None
    """
    from .helpers import LoadExperimentConfig, make_dirs
    from .fast5_to_pod5 import fast5_to_pod5
    import os
    bam_suffix = '.bam' # If different, change from here.
    split_dir = 'split_BAMs' # If different, change from here.
    strands = ['bottom', 'top'] # If different, change from here. Having both listed generally doesn't slow things down too much.
    conversions = ['unconverted'] # The name to use for the unconverted files. If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict

    conversion_types = var_dict.get('conversion_types')
    pod5_dir = var_dict.get('pod5_dir')
    output_directory = var_dict.get('output_directory')
    output_pod5 = var_dict.get('output_pod5')
    smf_modality = var_dict.get('smf_modality')
    fasta = var_dict.get('fasta')
    model = var_dict.get('model')
    barcode_kit = var_dict.get('barcode_kit') 
    mapping_threshold = var_dict.get('mapping_threshold')
    experiment_name = var_dict.get('experiment_name')
    filter_threshold = var_dict.get('filter_threshold')
    m6A_threshold = var_dict.get('m6A_threshold')
    m5C_threshold = var_dict.get('m5C_threshold')
    hm5C_threshold = var_dict.get('hm5C_threshold')
    mod_list = var_dict.get('mod_list')
    batch_size = var_dict.get('batch_size')

    conversions += conversion_types

    split_path = os.path.join(output_directory, split_dir)
    make_dirs([output_directory, split_path])
    os.chdir(output_directory)

    # Get the file names in the input pod5_dir
    nanopore_files = os.listdir(pod5_dir)
    input_is_pod5 = sum([True for file in nanopore_files if '.pod5' in file])
    input_is_fast5 = sum([True for file in nanopore_files if '.fast5' in file])

    # If the input files are not pod5 files, and they are fast5 files, convert the files to a pod5 file before proceeding.
    if input_is_fast5 and not input_is_pod5:
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        print(f'Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_directory}')
        fast5_to_pod5(pod5_dir, output_dir=output_directory, output_pod5='FAST5s_to_POD5.pod5')
        # Reassign the pod5_dir variable to point to the new pod5 file.
        pod5_dir = os.path.join(output_directory, output_pod5)

    if smf_modality == 'conversion':
        from .pod5_conversion import pod5_conversion
        pod5_conversion(fasta, output_directory, conversions, strands, model, pod5_dir, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix)
    elif smf_modality == 'direct':
        from .pod5_direct import pod5_direct
        thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
        pod5_direct(fasta, output_directory, mod_list, model, thresholds, pod5_dir, split_path, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size)
    else:
        print("Error")
