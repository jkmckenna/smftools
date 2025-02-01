# basecall_pod5s

def basecall_pod5s(config_path):
    """
    Basecall from pod5s given a config file.

    Parameters:
        config_path (str): File path to the basecall configuration file

    Returns:
        None
    """
    # Lazy importing of packages
    from .helpers import LoadExperimentConfig, make_dirs, canoncall, modcall
    from .fast5_to_pod5 import fast5_to_pod5
    import os
    from pathlib import Path

    # Default params
    bam_suffix = '.bam' # If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict

    # These below variables will point to default_value if they are empty in the experiment_config.csv or if the variable is fully omitted from the csv.
    default_value = None

    # General config variable init
    input_data_path = var_dict.get('input_data_path', default_value) # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = var_dict.get('output_directory', default_value) # Path to the output directory to make for the analysis. Necessary.
    model = var_dict.get('model', default_value) # needed for dorado basecaller
    barcode_kit = var_dict.get('barcode_kit', default_value) # needed for dorado basecaller
    barcode_both_ends = var_dict.get('barcode_both_ends', default_value) # dorado demultiplexing
    trim = var_dict.get('trim', default_value) # dorado adapter and barcode removal
    device = var_dict.get('device', 'auto')

    # Modified basecalling specific variable init
    filter_threshold = var_dict.get('filter_threshold', default_value)
    m6A_threshold = var_dict.get('m6A_threshold', default_value)
    m5C_threshold = var_dict.get('m5C_threshold', default_value)
    hm5C_threshold = var_dict.get('hm5C_threshold', default_value)
    thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
    mod_list = var_dict.get('mod_list', default_value)
    
    # Make initial output directory
    make_dirs([output_directory])
    os.chdir(output_directory)

    # Get the input filetype
    if Path(input_data_path).is_file():
        input_data_filetype = '.' + os.path.basename(input_data_path).split('.')[1].lower()
        input_is_pod5 = input_data_filetype in ['.pod5','.p5']
        input_is_fast5 = input_data_filetype in ['.fast5','.f5']

    elif Path(input_data_path).is_dir():
        # Get the file names in the input data dir
        input_files = os.listdir(input_data_path)
        input_is_pod5 = sum([True for file in input_files if '.pod5' in file or '.p5' in file])
        input_is_fast5 = sum([True for file in input_files if '.fast5' in file or '.f5' in file])

    # If the input files are not pod5 files, and they are fast5 files, convert the files to a pod5 file before proceeding.
    if input_is_fast5 and not input_is_pod5:
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        output_pod5 = os.path.join(output_directory, 'FAST5s_to_POD5.pod5')
        print(f'Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}')
        fast5_to_pod5(input_data_path, output_pod5)
        # Reassign the pod5_dir variable to point to the new pod5 file.
        input_data_path = output_pod5

    model_basename = os.path.basename(model)
    model_basename = model_basename.replace('.', '_')

    if mod_list:
        mod_string = "_".join(mod_list)
        bam=f"{output_directory}/{model_basename}_{mod_string}_calls"
        modcall(model, input_data_path, barcode_kit, mod_list, bam, bam_suffix, barcode_both_ends, trim, device)
    else:
        bam=f"{output_directory}/{model_basename}_canonical_basecalls"
        canoncall(model, input_data_path, barcode_kit, bam, bam_suffix, barcode_both_ends, trim, device)
