## pod5_to_adata
from .helpers import load_experiment_config
from.pod5_direct import pod5_direct
from.pod5_conversion import pod5_conversion

def pod5_to_adata(config_path, ):
    """
    
    """
    # Load experiment config parameters into global variables
    load_experiment_config(config_path)
    if smf_modality == 'conversion':
        (fasta, output_directory, conversion_types, strands, model, pod5_dir, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix)
    elif smf_modality == 'direct':
        pod5_direct(fasta, output_directory, mod_list, model, thresholds, pod5_dir, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size)
    else:
        print("Error")
