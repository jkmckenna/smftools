## pod5_conversion

def pod5_conversion(fasta, output_directory, conversion_types, strands, model, pod5_dir, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix):
    """
    Converts a POD5 file from a nanopore conversion SMF experiment to an adata object.

    Parameters:
        fasta (str): File path to the reference genome to align to.
        output_directory (str): A file path to the directory to output all the analyses.
        conversion_type (list): A list of strings of the conversion types to use in the analysis.
        strands (list): A list of converstion strands to use in the experiment.
        model (str): a string representing the file path to the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        bam_suffix (str): A suffix to add to the bam file.

    Returns:
        None
    """
    from .helpers import align_and_sort_BAM, canoncall, converted_BAM_to_adata, generate_converted_FASTA, split_and_index_BAM
    import os
    model_basename = os.path.basename(model)
    model_basename = model_basename.replace('.', '_')
    bam=f"{output_directory}/{model_basename}_canonical_basecalls"
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"

    os.chdir(output_directory)
    
    # 1) Convert FASTA file
    fasta_basename = os.path.basename(fasta)
    converted_FASTA_basename = fasta_basename.split('.fa')[0]+'_converted.fasta'
    converted_FASTA = os.path.join(output_directory, converted_FASTA_basename)
    if os.path.exists(converted_FASTA):
        print(converted_FASTA + ' already exists. Using existing converted FASTA.')
    else:
        generate_converted_FASTA(fasta, conversion_types, strands, converted_FASTA)

    # 2) Basecall from the input POD5 to generate a singular output BAM
    canoncall(model, pod5_dir, barcode_kit, bam, bam_suffix)

    # 3) Align the BAM to the converted reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    input_BAM = bam + bam_suffix
    align_and_sort_BAM(converted_FASTA, input_BAM, bam_suffix, output_directory)

    ### 4) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory###
    split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix)

    # 5) Take the converted BAM and load it into an adata object. 
    converted_BAM_to_adata(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix)