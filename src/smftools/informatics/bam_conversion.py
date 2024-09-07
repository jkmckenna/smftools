## bam_conversion

def bam_conversion(fasta, output_directory, conversion_types, strands, bam_path, split_dir, mapping_threshold, experiment_name, bam_suffix):
    """
    Converts a BAM file from a nanopore conversion SMF experiment to an adata object.

    Parameters:
        fasta (str): File path to the reference genome to align to.
        output_directory (str): A file path to the directory to output all the analyses.
        conversion_type (list): A list of strings of the conversion types to use in the analysis.
        strands (list): A list of converstion strands to use in the experiment.
        bam_path (str): a string representing the file path to the experiment BAM file.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        bam_suffix (str): A suffix to add to the bam file.

    Returns:
        None
    """
    from .helpers import align_BAM, converted_BAM_to_adata, generate_converted_FASTA, split_and_index_BAM
    import os
    input_bam_base = os.path.basename(bam_path)
    bam_basename = input_bam_base.split(bam_suffix)[0]
    output_bam=f"{output_directory}/{bam_basename}"
    aligned_BAM=f"{output_bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"

    # 1) Convert FASTA file
    converted_FASTA=fasta.split('.fa')[0]+'_converted.fasta'
    if os.path.exists(converted_FASTA):
        print(converted_FASTA + ' already exists. Using existing converted FASTA.')
    else:
        generate_converted_FASTA(fasta, conversion_types, strands, converted_FASTA)

    # 2) Align the BAM to the converted reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    input_bam = bam_path.split(bam_suffix)[0]
    align_BAM(converted_FASTA, input_bam, bam_suffix)

    ### 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory###
    split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix)

    # 4) Take the converted BAM and load it into an adata object. 
    converted_BAM_to_adata(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix)