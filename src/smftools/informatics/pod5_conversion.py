## pod5_conversion
from .helpers import align_BAM, canoncall, converted_BAM_to_adata, generate_converted_FASTA, split_and_index_BAM
import subprocess

def pod5_conversion(fasta, output_directory, conversion_types, strands, model, pod5_dir, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix):
    """
    Converts a POD5 file from a nanopore conversion SMF experiment to an adata object
    """
    bam=f"{output_directory}/HAC_basecalls"
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    # 1) Convert FASTA file
    converted_FASTA=fasta.split('.fa')[0]+'_converted.fasta'
    generate_converted_FASTA(fasta, conversion_types, strands, converted_FASTA)

    # 2) Basecall from the input POD5 to generate a singular output BAM
    canoncall(model, pod5_dir, barcode_kit, bam, bam_suffix)

    # 3) Align the BAM to the converted reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    align_BAM(converted_FASTA, bam, bam_suffix)

    ### 4) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory###
    split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix)

    # 5) Take the converted BAM and load it into an adata object. 
    converted_BAM_to_adata(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix)