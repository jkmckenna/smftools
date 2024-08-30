## pod5_direct
from .helpers import align_BAM, extract_mods, make_modbed, modcall, modkit_extract_to_adata, modQC, split_and_index_BAM

def pod5_direct(fasta, output_directory, mod_list, model, thresholds, pod5_dir, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size):
    """
    
    """
    bam=f"{output_directory}/HAC_mod_calls"
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    mod_bed_dir=f"{output_directory}/split_mod_beds"
    mod_tsv_dir=f"{output_directory}/split_mod_tsvs"

    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    mod_map = {'6mA': '6mA', '5mC_5hmC': '5mC'}
    mods = [mod_map[mod] for mod in mod_list]

    # 1) Basecall using dorado
    modcall(model, pod5_dir, barcode_kit, mod_list, bam, bam_suffix)
    # 2) Align the BAM to the converted reference FASTA. Also make an index and a bed file of mapped reads
    align_BAM(fasta, bam, bam_suffix)
    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix)
    # 4) Using nanopore modkit to work with modified BAM files ###
    modQC(aligned_sorted_output, thresholds) # get QC metrics for mod calls
    make_modbed(aligned_sorted_output, thresholds, mod_bed_dir) # Generate bed files of position methylation summaries for every sample
    extract_mods(thresholds, mod_tsv_dir, split_dir, bam_suffix) # Extract methylations calls for split BAM files into split TSV files
    #5 Load the modification data from TSVs into an adata object
    modkit_extract_to_adata(fasta, aligned_sorted_output, mapping_threshold, experiment_name, mods, batch_size)