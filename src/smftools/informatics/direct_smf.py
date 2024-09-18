## direct_smf

def direct_smf(fasta, output_directory, mod_list, model, thresholds, input_data_path, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix, batch_size, basecall):
    """
    Processes sequencing data from a direct methylation detection Nanopore SMF experiment to an AnnData object.

    Parameters:
        fasta (str): File path to the reference genome to align to.
        output_directory (str): A file path to the directory to output all the analyses.
        mod_list (list): A list of strings of the modification types to use in the analysis.
        model (str): a string representing the file path to the dorado basecalling model.
        thresholds (list): A list of floats to pass for call thresholds.
        input_data_path (str): a string representing the file path to the experiment directory containing the input sequencing files.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        bam_suffix (str): A suffix to add to the bam file.
        batch_size (int): An integer number of TSV files to analyze in memory at once while loading the final adata object.
        basecall (bool): Whether to basecall

    Returns:
        None   
    """
    from .helpers import align_and_sort_BAM, extract_mods, get_chromosome_lengths, make_modbed, modcall, modkit_extract_to_adata, modQC, split_and_index_BAM, make_dirs
    import os

    if basecall:
        model_basename = os.path.basename(model)
        model_basename = model_basename.replace('.', '_')
        mod_string = "_".join(mod_list)
        bam=f"{output_directory}/{model_basename}_{mod_string}_calls"
    else:
        bam_base=os.path.basename(input_data_path).split('.bam')[0]
        bam=os.path.join(output_directory, bam_base)
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    mod_bed_dir=f"{output_directory}/split_mod_beds"
    mod_tsv_dir=f"{output_directory}/split_mod_tsvs"

    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    mod_map = {'6mA': '6mA', '5mC_5hmC': '5mC'}
    mods = [mod_map[mod] for mod in mod_list]

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)

    os.chdir(output_directory)

    # 1) Basecall using dorado
    if basecall:
        modcall_output = bam + bam_suffix
        if os.path.exists(modcall_output):
            print(modcall_output + ' already exists. Using existing basecalled BAM.')
        else:
            modcall(model, input_data_path, barcode_kit, mod_list, bam, bam_suffix)
    else:
        modcall_output = input_data_path

    # 2) Align the BAM to the reference FASTA. Also make an index and a bed file of mapped reads
    aligned_output = aligned_BAM + bam_suffix
    sorted_output = aligned_sorted_BAM + bam_suffix
    if os.path.exists(aligned_output) and os.path.exists(sorted_output):
        print(sorted_output + ' already exists. Using existing aligned/sorted BAM.')
    else:
        align_and_sort_BAM(fasta, modcall_output, bam_suffix, output_directory)

    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if os.path.isdir(split_dir):
        print(split_dir + ' already exists. Using existing aligned/sorted/split BAMs.')
    else:
        make_dirs([split_dir])
        split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, output_directory, fasta)

    # 4) Using nanopore modkit to work with modified BAM files ###
    if os.path.isdir(mod_bed_dir):
        print(mod_bed_dir + ' already exists')
    else:
        make_dirs([mod_bed_dir])  
        modQC(aligned_sorted_output, thresholds) # get QC metrics for mod calls
        make_modbed(aligned_sorted_output, thresholds, mod_bed_dir) # Generate bed files of position methylation summaries for every sample
    if os.path.isdir(mod_tsv_dir):
        print(mod_tsv_dir + ' already exists')
    else:
        make_dirs([mod_tsv_dir])  
        extract_mods(thresholds, mod_tsv_dir, split_dir, bam_suffix) # Extract methylations calls for split BAM files into split TSV files

    #5 Load the modification data from TSVs into an adata object
    modkit_extract_to_adata(fasta, split_dir, mapping_threshold, experiment_name, mods, batch_size, mod_tsv_dir)