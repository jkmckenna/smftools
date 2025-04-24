## conversion_smf

def conversion_smf(fasta, output_directory, conversion_types, strands, model_dir, model, input_data_path, split_dir, barcode_kit, mapping_threshold, experiment_name, bam_suffix, basecall, barcode_both_ends, trim, device, make_bigwigs, threads, input_already_demuxed):
    """
    Processes sequencing data from a conversion SMF experiment to an adata object.

    Parameters:
        fasta (str): File path to the reference genome to align to.
        output_directory (str): A file path to the directory to output all the analyses.
        conversion_type (list): A list of strings of the conversion types to use in the analysis.
        strands (list): A list of converstion strands to use in the experiment.
        model_dir (str): a string representing the file path to the dorado basecalling model directory.
        model (str): a string representing the dorado basecalling model.
        input_data_path (str): a string representing the file path to the experiment directory/file containing sequencing data
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mapping_threshold (float): A value in between 0 and 1 to threshold the minimal fraction of aligned reads which map to the reference region. References with values above the threshold are included in the output adata.
        experiment_name (str): A string to provide an experiment name to the output adata file.
        bam_suffix (str): A suffix to add to the bam file.
        basecall (bool): Whether to go through basecalling or not.
        barcode_both_ends (bool): Whether to require a barcode detection on both ends for demultiplexing.
        trim (bool): Whether to trim barcodes, adapters, and primers from read ends.
        device (str): Device to use for basecalling. auto, metal, cpu, cuda
        make_bigwigs (bool): Whether to make bigwigs
        threads (int): cpu threads available for processing.
        input_already_demuxed (bool): Whether the input files were already demultiplexed

    Returns:
        final_adata_path (str): Path to the final adata object
        sorted_output (str): Path to the aligned, sorted BAM
    """
    from .helpers import align_and_sort_BAM, aligned_BAM_to_bed, canoncall, converted_BAM_to_adata_II, generate_converted_FASTA, get_chromosome_lengths, demux_and_index_BAM, make_dirs, bam_qc, run_multiqc, split_and_index_BAM
    import os
    import glob
    
    if basecall:
        model_basename = os.path.basename(model)
        model_basename = model_basename.replace('.', '_')
        bam=f"{output_directory}/{model_basename}_canonical_basecalls"
    else:
        bam_base=os.path.basename(input_data_path).split('.bam')[0]
        bam=os.path.join(output_directory, bam_base)
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"

    os.chdir(output_directory)
    
    # 1) Convert FASTA file
    fasta_basename = os.path.basename(fasta)
    converted_FASTA_basename = fasta_basename.split('.fa')[0]+'_converted.fasta'
    converted_FASTA = os.path.join(output_directory, converted_FASTA_basename)
    if 'converted.fa' in fasta:
        print(fasta + ' is already converted. Using existing converted FASTA.')
        converted_FASTA = fasta
    elif os.path.exists(converted_FASTA):
        print(converted_FASTA + ' already exists. Using existing converted FASTA.')
    else:
        generate_converted_FASTA(fasta, conversion_types, strands, converted_FASTA)

    # Make a FAI and .chrom.names file for the converted fasta
    get_chromosome_lengths(converted_FASTA)

    # 2) Basecall from the input POD5 to generate a singular output BAM
    if basecall:
        canoncall_output = bam + bam_suffix
        if os.path.exists(canoncall_output):
            print(canoncall_output + ' already exists. Using existing basecalled BAM.')
        else:
            canoncall(model_dir, model, input_data_path, barcode_kit, bam, bam_suffix, barcode_both_ends, trim, device)
    else:
        canoncall_output = input_data_path

    # 3) Align the BAM to the converted reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    aligned_output = aligned_BAM + bam_suffix
    sorted_output = aligned_sorted_BAM + bam_suffix
    if os.path.exists(aligned_output) and os.path.exists(sorted_output):
        print(sorted_output + ' already exists. Using existing aligned/sorted BAM.')
    else:
        align_and_sort_BAM(converted_FASTA, canoncall_output, bam_suffix, output_directory, make_bigwigs, threads)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(output_directory, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for ' + sorted_output)
    else:
        aligned_BAM_to_bed(aligned_output, output_directory, converted_FASTA, make_bigwigs, threads)

    ### 4) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory###
    if barcode_both_ends:
        split_dir = split_dir + '_both_ends_barcoded'
    else:
        split_dir = split_dir + '_at_least_one_end_barcoded'
    
    if os.path.isdir(split_dir):
        print(split_dir + ' already exists. Using existing demultiplexed BAMs.')
        bam_pattern = '*' + bam_suffix
        bam_files = glob.glob(os.path.join(split_dir, bam_pattern))
        bam_files = [bam for bam in bam_files if '.bai' not in bam and 'unclassified' not in bam]
        bam_files.sort()
    else:
        make_dirs([split_dir])
        if input_already_demuxed:
            bam_files = split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, output_directory) # custom for non-nanopore
        else:
            bam_files = demux_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, barcode_kit, barcode_both_ends, trim, fasta, make_bigwigs, threads)
        
    # Make beds and provide basic histograms
    bed_dir = os.path.join(split_dir, 'beds')
    if os.path.isdir(bed_dir):
        print(bed_dir + ' already exists. Skipping BAM -> BED conversion for demultiplexed bams')
    else:
        for bam in bam_files:
            aligned_BAM_to_bed(bam, split_dir, converted_FASTA, make_bigwigs, threads)

    # 5) Samtools QC metrics on split BAM files
    bam_qc_dir = f"{split_dir}/bam_qc"
    if os.path.isdir(bam_qc_dir):
        print(bam_qc_dir + ' already exists. Using existing BAM QC calculations.')
    else:
        make_dirs([bam_qc_dir])
        bam_qc(bam_files, bam_qc_dir, threads, modality='conversion')

    # multiqc ###
    if os.path.isdir(f"{split_dir}/multiqc"):
        print(f"{split_dir}/multiqc" + ' already exists, skipping multiqc')
    else:
        run_multiqc(split_dir, f"{split_dir}/multiqc")

    # 6) Take the converted BAM and load it into an adata object.
    final_adata, final_adata_path = converted_BAM_to_adata_II(converted_FASTA, split_dir, mapping_threshold, experiment_name, conversion_types, bam_suffix, device)

    return final_adata, final_adata_path, sorted_output, bam_files