## align_and_sort_BAM

def align_and_sort_BAM(fasta, input, bam_suffix, output_directory):
    """
    A wrapper for running dorado aligner and samtools functions
    
    Parameters:
        fasta (str): File path to the reference genome to align to.
        input (str): File path to the basecalled file to align. Works for .bam and .fastq files
        bam_suffix (str): The suffix to use for the BAM file.
        output_directory (str): A file path to the directory to output all the analyses.

    Returns:
        None
            The function writes out files for: 1) An aligned BAM, 2) and aligned_sorted BAM, 3) an index file for the aligned_sorted BAM, 4) A bed file for the aligned_sorted BAM, 5) A text file containing read names in the aligned_sorted BAM
    """
    import subprocess
    import os
    from .aligned_BAM_to_bed import aligned_BAM_to_bed
    from .extract_readnames_from_BAM import extract_readnames_from_BAM
    from .make_dirs import make_dirs
    input_basename = os.path.basename(input)
    input_suffix = '.' + input_basename.split('.')[1]

    output_path_minus_suffix = os.path.join(output_directory, input_basename.split(input_suffix)[0])
    
    aligned_BAM=f"{output_path_minus_suffix}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    aligned_output = aligned_BAM + bam_suffix
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    
    # Run dorado aligner
    subprocess.run(["dorado", "aligner", "--secondary", "no", fasta, input], stdout=open(aligned_output, "w"))

    # Sort the BAM on positional coordinates
    subprocess.run(["samtools", "sort", "-o", aligned_sorted_output, aligned_output])

    # Create a BAM index file
    subprocess.run(["samtools", "index", aligned_sorted_output])

    # Make a bed file of coordinates for the BAM
    plotting_dir = os.path.join(output_directory, 'coverage_and_readlength_histograms')
    bed_dir = os.path.join(output_directory, 'read_alignment_coordinates')
    make_dirs([plotting_dir, bed_dir])
    aligned_BAM_to_bed(aligned_sorted_output, plotting_dir, bed_dir, fasta)

    # Make a text file of reads for the BAM
    extract_readnames_from_BAM(aligned_sorted_output)