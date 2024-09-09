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
    input_basename = os.path.basename(input)
    input_suffix = '.' + input_basename.split('.')[1]

    output_path_minus_suffix = os.path.join(output_directory, input_basename.split(input_suffix)[0])
    
    aligned_BAM=f"{output_path_minus_suffix}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    aligned_output = aligned_BAM + bam_suffix
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    
    # Run dorado aligner
    subprocess.run(["dorado", "aligner", "--secondary=no", fasta, input], stdout=open(aligned_output, "w"))

    # Sort the BAM on positional coordinates
    subprocess.run(["samtools", "sort", "-o", aligned_sorted_output, aligned_output])

    # Create a BAM index file
    subprocess.run(["samtools", "index", aligned_sorted_output])

    # Make a bed file of coordinates for the BAM
    samtools_view = subprocess.Popen(["samtools", "view", aligned_sorted_output], stdout=subprocess.PIPE) 
    with open(f"{aligned_sorted_BAM}_bed.bed", "w") as output_file:
        awk_process = subprocess.Popen(["awk", '{print $3, $4, $4+length($10)-1}'], stdin=samtools_view.stdout, stdout=output_file)    
    samtools_view.stdout.close()
    awk_process.wait()
    samtools_view.wait()

    # Make a text file of reads for the BAM
    samtools_view = subprocess.Popen(["samtools", "view", aligned_sorted_output], stdout=subprocess.PIPE)
    with open(f"{aligned_sorted_BAM}_read_names.txt", "w") as output_file:
        cut_process = subprocess.Popen(["cut", "-f1"], stdin=samtools_view.stdout, stdout=output_file)   
    samtools_view.stdout.close()
    cut_process.wait()
    samtools_view.wait()
