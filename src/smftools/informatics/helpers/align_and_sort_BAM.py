## align_and_sort_BAM

def align_and_sort_BAM(fasta, input, bam_suffix='.bam', output_directory='aligned_outputs', make_bigwigs=False, threads=None):
    """
    A wrapper for running dorado aligner and samtools functions
    
    Parameters:
        fasta (str): File path to the reference genome to align to.
        input (str): File path to the basecalled file to align. Works for .bam and .fastq files
        bam_suffix (str): The suffix to use for the BAM file.
        output_directory (str): A file path to the directory to output all the analyses.
        make_bigwigs (bool): Whether to make bigwigs
        threads (int): Number of additional threads to use

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

    if threads:
        threads = str(threads)
    else:
        pass
    
    # Run dorado aligner
    print(f"Aligning BAM to Reference: {input}")
    if threads:
        alignment_command = ["dorado", "aligner", "-t", threads, '--mm2-opts', "-N 1", fasta, input]
    else:
        alignment_command = ["dorado", "aligner", '--mm2-opts', "-N 1", fasta, input]
    subprocess.run(alignment_command, stdout=open(aligned_output, "w"))

    # Sort the BAM on positional coordinates
    print(f"Sorting BAM: {aligned_output}")
    if threads:
        sort_command = ["samtools", "sort", "-@", threads, "-o", aligned_sorted_output, aligned_output]
    else:
        sort_command = ["samtools", "sort", "-o", aligned_sorted_output, aligned_output]
    subprocess.run(sort_command)

    # Create a BAM index file
    print(f"Indexing BAM: {aligned_sorted_output}")
    if threads:
        index_command = ["samtools", "index", "-@", threads, aligned_sorted_output]
    else:
        index_command = ["samtools", "index", aligned_sorted_output]
    subprocess.run(index_command)