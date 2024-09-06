## align_BAM

def align_BAM(fasta, bam, bam_suffix):
    """
    A wrapper for running dorado aligner and samtools functions
    
    Parameters:
        fasta (str): File path to the reference genome to align to.
        bam (str): File path to the BAM file to align (excluding the file suffix).
        bam_suffix (str): The suffix to use for the BAM file.

    Returns:
        None
            The function writes out files for: 1) An aligned BAM, 2) and aligned_sorted BAM, 3) an index file for the aligned_sorted BAM, 4) A bed file for the aligned_sorted BAM, 5) A text file containing read names in the aligned_sorted BAM
    """
    import subprocess
    
    aligned_BAM=f"{bam}_aligned"
    aligned_sorted_BAM=f"{aligned_BAM}_sorted"
    output = bam + bam_suffix
    aligned_output = aligned_BAM + bam_suffix
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    
    # Run dorado aligner
    subprocess.run([
        "dorado", "aligner",
        "--secondary=no",
        fasta,
        output
    ], stdout=open(aligned_output, "w"))

    # Sort the BAM on positional coordinates
    subprocess.run([
        "samtools", "sort",
        "-o", aligned_sorted_output,
        aligned_output
    ])

    # Create a BAM index file
    subprocess.run([
        "samtools", "index",
        aligned_sorted_output
    ])

    # Make a bed file of coordinates for the BAM
    subprocess.run([
        "samtools", "view",
        aligned_sorted_output
    ], stdout=subprocess.PIPE) | subprocess.run([
        "awk", '{print $3, $4, $4+length($10)-1}'
    ], stdin=subprocess.PIPE, stdout=open(f"{aligned_sorted_BAM}_bed.bed", "w"))

    # Make a text file of reads for the BAM
    subprocess.run([
        "samtools", "view",
        aligned_sorted_output
    ], stdout=subprocess.PIPE) | subprocess.run([
        "cut", "-f1"
    ], stdin=subprocess.PIPE, stdout=open(f"aligned_sorted_BAM_read_names.txt", "w"))