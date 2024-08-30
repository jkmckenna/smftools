## align_BAM
import subprocess

def align_BAM(fasta, bam, bam_suffix):
    """
    A wrapper for running dorado aligner and samtools functions
    """
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