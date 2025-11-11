# extract_readnames_from_BAM

def extract_readnames_from_BAM(aligned_BAM):
    """
    Takes a BAM and writes out a txt file containing read names from the BAM

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract read names from.

    Returns:
        None

    """
    import subprocess
    # Make a text file of reads for the BAM
    txt_output = aligned_BAM.split('.bam')[0] + '_read_names.txt'
    samtools_view = subprocess.Popen(["samtools", "view", aligned_BAM], stdout=subprocess.PIPE)
    with open(txt_output, "w") as output_file:
        cut_process = subprocess.Popen(["cut", "-f1"], stdin=samtools_view.stdout, stdout=output_file)   
    samtools_view.stdout.close()
    cut_process.wait()
    samtools_view.wait()
