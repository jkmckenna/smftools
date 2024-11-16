# extract_read_lengths_from_bam

def extract_read_lengths_from_bam(bam_file_path):
    """
    Make a dict of reads from a bam that points to read length
    Params:
        bam_file_path (str):
    Returns:
        read_lengths (dict)
    """
    import pysam
    # Open the BAM file
    print('Extracting read lengths from BAM')
    with pysam.AlignmentFile(bam_file_path, "rb") as bam_file:
        read_lengths = {}
        for read in bam_file:
            # Skip unmapped reads
            if read.is_unmapped:
                continue
            # Extract the read length
            read_lengths[read.query_name] = read.query_length

    return read_lengths
