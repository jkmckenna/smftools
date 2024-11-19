# extract_read_features_from_bam

def extract_read_features_from_bam(bam_file_path):
    """
    Make a dict of reads from a bam that points to a list of read metrics: read length, read median Q-score
    Params:
        bam_file_path (str):
    Returns:
        read_metrics (dict)
    """
    import pysam
    import numpy as np
    # Open the BAM file
    print('Extracting read features from BAM')
    with pysam.AlignmentFile(bam_file_path, "rb") as bam_file:
        read_metrics = {}
        for read in bam_file:
            # Skip unmapped reads
            if read.is_unmapped:
                continue
            # Extract the read metrics
            read_quality = read.query_qualities
            median_read_quality = np.median(read_quality)
            read_metrics[read.query_name] = [read.query_length, median_read_quality]

    return read_metrics
