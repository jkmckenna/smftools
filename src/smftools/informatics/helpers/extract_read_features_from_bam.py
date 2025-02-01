# extract_read_features_from_bam

def extract_read_features_from_bam(bam_file_path):
    """
    Make a dict of reads from a bam that points to a list of read metrics: read length, read median Q-score, reference length.
    Params:
        bam_file_path (str):
    Returns:
        read_metrics (dict)
    """
    import pysam
    import numpy as np
    # Open the BAM file
    print(f'Extracting read features from BAM: {bam_file_path}')
    with pysam.AlignmentFile(bam_file_path, "rb") as bam_file:
        read_metrics = {}
        reference_lengths = bam_file.lengths  # List of lengths for each reference (chromosome)
        for read in bam_file:
            # Skip unmapped reads
            if read.is_unmapped:
                continue
            # Extract the read metrics
            read_quality = read.query_qualities
            median_read_quality = np.median(read_quality)
            # Extract the reference (chromosome) name and its length
            reference_name = read.reference_name
            reference_index = bam_file.references.index(reference_name)
            reference_length = reference_lengths[reference_index]
            read_metrics[read.query_name] = [read.query_length, median_read_quality, reference_length]

    return read_metrics
