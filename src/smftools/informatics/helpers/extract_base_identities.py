def extract_base_identities(bam_file, chromosome, positions, max_reference_length):
    """
    Efficiently extracts base identities from mapped reads with reference coordinates.

    Parameters:
        bam_file (str): Path to the BAM file.
        chromosome (str): Name of the reference chromosome.
        positions (list): Positions to extract (0-based).
        max_reference_length (int): Maximum reference length for padding.

    Returns:
        dict: Base identities from forward mapped reads.
        dict: Base identities from reverse mapped reads.
    """
    import pysam
    import numpy as np
    from collections import defaultdict
    import time

    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    positions = set(positions)
    fwd_base_identities = defaultdict(lambda: np.full(max_reference_length, 'N', dtype='<U1'))
    rev_base_identities = defaultdict(lambda: np.full(max_reference_length, 'N', dtype='<U1'))

    #print(f"{timestamp} Reading reads from {chromosome} BAM file: {bam_file}")
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        total_reads = bam.mapped
        for read in bam.fetch(chromosome):
            if not read.is_mapped:
                continue  # Skip unmapped reads

            read_name = read.query_name
            query_sequence = read.query_sequence
            base_dict = rev_base_identities if read.is_reverse else fwd_base_identities

            # Use get_aligned_pairs directly with positions filtering
            aligned_pairs = read.get_aligned_pairs(matches_only=True)

            for read_position, reference_position in aligned_pairs:
                if reference_position in positions:
                    base_dict[read_name][reference_position] = query_sequence[read_position]

    return dict(fwd_base_identities), dict(rev_base_identities)
