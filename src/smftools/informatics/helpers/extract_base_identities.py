def extract_base_identities(bam_file, chromosome, positions, max_reference_length, sequence):
    """
    Efficiently extracts base identities from mapped reads with reference coordinates.

    Parameters:
        bam_file (str): Path to the BAM file.
        chromosome (str): Name of the reference chromosome.
        positions (list): Positions to extract (0-based).
        max_reference_length (int): Maximum reference length for padding.
        sequence (str): The sequence of the record fasta

    Returns:
        dict: Base identities from forward mapped reads.
        dict: Base identities from reverse mapped reads.
    """
    import pysam
    import numpy as np
    from collections import defaultdict
    import time
    from collections import defaultdict, Counter

    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    positions = set(positions)
    fwd_base_identities = defaultdict(lambda: np.full(max_reference_length, 'N', dtype='<U1'))
    rev_base_identities = defaultdict(lambda: np.full(max_reference_length, 'N', dtype='<U1'))
    mismatch_counts_per_read = defaultdict(lambda: defaultdict(Counter))

    #print(f"{timestamp} Reading reads from {chromosome} BAM file: {bam_file}")
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        total_reads = bam.mapped
        ref_seq = sequence.upper()
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
                    read_base = query_sequence[read_position]
                    ref_base = ref_seq[reference_position]

                    base_dict[read_name][reference_position] = read_base

                # Track mismatches (excluding Ns)
                if read_base != ref_base and read_base != 'N' and ref_base != 'N':
                    mismatch_counts_per_read[read_name][ref_base][read_base] += 1

    # Determine C→T vs G→A dominance per read
    mismatch_trend_per_read = {}
    for read_name, ref_dict in mismatch_counts_per_read.items():
        c_to_t = ref_dict.get("C", {}).get("T", 0)
        g_to_a = ref_dict.get("G", {}).get("A", 0)

        if abs(c_to_t - g_to_a) < 0.01 and c_to_t > 0:
            mismatch_trend_per_read[read_name] = "equal"
        elif c_to_t > g_to_a:
            mismatch_trend_per_read[read_name] = "C->T"
        elif g_to_a > c_to_t:
            mismatch_trend_per_read[read_name] = "G->A"
        else:
            mismatch_trend_per_read[read_name] = "none"

    return dict(fwd_base_identities), dict(rev_base_identities), dict(mismatch_counts_per_read), mismatch_trend_per_read
