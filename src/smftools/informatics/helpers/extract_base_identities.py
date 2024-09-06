## extract_base_identities

# General
def extract_base_identities(bam_file, chromosome, positions, max_reference_length):
    """
    Extracts the base identities from every position within the read that has a reference coordinate

    Parameters:
        bam (str): File path to the BAM file to align (excluding the file suffix).
        chromosome (str): A string representing the name of the record within the reference FASTA.
        positions (list): A list of position coordinates within the record to extract.
        max_reference_length (int): The maximum length of a record in the reference set.

    Returns:
        base_identities (dict): A dictionary, keyed by read name, that points to a list of base identities. If the read does not contain that position, fill the list at that index with a N value.
    
    """
    from .. import readwrite
    import pysam
    positions = set(positions)
    # Initialize a base identity dictionary that will hold key-value pairs that are: key (read-name) and value (list of base identities at positions of interest)
    base_identities = {}
    # Open the postion sorted BAM file
    print('{0}: Reading BAM file: {1}'.format(readwrite.time_string(), bam_file))
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Iterate over every read in the bam that comes from the chromosome of interest
        print('{0}: Iterating over reads in bam'.format(readwrite.time_string()))
        for read in bam.fetch(chromosome): 
            if read.query_name in base_identities:
                pass
                #print('Duplicate read found in BAM for read {}. Skipping duplicate'.format(read.query_name))
            else:          
                # Initialize the read key in the base_identities dictionary by pointing to a N filled list of length reference_length
                base_identities[read.query_name] = ['N'] * max_reference_length
                # Iterate over a list of tuples for the given read. The tuples contain the 0-indexed position relative to the read start, as well the 0-based index relative to the reference.
                for read_position, reference_position in read.get_aligned_pairs():
                    # If the aligned read's reference coordinate is in the positions set and if the read position was successfully mapped
                    if reference_position in positions and read_position:
                        # get the base_identity in the read corresponding to that position
                        base_identity = read.query_sequence[read_position]
                        # Add the base identity to array
                        base_identities[read.query_name][reference_position] = base_identity
    return base_identities