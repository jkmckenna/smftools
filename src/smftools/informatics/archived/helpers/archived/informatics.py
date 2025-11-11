## fasta_module
from .. import readwrite
# bioinformatic operations
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pysam

######################################################################################################
## FASTA functionality
# General

# Conversion specific
def modify_sequence_and_id(record, modification_type, strand):
    """
    Input: Takes a FASTA record, modification type, and strand as input
    Output: Returns a new seqrecord object with the conversions of interest
    """
    if modification_type == '5mC':
        if strand == 'top':
            # Replace every 'C' with 'T' in the sequence
            new_seq = record.seq.upper().replace('C', 'T')
        elif strand == 'bottom':
            # Replace every 'G' with 'A' in the sequence
            new_seq = record.seq.upper().replace('G', 'A')
        else:
            print('need to provide a valid strand string: top or bottom')        
    elif modification_type == '6mA':
        if strand == 'top':
            # Replace every 'A' with 'G' in the sequence
            new_seq = record.seq.upper().replace('A', 'G')
        elif strand == 'bottom':
            # Replace every 'T' with 'C' in the sequence
            new_seq = record.seq.upper().replace('T', 'C')
        else:
            print('need to provide a valid strand string: top or bottom')
    elif modification_type == 'unconverted':
        new_seq = record.seq.upper()
    else:
        print('need to provide a valid modification_type string: 5mC, 6mA, or unconverted')   
    new_id = '{0}_{1}_{2}'.format(record.id, modification_type, strand)      
    # Return a new SeqRecord with modified sequence and ID
    return record.__class__(new_seq, id=new_id, description=record.description)

def generate_converted_FASTA(input_fasta, modification_types, strands, output_fasta):
    """
    Input: Takes an input FASTA, modification types of interest, strands of interest, and an output FASTA name 
    Output: Writes out a new fasta with all stranded conversions
    Notes: Uses modify_sequence_and_id function on every record within the FASTA
    """
    with open(output_fasta, 'w') as output_handle:
        modified_records = []
        # Iterate over each record in the input FASTA
        for record in SeqIO.parse(input_fasta, 'fasta'):
            # Iterate over each modification type of interest
            for modification_type in modification_types:
                # Iterate over the strands of interest
                for i, strand in enumerate(strands):
                    if i > 0 and modification_type == 'unconverted': # This ensures that the unconverted only is added once and takes on the strand that is provided at the 0 index on strands.
                        pass
                    else:
                        # Add the modified record to the list of modified records
                        print(f'converting {modification_type} on the {strand} strand of record {record}')
                        modified_records.append(modify_sequence_and_id(record, modification_type, strand))
        # write out the concatenated FASTA file of modified sequences
        SeqIO.write(modified_records, output_handle, 'fasta')

def find_coordinates(fasta_file, modification_type):
    """
    A function to find genomic coordinates in every unconverted record contained within a FASTA file of every cytosine.
    If searching for adenine conversions, it will find coordinates of all adenines.
    Input: A FASTA file and the modification_types of interest
    Returns: 
    A dictionary called record_dict, which is keyed by unconverted record ids contained within the FASTA. Points to a list containing: 1) sequence length of the record, 2) top strand coordinate list, 3) bottom strand coorinate list, 4) sequence string
    """
    print('{0}: Finding positions of interest in reference FASTA > {1}'.format(time_string(), fasta_file))
    # Initialize lists to hold top and bottom strand positional coordinates of interest
    top_strand_coordinates = []
    bottom_strand_coordinates = []
    record_dict = {}
    print('{0}: Opening FASTA file {1}'.format(time_string(), fasta_file))
    # Open the FASTA record as read only
    with open(fasta_file, "r") as f:
        # Iterate over records in the FASTA
        for record in SeqIO.parse(f, "fasta"):
            # Only iterate over the unconverted records for the reference
            if 'unconverted' in record.id:
                print('{0}: Iterating over record {1} in FASTA file {2}'.format(time_string(), record, fasta_file))
                # Extract the sequence string of the record
                sequence = str(record.seq).upper()
                sequence_length = len(sequence)
                if modification_type == '5mC':
                    # Iterate over the sequence string from the record
                    for i in range(0, len(sequence)):
                        if sequence[i] == 'C':
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                        if sequence[i] == 'G':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for all cytosines'.format(time_string()))
                elif modification_type == '6mA':
                    # Iterate over the sequence string from the record
                    for i in range(0, len(sequence)):
                        if sequence[i] == 'A':
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                        if sequence[i] == 'T':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for adenines of interest'.format(time_string()))          
                else:
                    print('modification_type not found. Please try 5mC or 6mA')    
                record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence]
            else:
                pass  
    return record_dict

# Direct methylation specific
def get_references(fasta_file):
    """
    Input: A FASTA file
    Returns: 
    A dictionary called record_dict, which is keyed by record ids contained within the FASTA. Points to a list containing: 1) sequence length of the record, 2) sequence of the record
    """
    record_dict = {}
    print('{0}: Opening FASTA file {1}'.format(time_string(), fasta_file))
    # Open the FASTA record as read only
    with open(fasta_file, "r") as f:
        # Iterate over records in the FASTA
        for record in SeqIO.parse(f, "fasta"):
            # Extract the sequence string of the record
            sequence = str(record.seq).upper()
            sequence_length = len(sequence) 
            record_dict[record.id] = [sequence_length, sequence]
    return record_dict
######################################################################################################

######################################################################################################
## BAM functionality
# General
def separate_bam_by_bc(input_bam, output_prefix):
    """
    Input: Takes a single BAM input. Also takes an output prefix to append to the output file.
    Output: Splits the BAM based on the BC SAM tag value.
    """
    # Open the input BAM file for reading
    with pysam.AlignmentFile(input_bam, "rb") as bam:
        # Create a dictionary to store output BAM files
        output_files = {}
        # Iterate over each read in the BAM file
        for read in bam:
            try:
                # Get the barcode tag value
                bc_tag = read.get_tag("BC", with_value_type=True)[0].split('barcode')[1]
                # Open the output BAM file corresponding to the barcode
                if bc_tag not in output_files:
                    output_files[bc_tag] = pysam.AlignmentFile(f"{output_prefix}_{bc_tag}.bam", "wb", header=bam.header)
                # Write the read to the corresponding output BAM file
                output_files[bc_tag].write(read)
            except KeyError:
                 print(f"BC tag not present for read: {read.query_name}")
    # Close all output BAM files
    for output_file in output_files.values():
        output_file.close()

def count_aligned_reads(bam_file):
    """
    Input: A BAM alignment file.
    Output: The number of aligned/unaligned reads in the BAM file. Also returns a dictionary, keyed by reference id that points to a tuple. The tuple contains an integer number of mapped reads to that reference, followed by the proportion of mapped reads that map to that reference
    """
    print('{0}: Counting aligned reads in BAM > {1}'.format(time_string(), bam_file))
    aligned_reads_count = 0
    unaligned_reads_count = 0
    # Make a dictionary, keyed by the reference_name of reference chromosome that points to an integer number of read counts mapped to the chromosome, as well as the proportion of mapped reads in that chromosome
    record_counts = {}
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Iterate over reads to get the total mapped read counts and the reads that map to each reference
        for read in bam:
            if read.is_unmapped: 
                unaligned_reads_count += 1
            else: 
                aligned_reads_count += 1
                if read.reference_name in record_counts:
                    record_counts[read.reference_name] += 1
                else:
                    record_counts[read.reference_name] = 1
        # reformat the dictionary to contain read counts mapped to the reference, as well as the proportion of mapped reads in reference
        for reference in record_counts:
            proportion_mapped_reads_in_record = record_counts[reference] / aligned_reads_count
            record_counts[reference] = (record_counts[reference], proportion_mapped_reads_in_record)
    return aligned_reads_count, unaligned_reads_count, record_counts

def extract_base_identity_at_coordinates(bam_file, chromosome, positions, max_reference_length):
    """
    Input: A position sorted BAM file, chromosome number, position coordinate set, and reference length to extract the base identitity from the read.
    Output: A dictionary, keyed by read name, that points to a list of Base identities from each read.
    If the read does not contain that position, fill the list at that index with a N value.
    """
    positions = set(positions)
    # Initialize a base identity dictionary that will hold key-value pairs that are: key (read-name) and value (list of base identities at positions of interest)
    base_identities = {}
    # Open the postion sorted BAM file
    print('{0}: Reading BAM file: {1}'.format(time_string(), bam_file))
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Iterate over every read in the bam that comes from the chromosome of interest
        print('{0}: Iterating over reads in bam'.format(time_string()))
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

# Conversion SMF specific
def binarize_converted_base_identities(base_identities, strand, modification_type):
    """
    Input: The base identities dictionary returned by extract_base_identity_at_coordinates.
    Output: A binarized format of the dictionary, where 1 represents a methylated site. 0 represents an unmethylated site. NaN represents a site that does not carry SMF information.
    """
    binarized_base_identities = {}
    # Iterate over base identity keys to binarize the base identities
    for key in base_identities.keys():
        if strand == 'top':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'C' else 0 if x == 'T' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'A' else 0 if x == 'G' else np.nan for x in base_identities[key]]
        elif strand == 'bottom':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'G' else 0 if x == 'A' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'T' else 0 if x == 'C' else np.nan for x in base_identities[key]]
        else:
            pass
    return binarized_base_identities

# Direct methylation specific

######################################################################################################

######################################################################################################
# String encodings
def one_hot_encode(sequence):
    """
    Input: A sequence string of a read.
    Output: One hot encoding of the sequence string.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot_matrix = np.zeros((len(sequence), 5), dtype=int)
    for i, nucleotide in enumerate(sequence):
        one_hot_matrix[i, mapping[nucleotide]] = 1
    return one_hot_matrix
######################################################################################################