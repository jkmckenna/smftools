## find_conversion_sites

def find_conversion_sites(fasta_file, modification_type, conversion_types):
    """
    A function to find genomic coordinates in every unconverted record contained within a FASTA file of every cytosine.
    If searching for adenine conversions, it will find coordinates of all adenines.

    Parameters:
        fasta_file (str): A string representing the file path to the converted reference FASTA.
        modification_type (str): A string representing the modification type of interest (options are '5mC' and '6mA').
        conversion_types (list): A list of strings of the conversion types to use in the analysis. Used here to pass the unconverted record name.

    Returns: 
        record_dict (dict): A dictionary keyed by unconverted record ids contained within the FASTA. Points to a list containing: 1) sequence length of the record, 2) top strand coordinate list, 3) bottom strand coorinate list, 4) sequence string, 5) Complement sequence
    """
    from .. import readwrite
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    
    #print('{0}: Finding positions of interest in reference FASTA: {1}'.format(readwrite.time_string(), fasta_file))
    # Initialize lists to hold top and bottom strand positional coordinates of interest
    top_strand_coordinates = []
    bottom_strand_coordinates = []
    unconverted = conversion_types[0]
    record_dict = {}
    #print('{0}: Opening FASTA file {1}'.format(readwrite.time_string(), fasta_file))
    # Open the FASTA record as read only
    with open(fasta_file, "r") as f:
        # Iterate over records in the FASTA
        for record in SeqIO.parse(f, "fasta"):
            # Only iterate over the unconverted records for the reference
            if unconverted in record.id:
                #print('{0}: Iterating over record {1} in FASTA file {2}'.format(readwrite.time_string(), record, fasta_file))
                # Extract the sequence string of the record
                sequence = str(record.seq).upper()
                complement = str(record.seq.complement()).upper()
                sequence_length = len(sequence)
                if modification_type == '5mC':
                    # Iterate over the sequence string from the record
                    for i in range(0, len(sequence)):
                        if sequence[i] == 'C':
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                        if sequence[i] == 'G':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    #print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for all cytosines'.format(readwrite.time_string()))
                elif modification_type == '6mA':
                    # Iterate over the sequence string from the record
                    for i in range(0, len(sequence)):
                        if sequence[i] == 'A':
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                        if sequence[i] == 'T':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    #print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for adenines of interest'.format(readwrite.time_string()))          
                else:
                    #print('modification_type not found. Please try 5mC or 6mA') 
                    pass   
                record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence, complement]
            else:
                pass  
    return record_dict