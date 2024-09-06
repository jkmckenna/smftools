## generate_converted_FASTA

def convert_FASTA_record(record, modification_type, strand):
    """
    Takes a FASTA record and converts every instance of a base to the converted state.

    Parameters:
        record (str): The name of the record instance within the FASTA.
        modification_type (str): The modification type to convert for (options are '5mC' and '6mA').
        strand (str): The strand that is being converted in the experiment (options are 'top' and 'bottom').
    Returns:
        new_seq (str): Converted sequence string.
        new_id (str): Record id for the converted sequence string.
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
    return new_seq, new_id

def generate_converted_FASTA(input_fasta, modification_types, strands, output_fasta):
    """
    Uses modify_sequence_and_id function on every record within the FASTA to write out a converted FASTA.

    Parameters:
        input_FASTA (str): A string representing the path to the unconverted FASTA file.
        modification_types (list): A list of modification types to use in the experiment.
        strands (list): A list of converstion strands to use in the experiment.
        output_FASTA (str): A string representing the path to the converted FASTA output file.
    Returns:
        None
        Writes out a converted FASTA reference for the experiment.
    """
    from .. import readwrite
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
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
                    new_seq, new_id = convert_FASTA_record(record, modification_type, strand)
                    new_record = SeqRecord(Seq(new_seq), id=new_id)
                    modified_records.append(new_record)
    with open(output_fasta, 'w') as output_handle:
        # write out the concatenated FASTA file of modified sequences
        SeqIO.write(modified_records, output_handle, 'fasta')