## generate_converted_FASTA
from .. import readwrite
# bioinformatic operations
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def convert_FASTA_record(record, modification_type, strand):
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
                        modified_records.append(convert_FASTA_record(record, modification_type, strand))
        # write out the concatenated FASTA file of modified sequences
        SeqIO.write(modified_records, output_handle, 'fasta')