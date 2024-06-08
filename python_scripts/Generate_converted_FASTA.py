######################################################################################################
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from datetime import datetime
import os
import argparse
######################################################################################################
# Get the current date
current_date = datetime.now()
# Format the date as a string
date_string = current_date.strftime("%Y%m%d")
date_string = date_string[2:]
def time_string():
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")
######################################################################################################
# Function to modify sequence and ID
def modify_sequence_and_id(record, modification_type, strand):
    """
    Takes a FASTA record, modification type, and strand as input
    Returns a new seqrecord object with the conversions of interest
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
# Read the input FASTA file, modify sequences and IDs, and write to output FASTA file

if __name__ == "__main__":
    ### Parse Inputs ###
    parser = argparse.ArgumentParser(description="Convert FASTA files")
    parser.add_argument("input_fasta", help="FASTA file to convert")
    parser.add_argument("modification_types", help="Indicate types of modifications you want to detect")
    parser.add_argument("strands", help="Indicate strands that could have been converted in your experiment (relative to the FASTA orientation)")
    parser.add_argument("output_fasta", help="converted FASTA file output path")
    args = parser.parse_args()
    modification_types = args.modification_types
    strands = args.strands
    input_fasta = args.input_fasta
    output_fasta = args.output_fasta
    ####################
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
                        modified_records.append(modify_sequence_and_id(record, modification_type, strand))
        # write out the concatenated FASTA file of modified sequences
        SeqIO.write(modified_records, output_handle, 'fasta')
######################################################################################################

