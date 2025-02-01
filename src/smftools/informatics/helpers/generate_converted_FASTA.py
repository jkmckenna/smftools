import numpy as np
import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from concurrent.futures import ProcessPoolExecutor

def convert_FASTA_record(record, modification_type, strand, unconverted):
    """
    Converts a FASTA record by replacing specific bases based on modification type and strand.

    Parameters:
        record (SeqRecord): The FASTA sequence record.
        modification_type (str): Modification type ('5mC', '6mA', or unconverted).
        strand (str): Strand being converted ('top' or 'bottom').
        unconverted (str): The unconverted modification type.

    Returns:
        tuple: Converted sequence and modified record ID.
    """
    # Define conversion mappings
    conversion_maps = {
        ('5mC', 'top'): ('C', 'T'),
        ('5mC', 'bottom'): ('G', 'A'),
        ('6mA', 'top'): ('A', 'G'),
        ('6mA', 'bottom'): ('T', 'C')
    }

    sequence = str(record.seq).upper()  # Convert sequence to uppercase once

    if modification_type == unconverted:
        return sequence, f"{record.id}_{modification_type}_top"

    if (modification_type, strand) not in conversion_maps:
        raise ValueError("Invalid combination of modification_type and strand")

    original_base, converted_base = conversion_maps[(modification_type, strand)]
    new_seq = sequence.replace(original_base, converted_base)  # Perform replacement

    new_id = f"{record.id}_{modification_type}_{strand}"
    return new_seq, new_id


def process_fasta_record(record, modification_types, strands, unconverted):
    """
    Processes a single FASTA record, generating converted sequences for all modification types and strands.

    Parameters:
        record (SeqRecord): A FASTA sequence record.
        modification_types (list): List of modification types ('5mC', '6mA', or unconverted).
        strands (list): List of strands ('top', 'bottom').
        unconverted (str): The unconverted modification type.

    Returns:
        list: List of modified SeqRecord objects.
    """
    modified_records = []
    record_description = record.description

    for modification_type in modification_types:
        for i, strand in enumerate(strands):
            if i > 0 and modification_type == unconverted:
                continue  # Ensure unconverted is added only once

            new_seq, new_id = convert_FASTA_record(record, modification_type, strand, unconverted)
            new_record = SeqRecord(Seq(new_seq), id=new_id, description=record_description)
            modified_records.append(new_record)

    return modified_records


def generate_converted_FASTA(input_fasta, modification_types, strands, output_fasta, num_threads=4):
    """
    Converts an input FASTA file and writes a new converted FASTA file.

    Parameters:
        input_fasta (str): Path to the unconverted FASTA file.
        modification_types (list): List of modification types ('5mC', '6mA', or unconverted).
        strands (list): List of strands ('top', 'bottom').
        output_fasta (str): Path to the converted FASTA output file.
        num_threads (int): Number of parallel threads to use.

    Returns:
        None (Writes the converted FASTA file).
    """
    modified_records = []
    unconverted = modification_types[0]

    # Determine whether the file is gzipped
    open_func = gzip.open if input_fasta.endswith('.gz') else open
    file_mode = 'rt' if input_fasta.endswith('.gz') else 'r'

    # Read the FASTA file and process records in parallel
    with open_func(input_fasta, file_mode) as handle:
        records = list(SeqIO.parse(handle, 'fasta'))

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda r: process_fasta_record(r, modification_types, strands, unconverted), records)

    # Flatten the list of lists
    modified_records = [record for sublist in results for record in sublist]

    # Write to output FASTA
    with open(output_fasta, 'w') as output_handle:
        SeqIO.write(modified_records, output_handle, 'fasta')
