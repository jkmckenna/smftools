import numpy as np
import gzip
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

def convert_FASTA_record(record, modification_type, strand, unconverted):
    """ Converts a FASTA record based on modification type and strand. """
    conversion_maps = {
        ('5mC', 'top'): ('C', 'T'),
        ('5mC', 'bottom'): ('G', 'A'),
        ('6mA', 'top'): ('A', 'G'),
        ('6mA', 'bottom'): ('T', 'C')
    }

    sequence = str(record.seq).upper()

    if modification_type == unconverted:
        return SeqRecord(Seq(sequence), id=f"{record.id}_{modification_type}_top", description=record.description)

    if (modification_type, strand) not in conversion_maps:
        raise ValueError(f"Invalid combination: {modification_type}, {strand}")

    original_base, converted_base = conversion_maps[(modification_type, strand)]
    new_seq = sequence.replace(original_base, converted_base)

    return SeqRecord(Seq(new_seq), id=f"{record.id}_{modification_type}_{strand}", description=record.description)


def process_fasta_record(args):
    """
    Processes a single FASTA record for parallel execution.
    Args:
        args (tuple): (record, modification_types, strands, unconverted)
    Returns:
        list of modified SeqRecord objects.
    """
    record, modification_types, strands, unconverted = args
    modified_records = []
    
    for modification_type in modification_types:
        for i, strand in enumerate(strands):
            if i > 0 and modification_type == unconverted:
                continue  # Ensure unconverted is added only once

            modified_records.append(convert_FASTA_record(record, modification_type, strand, unconverted))

    return modified_records


def generate_converted_FASTA(input_fasta, modification_types, strands, output_fasta, num_threads=4, chunk_size=500):
    """
    Converts an input FASTA file and writes a new converted FASTA file efficiently.

    Parameters:
        input_fasta (str): Path to the unconverted FASTA file.
        modification_types (list): List of modification types ('5mC', '6mA', or unconverted).
        strands (list): List of strands ('top', 'bottom').
        output_fasta (str): Path to the converted FASTA output file.
        num_threads (int): Number of parallel threads to use.
        chunk_size (int): Number of records to process per write batch.

    Returns:
        None (Writes the converted FASTA file).
    """
    unconverted = modification_types[0]

    # Detect if input is gzipped
    open_func = gzip.open if input_fasta.endswith('.gz') else open
    file_mode = 'rt' if input_fasta.endswith('.gz') else 'r'

    def fasta_record_generator():
        """ Lazily yields FASTA records from file. """
        with open_func(input_fasta, file_mode) as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                yield record

    with open(output_fasta, 'w') as output_handle, ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Process records in parallel using a named function (avoiding lambda)
        results = executor.map(
            process_fasta_record,
            ((record, modification_types, strands, unconverted) for record in fasta_record_generator())
        )

        buffer = []
        for modified_records in results:
            buffer.extend(modified_records)

            # Write out in chunks to save memory
            if len(buffer) >= chunk_size:
                SeqIO.write(buffer, output_handle, 'fasta')
                buffer.clear()

        # Write any remaining records
        if buffer:
            SeqIO.write(buffer, output_handle, 'fasta')