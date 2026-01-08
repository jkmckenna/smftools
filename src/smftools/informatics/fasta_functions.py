import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pysam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfaidx import Fasta

from ..readwrite import time_string


def _convert_FASTA_record(record, modification_type, strand, unconverted):
    """Converts a FASTA record based on modification type and strand."""
    conversion_maps = {
        ("5mC", "top"): ("C", "T"),
        ("5mC", "bottom"): ("G", "A"),
        ("6mA", "top"): ("A", "G"),
        ("6mA", "bottom"): ("T", "C"),
    }

    sequence = str(record.seq).upper()

    if modification_type == unconverted:
        return SeqRecord(
            Seq(sequence), id=f"{record.id}_{modification_type}_top", description=record.description
        )

    if (modification_type, strand) not in conversion_maps:
        raise ValueError(f"Invalid combination: {modification_type}, {strand}")

    original_base, converted_base = conversion_maps[(modification_type, strand)]
    new_seq = sequence.replace(original_base, converted_base)

    return SeqRecord(
        Seq(new_seq), id=f"{record.id}_{modification_type}_{strand}", description=record.description
    )


def _process_fasta_record(args):
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

            modified_records.append(
                _convert_FASTA_record(record, modification_type, strand, unconverted)
            )

    return modified_records


def generate_converted_FASTA(
    input_fasta, modification_types, strands, output_fasta, num_threads=4, chunk_size=500
):
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
    input_fasta = str(input_fasta)
    output_fasta = str(output_fasta)

    # Detect if input is gzipped
    open_func = gzip.open if input_fasta.endswith(".gz") else open
    file_mode = "rt" if input_fasta.endswith(".gz") else "r"

    def _fasta_record_generator():
        """Lazily yields FASTA records from file."""
        with open_func(input_fasta, file_mode) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                yield record

    with (
        open(output_fasta, "w") as output_handle,
        ProcessPoolExecutor(max_workers=num_threads) as executor,
    ):
        # Process records in parallel using a named function (avoiding lambda)
        results = executor.map(
            _process_fasta_record,
            (
                (record, modification_types, strands, unconverted)
                for record in _fasta_record_generator()
            ),
        )

        buffer = []
        for modified_records in results:
            buffer.extend(modified_records)

            # Write out in chunks to save memory
            if len(buffer) >= chunk_size:
                SeqIO.write(buffer, output_handle, "fasta")
                buffer.clear()

        # Write any remaining records
        if buffer:
            SeqIO.write(buffer, output_handle, "fasta")


def index_fasta(fasta: str | Path, write_chrom_sizes: bool = True) -> Path:
    fasta = Path(fasta)
    pysam.faidx(str(fasta))  # creates <fasta>.fai

    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if write_chrom_sizes:
        chrom_sizes = fasta.with_suffix(".chrom.sizes")
        with fai.open() as f_in, chrom_sizes.open("w") as out:
            for line in f_in:
                chrom, size = line.split()[:2]
                out.write(f"{chrom}\t{size}\n")
        return chrom_sizes
    return fai


def get_chromosome_lengths(fasta: str | Path) -> Path:
    """
    Create (or reuse) <fasta>.chrom.sizes, derived from the FASTA index.
    """
    fasta = Path(fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if not fai.exists():
        index_fasta(fasta, write_chrom_sizes=True)  # will also create .chrom.sizes
    chrom_sizes = fasta.with_suffix(".chrom.sizes")
    if chrom_sizes.exists():
        print(f"Using existing chrom length file: {chrom_sizes}")
        return chrom_sizes

    # Build chrom.sizes from .fai
    with fai.open() as f_in, chrom_sizes.open("w") as out:
        for line in f_in:
            chrom, size = line.split()[:2]
            out.write(f"{chrom}\t{size}\n")
    return chrom_sizes


def get_native_references(fasta_file: str | Path) -> Dict[str, Tuple[int, str]]:
    """
    Return {record_id: (length, sequence)} from a FASTA.
    Direct methylation specific
    """
    fasta_file = Path(fasta_file)
    print(f"{time_string()}: Opening FASTA file {fasta_file}")
    record_dict: Dict[str, Tuple[int, str]] = {}
    with fasta_file.open("r") as f:
        for rec in SeqIO.parse(f, "fasta"):
            seq = str(rec.seq).upper()
            record_dict[rec.id] = (len(seq), seq)
    return record_dict


def find_conversion_sites(fasta_file, modification_type, conversions, deaminase_footprinting=False):
    """
    Finds genomic coordinates of modified bases (5mC or 6mA) in a reference FASTA file.

    Parameters:
        fasta_file (str): Path to the converted reference FASTA.
        modification_type (str): Modification type ('5mC' or '6mA') or 'unconverted'.
        conversions (list): List of conversion types. The first element is the unconverted record type.
        deaminase_footprinting (bool): Whether the footprinting was done with a direct deamination chemistry.

    Returns:
        dict: Dictionary where keys are **both unconverted & converted record names**.
              Values contain:
              [sequence length, top strand coordinates, bottom strand coordinates, sequence, complement sequence].
    """
    unconverted = conversions[0]
    record_dict = {}

    # Define base mapping based on modification type
    base_mappings = {
        "5mC": ("C", "G"),  # Cytosine and Guanine
        "6mA": ("A", "T"),  # Adenine and Thymine
    }

    # Read FASTA file and process records
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            if unconverted in record.id or deaminase_footprinting:
                sequence = str(record.seq).upper()
                complement = str(record.seq.complement()).upper()
                sequence_length = len(sequence)

                # Unconverted case: store the full sequence without coordinate filtering
                if modification_type == unconverted:
                    record_dict[record.id] = [sequence_length, [], [], sequence, complement]

                # Process converted records: extract modified base positions
                elif modification_type in base_mappings:
                    top_base, bottom_base = base_mappings[modification_type]
                    seq_array = np.array(list(sequence))
                    top_strand_coordinates = np.where(seq_array == top_base)[0].tolist()
                    bottom_strand_coordinates = np.where(seq_array == bottom_base)[0].tolist()

                    record_dict[record.id] = [
                        sequence_length,
                        top_strand_coordinates,
                        bottom_strand_coordinates,
                        sequence,
                        complement,
                    ]

                else:
                    raise ValueError(
                        f"Invalid modification_type: {modification_type}. Choose '5mC', '6mA', or 'unconverted'."
                    )

    return record_dict


def subsample_fasta_from_bed(
    input_FASTA: str | Path,
    input_bed: str | Path,
    output_directory: str | Path,
    output_FASTA: str | Path,
) -> None:
    """
    Take a genome-wide FASTA file and a BED file containing
    coordinate windows of interest. Outputs a subsampled FASTA.
    """

    # Normalize everything to Path
    input_FASTA = Path(input_FASTA)
    input_bed = Path(input_bed)
    output_directory = Path(output_directory)
    output_FASTA = Path(output_FASTA)

    # Ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Load the FASTA file using pyfaidx
    fasta = Fasta(str(input_FASTA))  # pyfaidx requires string paths

    # Open BED + output FASTA
    with input_bed.open("r") as bed, output_FASTA.open("w") as out_fasta:
        for line in bed:
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])  # BED is 0-based
            end = int(fields[2])  # BED is 0-based and end is exclusive
            desc = " ".join(fields[3:]) if len(fields) > 3 else ""

            if chrom not in fasta:
                print(f"Warning: {chrom} not found in FASTA")
                continue

            # pyfaidx is 1-based indexing internally, but [start:end] works with BED coords
            sequence = fasta[chrom][start:end].seq

            header = f">{chrom}:{start}-{end}"
            if desc:
                header += f"    {desc}"

            out_fasta.write(f"{header}\n{sequence}\n")
