from __future__ import annotations

import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from smftools.logging_utils import get_logger

from ..readwrite import time_string

logger = get_logger(__name__)

try:
    import pysam
except Exception:
    pysam = None  # type: ignore

try:
    import shutil
    import subprocess
except Exception:  # pragma: no cover - stdlib
    shutil = None  # type: ignore
    subprocess = None  # type: ignore


def _resolve_fasta_backend() -> str:
    """Resolve the backend to use for FASTA access."""
    if shutil is not None and shutil.which("samtools"):
        return "cli"
    if pysam is not None:
        return "python"
    raise RuntimeError("FASTA access requires pysam or samtools in PATH.")


def _ensure_fasta_index(fasta: Path) -> None:
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        return
    if subprocess is None or shutil is None or not shutil.which("samtools"):
        if pysam is not None:
            pysam.faidx(str(fasta))
            return
        raise RuntimeError("FASTA indexing requires pysam or samtools in PATH.")
    cp = subprocess.run(
        ["samtools", "faidx", str(fasta)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"samtools faidx failed (exit {cp.returncode}):\n{cp.stderr}")


def _bed_to_faidx_region(chrom: str, start: int, end: int) -> str:
    """Convert 0-based half-open BED coords to samtools faidx region."""
    start1 = start + 1
    end1 = end
    if start1 > end1:
        start1, end1 = end1, start1
    return f"{chrom}:{start1}-{end1}"


def _fetch_sequence_with_samtools(fasta: Path, chrom: str, start: int, end: int) -> str:
    if subprocess is None or shutil is None:
        raise RuntimeError("samtools backend is unavailable.")
    if not shutil.which("samtools"):
        raise RuntimeError("samtools is required but not available in PATH.")
    region = _bed_to_faidx_region(chrom, start, end)
    cp = subprocess.run(
        ["samtools", "faidx", str(fasta), region],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"samtools faidx failed (exit {cp.returncode}):\n{cp.stderr}")
    lines = [line.strip() for line in cp.stdout.splitlines() if line and not line.startswith(">")]
    return "".join(lines)


def _convert_FASTA_record(
    record: SeqRecord,
    modification_type: str,
    strand: str,
    unconverted: str,
) -> SeqRecord:
    """Convert a FASTA record based on modification type and strand.

    Args:
        record: Input FASTA record.
        modification_type: Modification type (e.g., ``5mC`` or ``6mA``).
        strand: Strand label (``top`` or ``bottom``).
        unconverted: Label for the unconverted record type.

    Returns:
        Bio.SeqRecord.SeqRecord: Converted FASTA record.

    Raises:
        ValueError: If the modification type/strand combination is invalid.
    """
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


def _process_fasta_record(
    args: tuple[SeqRecord, Iterable[str], Iterable[str], str],
) -> list[SeqRecord]:
    """Process a single FASTA record for parallel conversion.

    Args:
        args: Tuple containing ``(record, modification_types, strands, unconverted)``.

    Returns:
        list[Bio.SeqRecord.SeqRecord]: Converted FASTA records.
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
    input_fasta: str | Path,
    modification_types: list[str],
    strands: list[str],
    output_fasta: str | Path,
    num_threads: int = 4,
    chunk_size: int = 500,
) -> None:
    """Convert a FASTA file and write converted records to disk.

    Args:
        input_fasta: Path to the unconverted FASTA file.
        modification_types: List of modification types (``5mC``, ``6mA``, or unconverted).
        strands: List of strands (``top``, ``bottom``).
        output_fasta: Path to the converted FASTA output file.
        num_threads: Number of parallel workers to use.
        chunk_size: Number of records to process per write batch.
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
    """Index a FASTA file and optionally write chromosome sizes.

    Args:
        fasta: Path to the FASTA file.
        write_chrom_sizes: Whether to write a ``.chrom.sizes`` file.

    Returns:
        Path: Path to the index file or chromosome sizes file.
    """
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
    """Create or reuse ``<fasta>.chrom.sizes`` derived from the FASTA index.

    Args:
        fasta: Path to the FASTA file.

    Returns:
        Path: Path to the chromosome sizes file.
    """
    fasta = Path(fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if not fai.exists():
        index_fasta(fasta, write_chrom_sizes=True)  # will also create .chrom.sizes
    chrom_sizes = fasta.with_suffix(".chrom.sizes")
    if chrom_sizes.exists():
        logger.debug(f"Using existing chrom length file: {chrom_sizes}")
        return chrom_sizes

    # Build chrom.sizes from .fai
    with fai.open() as f_in, chrom_sizes.open("w") as out:
        for line in f_in:
            chrom, size = line.split()[:2]
            out.write(f"{chrom}\t{size}\n")
    return chrom_sizes


def get_native_references(fasta_file: str | Path) -> Dict[str, Tuple[int, str]]:
    """Return record lengths and sequences from a FASTA file.

    Args:
        fasta_file: Path to the FASTA file.

    Returns:
        dict[str, tuple[int, str]]: Mapping of record ID to ``(length, sequence)``.
    """
    fasta_file = Path(fasta_file)
    print(f"{time_string()}: Opening FASTA file {fasta_file}")
    record_dict: Dict[str, Tuple[int, str]] = {}
    with fasta_file.open("r") as f:
        for rec in SeqIO.parse(f, "fasta"):
            seq = str(rec.seq).upper()
            record_dict[rec.id] = (len(seq), seq)
    return record_dict


def find_conversion_sites(
    fasta_file: str | Path,
    modification_type: str,
    conversions: list[str],
    deaminase_footprinting: bool = False,
) -> dict[str, list]:
    """Find genomic coordinates of modified bases in a reference FASTA.

    Args:
        fasta_file: Path to the converted reference FASTA.
        modification_type: Modification type (``5mC``, ``6mA``, or ``unconverted``).
        conversions: List of conversion types (first entry is the unconverted record type).
        deaminase_footprinting: Whether the footprinting used direct deamination chemistry.

    Returns:
        dict[str, list]: Mapping of record name to
        ``[sequence length, top strand coordinates, bottom strand coordinates, sequence, complement]``.

    Raises:
        ValueError: If the modification type is invalid.
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
    """Subsample a FASTA using BED coordinates.

    Args:
        input_FASTA: Genome-wide FASTA path.
        input_bed: BED file path containing coordinate windows of interest.
        output_directory: Directory to write the subsampled FASTA.
        output_FASTA: Output FASTA path.
    """

    # Normalize everything to Path
    input_FASTA = Path(input_FASTA)
    input_bed = Path(input_bed)
    output_directory = Path(output_directory)
    output_FASTA = Path(output_FASTA)

    # Ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    backend = _resolve_fasta_backend()
    _ensure_fasta_index(input_FASTA)

    fasta_handle = None
    if backend == "python":
        assert pysam is not None
        fasta_handle = pysam.FastaFile(str(input_FASTA))

    # Open BED + output FASTA
    with input_bed.open("r") as bed, output_FASTA.open("w") as out_fasta:
        for line in bed:
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])  # BED is 0-based
            end = int(fields[2])  # BED is 0-based and end is exclusive
            desc = " ".join(fields[3:]) if len(fields) > 3 else ""

            if backend == "python":
                assert fasta_handle is not None
                if chrom not in fasta_handle.references:
                    logger.warning(f"{chrom} not found in FASTA")
                    continue
                sequence = fasta_handle.fetch(chrom, start, end)
            else:
                sequence = _fetch_sequence_with_samtools(input_FASTA, chrom, start, end)

            if not sequence:
                logger.warning(f"{chrom} not found in FASTA")
                continue

            header = f">{chrom}:{start}-{end}"
            if desc:
                header += f"    {desc}"

            out_fasta.write(f"{header}\n{sequence}\n")

    if fasta_handle is not None:
        fasta_handle.close()
