from pathlib import Path
from pyfaidx import Fasta

def subsample_fasta_from_bed(
    input_FASTA: str | Path,
    input_bed: str | Path,
    output_directory: str | Path,
    output_FASTA: str | Path
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

    output_FASTA_path = output_directory / output_FASTA

    # Load the FASTA file using pyfaidx
    fasta = Fasta(str(input_FASTA))   # pyfaidx requires string paths

    # Open BED + output FASTA
    with input_bed.open("r") as bed, output_FASTA_path.open("w") as out_fasta:
        for line in bed:
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1]) # BED is 0-based
            end   = int(fields[2]) # BED is 0-based and end is exclusive
            desc  = " ".join(fields[3:]) if len(fields) > 3 else ""

            if chrom not in fasta:
                print(f"Warning: {chrom} not found in FASTA")
                continue

            # pyfaidx is 1-based indexing internally, but [start:end] works with BED coords
            sequence = fasta[chrom][start:end].seq

            header = f">{chrom}:{start}-{end}"
            if desc:
                header += f"    {desc}"

            out_fasta.write(f"{header}\n{sequence}\n")