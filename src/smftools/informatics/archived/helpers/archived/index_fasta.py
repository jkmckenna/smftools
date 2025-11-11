import pysam
from pathlib import Path

def index_fasta(fasta: str | Path, write_chrom_sizes: bool = True) -> Path:
    """
    Index a FASTA and optionally write <fasta>.chrom.sizes for bigwig/bedgraph work.

    Returns
    -------
    Path: path to chrom.sizes file (if requested), else .fai
    """
    fasta = Path(fasta)
    pysam.faidx(str(fasta))   # makes fasta.fai

    if write_chrom_sizes:
        fai = fasta.with_suffix(fasta.suffix + ".fai")
        chrom_sizes = fasta.with_suffix(".chrom.sizes")
        with open(fai) as f_in, open(chrom_sizes, "w") as out:
            for line in f_in:
                chrom, size = line.split()[:2]
                out.write(f"{chrom}\t{size}\n")
        return chrom_sizes

    return fasta.with_suffix(fasta.suffix + ".fai")