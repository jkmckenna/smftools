from pathlib import Path
from smftools.optional_imports import require

pybedtools = require("pybedtools", extra="informatics", purpose="archived bed to bigwig")
pyBigWig = require("pyBigWig", extra="informatics", purpose="archived bed to bigwig")

def bed_to_bigwig(fasta: str, bed: str) -> str:
    """
    BED → bedGraph → bigWig
    Requires:
      - FASTA must have .fai index present
    """

    bed = Path(bed)
    fa = Path(fasta)  # path to .fa
    parent = bed.parent
    stem = bed.stem
    fa_stem = fa.stem
    fai = parent / f"{fa_stem}.fai"

    bedgraph = parent / f"{stem}.bedgraph"
    bigwig = parent / f"{stem}.bw"

    # 1) Compute coverage → bedGraph
    print(f"[pybedtools] generating coverage bedgraph from {bed}")
    bt = pybedtools.BedTool(str(bed))
    # bedtools genomecov -bg
    coverage = bt.genome_coverage(bg=True, genome=str(fai))
    coverage.saveas(str(bedgraph))

    # 2) Convert bedGraph → BigWig via pyBigWig
    print(f"[pyBigWig] converting bedgraph → bigwig: {bigwig}")

    # read chrom sizes from the FASTA .fai index
    chrom_sizes = {}
    with open(fai) as f:
        for line in f:
            fields = line.strip().split("\t")
            chrom = fields[0]
            size = int(fields[1])
            chrom_sizes[chrom] = size

    bw = pyBigWig.open(str(bigwig), "w")
    bw.addHeader(list(chrom_sizes.items()))

    with open(bedgraph) as f:
        for line in f:
            chrom, start, end, coverage = line.strip().split()
            bw.addEntries(chrom, int(start), ends=int(end), values=float(coverage))

    bw.close()

    print(f"BigWig written: {bigwig}")
    return str(bigwig)

# def bed_to_bigwig(fasta, bed):
#     """
#     Takes a bed file of reads and makes a bedgraph plus a bigwig

#     Parameters:
#         fasta (str): File path to the reference genome to align to.
#         bed (str): File path to the input bed.
#     Returns:
#         None
#     """
#     import os
#     import subprocess

#     bed_basename = os.path.basename(bed)
#     parent_dir = os.path.dirname(bed)
#     bed_basename_minus_suffix = bed_basename.split('.bed')[0]
#     fasta_basename = os.path.basename(fasta)
#     fasta_dir = os.path.dirname(fasta)
#     fasta_basename_minus_suffix = fasta_basename.split('.fa')[0]
#     chrom_basename = fasta_basename_minus_suffix + '.chrom.sizes'
#     chrom_path = os.path.join(fasta_dir, chrom_basename)
#     bedgraph_basename = bed_basename_minus_suffix + '_bedgraph.bedgraph'
#     bedgraph_output = os.path.join(parent_dir, bedgraph_basename)
#     bigwig_basename = bed_basename_minus_suffix + '_bigwig.bw'
#     bigwig_output = os.path.join(parent_dir, bigwig_basename)

#     # Make the bedgraph
#     with open(bedgraph_output, 'w') as outfile:
#         # Command as a list
#         command = ["bedtools", "genomecov", "-i", bed, "-g", chrom_path, "-bg"]
#         print(f'Making bedgraph from {bed_basename}')
#         subprocess.run(command, stdout=outfile)

#     # Make the bigwig
#     command = ["bedGraphToBigWig", bedgraph_output, chrom_path, bigwig_output]
#     print(f'Making bigwig from {bedgraph_basename}')
#     subprocess.run(command)
