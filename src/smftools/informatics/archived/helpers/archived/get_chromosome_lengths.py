# get_chromosome_lengths

def get_chromosome_lengths(fasta):
    """
    Generates a file containing chromosome lengths within an input FASTA.

    Parameters:
        fasta (str): Path to the input fasta
    """
    import os
    from pathlib import Path
    import subprocess
    from .index_fasta import index_fasta

    # Make a fasta index file if one isn't already available
    index_path = fasta / '.fai'
    if index_path.exists():
        print(f'Using existing fasta index file: {index_path}')
    else:
        index_fasta(fasta)

    parent_dir = fasta.parent
    fasta_basename = fasta.name
    chrom_basename = fasta.stem + '.chrom.sizes'
    chrom_path = parent_dir / chrom_basename

    # Make a chromosome length file
    if chrom_path.exists():
        print(f'Using existing chrom length index file: {chrom_path}')
    else:
        with open(chrom_path, 'w') as outfile:
            command = ["cut", "-f1,2", str(index_path)]
            subprocess.run(command, stdout=outfile)
