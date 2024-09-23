# get_chromosome_lengths

def get_chromosome_lengths(fasta):
    """
    Generates a file containing chromosome lengths within an input FASTA.

    Parameters:
        fasta (str): Path to the input fasta
    """
    import os
    import subprocess
    from .index_fasta import index_fasta

    # Make a fasta index file if one isn't already available
    index_path = f'{fasta}.fai'
    if os.path.exists(index_path):
        print(f'Using existing fasta index file: {index_path}')
    else:
        index_fasta(fasta)

    parent_dir = os.path.dirname(fasta)
    fasta_basename = os.path.basename(fasta)
    chrom_basename = fasta_basename.split('.fa')[0] + '.chrom.sizes'
    chrom_path = os.path.join(parent_dir, chrom_basename)

    # Make a chromosome length file
    if os.path.exists(chrom_path):
        print(f'Using existing chrom length index file: {chrom_path}')
    else:
        with open(chrom_path, 'w') as outfile:
            command = ["cut", "-f1,2", index_path]
            subprocess.run(command, stdout=outfile)
