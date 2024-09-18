# index_fasta

def index_fasta(fasta):
    """
    Generate a FASTA index file for an input fasta.

    Parameters:
        fasta (str): Path to the input fasta to make an index file for.
    """
    import subprocess

    subprocess.run(["samtools", "faidx", fasta])